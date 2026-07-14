import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from easy_agentic_data.agent import AgentBudgets
from easy_agentic_data.config import LLMConfig
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.models import LLMResponse, Message, stable_id
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_rollouts import (
    RolloutArtifactPaths,
    RolloutValidationReceipt,
    publish_registry_rollout,
    run_registry_rollout,
)
from easy_agentic_data.sandbox import CommandResult, MemorySandbox
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed
from easy_agentic_data.traces import replay_trace


class RegistryRolloutTests(unittest.TestCase):
    def _stage_rollout(self, root: Path, name: str = "job.jsonl"):
        registry = ScenarioRegistry(root / "registry")
        scenario = _scenario()
        registry.add_scenario(scenario)
        return run_registry_rollout(
            registry,
            scenario.scenario_id,
            _config(),
            root / "staging-targets" / name,
            random_seed=0,
            budgets=AgentBudgets(max_turns=4),
            sandbox_factory=lambda _scenario, _source: _PatchMemorySandbox(),
            client_builder=lambda _config: _RepairClient(),
            source_materializer=_empty_source,
            publish=False,
        )

    def test_staged_rollout_is_not_canonical_until_explicit_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ScenarioRegistry(root / "registry")
            scenario = _scenario()
            registry.add_scenario(scenario)
            trace_path = root / "traces" / "job.jsonl"
            staged = run_registry_rollout(
                registry,
                scenario.scenario_id,
                _config(),
                trace_path,
                random_seed=0,
                budgets=AgentBudgets(max_turns=4),
                sandbox_factory=lambda _scenario, _source: _PatchMemorySandbox(),
                client_builder=lambda _config: _RepairClient(),
                source_materializer=_empty_source,
                publish=False,
            )

            self.assertFalse(trace_path.exists())
            self.assertTrue(staged.artifacts.trace.is_file())
            self.assertTrue(staged.artifacts.private_evaluation.is_file())

            published = publish_registry_rollout(staged, trace_path)

            self.assertTrue(trace_path.is_file())
            self.assertEqual(published.trace.path, trace_path)
            self.assertEqual(published.trace.trace_id, staged.trace.trace_id)
            self.assertTrue(published.artifacts.private_evaluation.is_file())

    def test_nonpilot_staged_rollout_cannot_enter_reserved_pilot_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._stage_rollout(root)
            canonical = root / "traces" / f"rollout_{'a' * 20}.jsonl"

            with self.assertRaisesRegex(ValueError, "validation receipt"):
                publish_registry_rollout(staged, canonical)

            self.assertFalse(canonical.exists())
            self.assertTrue(staged.artifacts.trace.is_file())
            self.assertTrue(staged.artifacts.candidate_patch.is_file())

    def test_pilot_evidence_requires_a_strict_validator_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._stage_rollout(root)
            contract_id = "pilot_publication_test"
            _set_run_contract_id(staged.artifacts.run_evidence, contract_id)
            canonical = root / "traces" / "job.jsonl"

            with self.assertRaisesRegex(ValueError, "validation receipt"):
                publish_registry_rollout(staged, canonical)

            with self.assertRaisesRegex(TypeError, "strict validator"):
                RolloutValidationReceipt(
                    contract_id=contract_id,
                    job_id=canonical.stem,
                    trace_id=staged.trace.trace_id,
                    artifact_sha256={
                        "trace": "a" * 64,
                        "candidate_patch": "b" * 64,
                        "private_evaluation": "c" * 64,
                        "run_evidence": "d" * 64,
                    },
                )

            self.assertFalse(canonical.exists())

    def test_existing_sidecar_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._stage_rollout(root)
            canonical = root / "traces" / "job.jsonl"
            canonical_artifacts = RolloutArtifactPaths.for_trace(canonical)
            canonical_artifacts.private_evaluation.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            original = b"pre-existing-private-evaluation\n"
            canonical_artifacts.private_evaluation.write_bytes(original)

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish_registry_rollout(staged, canonical)

            self.assertEqual(canonical_artifacts.private_evaluation.read_bytes(), original)
            self.assertFalse(canonical.exists())
            self.assertFalse(canonical_artifacts.candidate_patch.exists())
            self.assertFalse(canonical_artifacts.run_evidence.exists())
            self.assertTrue(staged.artifacts.trace.is_file())

    def test_matching_partial_sidecars_are_resumed_before_trace_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._stage_rollout(root)
            canonical = root / "traces" / "job.jsonl"
            canonical_artifacts = RolloutArtifactPaths.for_trace(canonical)
            partial_pairs = (
                (
                    staged.artifacts.candidate_patch,
                    canonical_artifacts.candidate_patch,
                ),
                (
                    staged.artifacts.private_evaluation,
                    canonical_artifacts.private_evaluation,
                ),
            )
            original_inodes = {}
            for source, destination in partial_pairs:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, destination)
                original_inodes[destination] = destination.stat().st_ino

            published = publish_registry_rollout(staged, canonical)

            self.assertTrue(published.artifacts.trace.is_file())
            self.assertTrue(published.artifacts.run_evidence.is_file())
            for destination, inode in original_inodes.items():
                self.assertEqual(destination.stat().st_ino, inode)
            self.assertFalse(staged.artifacts.trace.exists())

    def test_same_content_link_race_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._stage_rollout(root)
            canonical = root / "traces" / "job.jsonl"
            canonical_artifacts = RolloutArtifactPaths.for_trace(canonical)
            raced_destination = canonical_artifacts.candidate_patch
            real_link = os.link
            raced = False

            def publish_concurrently(source, destination, *, follow_symlinks=True):
                nonlocal raced
                if Path(destination) == raced_destination and not raced:
                    raced = True
                    real_link(
                        source,
                        destination,
                        follow_symlinks=follow_symlinks,
                    )
                    raise FileExistsError("simulated same-content publication race")
                return real_link(
                    source,
                    destination,
                    follow_symlinks=follow_symlinks,
                )

            with patch(
                "easy_agentic_data.registry_rollouts.os.link",
                side_effect=publish_concurrently,
            ):
                published = publish_registry_rollout(staged, canonical)

            self.assertTrue(raced)
            self.assertTrue(published.artifacts.trace.is_file())
            self.assertTrue(published.artifacts.candidate_patch.is_file())

    def test_existing_symlink_sidecar_is_rejected_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._stage_rollout(root)
            canonical = root / "traces" / "job.jsonl"
            canonical_artifacts = RolloutArtifactPaths.for_trace(canonical)
            canonical_artifacts.candidate_patch.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            canonical_artifacts.candidate_patch.symlink_to(staged.artifacts.candidate_patch)

            with self.assertRaisesRegex(FileExistsError, "symlink"):
                publish_registry_rollout(staged, canonical)

            self.assertTrue(canonical_artifacts.candidate_patch.is_symlink())
            self.assertFalse(canonical.exists())

    def test_sidecar_directory_cannot_redirect_publication_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._stage_rollout(root)
            canonical = root / "traces" / "job.jsonl"
            canonical_artifacts = RolloutArtifactPaths.for_trace(canonical)
            canonical.parent.mkdir(parents=True)
            outside = root / "outside-candidate-patches"
            outside.mkdir()
            canonical_artifacts.candidate_patch.parent.symlink_to(
                outside,
                target_is_directory=True,
            )

            with self.assertRaisesRegex(ValueError, "outside its root"):
                publish_registry_rollout(staged, canonical)

            self.assertFalse(canonical.exists())
            self.assertEqual(list(outside.iterdir()), [])

    def test_trace_link_race_rolls_back_only_new_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._stage_rollout(root)
            canonical = root / "traces" / "job.jsonl"
            canonical_artifacts = RolloutArtifactPaths.for_trace(canonical)
            real_link = os.link

            def race_on_trace(source, destination, *, follow_symlinks=True):
                if Path(destination) == canonical:
                    raise FileExistsError("simulated trace publication race")
                return real_link(
                    source,
                    destination,
                    follow_symlinks=follow_symlinks,
                )

            with (
                patch(
                    "easy_agentic_data.registry_rollouts.os.link",
                    side_effect=race_on_trace,
                ),
                self.assertRaisesRegex(FileExistsError, "appeared during publication"),
            ):
                publish_registry_rollout(staged, canonical)

            self.assertFalse(canonical.exists())
            self.assertFalse(canonical_artifacts.candidate_patch.exists())
            self.assertFalse(canonical_artifacts.private_evaluation.exists())
            self.assertFalse(canonical_artifacts.run_evidence.exists())
            self.assertTrue(staged.artifacts.trace.is_file())

    def test_pilot_contract_rejects_execution_dependency_injection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ScenarioRegistry(root / "registry")
            scenario = _scenario()
            registry.add_scenario(scenario)
            trace_path = root / "traces" / "job.jsonl"
            rejected_overrides = (
                (
                    "sandbox_factory",
                    {"sandbox_factory": lambda _scenario, _source: _PatchMemorySandbox()},
                ),
                (
                    "client_builder",
                    {"client_builder": lambda _config: _RepairClient()},
                ),
                ("source_materializer", {"source_materializer": _empty_source}),
            )

            for parameter, overrides in rejected_overrides:
                with self.subTest(parameter=parameter):
                    with self.assertRaisesRegex(ValueError, parameter):
                        run_registry_rollout(
                            registry,
                            scenario.scenario_id,
                            _config(),
                            trace_path,
                            random_seed=0,
                            run_contract_id="pilot_contract_test",
                            publish=False,
                            **overrides,
                        )

    def test_pilot_contract_rejects_direct_canonical_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ScenarioRegistry(root / "registry")
            scenario = _scenario()
            registry.add_scenario(scenario)

            with self.assertRaisesRegex(ValueError, "publish=False"):
                run_registry_rollout(
                    registry,
                    scenario.scenario_id,
                    _config(),
                    root / "traces" / "job.jsonl",
                    random_seed=0,
                    run_contract_id="pilot_contract_test",
                )

    def test_candidate_is_verified_in_fresh_sandbox_and_artifacts_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ScenarioRegistry(root / "registry")
            scenario = _scenario()
            registry.add_scenario(scenario)
            sandboxes = []

            def sandbox_factory(_scenario, _source):
                sandbox = _PatchMemorySandbox()
                sandboxes.append(sandbox)
                return sandbox

            trace_path = root / "traces" / "job.jsonl"
            result = run_registry_rollout(
                registry,
                scenario.scenario_id,
                _config(),
                trace_path,
                random_seed=0,
                budgets=AgentBudgets(max_turns=4),
                sandbox_factory=sandbox_factory,
                client_builder=lambda _config: _RepairClient(),
                source_materializer=_empty_source,
                cost_calculator=lambda usage: usage["total_tokens"] / 1000,
            )

            public_trace = trace_path.read_text(encoding="utf-8")
            private_report = result.artifacts.private_evaluation.read_text(encoding="utf-8")
            run_evidence = result.artifacts.run_evidence.read_text(encoding="utf-8")

        self.assertEqual(len(sandboxes), 2)
        self.assertIsNot(sandboxes[0], sandboxes[1])
        self.assertNotIn("python verify.py", sandboxes[0].commands_run)
        self.assertIn("python verify.py", sandboxes[1].commands_run)
        self.assertTrue(result.report.success)
        self.assertEqual(result.cost, 0.03)
        self.assertEqual(
            replay_trace(result.trace).terminal_state_hash,
            result.run_result.final_state_hash,
        )
        self.assertNotIn("secret verifier diagnostic", public_trace)
        self.assertIn("secret verifier diagnostic", private_report)
        self.assertNotIn("https://private.invalid/v1", run_evidence)
        self.assertEqual(json.loads(run_evidence)["run_contract_id"], "")

    def test_failed_attempt_is_archived_without_publishing_canonical_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ScenarioRegistry(root / "registry")
            scenario = _scenario()
            registry.add_scenario(scenario)
            trace_path = root / "traces" / "job.jsonl"

            with self.assertRaisesRegex(RuntimeError, "private.invalid"):
                run_registry_rollout(
                    registry,
                    scenario.scenario_id,
                    _config(),
                    trace_path,
                    random_seed=0,
                    sandbox_factory=lambda _scenario, _source: _PatchMemorySandbox(),
                    client_builder=lambda _config: _FailingClient(),
                    source_materializer=_empty_source,
                )

            errors = list((trace_path.parent / ".attempts" / trace_path.stem).glob("*.error.json"))
            self.assertFalse(trace_path.exists())
            self.assertEqual(len(errors), 1)
            error_payload = json.loads(errors[0].read_text(encoding="utf-8"))

        self.assertNotIn("https://private.invalid", error_payload["error"])
        self.assertIn("[redacted endpoint]", error_payload["error"])


class _PatchMemorySandbox(MemorySandbox):
    def __init__(self) -> None:
        super().__init__(
            {"app.py": "value = 1\n"},
            {"python verify.py": _verify_candidate},
        )
        self.commands_run = []

    def execute(self, command, *, timeout_seconds=None):
        self.commands_run.append(" ".join(command))
        return super().execute(command, timeout_seconds=timeout_seconds)

    def execute_as_root(self, command, *, timeout_seconds=None):
        return self.execute(command, timeout_seconds=timeout_seconds)

    def prepare_git_baseline(self) -> str:
        self.initial_files = dict(self.files)
        return self.state_hash()

    def candidate_patch(self) -> str:
        return json.dumps(self.files, sort_keys=True)

    def apply_candidate_patch(self, patch: str) -> str:
        self.files = json.loads(patch)
        return self.state_hash()


class _RepairClient:
    model = "scripted-repair"

    def __init__(self) -> None:
        self.index = 0

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        script = [
            _tool("read", "read_file", {"path": "app.py"}),
            _tool(
                "patch",
                "apply_patch",
                {"path": "app.py", "old": "value = 1", "new": "value = 2"},
            ),
            Message("assistant", "Updated app.py."),
        ]
        message = script[self.index]
        self.index += 1
        return LLMResponse(message, self.model, {"total_tokens": 10})


class _FailingClient:
    model = "scripted-failure"

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        raise RuntimeError("request to https://private.invalid/v1 failed")


def _verify_candidate(sandbox: MemorySandbox) -> CommandResult:
    passed = sandbox.read("app.py") == "value = 2\n"
    return CommandResult(
        0 if passed else 1,
        "secret verifier diagnostic\n",
        "" if passed else "candidate failed\n",
        1.0,
    )


def _tool(call_id: str, name: str, arguments: dict) -> Message:
    return Message(
        "assistant",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


def _scenario() -> Scenario:
    return Scenario(
        QuerySeed(PublicTaskContext("Change app.py so value equals 2.")),
        EnvironmentSpec(
            name="registry-rollout-test",
            version="1",
            image_digest="sha256:" + "a" * 64,
            capability_packs=["read_file", "apply_patch"],
        ),
        HiddenEvaluatorContext(hidden_tests=["python verify.py"]),
    )


def _config() -> LLMConfig:
    return LLMConfig(
        provider="openai_compatible",
        model="scripted-repair",
        base_url="https://private.invalid/v1",
    )


def _set_run_contract_id(path: Path, contract_id: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("evidence_id")
    payload["run_contract_id"] = contract_id
    payload["evidence_id"] = stable_id("run_evidence", payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _empty_source(environment, destination, *, run_health_checks=False):
    del environment, run_health_checks
    return Path(destination)
