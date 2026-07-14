from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import easy_agentic_data.gold20_runtime as gold20_runtime
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.gold20 import (
    GOLD20_FREEZE_CONFIG_SCHEMA_VERSION,
    GOLD20_MATERIALIZATION_SCHEMA_VERSION,
    GOLD20_REFERENCE_REPAIRS_SCHEMA_VERSION,
)
from easy_agentic_data.gold20_runtime import replay_gold20_runtime
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.sandbox import SandboxLimits
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


class Gold20RuntimeReplayTests(unittest.TestCase):
    def test_valid_exact_replay_is_published_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            with _mock_runtime_execution():
                replay = replay_gold20_runtime(fixture["config"])

            self.assertTrue(replay["valid"])
            self.assertEqual(replay["counts"], {"records": 20, "valid": 20, "invalid": 0})
            self.assertEqual(
                json.loads(fixture["output"].read_text(encoding="utf-8")),
                replay,
            )
            self.assertEqual(
                set(replay["producer"]["component_sha256s"]),
                {"evaluation.py", "registry.py", "scenarios.py"},
            )
            for build in replay["runtime_builds"]:
                self.assertEqual(build["image_id"], build["image_digest"])
                self.assertEqual(
                    build["build_verification_mode"],
                    "local_image_id_plus_declared_spec",
                )

    def test_invalid_replay_does_not_replace_valid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            with _mock_runtime_execution():
                replay_gold20_runtime(fixture["config"])
            original = fixture["output"].read_bytes()

            def invalid_replay(**kwargs):
                return {
                    "source_instance_id": kwargs["source_id"],
                    "scenario_id": kwargs["scenario"].scenario_id,
                    "valid": kwargs["source_id"] != "source-0",
                }

            with _mock_runtime_execution(replay_side_effect=invalid_replay):
                replay = replay_gold20_runtime(fixture["config"])

            self.assertFalse(replay["valid"])
            self.assertEqual(replay["counts"], {"records": 20, "valid": 19, "invalid": 1})
            self.assertEqual(fixture["output"].read_bytes(), original)

    def test_runtime_build_set_must_match_registry_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            config = json.loads(fixture["config"].read_text(encoding="utf-8"))
            config["container_runtime_builds"] = config["container_runtime_builds"][:1]
            fixture["config"].write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with (
                _mock_runtime_execution(),
                self.assertRaisesRegex(
                    ValueError,
                    "image set does not match",
                ),
            ):
                replay_gold20_runtime(fixture["config"])

    def test_runtime_replay_requires_disabled_network_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            for scenario_path in (fixture["registry"] / "scenarios").glob("*.json"):
                scenario = _read_json(scenario_path)
                scenario["environment"]["network_policy"] = "enabled"
                _write_json(scenario_path, scenario)

            with (
                _mock_runtime_metadata(),
                self.assertRaisesRegex(
                    ValueError,
                    "disabled network access",
                ),
            ):
                replay_gold20_runtime(fixture["config"])

    def test_runtime_replay_rejects_unsupported_evaluator_surfaces(self) -> None:
        cases = (
            ("required_state", {"private": "required"}),
            ("forbidden_state", {"private": "forbidden"}),
            ("retrieval_requirements", ["inspect a file"]),
            ("trace_quality_rubric", ["use a tool"]),
        )
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = _build_fixture(Path(directory))
                for scenario_path in (fixture["registry"] / "scenarios").glob("*.json"):
                    scenario = _read_json(scenario_path)
                    if field in {"required_state", "forbidden_state"}:
                        scenario["hidden_evaluator"][field] = value
                    else:
                        scenario["hidden_evaluator"]["metadata"][field] = value
                    _write_json(scenario_path, scenario)

                with (
                    _mock_runtime_metadata(),
                    self.assertRaisesRegex(
                        ValueError,
                        "unsupported evaluator surface",
                    ),
                ):
                    replay_gold20_runtime(fixture["config"])

    def test_health_result_drift_invalidates_replay_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(Path(directory))
            with _mock_runtime_health_drift(fixture["root"]):
                replay = replay_gold20_runtime(fixture["config"])

            self.assertFalse(replay["valid"])
            self.assertEqual(replay["counts"], {"records": 20, "valid": 0, "invalid": 20})
            self.assertFalse(fixture["output"].exists())


def _mock_runtime_execution(*, replay_side_effect=None):
    def valid_replay(**kwargs):
        return {
            "source_instance_id": kwargs["source_id"],
            "scenario_id": kwargs["scenario"].scenario_id,
            "valid": True,
        }

    replay = replay_side_effect or valid_replay
    stack = _mock_runtime_metadata()
    stack.enter_context(
        patch("easy_agentic_data.gold20_runtime._replay_scenario", side_effect=replay)
    )
    return stack


def _mock_runtime_metadata() -> ExitStack:
    def inspect_image(build):
        return {
            "image_digest": build["image_digest"],
            "image_id": build["image_digest"],
            "platform": build["platform"],
            "image_size_bytes": 1,
            "build_spec_sha256": build["build_spec_sha256"],
            "build_verification_mode": "local_image_id_plus_declared_spec",
        }

    stack = ExitStack()
    stack.enter_context(
        patch(
            "easy_agentic_data.gold20_runtime._inspect_runtime_image",
            side_effect=inspect_image,
        )
    )
    stack.enter_context(
        patch(
            "easy_agentic_data.gold20_runtime._docker_server_version",
            return_value="test",
        )
    )
    return stack


def _mock_runtime_health_drift(root: Path) -> ExitStack:
    def execute_variant(**kwargs):
        scenario = kwargs["scenario"]
        repaired = kwargs["repair_patch"] is not None
        limits = SandboxLimits(**scenario.environment.resource_limits)
        hidden_exit = 0 if repaired else 1
        return {
            "scenario_instance_id": "instance_fixture",
            "sandbox_policy": gold20_runtime._expected_sandbox_policy(
                limits,
                scenario.environment.image_digest,
            ),
            "initial_state_sha256": "d" * 64,
            "setup_exit_codes": [],
            "health_exit_codes": [0],
            "post_health_exit_codes": [0],
            "health_result_sha256s": [("b" if repaired else "a") * 64],
            "repair_check_exit": 0 if repaired else None,
            "repair_apply_exit": 0 if repaired else None,
            "hidden_patch_exit": 0,
            "hidden_patch_infrastructure_failure": False,
            "hidden_test_exit_codes": [hidden_exit],
            "hidden_test_infrastructure_failures": [False],
            "hidden_test_result_sha256s": [
                gold20_runtime._hidden_test_result_sha256(
                    exit_code=hidden_exit,
                    infrastructure_failure=False,
                )
            ],
        }

    stack = _mock_runtime_metadata()
    stack.enter_context(
        patch(
            "easy_agentic_data.gold20_runtime.materialize_environment_source",
            return_value=root,
        )
    )
    stack.enter_context(
        patch(
            "easy_agentic_data.gold20_runtime._workspace_tree_hash",
            return_value="c" * 64,
        )
    )
    stack.enter_context(
        patch(
            "easy_agentic_data.gold20_runtime._execute_variant",
            side_effect=execute_variant,
        )
    )
    return stack


def _build_fixture(root: Path) -> dict[str, Path]:
    registry_root = root / "registry"
    registry = ScenarioRegistry(registry_root)
    images = [f"sha256:{'a' * 64}", f"sha256:{'b' * 64}"]
    materialization_records = []
    repair_records = []
    for index in range(20):
        source_id = f"source-{index}"
        repair_patch = (
            f"diff --git a/service-{index}.py b/service-{index}.py\n"
            f"--- a/service-{index}.py\n"
            f"+++ b/service-{index}.py\n"
            "@@ -1 +1 @@\n"
            "-broken\n"
            "+fixed\n"
        )
        seed = QuerySeed(
            public=PublicTaskContext(f"Repair fixture {index}"),
            provenance=f"fixture:{source_id}",
            metadata={"source_instance_id": source_id},
        )
        environment = EnvironmentSpec(
            name=f"fixture-{index}",
            version="1",
            image_digest=images[index % 2],
            health_check=["python -c 'pass'"],
        )
        scenario = Scenario(
            query_seed=seed,
            environment=environment,
            hidden_evaluator=HiddenEvaluatorContext(
                hidden_tests=["python -c 'raise SystemExit(1)'"],
                metadata={"test_patch": "diff --git a/a b/a\n+a\n"},
            ),
            metadata={"source_instance_id": source_id},
        )
        registry.add_scenario(scenario)
        materialization_records.append(
            {
                "scenario_id": scenario.scenario_id,
                "workspace_tree_hashes": ["c" * 64, "c" * 64],
            }
        )
        repair_records.append(
            {
                "source_instance_id": source_id,
                "scenario_id": scenario.scenario_id,
                "seed_id": scenario.query_seed.seed_id,
                "environment_id": scenario.environment.environment_id,
                "source_revision": scenario.environment.source_revision,
                "repair_patch": repair_patch,
                "repair_patch_sha256": hashlib.sha256(repair_patch.encode("utf-8")).hexdigest(),
            }
        )

    materialization = root / "materialization.json"
    materialization.write_text(
        json.dumps(
            {
                "schema_version": GOLD20_MATERIALIZATION_SCHEMA_VERSION,
                "records": materialization_records,
                "valid": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    repairs = root / "repairs.json"
    repairs.write_text(
        json.dumps(
            {
                "schema_version": GOLD20_REFERENCE_REPAIRS_SCHEMA_VERSION,
                "records": repair_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_builds = []
    for index, image in enumerate(images):
        build_spec = root / f"runtime-{index}.Dockerfile"
        build_spec.write_text("FROM scratch\n", encoding="utf-8")
        runtime_builds.append(
            {
                "image_digest": image,
                "platform": "linux/arm64",
                "build_spec": str(build_spec),
            }
        )
    output = root / "container-replay.json"
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": GOLD20_FREEZE_CONFIG_SCHEMA_VERSION,
                "registry_root": str(registry_root),
                "materialization_reset_evidence": str(materialization),
                "reference_repair_evidence": str(repairs),
                "container_replay_evidence": str(output),
                "container_runtime_builds": runtime_builds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "root": root,
        "registry": registry_root,
        "config": config,
        "output": output,
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
