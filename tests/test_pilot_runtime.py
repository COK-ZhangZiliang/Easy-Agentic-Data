from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import easy_agentic_data.llm.observability as llm_observability_module
import easy_agentic_data.llm.openai_compatible as llm_adapter_module
import easy_agentic_data.pilot_contract as pilot_contract_module
import easy_agentic_data.registry_rollouts as registry_rollouts_module
import easy_agentic_data.traces.events as trace_events_module
import easy_agentic_data.trajectory_review as trajectory_review_module
from easy_agentic_data.batch import (
    ConsumedUsageTotals,
    PersistentScheduler,
    RolloutJob,
    RolloutOutcome,
)
from easy_agentic_data.cli import main
from easy_agentic_data.config import LLMConfig
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.models import LLMResponse, Message, stable_id
from easy_agentic_data.pilot_contract import (
    GOLD20_REQUIRED_VALIDATION_GATES,
    Gold20Binding,
    PilotBudgets,
    PilotRunContract,
    PricingSpec,
    ProviderConfigBinding,
    canonical_sha256,
)
from easy_agentic_data.pilot_usage_ledger import (
    PilotUsageAttempt,
    UnknownProviderUsageError,
)
from easy_agentic_data.pilot_workflow import (
    PilotRolloutWorker,
    _aggregate_usage_values,
    _config_for_assignment,
    _registry_snapshot_sha256,
    current_pilot_versions,
    reconcile_pilot_usage_ledger,
    submit_pilot_jobs,
    validate_pilot_versions,
    write_pilot_run_contract,
)
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_rollouts import RolloutArtifactPaths, run_registry_rollout
from easy_agentic_data.sandbox import CommandResult, MemorySandbox
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed
from easy_agentic_data.traces import load_trace
from easy_agentic_data.trajectory_review import ReviewDecision


class PilotRuntimeTests(unittest.TestCase):
    def test_worker_uses_deterministic_provider_without_unsupported_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)
            assignments = contract.rollouts[:2]
            received_configs = []
            received_kwargs = []

            def fake_rollout(_registry, _scenario_id, run_config, *_args, **kwargs):
                received_configs.append(run_config)
                received_kwargs.append(kwargs)
                _record_usage_attempt(
                    kwargs["usage_attempt"],
                    response_id=f"completion_{len(received_configs)}",
                )
                staged_trace = Path(_args[0])
                artifacts = SimpleNamespace(trace=staged_trace)
                return SimpleNamespace(
                    trace=SimpleNamespace(
                        trace_id=f"trace_{len(received_configs)}",
                        path=staged_trace,
                    ),
                    report=SimpleNamespace(success=True, infrastructure_failure=False),
                    run_result=SimpleNamespace(tokens=15),
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                    cost=0.0,
                    metrics={"elapsed_ms": 1.0},
                    artifacts=artifacts,
                )

            with (
                patch(
                    "easy_agentic_data.pilot_workflow.run_registry_rollout",
                    side_effect=fake_rollout,
                ),
                patch(
                    "easy_agentic_data.pilot_workflow."
                    "pilot_artifacts_module.validate_pilot_rollout_artifact",
                    return_value=SimpleNamespace(validation_receipt=object()),
                ) as validate_artifact,
                patch(
                    "easy_agentic_data.pilot_workflow.load_pilot_job_usage",
                    return_value=SimpleNamespace(totals=ConsumedUsageTotals(15, 0.00002, 1.0)),
                ),
                patch(
                    "easy_agentic_data.pilot_workflow.publish_registry_rollout",
                    side_effect=lambda result, _path, **_kwargs: result,
                ) as publish_artifact,
            ):
                outcomes = [worker.run(_job(contract, assignment)) for assignment in assignments]

        self.assertTrue(all(outcome.trace_id for outcome in outcomes))
        self.assertTrue(all("seed" not in item.request_body for item in received_configs))
        self.assertEqual(config.request_body, {"thinking": {"type": "disabled"}})
        self.assertIsNot(received_configs[0], received_configs[1])
        self.assertTrue(all(item["publish"] is False for item in received_kwargs))
        self.assertEqual(validate_artifact.call_count, 2)
        self.assertEqual(publish_artifact.call_count, 2)
        self.assertTrue(
            all(item["version_hashes"] == contract.versions.to_dict() for item in received_kwargs)
        )
        self.assertTrue(
            all(item["provider_binding"] == contract.provider.to_dict() for item in received_kwargs)
        )

    def test_assignment_seed_is_injected_only_when_provider_declares_field(self) -> None:
        config = LLMConfig(
            provider="openai_compatible",
            model="seeded-model",
            temperature=0.7,
            request_body={"thinking": {"type": "disabled"}},
            seed_request_field="seed",
        )

        first = _config_for_assignment(config, 101)
        second = _config_for_assignment(config, 202)

        self.assertEqual(first.request_body["seed"], 101)
        self.assertEqual(second.request_body["seed"], 202)
        self.assertIsNone(first.seed_request_field)
        self.assertEqual(config.request_body, {"thinking": {"type": "disabled"}})
        self.assertEqual(config.seed_request_field, "seed")

    def test_worker_does_not_retry_programming_or_integrity_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)
            assignments = contract.rollouts[:2]
            with patch(
                "easy_agentic_data.pilot_workflow.run_registry_rollout",
                side_effect=[ValueError("invalid lineage"), RuntimeError("provider timeout")],
            ):
                integrity = worker.run(_job(contract, assignments[0]))
                infrastructure = worker.run(_job(contract, assignments[1]))

        self.assertFalse(integrity.infrastructure_failure)
        self.assertTrue(infrastructure.infrastructure_failure)
        self.assertIn("ValueError", integrity.error)
        self.assertIn("RuntimeError", infrastructure.error)

    def test_worker_preserves_known_usage_on_failed_post_model_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)
            assignment = contract.rollouts[0]

            def fail_after_completed_call(*_args, **kwargs):
                _record_usage_attempt(
                    kwargs["usage_attempt"],
                    response_id="completion_failed_verifier",
                )
                raise RuntimeError("verifier host unavailable")

            with patch(
                "easy_agentic_data.pilot_workflow.run_registry_rollout",
                side_effect=fail_after_completed_call,
            ):
                outcome = worker.run(_job(contract, assignment))

        self.assertTrue(outcome.infrastructure_failure)
        self.assertEqual(outcome.tokens, 15)
        self.assertEqual(outcome.cost, 0.00002)
        self.assertEqual(outcome.metrics["failed_attempt_usage_known"], 1.0)

    def test_completed_terminal_recovers_once_after_scheduler_finish_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")
            submit_pilot_jobs(scheduler, contract)
            assignment = contract.rollouts[0]
            scheduler._mark_running(assignment.job_id)
            attempt = PilotUsageAttempt(
                root / "traces",
                contract_id=contract.contract_id,
                job_id=assignment.job_id,
                attempt_id="attempt_1",
            )
            _record_usage_attempt(attempt, response_id="completion_recovered")
            attempt.finalize(
                RolloutOutcome(
                    infrastructure_failure=True,
                    tokens=15,
                    cost=0.00002,
                    error="verifier host unavailable",
                ),
                elapsed_ms=25.0,
            )
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)

            first = reconcile_pilot_usage_ledger(scheduler, contract, worker)
            first_row = next(row for row in scheduler.rows() if row["job_id"] == assignment.job_id)
            second = reconcile_pilot_usage_ledger(scheduler, contract, worker)
            second_row = next(row for row in scheduler.rows() if row["job_id"] == assignment.job_id)

        self.assertEqual(first, second)
        self.assertEqual(first_row["status"], "pending")
        self.assertEqual(first_row["attempts"], 1)
        self.assertEqual(first_row["consumed_tokens"], 15)
        self.assertAlmostEqual(first_row["consumed_cost"], 0.00002)
        self.assertEqual(first_row["consumed_elapsed_ms"], 25.0)
        self.assertEqual(second_row, first_row)

    def test_running_before_ledger_admission_recovers_zero_call_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")
            submit_pilot_jobs(scheduler, contract)
            assignment = contract.rollouts[0]
            scheduler._mark_running(assignment.job_id)
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)

            first = reconcile_pilot_usage_ledger(scheduler, contract, worker)
            first_row = next(row for row in scheduler.rows() if row["job_id"] == assignment.job_id)
            second = reconcile_pilot_usage_ledger(scheduler, contract, worker)
            second_row = next(row for row in scheduler.rows() if row["job_id"] == assignment.job_id)
            terminals = list((root / "traces" / ".pilot-usage-ledger").rglob("terminal.*.json"))
            terminal = json.loads(terminals[0].read_text(encoding="utf-8"))

        self.assertEqual(first, second)
        self.assertEqual(first_row, second_row)
        self.assertEqual(first_row["status"], "pending")
        self.assertEqual(first_row["attempts"], 1)
        self.assertEqual(first_row["consumed_tokens"], 0)
        self.assertEqual(first_row["consumed_cost"], 0.0)
        self.assertEqual(first_row["consumed_elapsed_ms"], 60_000.0)
        self.assertEqual(len(terminals), 1)
        self.assertFalse(terminal["success"])
        self.assertTrue(terminal["infrastructure_failure"])
        self.assertEqual(terminal["started_call_record_sha256s"], [])
        self.assertEqual(terminal["completed_call_record_sha256s"], [])

    def test_crash_while_recovering_missing_admission_keeps_elapsed_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")
            submit_pilot_jobs(scheduler, contract)
            assignment = contract.rollouts[0]
            scheduler._mark_running(assignment.job_id)
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)

            with patch.object(
                PilotUsageAttempt,
                "finalize",
                side_effect=RuntimeError("crash during recovery"),
            ):
                with self.assertRaisesRegex(RuntimeError, "crash during recovery"):
                    reconcile_pilot_usage_ledger(scheduler, contract, worker)

            attempt_root = root / "traces" / ".pilot-usage-ledger" / assignment.job_id
            attempt_directories = list(attempt_root.glob("attempt_*"))
            self.assertEqual(len(attempt_directories), 1)
            self.assertEqual(
                list(attempt_directories[0].glob("terminal.*.json")),
                [],
            )

            reconcile_pilot_usage_ledger(scheduler, contract, worker)
            recovered_row = next(
                row for row in scheduler.rows() if row["job_id"] == assignment.job_id
            )

        self.assertEqual(recovered_row["status"], "pending")
        self.assertEqual(recovered_row["consumed_tokens"], 0)
        self.assertEqual(recovered_row["consumed_cost"], 0.0)
        self.assertGreaterEqual(recovered_row["consumed_elapsed_ms"], 60_000.0)

    def test_completed_calls_without_terminal_recover_as_infrastructure_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")
            submit_pilot_jobs(scheduler, contract)
            assignment = contract.rollouts[0]
            scheduler._mark_running(assignment.job_id)
            with patch(
                "easy_agentic_data.pilot_usage_ledger.utc_now",
                side_effect=[
                    "2026-07-14T00:00:00+00:00",
                    "2026-07-14T00:00:01+00:00",
                    "2026-07-14T00:00:05+00:00",
                ],
            ):
                attempt = PilotUsageAttempt(
                    root / "traces",
                    contract_id=contract.contract_id,
                    job_id=assignment.job_id,
                    attempt_id="attempt_1",
                )
                _record_usage_attempt(attempt, response_id="completion_interrupted")
                worker = PilotRolloutWorker(registry, config, root / "traces", contract)
                first = reconcile_pilot_usage_ledger(scheduler, contract, worker)
            first_row = next(row for row in scheduler.rows() if row["job_id"] == assignment.job_id)
            second = reconcile_pilot_usage_ledger(scheduler, contract, worker)
            second_row = next(row for row in scheduler.rows() if row["job_id"] == assignment.job_id)
            terminal_path = next(attempt.directory.glob("terminal.*.json"))
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))

        self.assertEqual(first, second)
        self.assertEqual(first_row, second_row)
        self.assertEqual(first_row["status"], "pending")
        self.assertEqual(first_row["attempts"], 1)
        self.assertEqual(first_row["success"], 0)
        self.assertEqual(first_row["consumed_tokens"], 15)
        self.assertAlmostEqual(first_row["consumed_cost"], 0.00002)
        self.assertEqual(first_row["consumed_elapsed_ms"], 5_000.0)
        self.assertFalse(terminal["success"])
        self.assertTrue(terminal["infrastructure_failure"])
        self.assertEqual(terminal["attempt_tokens"], 15)
        self.assertEqual(terminal["attempt_cost_usd"], "0.00002")

    def test_started_only_usage_blocks_pilot_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")
            submit_pilot_jobs(scheduler, contract)
            assignment = contract.rollouts[0]
            scheduler._mark_running(assignment.job_id)
            attempt = PilotUsageAttempt(
                root / "traces",
                contract_id=contract.contract_id,
                job_id=assignment.job_id,
                attempt_id="attempt_1",
            )
            _record_started_usage_attempt(attempt)
            contract_path = root / "contract.json"
            config_path = root / "config.json"
            write_pilot_run_contract(contract_path, contract)
            config_path.write_text(
                json.dumps({"llm": asdict(config)}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(UnknownProviderUsageError, "completed receipt"):
                main(
                    [
                        "pilot",
                        "run",
                        "--contract",
                        str(contract_path),
                        "--registry",
                        str(registry.root),
                        "--database",
                        str(scheduler.database),
                        "--config",
                        str(config_path),
                        "--trace-directory",
                        str(root / "traces"),
                        "--dry-run",
                    ]
                )

    def test_canonical_symlink_does_not_enable_interrupted_attempt_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")
            submit_pilot_jobs(scheduler, contract)
            assignment = contract.rollouts[0]
            scheduler._mark_running(assignment.job_id)
            trace_root = root / "traces"
            trace_root.mkdir()
            outside = root / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            (trace_root / f"{assignment.job_id}.jsonl").symlink_to(outside)
            worker = PilotRolloutWorker(registry, config, trace_root, contract)

            with self.assertRaisesRegex(ValueError, "canonical trace must not be a symlink"):
                reconcile_pilot_usage_ledger(scheduler, contract, worker)

            self.assertFalse((trace_root / ".pilot-usage-ledger").exists())

    def test_canonical_publish_before_scheduler_finish_recovers_without_new_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")
            submit_pilot_jobs(scheduler, contract)
            assignment = contract.rollouts[0]
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)
            with (
                patch.object(
                    registry_rollouts_module,
                    "_docker_sandbox",
                    side_effect=lambda _scenario, _source: _RuntimeMemorySandbox(),
                ),
                patch.object(
                    registry_rollouts_module,
                    "_build_llm_client",
                    side_effect=lambda _config: _CompletingClient(),
                ),
                patch.object(
                    registry_rollouts_module,
                    "materialize_environment_source",
                    side_effect=_empty_source,
                ),
                patch.object(
                    scheduler,
                    "_finish",
                    side_effect=RuntimeError("crash before scheduler finish"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "scheduler finish"):
                    scheduler.run(worker, job_ids=[assignment.job_id])

            crashed_row = next(
                row for row in scheduler.rows() if row["job_id"] == assignment.job_id
            )
            self.assertEqual(crashed_row["status"], "running")
            self.assertEqual(crashed_row["consumed_tokens"], 0)
            self.assertTrue((root / "traces" / f"{assignment.job_id}.jsonl").is_file())

            reconcile_pilot_usage_ledger(scheduler, contract, worker)
            recovered_row = next(
                row for row in scheduler.rows() if row["job_id"] == assignment.job_id
            )
            reconcile_pilot_usage_ledger(scheduler, contract, worker)
            repeated_row = next(
                row for row in scheduler.rows() if row["job_id"] == assignment.job_id
            )

        self.assertEqual(recovered_row["status"], "completed")
        self.assertEqual(recovered_row["attempts"], 1)
        self.assertEqual(recovered_row["consumed_tokens"], 15)
        self.assertEqual(repeated_row, recovered_row)

    def test_terminal_before_publication_failure_still_accounts_usage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            assignment = contract.rollouts[0]
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)
            with (
                patch.object(
                    registry_rollouts_module,
                    "_docker_sandbox",
                    side_effect=lambda _scenario, _source: _RuntimeMemorySandbox(),
                ),
                patch.object(
                    registry_rollouts_module,
                    "_build_llm_client",
                    side_effect=lambda _config: _CompletingClient(),
                ),
                patch.object(
                    registry_rollouts_module,
                    "materialize_environment_source",
                    side_effect=_empty_source,
                ),
                patch(
                    "easy_agentic_data.pilot_workflow.publish_registry_rollout",
                    side_effect=RuntimeError("publication host failure"),
                ),
            ):
                outcome = worker.run(_job(contract, assignment))

        self.assertFalse(outcome.trace_id)
        self.assertTrue(outcome.infrastructure_failure)
        self.assertEqual(outcome.tokens, 15)
        self.assertAlmostEqual(outcome.cost, 0.00002)
        self.assertIsNotNone(outcome.absolute_consumed_usage)
        self.assertEqual(outcome.absolute_consumed_usage.tokens, 15)
        self.assertIn("publication", outcome.error.lower())

    def test_worker_rejects_nested_provider_config_drift_before_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)
            config.base_url = "https://drift.invalid/v1"
            config.request_body["thinking"]["type"] = "enabled"

            with patch("easy_agentic_data.pilot_workflow.run_registry_rollout") as run_rollout:
                outcome = worker.run(_job(contract, contract.rollouts[0]))

        run_rollout.assert_not_called()
        self.assertFalse(outcome.trace_id)
        self.assertFalse(outcome.infrastructure_failure)
        self.assertIn("provider configuration", outcome.error.lower())
        self.assertIsNotNone(outcome.absolute_consumed_usage)
        self.assertEqual(outcome.absolute_consumed_usage.tokens, 0)

    def test_execution_recomputes_provider_binding_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            assignment = contract.rollouts[0]
            run_config = _config_for_assignment(config, assignment.random_seed)
            run_config.base_url = "https://drift.invalid/v1"
            usage_attempt = PilotUsageAttempt(
                root / "traces",
                contract_id=contract.contract_id,
                job_id=assignment.job_id,
            )

            with self.assertRaisesRegex(ValueError, "provider configuration"):
                run_registry_rollout(
                    registry,
                    assignment.scenario_id,
                    run_config,
                    root / "traces" / f"{assignment.job_id}.jsonl",
                    assignment.random_seed,
                    run_contract_id=contract.contract_id,
                    provider_binding_sha256=contract.provider.config_sha256,
                    provider_binding=contract.provider.to_dict(),
                    version_hashes=contract.versions.to_dict(),
                    usage_attempt=usage_attempt,
                    publish=False,
                )

    def test_model_response_with_empty_usage_is_an_integrity_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _aggregate_usage_values([{}])

    def test_enqueue_rejects_an_existing_contract_row_with_changed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, contract, _ = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")
            submit_pilot_jobs(scheduler, contract)
            first_job_id = contract.rollouts[0].job_id
            with sqlite3.connect(scheduler.database) as connection:
                connection.execute(
                    "UPDATE jobs SET scenario_id = ? WHERE job_id = ?",
                    ("scenario_tampered", first_job_id),
                )

            with self.assertRaisesRegex(ValueError, "queue row mismatch"):
                submit_pilot_jobs(scheduler, contract)
            row_count = len(scheduler.rows())

        self.assertEqual(row_count, 40)

    def test_runtime_version_hash_binds_execution_and_review_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, _ = _registry_and_contract(root)
            modules = (
                (llm_adapter_module, "environment_sha256"),
                (llm_observability_module, "environment_sha256"),
                (pilot_contract_module, "environment_sha256"),
                (trace_events_module, "environment_sha256"),
                (trajectory_review_module, "exporter_sha256"),
            )
            for index, (module, field) in enumerate(modules):
                changed_source = root / f"changed-{index}.py"
                changed_source.write_text(f"# changed {index}\n", encoding="utf-8")
                with self.subTest(module=module.__name__):
                    with patch.object(module, "__file__", str(changed_source)):
                        changed = current_pilot_versions(contract.corpus, registry)
                    self.assertNotEqual(
                        getattr(changed, field),
                        getattr(contract.versions, field),
                    )

    def test_version_validation_rejects_changed_review_implementation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, _ = _registry_and_contract(root)
            changed_source = root / "changed-review.py"
            changed_source.write_text("# changed review implementation\n", encoding="utf-8")

            validate_pilot_versions(contract, registry)
            with patch.object(
                trajectory_review_module,
                "__file__",
                str(changed_source),
            ):
                with self.assertRaisesRegex(ValueError, "implementation versions"):
                    validate_pilot_versions(contract, registry)

    def test_existing_canonical_rollout_requires_all_bound_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            worker = PilotRolloutWorker(registry, config, root / "traces", contract)
            assignment = contract.rollouts[0]
            trace_path = root / "traces" / f"{assignment.job_id}.jsonl"
            with (
                patch.object(
                    registry_rollouts_module,
                    "_docker_sandbox",
                    side_effect=lambda _scenario, _source: _RuntimeMemorySandbox(),
                ),
                patch.object(
                    registry_rollouts_module,
                    "_build_llm_client",
                    side_effect=lambda _config: _CompletingClient(),
                ),
                patch.object(
                    registry_rollouts_module,
                    "materialize_environment_source",
                    side_effect=_empty_source,
                ),
            ):
                initial = worker.run(_job(contract, assignment))
            self.assertTrue(initial.trace_id, initial.error)
            produced = SimpleNamespace(
                trace=load_trace(trace_path),
                artifacts=RolloutArtifactPaths.for_trace(trace_path),
            )
            recovered = worker.run(_job(contract, assignment))
            original_private = produced.artifacts.private_evaluation.read_text(encoding="utf-8")
            invalid_evaluator = json.loads(original_private)
            invalid_evaluator["report"]["results"][0]["evaluator"] = "unexpected"
            invalid_evaluator.pop("private_evaluation_id")
            invalid_evaluator["private_evaluation_id"] = stable_id(
                "private_evaluation",
                invalid_evaluator,
            )
            produced.artifacts.private_evaluation.write_text(
                json.dumps(invalid_evaluator),
                encoding="utf-8",
            )
            evaluator_rejected = worker.run(_job(contract, assignment))
            produced.artifacts.private_evaluation.write_text(
                original_private,
                encoding="utf-8",
            )
            invalid_aggregate = json.loads(original_private)
            invalid_aggregate["report"]["success"] = False
            invalid_aggregate["report"]["reward"] = 0
            invalid_aggregate.pop("private_evaluation_id")
            invalid_aggregate["private_evaluation_id"] = stable_id(
                "private_evaluation",
                invalid_aggregate,
            )
            produced.artifacts.private_evaluation.write_text(
                json.dumps(invalid_aggregate),
                encoding="utf-8",
            )
            aggregate_rejected = worker.run(_job(contract, assignment))
            produced.artifacts.private_evaluation.write_text(
                original_private,
                encoding="utf-8",
            )
            original_evidence = produced.artifacts.run_evidence.read_text(encoding="utf-8")
            invalid_call = json.loads(original_evidence)
            invalid_call["observed_calls"][0]["response_model"] = "different-model"
            invalid_call.pop("evidence_id")
            invalid_call["evidence_id"] = stable_id("run_evidence", invalid_call)
            produced.artifacts.run_evidence.write_text(
                json.dumps(invalid_call),
                encoding="utf-8",
            )
            call_rejected = worker.run(_job(contract, assignment))
            produced.artifacts.run_evidence.write_text(
                original_evidence,
                encoding="utf-8",
            )
            produced.artifacts.candidate_patch.write_text(
                produced.artifacts.candidate_patch.read_text(encoding="utf-8") + "tampered",
                encoding="utf-8",
            )
            rejected = worker.run(_job(contract, assignment))
            canonical_exists = trace_path.exists()

        self.assertTrue(recovered.trace_id, recovered.error)
        self.assertEqual(recovered.trace_id, produced.trace.trace_id)
        self.assertFalse(evaluator_rejected.trace_id)
        self.assertIn("evaluator set", evaluator_rejected.error.lower())
        self.assertFalse(aggregate_rejected.trace_id)
        self.assertIn("aggregate fields", aggregate_rejected.error.lower())
        self.assertFalse(call_rejected.trace_id)
        self.assertIn("public response", call_rejected.error.lower())
        self.assertFalse(rejected.trace_id)
        self.assertFalse(rejected.infrastructure_failure)
        self.assertIn("canonical rollout is invalid", rejected.error.lower())
        self.assertTrue(canonical_exists)

    def test_review_cli_binds_queue_gate_and_trace_locations_to_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, _ = _registry_and_contract(root)
            contract_path = root / "contract.json"
            quality_path = root / "quality.json"
            queue_path = root / "review-queue.json"
            decisions_path = root / "decisions.json"
            gate_path = root / "review-gate.json"
            quarantine_path = root / "quarantine.json"
            reproduction_path = root / "reproduction.json"
            database = root / "pilot.sqlite3"
            write_pilot_run_contract(contract_path, contract)
            submit_pilot_jobs(PersistentScheduler(database), contract)
            reproduction_path.write_text("{}\n", encoding="utf-8")
            summaries = []
            for index, assignment in enumerate(contract.rollouts):
                summaries.append(
                    {
                        "contract_id": contract.contract_id,
                        "job_id": assignment.job_id,
                        "trace_path": f"{assignment.job_id}.jsonl",
                        "trace_id": f"trace_{index:02d}",
                        "scenario_id": assignment.scenario_id,
                        "repository": f"org/repo-{index % 8}",
                        "language": "python",
                        "success": True,
                        "termination_reason": "completed",
                        "schema_valid": True,
                        "replay_valid": True,
                    }
                )
            quality_path.write_text(
                json.dumps(
                    {
                        "contract_id": contract.contract_id,
                        "review_summaries": summaries,
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                queue_code = main(
                    [
                        "pilot",
                        "review-queue",
                        "--contract",
                        str(contract_path),
                        "--registry",
                        str(registry.root),
                        "--quality-report",
                        str(quality_path),
                        "--output",
                        str(queue_path),
                    ]
                )
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            decisions_path.write_text(
                json.dumps(
                    [
                        ReviewDecision(
                            trace_id=item["trace_id"],
                            reviewer_alias="reviewer-a",
                            timestamp="2026-07-14T00:00:00Z",
                            verdict="acceptable",
                        ).to_dict()
                        for item in queue["items"]
                    ]
                ),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                gate_code = main(
                    [
                        "pilot",
                        "review-gate",
                        "--contract",
                        str(contract_path),
                        "--registry",
                        str(registry.root),
                        "--queue",
                        str(queue_path),
                        "--decisions",
                        str(decisions_path),
                        "--output",
                        str(gate_path),
                        "--quarantine-output",
                        str(quarantine_path),
                    ]
                )
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
            export_args = [
                "pilot",
                "export",
                "--contract",
                str(contract_path),
                "--registry",
                str(registry.root),
                "--database",
                str(database),
                "--trace-directory",
                str(root / "traces"),
                "--reproduction",
                str(reproduction_path),
                "--review-queue",
                str(queue_path),
                "--review-gate",
                str(gate_path),
                "--quarantine",
                str(quarantine_path),
                "--output-directory",
                str(root / "exports"),
            ]
            with (
                patch(
                    "easy_agentic_data.cli.write_pilot_exports",
                    return_value={"export_manifest_id": "pilot_exports_test"},
                ) as export_mock,
                redirect_stdout(io.StringIO()),
            ):
                export_code = main(export_args)

            tampered_quarantine = dict(quarantine)
            tampered_quarantine["trace_ids"] = [queue["items"][0]["trace_id"]]
            quarantine_path.write_text(
                json.dumps(tampered_quarantine),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                with redirect_stdout(io.StringIO()):
                    main(export_args)

        self.assertEqual(queue_code, 0)
        self.assertEqual(gate_code, 0)
        self.assertEqual(export_code, 0)
        export_mock.assert_called_once()
        self.assertEqual(queue["contract_id"], contract.contract_id)
        self.assertEqual(gate["contract_id"], contract.contract_id)
        self.assertEqual(quarantine["contract_id"], contract.contract_id)
        self.assertTrue(
            all(item["trace_path"] == f"{item['job_id']}.jsonl" for item in queue["items"])
        )

    def test_post_run_cli_commands_fail_closed_on_version_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, _ = _registry_and_contract(root)
            contract_path = root / "contract.json"
            database = root / "pilot.sqlite3"
            write_pilot_run_contract(contract_path, contract)
            submit_pilot_jobs(PersistentScheduler(database), contract)
            changed_source = root / "changed-review.py"
            changed_source.write_text("# changed review implementation\n", encoding="utf-8")
            common = [
                "--contract",
                str(contract_path),
                "--registry",
                str(registry.root),
            ]
            commands = (
                [
                    "pilot",
                    "reproduce",
                    *common,
                    "--database",
                    str(database),
                    "--trace-directory",
                    str(root / "traces"),
                    "--output",
                    str(root / "reproduction.json"),
                ],
                [
                    "pilot",
                    "export",
                    *common,
                    "--database",
                    str(database),
                    "--trace-directory",
                    str(root / "traces"),
                    "--reproduction",
                    str(root / "reproduction.json"),
                    "--review-queue",
                    str(root / "review-queue.json"),
                    "--review-gate",
                    str(root / "review-gate.json"),
                    "--quarantine",
                    str(root / "quarantine.json"),
                    "--output-directory",
                    str(root / "exports"),
                ],
                [
                    "pilot",
                    "quality-report",
                    *common,
                    "--database",
                    str(database),
                    "--trace-directory",
                    str(root / "traces"),
                    "--output",
                    str(root / "quality.json"),
                ],
                [
                    "pilot",
                    "review-queue",
                    *common,
                    "--quality-report",
                    str(root / "quality.json"),
                    "--output",
                    str(root / "review-queue.json"),
                ],
                [
                    "pilot",
                    "review-gate",
                    *common,
                    "--queue",
                    str(root / "review-queue.json"),
                    "--decisions",
                    str(root / "decisions.json"),
                    "--output",
                    str(root / "review-gate.json"),
                ],
            )

            with patch.object(
                trajectory_review_module,
                "__file__",
                str(changed_source),
            ):
                for command in commands:
                    with self.subTest(command=command[1]):
                        with self.assertRaisesRegex(ValueError, "implementation versions"):
                            main(command)

    def test_pilot_dry_run_requires_an_existing_bound_provider_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            contract_path = root / "contract.json"
            database = root / "pilot.sqlite3"
            config_path = root / "config.json"
            write_pilot_run_contract(contract_path, contract)
            submit_pilot_jobs(PersistentScheduler(database), contract)
            config_path.write_text(
                json.dumps({"llm": asdict(config)}),
                encoding="utf-8",
            )
            arguments = [
                "pilot",
                "run",
                "--contract",
                str(contract_path),
                "--registry",
                str(registry.root),
                "--database",
                str(database),
                "--config",
                str(config_path),
                "--trace-directory",
                str(root / "traces"),
                "--dry-run",
            ]

            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)

            config_path.unlink()
            with self.assertRaises(FileNotFoundError):
                main(arguments)

    def test_pilot_dry_run_rejects_provider_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, config = _registry_and_contract(root)
            contract_path = root / "contract.json"
            database = root / "pilot.sqlite3"
            config_path = root / "drifted-config.json"
            write_pilot_run_contract(contract_path, contract)
            submit_pilot_jobs(PersistentScheduler(database), contract)
            config_path.write_text(
                json.dumps({"llm": asdict(replace(config, model="other-model"))}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "provider configuration"):
                main(
                    [
                        "pilot",
                        "run",
                        "--contract",
                        str(contract_path),
                        "--registry",
                        str(registry.root),
                        "--database",
                        str(database),
                        "--config",
                        str(config_path),
                        "--trace-directory",
                        str(root / "traces"),
                        "--dry-run",
                    ]
                )


def _job(contract: PilotRunContract, assignment) -> RolloutJob:
    return RolloutJob(
        assignment.scenario_id,
        assignment.rollout_index,
        contract.provider.model,
        contract.contract_id,
        assignment.job_id,
    )


def _registry_and_contract(
    root: Path,
) -> tuple[ScenarioRegistry, PilotRunContract, LLMConfig]:
    registry = ScenarioRegistry(root / "registry")
    records = []
    for index in range(20):
        scenario = Scenario(
            QuerySeed(PublicTaskContext(f"Repair task {index}.")),
            EnvironmentSpec(
                name=f"fixture-{index}",
                version="1",
                image_digest="sha256:" + f"{index:064x}",
            ),
            HiddenEvaluatorContext(hidden_tests=["python hidden.py"]),
            scenario_id=f"scenario_{index:02d}",
        )
        registry.add_scenario(scenario)
        record = {
            "scenario_id": scenario.scenario_id,
            "seed_id": scenario.query_seed.seed_id,
            "environment_id": scenario.environment.environment_id,
            "valid": True,
            "hashes": {
                "scenario_sha256": canonical_sha256(scenario.to_dict()),
                "environment_sha256": canonical_sha256(scenario.environment.to_dict()),
                "evaluator_sha256": canonical_sha256(scenario.hidden_evaluator.to_dict()),
            },
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
    manifest = {
        "schema_version": "easy_agentic_data.gold20_manifest.v1",
        "corpus_id": "gold20_test",
        "expected_seed_count": 20,
        "valid": True,
        "issues": [],
        "validation": dict.fromkeys(GOLD20_REQUIRED_VALIDATION_GATES, True),
        "evidence": {"registry_snapshot_sha256": _registry_snapshot_sha256(registry.root)},
        "records": records,
    }
    corpus = Gold20Binding.from_manifest(manifest)
    config = LLMConfig(
        provider="local_openai_compatible",
        model="fixture-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env=None,
        temperature=0.0,
        request_body={"thinking": {"type": "disabled"}},
    )
    contract = PilotRunContract(
        corpus=corpus,
        provider=ProviderConfigBinding.from_config(config),
        budgets=PilotBudgets(
            max_agent_turns=10,
            max_agent_tool_calls=20,
            max_agent_tokens=10_000,
            max_agent_seconds=60,
            max_total_tokens=1_000_000,
            max_total_cost_usd="10",
            max_total_seconds=3600,
        ),
        versions=current_pilot_versions(corpus, registry),
        pricing=PricingSpec(
            input_usd_per_million_tokens="1",
            cached_input_usd_per_million_tokens="0.1",
            output_usd_per_million_tokens="2",
        ),
        rollout_seeds=(101, 202),
    )
    return registry, contract, config


class _RuntimeMemorySandbox(MemorySandbox):
    def __init__(self) -> None:
        super().__init__(
            {"app.py": "value = 1\n"},
            {
                "python hidden.py": lambda _sandbox: CommandResult(
                    0,
                    "passed\n",
                    "",
                    1.0,
                )
            },
        )

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


class _CompletingClient:
    model = "fixture-model"
    temperature = 0.0
    max_tokens = 2048

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        return LLMResponse(
            Message("assistant", "No source change was required."),
            self.model,
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            raw={
                "id": "completion_fixture",
                "created": 1,
                "object": "chat.completion",
                "model": self.model,
            },
        )


def _empty_source(environment, destination, *, run_health_checks=False):
    del environment, run_health_checks
    return Path(destination)


def _record_usage_attempt(attempt, *, response_id: str) -> None:
    identity = {
        "id": response_id,
        "created": 1,
        "object": "chat.completion",
        "model": "fixture-model",
    }
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    _record_started_usage_attempt(attempt)
    attempt.call_completed(
        {
            "call_index": 0,
            "response_model": "fixture-model",
            "usage": usage,
            "retry_count": 0,
            "latency_ms": 1.0,
            "provider_response_identity": identity,
            "provider_response_identity_sha256": canonical_sha256(identity),
            "provider_response_sha256": canonical_sha256(identity),
        }
    )


def _record_started_usage_attempt(attempt) -> None:
    attempt.call_started(
        {
            "call_index": 0,
            "started_at": "2026-07-14T00:00:00Z",
            "model": "fixture-model",
            "prompt_hash": canonical_sha256("fixture-prompt"),
            "message_count": 2,
            "tool_count": 0,
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": None,
        }
    )


if __name__ == "__main__":
    unittest.main()
