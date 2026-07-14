import hashlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import easy_agentic_data.registry_rollouts as registry_rollouts_module
from easy_agentic_data.agent import DEFAULT_SYSTEM_PROMPT, AgentBudgets
from easy_agentic_data.batch import ConsumedUsageTotals, PersistentScheduler, RolloutOutcome
from easy_agentic_data.cli import main
from easy_agentic_data.config import LLMConfig
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.evaluation import (
    EvaluationEvidence,
    EvaluationReport,
    evaluation_result_metrics,
    public_evaluation_result,
)
from easy_agentic_data.llm import trace_prompt_fingerprints
from easy_agentic_data.models import LLMResponse, Message, stable_id
from easy_agentic_data.pilot_artifacts import (
    _row_metrics,
    _scenario_tool_schema_sha256,
    _scenario_tool_schemas,
    build_pilot_quality_report,
    load_pilot_trace_artifacts,
    validate_pilot_export_manifest,
    validate_pilot_reproduction,
    validate_pilot_rollout_artifact,
    write_pilot_exports,
)
from easy_agentic_data.pilot_contract import (
    GOLD20_REQUIRED_VALIDATION_GATES,
    Gold20Binding,
    PilotBudgets,
    PilotRunContract,
    PricingSpec,
    ProviderConfigBinding,
    canonical_sha256,
)
from easy_agentic_data.pilot_usage_ledger import PilotUsageAttempt
from easy_agentic_data.pilot_workflow import (
    _registry_snapshot_sha256,
    current_pilot_versions,
    submit_pilot_jobs,
    validate_pilot_runtime,
    write_pilot_run_contract,
)
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_rollouts import (
    RolloutArtifactPaths,
    publish_registry_rollout,
    run_registry_rollout,
)
from easy_agentic_data.sandbox import CommandResult, MemorySandbox
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed
from easy_agentic_data.traces import EventType, TerminationReason, TraceRecorder, load_trace
from easy_agentic_data.trajectory_review import (
    ReviewDecision,
    build_trajectory_review_queue,
    summarize_review_gate,
)


class PilotArtifactTests(unittest.TestCase):
    def test_scheduler_metrics_allow_signed_turn_reward_summaries(self) -> None:
        metrics = _row_metrics(
            {
                "metrics": {
                    "turn_reward_total": -0.4,
                    "turn_reward_mean": -0.1,
                    "negative_turn_rewards": 4.0,
                    "tokens": 100.0,
                }
            }
        )

        self.assertEqual(metrics["turn_reward_total"], -0.4)
        self.assertEqual(metrics["turn_reward_mean"], -0.1)
        with self.assertRaisesRegex(ValueError, "tokens must be finite and non-negative"):
            _row_metrics({"metrics": {"tokens": -1.0}})

    def test_pilot_cli_enqueues_and_dry_runs_exact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, contract = _registry_and_contract(root)
            contract_path = root / "contract.json"
            config_path = root / "config.json"
            database = root / "pilot.sqlite3"
            write_pilot_run_contract(contract_path, contract)
            config_path.write_text(
                json.dumps({"llm": asdict(_provider_config())}),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                enqueue_code = main(
                    [
                        "pilot",
                        "enqueue",
                        "--contract",
                        str(contract_path),
                        "--database",
                        str(database),
                    ]
                )
                dry_run_code = main(
                    [
                        "pilot",
                        "run",
                        "--contract",
                        str(contract_path),
                        "--registry",
                        str(root / "registry"),
                        "--database",
                        str(database),
                        "--config",
                        str(config_path),
                        "--trace-directory",
                        str(root / "traces"),
                        "--dry-run",
                    ]
                )

            payloads = _concatenated_json_objects(output.getvalue())

        self.assertEqual(enqueue_code, 0)
        self.assertEqual(dry_run_code, 0)
        self.assertEqual(payloads[0]["submitted_job_count"], 40)
        self.assertEqual(payloads[1]["runnable_job_count"], 40)

    def test_pilot_queue_is_exactly_contract_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            scheduler = PersistentScheduler(root / "pilot.sqlite3")

            submitted = submit_pilot_jobs(scheduler, contract)
            submit_pilot_jobs(scheduler, contract)
            validate_pilot_runtime(contract, registry, _provider_config())
            rows = scheduler.rows()

        self.assertEqual(len(submitted), 40)
        self.assertEqual(len(rows), 40)
        self.assertEqual({row["job_id"] for row in rows}, set(submitted))
        self.assertEqual({row["config_hash"] for row in rows}, {contract.contract_id})

    def test_real_rollout_sidecars_satisfy_strict_pilot_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, assignment, trace_path, usage_attempt, result, row = (
                _stage_real_pilot_rollout(root)
            )
            staged = validate_pilot_rollout_artifact(
                contract,
                registry,
                assignment,
                row,
                result.artifacts.trace,
                artifact_paths=result.artifacts,
            )
            self.assertEqual(staged.trace_id, result.trace.trace_id)
            self.assertFalse(trace_path.exists())
            publish_registry_rollout(
                result,
                trace_path,
                validation_receipt=staged.validation_receipt,
            )
            usage_attempt.finalize(
                RolloutOutcome(
                    trace_id=result.trace.trace_id,
                    success=result.report.success,
                    tokens=row["tokens"],
                    cost=row["cost"],
                    metrics=result.metrics,
                ),
                elapsed_ms=result.elapsed_ms,
            )

            loaded = load_pilot_trace_artifacts(
                contract,
                registry,
                [row],
                root / "traces",
                require_complete=False,
            )

        self.assertEqual(len(loaded), 1)
        self.assertTrue(loaded[0].report.success)

    def test_receipt_requires_strict_validation_and_rejects_post_validation_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, assignment, _, _, result, row = _stage_real_pilot_rollout(root)
            evidence = json.loads(result.artifacts.run_evidence.read_text(encoding="utf-8"))
            evidence["elapsed_ms"] += 1
            result.artifacts.run_evidence.write_text(
                json.dumps(evidence),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Content ID mismatch"):
                validate_pilot_rollout_artifact(
                    contract,
                    registry,
                    assignment,
                    row,
                    result.artifacts.trace,
                    artifact_paths=result.artifacts,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract, assignment, trace_path, _, result, row = _stage_real_pilot_rollout(
                root
            )
            validated = validate_pilot_rollout_artifact(
                contract,
                registry,
                assignment,
                row,
                result.artifacts.trace,
                artifact_paths=result.artifacts,
            )
            wrong_job_path = trace_path.with_name("rollout_wrong_job.jsonl")
            with self.assertRaisesRegex(ValueError, "job_id"):
                publish_registry_rollout(
                    result,
                    wrong_job_path,
                    validation_receipt=validated.validation_receipt,
                )
            result.artifacts.candidate_patch.write_text(
                result.artifacts.candidate_patch.read_text(encoding="utf-8") + "tampered\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "artifact_sha256"):
                publish_registry_rollout(
                    result,
                    trace_path,
                    validation_receipt=validated.validation_receipt,
                )

            self.assertFalse(trace_path.exists())

    def test_loader_rejects_symlinked_canonical_trace_and_sidecar(self) -> None:
        for artifact_name in ("trace", "run_evidence"):
            with (
                self.subTest(artifact_name=artifact_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                registry, contract = _registry_and_contract(root)
                trace_root = root / "traces"
                rows, _ = _write_artifacts(registry, contract, trace_root)
                assignment = contract.rollouts[0]
                paths = RolloutArtifactPaths.for_trace(trace_root / f"{assignment.job_id}.jsonl")
                artifact_path = getattr(paths, artifact_name)
                outside = root / "outside" / artifact_path.name
                outside.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.replace(outside)
                artifact_path.symlink_to(outside)

                with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                    load_pilot_trace_artifacts(
                        contract,
                        registry,
                        rows,
                        trace_root,
                    )

    def test_loader_rejects_symlinked_trace_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            real_trace_root = root / "real-traces"
            rows, _ = _write_artifacts(registry, contract, real_trace_root)
            trace_root = root / "traces"
            trace_root.symlink_to(real_trace_root, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "non-symlink directory"):
                load_pilot_trace_artifacts(
                    contract,
                    registry,
                    rows,
                    trace_root,
                )

    def test_exports_and_quality_report_enforce_m2_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            trace_root = root / "traces"
            rows, reproduction = _write_artifacts(registry, contract, trace_root)

            loaded = load_pilot_trace_artifacts(contract, registry, rows, trace_root)
            with _trust_synthetic_reproduction(reproduction):
                preliminary = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    trace_root,
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                )
            queue = build_trajectory_review_queue(preliminary["review_summaries"])
            review_gate = summarize_review_gate(queue, _acceptable_decisions(queue))
            export_manifest = write_pilot_exports(
                contract,
                registry,
                rows,
                trace_root,
                root / "exports",
                reproduction=reproduction,
                private_reproduction_directory=root / "private-reproductions",
                quarantined_trace_ids=review_gate["quarantined_trace_ids"],
            )
            with _trust_synthetic_reproduction(reproduction):
                report = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    trace_root,
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                    export_manifest=export_manifest,
                    export_directory=root / "exports",
                    review_gate=review_gate,
                )

            export_lines = {
                name: (root / "exports" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
                for name in ("analysis", "rl", "sft", "preference")
            }

        self.assertEqual(len(loaded), 40)
        self.assertEqual(len(export_lines["analysis"]), 40)
        self.assertEqual(len(export_lines["rl"]), 40)
        self.assertEqual(len(export_lines["sft"]), 20)
        self.assertEqual(len(export_lines["preference"]), 20)
        self.assertTrue(all(json.loads(line)["margin"] > 0 for line in export_lines["preference"]))
        self.assertEqual(report["canonical_trajectories"], 40)
        self.assertEqual(report["successes"], 20)
        self.assertEqual(report["infrastructure_failure_rate"], 0.0)
        self.assertEqual(report["leak_trace_ids"], [])
        self.assertEqual(report["hard_bypass_trace_ids"], [])
        self.assertEqual(len(report["review_summaries"]), 40)
        self.assertTrue(report["gates"]["independent_clean_reset_reproduction_verified"])
        self.assertTrue(report["gates"]["immutable_usage_ledger_reconciled"])
        self.assertEqual(report["usage_ledger"]["job_count"], 40)
        self.assertEqual(
            report["reproduction_reverification"]["executed_success_count"],
            20,
        )
        self.assertEqual(
            report["reproduction_reverification"]["fresh_reproduction_sha256"],
            reproduction["reproduction_sha256"],
        )
        self.assertTrue(
            all(
                summary["trace_path"] == f"{summary['job_id']}.jsonl"
                and summary["contract_id"] == contract.contract_id
                for summary in report["review_summaries"]
            )
        )
        self.assertTrue(report["passed"])

    def test_quality_targets_allow_all_success_without_preference_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, reproduction = _write_artifacts(
                registry,
                contract,
                root / "traces",
                success_rollout_indices={0, 1},
            )
            with _trust_synthetic_reproduction(reproduction):
                preliminary = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    root / "traces",
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                )
            queue = build_trajectory_review_queue(preliminary["review_summaries"])
            review_gate = summarize_review_gate(queue, _acceptable_decisions(queue))
            manifest = write_pilot_exports(
                contract,
                registry,
                rows,
                root / "traces",
                root / "exports",
                reproduction=reproduction,
                private_reproduction_directory=root / "private-reproductions",
            )
            with _trust_synthetic_reproduction(reproduction):
                report = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    root / "traces",
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                    export_manifest=manifest,
                    export_directory=root / "exports",
                    review_gate=review_gate,
                )

        self.assertEqual(report["successes"], 40)
        self.assertEqual(
            report["export_record_counts"],
            {"analysis": 40, "rl": 40, "sft": 40, "preference": 0},
        )
        self.assertEqual(report["eligible_positive_margin_scenario_count"], 0)
        self.assertTrue(report["gates"]["preference_count_matches_positive_margin_eligibility"])
        self.assertTrue(report["passed"])

    def test_quality_targets_reject_zero_success_vacuous_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, reproduction = _write_artifacts(
                registry,
                contract,
                root / "traces",
                success_rollout_indices=set(),
            )
            with _trust_synthetic_reproduction(reproduction):
                preliminary = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    root / "traces",
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                )
            queue = build_trajectory_review_queue(preliminary["review_summaries"])
            review_gate = summarize_review_gate(queue, _acceptable_decisions(queue))
            manifest = write_pilot_exports(
                contract,
                registry,
                rows,
                root / "traces",
                root / "exports",
                reproduction=reproduction,
                private_reproduction_directory=root / "private-reproductions",
            )
            with _trust_synthetic_reproduction(reproduction):
                report = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    root / "traces",
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                    export_manifest=manifest,
                    export_directory=root / "exports",
                    review_gate=review_gate,
                )

        self.assertEqual(report["successes"], 0)
        self.assertEqual(report["export_record_counts"]["sft"], 0)
        self.assertFalse(report["gates"]["minimum_successes_met"])
        self.assertFalse(report["gates"]["minimum_sft_records_met"])
        self.assertFalse(report["gates"]["independent_clean_reset_reproduction_verified"])
        self.assertFalse(report["passed"])

    def test_strict_loader_rejects_incomplete_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, _ = _write_artifacts(registry, contract, root / "traces")
            rows[-1]["status"] = "pending"

            with self.assertRaisesRegex(ValueError, "missing 1"):
                load_pilot_trace_artifacts(contract, registry, rows, root / "traces")

    def test_loader_rejects_private_report_not_bound_to_public_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, _ = _write_artifacts(registry, contract, root / "traces")
            assignment = contract.rollouts[0]
            paths = RolloutArtifactPaths.for_trace(root / "traces" / f"{assignment.job_id}.jsonl")
            private = json.loads(paths.private_evaluation.read_text(encoding="utf-8"))
            private["report"]["results"][0]["evidence"]["stdout"] = "/private/evaluator/oracle.txt"
            private.pop("private_evaluation_id")
            private["private_evaluation_id"] = stable_id(
                "private_evaluation",
                private,
            )
            paths.private_evaluation.write_text(json.dumps(private), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Public and private"):
                load_pilot_trace_artifacts(contract, registry, rows, root / "traces")

    def test_loader_recomputes_provider_prompt_hashes_from_complete_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, _ = _write_artifacts(registry, contract, root / "traces")
            assignment = contract.rollouts[0]
            paths = RolloutArtifactPaths.for_trace(root / "traces" / f"{assignment.job_id}.jsonl")
            evidence = json.loads(paths.run_evidence.read_text(encoding="utf-8"))
            evidence["observed_calls"][0]["prompt_hash"] = "f" * 64
            evidence.pop("evidence_id")
            evidence["evidence_id"] = stable_id("run_evidence", evidence)
            paths.run_evidence.write_text(json.dumps(evidence), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Observed prompt lineage mismatch"):
                load_pilot_trace_artifacts(contract, registry, rows, root / "traces")

    def test_loader_rejects_provider_response_id_reused_across_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, _ = _write_artifacts(registry, contract, root / "traces")
            first, second = contract.rollouts[:2]
            first_evidence = json.loads(
                RolloutArtifactPaths.for_trace(
                    root / "traces" / f"{first.job_id}.jsonl"
                ).run_evidence.read_text(encoding="utf-8")
            )
            second_path = RolloutArtifactPaths.for_trace(
                root / "traces" / f"{second.job_id}.jsonl"
            ).run_evidence
            second_evidence = json.loads(second_path.read_text(encoding="utf-8"))
            reused_id = first_evidence["observed_calls"][0]["provider_response_identity"]["id"]
            identity = second_evidence["observed_calls"][0]["provider_response_identity"]
            identity["id"] = reused_id
            second_evidence["observed_calls"][0]["provider_response_identity_sha256"] = (
                canonical_sha256(identity)
            )
            second_evidence.pop("evidence_id")
            second_evidence["evidence_id"] = stable_id(
                "run_evidence",
                second_evidence,
            )
            second_path.write_text(json.dumps(second_evidence), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "globally unique"):
                load_pilot_trace_artifacts(contract, registry, rows, root / "traces")

    def test_reproduction_rejects_truthy_string_and_recomputed_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, reproduction = _write_artifacts(registry, contract, root / "traces")
            artifacts = load_pilot_trace_artifacts(contract, registry, rows, root / "traces")
            tampered = json.loads(json.dumps(reproduction))
            item = tampered["items"][0]
            item["reproduced"] = "false"
            item.pop("reproduction_id")
            item["reproduction_id"] = stable_id("reproduction", item)
            tampered["reproduced_count"] -= 1
            tampered["all_successes_reproduced"] = False
            tampered.pop("reproduction_sha256")
            tampered["reproduction_sha256"] = canonical_sha256(tampered)

            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                validate_pilot_reproduction(
                    contract,
                    artifacts,
                    tampered,
                    root / "private-reproductions",
                )

    def test_reproduction_requires_content_bound_private_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, reproduction = _write_artifacts(registry, contract, root / "traces")
            artifacts = load_pilot_trace_artifacts(contract, registry, rows, root / "traces")
            missing_job = reproduction["items"][0]["job_id"]
            (root / "private-reproductions" / f"{missing_job}.json").unlink()

            with self.assertRaisesRegex(ValueError, "Missing private reproduction"):
                validate_pilot_reproduction(
                    contract,
                    artifacts,
                    reproduction,
                    root / "private-reproductions",
                )

    def test_quality_report_reexecutes_and_rejects_different_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, reproduction = _write_artifacts(registry, contract, root / "traces")
            fresh = json.loads(json.dumps(reproduction))
            fresh["items"].reverse()
            fresh.pop("reproduction_sha256")
            fresh["reproduction_sha256"] = canonical_sha256(fresh)

            def fake_reproduction(
                _contract,
                _registry,
                _rows,
                _trace_directory,
                output,
            ):
                output_path = Path(output)
                private_root = output_path.parent / "private-reproductions"
                private_root.mkdir(parents=True)
                for item in fresh["items"]:
                    name = f"{item['job_id']}.json"
                    source = root / "private-reproductions" / name
                    (private_root / name).write_bytes(source.read_bytes())
                output_path.write_text(json.dumps(fresh), encoding="utf-8")
                return fresh

            with (
                _trust_synthetic_usage(),
                patch(
                    "easy_agentic_data.pilot_artifacts.reproduce_successful_trajectories",
                    side_effect=fake_reproduction,
                ) as rerun,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "does not match the declared reproduction",
                ):
                    build_pilot_quality_report(
                        contract,
                        registry,
                        rows,
                        root / "traces",
                        reproduction=reproduction,
                        private_reproduction_directory=root / "private-reproductions",
                    )

            rerun.assert_called_once()

    def test_export_validator_rejects_file_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, reproduction = _write_artifacts(registry, contract, root / "traces")
            artifacts = load_pilot_trace_artifacts(contract, registry, rows, root / "traces")
            manifest = write_pilot_exports(
                contract,
                registry,
                rows,
                root / "traces",
                root / "exports",
                reproduction=reproduction,
                private_reproduction_directory=root / "private-reproductions",
            )
            sft_path = root / "exports" / "sft.jsonl"
            sft_path.write_text(
                sft_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "content does not match"):
                validate_pilot_export_manifest(
                    contract,
                    artifacts,
                    reproduction,
                    root / "private-reproductions",
                    manifest,
                    root / "exports",
                )

    def test_quality_gate_binds_review_quarantine_to_exports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root)
            rows, reproduction = _write_artifacts(registry, contract, root / "traces")
            with _trust_synthetic_reproduction(reproduction):
                preliminary = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    root / "traces",
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                )
            queue = build_trajectory_review_queue(preliminary["review_summaries"])
            decisions = _acceptable_decisions(queue)
            critical_index = next(
                index for index, item in enumerate(queue["items"]) if item["success"]
            )
            decisions[critical_index] = ReviewDecision(
                trace_id=queue["items"][critical_index]["trace_id"],
                reviewer_alias="fixture-reviewer",
                timestamp="2026-07-14T00:00:00Z",
                verdict="critical",
                issue_codes=("incorrect.repair",),
                quarantine=True,
            )
            review_gate = summarize_review_gate(queue, decisions)
            manifest = write_pilot_exports(
                contract,
                registry,
                rows,
                root / "traces",
                root / "exports",
                reproduction=reproduction,
                private_reproduction_directory=root / "private-reproductions",
            )
            with _trust_synthetic_reproduction(reproduction):
                report = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    root / "traces",
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                    export_manifest=manifest,
                    export_directory=root / "exports",
                    review_gate=review_gate,
                )

        self.assertTrue(review_gate["passed"])
        self.assertFalse(report["gates"]["review_quarantine_matches_exports"])
        self.assertFalse(report["passed"])

    def test_quality_report_fails_closed_when_total_token_budget_is_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry, contract = _registry_and_contract(root, max_total_tokens=500)
            rows, reproduction = _write_artifacts(registry, contract, root / "traces")
            with _trust_synthetic_reproduction(reproduction):
                report = build_pilot_quality_report(
                    contract,
                    registry,
                    rows,
                    root / "traces",
                    reproduction=reproduction,
                    private_reproduction_directory=root / "private-reproductions",
                )

        self.assertEqual(report["usage"]["total_tokens"], 600)
        self.assertFalse(report["gates"]["within_total_token_budget"])
        self.assertFalse(report["passed"])


def _registry_and_contract(
    root: Path,
    *,
    max_total_tokens: int = 1_000_000,
) -> tuple[ScenarioRegistry, PilotRunContract]:
    registry = ScenarioRegistry(root / "registry")
    records = []
    for index in range(20):
        scenario = Scenario(
            QuerySeed(PublicTaskContext(f"Repair task {index}.")),
            EnvironmentSpec(
                name=f"fixture-{index}",
                version="1",
                image_digest="sha256:" + f"{index:064x}",
                metadata={
                    "repository": f"owner/repo-{index % 8}",
                    "language": "Python" if index % 2 == 0 else "JavaScript",
                },
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
    provider = ProviderConfigBinding.from_config(_provider_config())
    contract = PilotRunContract(
        corpus=corpus,
        provider=provider,
        budgets=PilotBudgets(
            max_agent_turns=10,
            max_agent_tool_calls=20,
            max_agent_tokens=10_000,
            max_agent_seconds=60,
            max_total_tokens=max_total_tokens,
            max_total_cost_usd="10",
            max_total_seconds=3600,
        ),
        versions=current_pilot_versions(corpus, registry),
        pricing=PricingSpec(
            input_usd_per_million_tokens="1",
            cached_input_usd_per_million_tokens="0.1",
            output_usd_per_million_tokens="2",
        ),
        rollout_seeds=(0, 1),
    )
    return registry, contract


def _provider_config() -> LLMConfig:
    return LLMConfig(
        provider="local_openai_compatible",
        model="fixture-model",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env=None,
        temperature=0.0,
    )


class _PilotMemorySandbox(MemorySandbox):
    def __init__(self) -> None:
        super().__init__(
            {"app.py": "value = 1\n"},
            {
                "python hidden.py": lambda _sandbox: CommandResult(
                    0,
                    "private verifier passed\n",
                    "",
                    1.0,
                )
            },
        )

    def execute_as_root(self, command, *, timeout_seconds=None):
        return self.execute(command, timeout_seconds=timeout_seconds)

    def prepare_git_baseline(self) -> str:
        return self.state_hash()

    def candidate_patch(self) -> str:
        return json.dumps(self.files, sort_keys=True)

    def apply_candidate_patch(self, patch: str) -> str:
        self.files = json.loads(patch)
        return self.state_hash()


class _PilotOneShotClient:
    model = "fixture-model"
    temperature = 0.0
    max_tokens = 2048

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        return LLMResponse(
            Message("assistant", "No public changes are required for this fixture."),
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


def _concatenated_json_objects(value: str) -> list[dict]:
    decoder = json.JSONDecoder()
    items = []
    offset = 0
    while offset < len(value):
        while offset < len(value) and value[offset].isspace():
            offset += 1
        if offset >= len(value):
            break
        item, offset = decoder.raw_decode(value, offset)
        items.append(item)
    return items


def _acceptable_decisions(queue: dict) -> list[ReviewDecision]:
    return [
        ReviewDecision(
            trace_id=item["trace_id"],
            reviewer_alias="fixture-reviewer",
            timestamp="2026-07-14T00:00:00Z",
            verdict="acceptable",
        )
        for item in queue["items"]
    ]


@contextmanager
def _trust_synthetic_reproduction(reproduction: dict):
    reproduced = {item["trace_id"]: item["reproduced"] for item in reproduction["items"]}

    with (
        patch(
            "easy_agentic_data.pilot_artifacts._reverify_reproduction_from_clean_reset",
            return_value=(dict(reproduction), reproduced),
        ),
        _trust_synthetic_usage(),
    ):
        yield


@contextmanager
def _trust_synthetic_usage():
    def synthetic_usage_audit(_contract, rows, _trace_directory, **_kwargs):
        states = {}
        for row in rows:
            metrics = json.loads(row.get("metrics", "{}"))
            totals = ConsumedUsageTotals(
                int(row.get("consumed_tokens", row.get("tokens", 0))),
                float(row.get("consumed_cost", row.get("cost", 0.0))),
                float(row.get("consumed_elapsed_ms", metrics.get("elapsed_ms", 0.0))),
            )
            states[row["job_id"]] = SimpleNamespace(totals=totals)
        return SimpleNamespace(
            jobs=states,
            to_evidence=lambda: {
                "contract_id": _contract.contract_id,
                "job_count": len(states),
                "attempt_count": len(states),
                "call_count": len(states),
                "ledger_sha256": canonical_sha256("synthetic-ledger"),
                "jobs_sha256": canonical_sha256(sorted(states)),
            },
        )

    with patch(
        "easy_agentic_data.pilot_artifacts.audit_pilot_usage_ledger",
        side_effect=synthetic_usage_audit,
    ):
        yield


def _stage_real_pilot_rollout(root: Path):
    registry, contract = _registry_and_contract(root)
    assignment = contract.rollouts[0]
    trace_path = root / "traces" / f"{assignment.job_id}.jsonl"
    budgets = AgentBudgets(
        max_turns=contract.budgets.max_agent_turns,
        max_tool_calls=contract.budgets.max_agent_tool_calls,
        max_tokens=contract.budgets.max_agent_tokens,
        max_seconds=contract.budgets.max_agent_seconds,
        malformed_tool_retries=contract.budgets.malformed_tool_retries,
    )
    usage_attempt = PilotUsageAttempt(
        root / "traces",
        contract_id=contract.contract_id,
        job_id=assignment.job_id,
    )
    with (
        patch.object(
            registry_rollouts_module,
            "_docker_sandbox",
            side_effect=lambda _scenario, _source: _PilotMemorySandbox(),
        ),
        patch.object(
            registry_rollouts_module,
            "_build_llm_client",
            side_effect=lambda _config: _PilotOneShotClient(),
        ),
        patch.object(
            registry_rollouts_module,
            "materialize_environment_source",
            side_effect=_empty_source,
        ),
    ):
        result = run_registry_rollout(
            registry,
            assignment.scenario_id,
            _provider_config(),
            trace_path,
            assignment.random_seed,
            budgets,
            cost_calculator=lambda usage: float(contract.pricing.calculate_cost(usage).cost_usd),
            run_contract_id=contract.contract_id,
            provider_binding_sha256=contract.provider.config_sha256,
            provider_binding=contract.provider.to_dict(),
            version_hashes=contract.versions.to_dict(),
            usage_attempt=usage_attempt,
            publish=False,
        )
    usage_cost = contract.pricing.calculate_cost(result.usage)
    row = {
        "job_id": assignment.job_id,
        "scenario_id": assignment.scenario_id,
        "rollout_index": assignment.rollout_index,
        "model": contract.provider.model,
        "config_hash": contract.contract_id,
        "status": "completed",
        "trace_id": result.trace.trace_id,
        "success": int(result.report.success),
        "tokens": usage_cost.total_tokens,
        "cost": float(usage_cost.cost_usd),
        "metrics": json.dumps(result.metrics, sort_keys=True),
    }
    return registry, contract, assignment, trace_path, usage_attempt, result, row


def _write_artifacts(
    registry: ScenarioRegistry,
    contract: PilotRunContract,
    trace_root: Path,
    *,
    success_rollout_indices: set[int] | None = None,
) -> tuple[list[dict], dict]:
    rows = []
    reproduction_items = []
    successful_indices = {0} if success_rollout_indices is None else success_rollout_indices
    private_reproduction_root = trace_root.parent / "private-reproductions"
    private_reproduction_root.mkdir(parents=True, exist_ok=True)
    for assignment in contract.rollouts:
        success = assignment.rollout_index in successful_indices
        initial_hash = f"state_{assignment.scenario_id}"
        instance = registry.materialize(
            assignment.scenario_id,
            random_seed=assignment.random_seed,
            initial_state_hash=initial_hash,
        )
        trace_path = trace_root / f"{assignment.job_id}.jsonl"
        hard = EvaluationEvidence(
            "hidden_command",
            success,
            1.0 if success else 0.0,
            "private verifier output",
            {"stdout": "private hidden diagnostic"},
        )
        termination = EvaluationEvidence(
            "agent_termination",
            True,
            1.0,
            "Agent completed normally",
            {"termination_reason": "agent_stop"},
        )
        report = EvaluationReport(
            instance.instance_id,
            [hard, termination],
            success,
            1 if success else 0,
            False,
            {
                "turns": 1.0,
                "tool_calls": 0.0,
                "tokens": 15.0,
                "agent_elapsed_ms": 1.0,
            },
        )
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        with TraceRecorder(
            trace_path,
            session_id=f"session_{assignment.job_id}",
            scenario_instance=instance,
        ) as recorder:
            recorder.start(instance, system_prompt=DEFAULT_SYSTEM_PROMPT)
            recorder.record(
                EventType.USER_MESSAGE,
                {"message_id": "user_0", "content": instance.public_task.query},
            )
            recorder.record(
                EventType.MODEL_RESPONSE,
                {
                    "message_id": "assistant_0",
                    "content": "Applied the candidate repair.",
                    "model": "fixture-model",
                    "usage": usage,
                },
            )
            for result in report.results:
                public = public_evaluation_result(result)
                recorder.record(
                    EventType.VERIFICATION_RESULT,
                    {"verifier": public.pop("evaluator"), **public},
                )
            recorder.record(
                EventType.SESSION_FINISHED,
                {
                    "termination_reason": (
                        TerminationReason.SUCCESS.value
                        if success
                        else TerminationReason.AGENT_STOP.value
                    ),
                    "final_state_hash": initial_hash,
                    "success": success,
                },
            )
        trace = load_trace(trace_path)
        prompt_fingerprint = trace_prompt_fingerprints(
            trace,
            DEFAULT_SYSTEM_PROMPT,
            _scenario_tool_schemas(registry.get_scenario(assignment.scenario_id)),
        )[0]
        paths = RolloutArtifactPaths.for_trace(trace_path)
        paths.candidate_patch.parent.mkdir(parents=True, exist_ok=True)
        paths.private_evaluation.parent.mkdir(parents=True, exist_ok=True)
        paths.run_evidence.parent.mkdir(parents=True, exist_ok=True)
        paths.candidate_patch.write_text("", encoding="utf-8")
        private_evaluation = {
            "schema": "easy_agentic_data.private_evaluation.v1",
            "trace_id": trace.trace_id,
            "candidate_patch_sha256": hashlib.sha256(b"").hexdigest(),
            "clean_reset": True,
            "report": report.to_dict(),
        }
        private_evaluation["private_evaluation_id"] = stable_id(
            "private_evaluation",
            private_evaluation,
        )
        paths.private_evaluation.write_text(
            json.dumps(private_evaluation),
            encoding="utf-8",
        )
        cost = contract.pricing.calculate_cost(usage)
        provider_response_identity = {
            "id": f"completion_{assignment.job_id}",
            "created": assignment.random_seed,
            "object": "chat.completion",
            "model": "fixture-model",
        }
        evidence = {
            "schema": "easy_agentic_data.registry_rollout_evidence.v1",
            "run_contract_id": contract.contract_id,
            "provider_binding_sha256": contract.provider.config_sha256,
            "usage_attempt_id": f"attempt_fixture_{assignment.job_id}",
            "trace_id": trace.trace_id,
            "scenario_id": assignment.scenario_id,
            "scenario_instance_id": instance.instance_id,
            "environment_id": instance.environment_id,
            "image_digest": registry.get_scenario(assignment.scenario_id).environment.image_digest,
            "random_seed": assignment.random_seed,
            "provider_config": contract.provider.to_dict(),
            "provider_runtime_sha256": canonical_sha256(contract.provider.to_dict()),
            "contract_versions": contract.versions.to_dict(),
            "budgets": {
                "max_turns": contract.budgets.max_agent_turns,
                "max_tool_calls": contract.budgets.max_agent_tool_calls,
                "max_tokens": contract.budgets.max_agent_tokens,
                "max_seconds": contract.budgets.max_agent_seconds,
                "malformed_tool_retries": contract.budgets.malformed_tool_retries,
            },
            "prompt_sha256": contract.versions.prompt_sha256,
            "tool_schema_sha256": _scenario_tool_schema_sha256(
                registry.get_scenario(assignment.scenario_id)
            ),
            "evaluator_names": [result.evaluator for result in report.results],
            "evaluator_set_sha256": canonical_sha256(
                [result.evaluator for result in report.results]
            ),
            "candidate_patch_sha256": hashlib.sha256(b"").hexdigest(),
            "initial_state_hash": initial_hash,
            "candidate_state_hash": initial_hash,
            "clean_reset_verification": True,
            "success": report.success,
            "infrastructure_failure": report.infrastructure_failure,
            "reward": report.reward,
            "termination_reason": "agent_stop",
            "turns": 1,
            "tool_calls": 0,
            "usage": usage,
            "cost": float(cost.cost_usd),
            "retry_count": 0,
            "observed_calls": [
                {
                    "call_index": 0,
                    "started_at": "2026-07-14T00:00:00Z",
                    "model": contract.provider.model,
                    "message_count": prompt_fingerprint["message_count"],
                    "tool_count": prompt_fingerprint["tool_count"],
                    "temperature": contract.provider.temperature,
                    "max_tokens": contract.provider.max_tokens,
                    "retry_count": 0,
                    "prompt_hash": prompt_fingerprint["prompt_hash"],
                    "prompt_token_upper_bound": prompt_fingerprint["prompt_token_upper_bound"],
                    "response_format": None,
                    "status": "completed",
                    "response_model": "fixture-model",
                    "provider_response_identity": provider_response_identity,
                    "provider_response_identity_sha256": canonical_sha256(
                        provider_response_identity
                    ),
                    "provider_response_sha256": canonical_sha256(provider_response_identity),
                    "usage": usage,
                    "latency_ms": 1.0,
                }
            ],
            "started_at": "2026-07-14T00:00:00Z",
            "elapsed_ms": 1.0,
        }
        evidence["evidence_id"] = stable_id("run_evidence", evidence)
        paths.run_evidence.write_text(
            json.dumps(evidence),
            encoding="utf-8",
        )
        scheduler_metrics = {
            **report.metrics,
            **evaluation_result_metrics(report),
            "elapsed_ms": 1.0,
        }
        rows.append(
            {
                "job_id": assignment.job_id,
                "scenario_id": assignment.scenario_id,
                "rollout_index": assignment.rollout_index,
                "model": contract.provider.model,
                "config_hash": contract.contract_id,
                "status": "completed",
                "trace_id": trace.trace_id,
                "success": int(success),
                "tokens": cost.total_tokens,
                "cost": float(cost.cost_usd),
                "metrics": json.dumps(scheduler_metrics, sort_keys=True),
            }
        )
        if success:
            private_reproduction = {
                "schema": "easy_agentic_data.private_reproduction.v1",
                "job_id": assignment.job_id,
                "trace_id": trace.trace_id,
                "report": report.to_dict(),
            }
            private_reproduction["private_reproduction_id"] = stable_id(
                "private_reproduction",
                private_reproduction,
            )
            (private_reproduction_root / f"{assignment.job_id}.json").write_text(
                json.dumps(private_reproduction),
                encoding="utf-8",
            )
            reproduction_item = {
                "job_id": assignment.job_id,
                "trace_id": trace.trace_id,
                "scenario_id": assignment.scenario_id,
                "random_seed": assignment.random_seed,
                "candidate_patch_sha256": hashlib.sha256(b"").hexdigest(),
                "reproduced": True,
                "signature_matches": True,
                "infrastructure_failure": False,
                "private_reproduction_sha256": canonical_sha256(private_reproduction),
                "results": [public_evaluation_result(result) for result in report.results],
            }
            reproduction_item["reproduction_id"] = stable_id(
                "reproduction",
                reproduction_item,
            )
            reproduction_items.append(reproduction_item)
    reproduction_material = {
        "schema": "easy_agentic_data.pilot_reproduction.v1",
        "contract_id": contract.contract_id,
        "required_success_count": len(reproduction_items),
        "reproduction_count": len(reproduction_items),
        "reproduced_count": len(reproduction_items),
        "all_successes_reproduced": True,
        "items": reproduction_items,
    }
    reproduction = {
        **reproduction_material,
        "reproduction_sha256": canonical_sha256(reproduction_material),
    }
    return rows, reproduction
