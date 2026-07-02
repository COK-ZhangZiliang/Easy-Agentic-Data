import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.batch import (
    JsonCallCache,
    PersistentScheduler,
    ResourceGates,
    RolloutJob,
    RolloutOutcome,
    RunBudget,
    audit_trace_logic,
    enqueue_human_review,
    estimate_scale_run,
    load_human_reviews,
    planned_batch_run,
    quality_report,
    retry_with_backoff,
    scale_continuation_decision,
    scale_readiness_summary,
    scenario_quality_report,
    select_scale_candidates,
    selected_job_status,
    worker_health,
    write_release_manifest,
)
from easy_agentic_data.cli import (
    _rollout_outcome_from_existing_trace,
    _selected_job_ids_for_run,
    main,
)
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.scenarios import Scenario
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


class BatchTests(unittest.TestCase):
    def test_interrupted_batch_resumes_without_duplicate_traces(self) -> None:
        class Worker:
            def __init__(self):
                self.calls = []

            def run(self, job):
                self.calls.append(job.job_id)
                return RolloutOutcome(trace_id=f"trace_{job.job_id}", success=True, tokens=10)

        with tempfile.TemporaryDirectory() as directory:
            scheduler = PersistentScheduler(Path(directory) / "jobs.sqlite3")
            jobs = [RolloutJob(f"scenario_{index}", 0, "model", "config") for index in range(3)]
            scheduler.submit(jobs)
            worker = Worker()
            scheduler.run(worker, max_jobs=1)
            scheduler.submit(jobs)
            scheduler.run(worker)
            scheduler.run(worker)
            rows = scheduler.completed_rows()

        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["trace_id"] for row in rows}), 3)
        self.assertEqual(len(worker.calls), 3)

    def test_scheduler_runs_only_selected_job_ids(self) -> None:
        class Worker:
            def __init__(self):
                self.calls = []

            def run(self, job):
                self.calls.append(job.job_id)
                return RolloutOutcome(trace_id=f"trace_{job.job_id}", success=True)

        with tempfile.TemporaryDirectory() as directory:
            scheduler = PersistentScheduler(Path(directory) / "jobs.sqlite3")
            jobs = [RolloutJob(f"scenario_{index}", 0, "model", "config") for index in range(3)]
            scheduler.submit(jobs)
            worker = Worker()
            summary = scheduler.run(worker, job_ids=[jobs[1].job_id])

            rows = scheduler.rows()

        self.assertEqual(summary["processed"], 1)
        self.assertEqual(worker.calls, [jobs[1].job_id])
        self.assertEqual(
            {row["job_id"] for row in rows if row["status"] == "completed"},
            {jobs[1].job_id},
        )

    def test_infrastructure_failure_retries_then_completes(self) -> None:
        class FlakyWorker:
            def __init__(self):
                self.calls = 0

            def run(self, job):
                del job
                self.calls += 1
                if self.calls == 1:
                    return RolloutOutcome(infrastructure_failure=True, error="temporary")
                return RolloutOutcome(trace_id="trace_ok", success=True)

        with tempfile.TemporaryDirectory() as directory:
            scheduler = PersistentScheduler(Path(directory) / "jobs.sqlite3")
            scheduler.submit([RolloutJob("scenario", 0, "model", "config")])
            worker = FlakyWorker()
            scheduler.run(worker, max_retries=2)
            scheduler.run(worker, max_retries=2)
            rows = scheduler.completed_rows()
        self.assertEqual(worker.calls, 2)
        self.assertEqual(rows[0]["trace_id"], "trace_ok")

    def test_worker_exception_is_recorded_and_retried(self) -> None:
        class RaisingWorker:
            def __init__(self):
                self.calls = 0

            def run(self, job):
                del job
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("worker crashed")
                return RolloutOutcome(trace_id="trace_recovered", success=True)

        with tempfile.TemporaryDirectory() as directory:
            scheduler = PersistentScheduler(Path(directory) / "jobs.sqlite3")
            scheduler.submit([RolloutJob("scenario", 0, "model", "config")])
            worker = RaisingWorker()
            first = scheduler.run(worker, max_retries=1)
            second = scheduler.run(worker, max_retries=1)

        self.assertEqual(first["status_counts"]["pending"], 1)
        self.assertEqual(second["status_counts"]["completed"], 1)
        self.assertEqual(worker.calls, 2)

    def test_cache_manifest_quality_and_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = JsonCallCache(root / "cache.json")
            calls = []
            first = cache.get_or_compute({"prompt": "x"}, lambda: calls.append(1) or {"v": 1})
            second = cache.get_or_compute({"prompt": "x"}, lambda: calls.append(2) or {"v": 2})
            manifest = root / "manifest.json"
            write_release_manifest(
                manifest,
                scenarios={"version": "1"},
                models={"agent": "m"},
                prompts={"agent": "p1"},
                tools={"coding": "1"},
                images={"fixture": "sha256:x"},
                evaluators={"hidden": "1"},
                exporters={"rl": "1"},
            )
            review = root / "review.jsonl"
            enqueue_human_review(review, {"trace_id": "trace_a", "reason": "sample"})
            report = quality_report(
                [
                    {
                        "trace_id": "trace_a",
                        "scenario_id": "scenario_a",
                        "success": 1,
                        "reward": 1.0,
                        "status": "completed",
                        "goal_type": "refund",
                        "metrics": {
                            "simulator_error_rate": 0.0,
                            "goal_alignment": 1.0,
                            "verifier_hidden_command_passed": 1.0,
                        },
                    },
                    {
                        "trace_id": "trace_a",
                        "scenario_id": "scenario_a",
                        "success": 0,
                        "reward": 0.0,
                        "status": "failed",
                        "goal_type": "refund",
                        "metrics": {
                            "simulator_error_rate": 0.5,
                            "goal_alignment": 0.5,
                            "verifier_hidden_command_passed": 0.0,
                        },
                    },
                ]
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            review_lines = review.read_text(encoding="utf-8").splitlines()
            reviews = load_human_reviews(review)

        self.assertEqual(first, second)
        self.assertEqual(calls, [1])
        self.assertEqual(payload["models"]["agent"], "m")
        self.assertEqual(report["duplicate_traces"], 1)
        self.assertEqual(report["scenario_count"], 1)
        self.assertEqual(report["pass_at_observed_k"], 1.0)
        self.assertEqual(report["episode_reward_mean"], 0.5)
        self.assertEqual(report["success_by_goal_type"]["refund"], 0.5)
        self.assertGreater(report["in_group_reward_std_mean"], 0.0)
        self.assertEqual(report["average_metrics"]["goal_alignment"], 0.75)
        self.assertEqual(report["average_metrics"]["verifier_hidden_command_passed"], 0.5)
        self.assertEqual(len(review_lines), 1)
        self.assertEqual(reviews[0]["trace_id"], "trace_a")

    def test_batch_report_cli_writes_quality_summary_and_review_sample(self) -> None:
        class MixedWorker:
            def run(self, job):
                if job.scenario_id == "scenario_b":
                    return RolloutOutcome(trace_id="trace_b", success=False, tokens=7)
                return RolloutOutcome(
                    trace_id="trace_a",
                    success=True,
                    tokens=11,
                    metrics={"tool_calls": 3.0, "turns": 2.0},
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "jobs.sqlite3"
            report_path = root / "quality.json"
            review_path = root / "review.jsonl"
            trace_dir = root / "traces"
            review_path.write_text(json.dumps({"job_id": "old"}) + "\n", encoding="utf-8")
            scheduler = PersistentScheduler(database)
            scheduler.submit(
                [
                    RolloutJob("scenario_a", 0, "deepseek-v4-pro", "config"),
                    RolloutJob("scenario_b", 0, "deepseek-v4-pro", "config"),
                ]
            )
            scheduler.run(MixedWorker(), max_retries=0)
            scheduler.submit([RolloutJob("scenario_pending", 0, "deepseek-v4-pro", "config")])
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "report",
                        "--database",
                        str(database),
                        "--output",
                        str(report_path),
                        "--trace-directory",
                        str(trace_dir),
                        "--review-sample",
                        str(review_path),
                        "--overwrite-review-sample",
                        "--sample-size",
                        "2",
                    ]
                )

            printed = json.loads(output.getvalue())
            written = json.loads(report_path.read_text(encoding="utf-8"))
            samples = load_human_reviews(review_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["rollouts"], 2)
        self.assertEqual(written["scenario_count"], 2)
        self.assertEqual(written["success_rate"], 0.5)
        self.assertEqual(written["pass_at_observed_k"], 0.5)
        self.assertEqual(written["unique_traces"], 2)
        self.assertEqual(len(samples), 2)
        self.assertNotIn("old", {sample["job_id"] for sample in samples})
        self.assertIn("trace_path", samples[0])

    def test_batch_report_cli_filters_by_estimate_shard(self) -> None:
        class MixedWorker:
            def run(self, job):
                return RolloutOutcome(
                    trace_id=f"trace_{job.scenario_id}",
                    success=job.scenario_id == "scenario_a",
                    tokens=5,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "jobs.sqlite3"
            report_path = root / "quality.json"
            review_path = root / "review.jsonl"
            scheduler = PersistentScheduler(database)
            jobs = [
                RolloutJob("scenario_a", 0, "model", "config"),
                RolloutJob("scenario_b", 0, "model", "config"),
            ]
            scheduler.submit(jobs)
            scheduler.run(MixedWorker(), max_retries=0)
            estimate_path = root / "estimate.json"
            estimate_path.write_text(
                json.dumps({"shards": [{"job_ids": [jobs[1].job_id]}]}),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "report",
                        "--database",
                        str(database),
                        "--output",
                        str(report_path),
                        "--review-sample",
                        str(review_path),
                        "--sample-size",
                        "5",
                        "--job-id-file",
                        str(estimate_path),
                        "--shard-index",
                        "0",
                    ]
                )

            printed = json.loads(output.getvalue())
            written = json.loads(report_path.read_text(encoding="utf-8"))
            samples = load_human_reviews(review_path)

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["rollouts"], 1)
        self.assertEqual(written["scenario_count"], 1)
        self.assertEqual(written["success_rate"], 0.0)
        self.assertEqual(samples[0]["job_id"], jobs[1].job_id)

    def test_audit_trace_logic_classifies_completed_coding_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_dir = root / "traces"
            trace_dir.mkdir()
            job = RolloutJob("scenario_a", 0, "model", "config")
            _write_trace(
                trace_dir / f"{job.job_id}.jsonl",
                job_id=job.job_id,
                scenario_id=job.scenario_id,
                success=True,
                termination_reason="success",
            )
            rows = [
                {
                    "job_id": job.job_id,
                    "scenario_id": job.scenario_id,
                    "status": "completed",
                    "success": 1,
                }
            ]

            audit = audit_trace_logic(rows, trace_dir)

        self.assertEqual(audit["completed_jobs_reviewed"], 1)
        self.assertEqual(audit["verdict_counts"], {"high_quality": 1})
        self.assertEqual(audit["high_quality_rate"], 1.0)
        self.assertEqual(audit["items"][0]["closed_loop"], True)
        self.assertEqual(audit["items"][0]["multi_step_complex"], True)

    def test_batch_audit_traces_cli_filters_by_estimate_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "jobs.sqlite3"
            trace_dir = root / "traces"
            trace_dir.mkdir()
            output_path = root / "audit.json"
            scheduler = PersistentScheduler(database)
            jobs = [
                RolloutJob("scenario_a", 0, "model", "config"),
                RolloutJob("scenario_b", 0, "model", "config"),
            ]
            scheduler.submit(jobs)
            scheduler.run(_StaticWorker(RolloutOutcome(trace_id="trace_a", success=True)))
            for job in jobs:
                _write_trace(
                    trace_dir / f"{job.job_id}.jsonl",
                    job_id=job.job_id,
                    scenario_id=job.scenario_id,
                    success=True,
                    termination_reason="success",
                )
            estimate_path = root / "estimate.json"
            estimate_path.write_text(
                json.dumps({"shards": [{"job_ids": [jobs[1].job_id]}]}),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "audit-traces",
                        "--database",
                        str(database),
                        "--trace-directory",
                        str(trace_dir),
                        "--job-id-file",
                        str(estimate_path),
                        "--shard-index",
                        "0",
                        "--summary-only",
                        "--output",
                        str(output_path),
                    ]
                )

            printed = json.loads(output.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["completed_jobs_reviewed"], 1)
        self.assertEqual(written["selected_job_count"], 1)
        self.assertNotIn("items", written)
        self.assertEqual(written["verdict_counts"], {"high_quality": 1})

    def test_select_scale_candidates_groups_scenarios_by_quality(self) -> None:
        rows = [
            _quality_row(
                "scenario_strong",
                success=True,
                metrics={
                    "tool_calls": 12.0,
                    "verifier_hidden_command_passed": 1.0,
                    "verifier_all_non_agent_passed": 1.0,
                    "verifier_agent_termination_passed": 1.0,
                },
            ),
            _quality_row(
                "scenario_strong",
                success=False,
                metrics={
                    "tool_calls": 10.0,
                    "verifier_hidden_command_passed": 1.0,
                    "verifier_all_non_agent_passed": 1.0,
                    "verifier_agent_termination_passed": 0.0,
                },
            ),
            _quality_row(
                "scenario_weak",
                success=False,
                metrics={
                    "tool_calls": 14.0,
                    "verifier_hidden_command_passed": 0.0,
                    "verifier_all_non_agent_passed": 0.0,
                },
            ),
            _quality_row(
                "scenario_weak",
                success=False,
                metrics={
                    "tool_calls": 15.0,
                    "verifier_hidden_command_passed": 0.0,
                    "verifier_all_non_agent_passed": 0.0,
                },
            ),
        ]

        reports = scenario_quality_report(rows)
        selection = select_scale_candidates(rows)

        self.assertEqual(reports[0]["scenario_id"], "scenario_strong")
        self.assertEqual(reports[0]["hidden_command_pass_rate"], 1.0)
        self.assertEqual(selection["candidates"], ["scenario_strong"])
        weak = next(
            report
            for report in selection["scenario_reports"]
            if report["scenario_id"] == "scenario_weak"
        )
        self.assertIn("min_success_rate", weak["candidate_failures"])

        strict_selection = select_scale_candidates(rows, min_agent_stop_rate=0.75)
        strict_strong = next(
            report
            for report in strict_selection["scenario_reports"]
            if report["scenario_id"] == "scenario_strong"
        )

        self.assertEqual(strict_selection["candidates"], [])
        self.assertIn("min_agent_stop_rate", strict_strong["candidate_failures"])

        audit_selection = select_scale_candidates(
            rows,
            audit={
                "scenario_reports": [
                    {
                        "scenario_id": "scenario_strong",
                        "rollouts": 2,
                        "high_quality": 2,
                        "closed_loop_rate": 1.0,
                        "multi_step_complex_rate": 1.0,
                    },
                    {
                        "scenario_id": "scenario_weak",
                        "rollouts": 2,
                        "high_quality": 0,
                        "closed_loop_rate": 0.5,
                        "multi_step_complex_rate": 1.0,
                    },
                ]
            },
            min_high_quality_rate=0.75,
            min_closed_loop_rate=0.75,
            min_multi_step_complex_rate=0.75,
        )
        audited_weak = next(
            report
            for report in audit_selection["scenario_reports"]
            if report["scenario_id"] == "scenario_weak"
        )

        self.assertEqual(audit_selection["candidates"], ["scenario_strong"])
        self.assertEqual(audit_selection["scenario_reports"][0]["high_quality_rate"], 1.0)
        self.assertIn("min_high_quality_rate", audited_weak["candidate_failures"])

    def test_batch_select_scale_candidates_cli_writes_selection(self) -> None:
        class QualityWorker:
            def run(self, job):
                return RolloutOutcome(
                    trace_id=f"trace_{job.scenario_id}_{job.rollout_index}",
                    success=job.scenario_id == "scenario_a",
                    tokens=10,
                    metrics={
                        "tool_calls": 8.0,
                        "verifier_hidden_command_passed": (
                            1.0 if job.scenario_id == "scenario_a" else 0.0
                        ),
                        "verifier_all_non_agent_passed": (
                            1.0 if job.scenario_id == "scenario_a" else 0.0
                        ),
                    },
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "jobs.sqlite3"
            output_path = root / "selection.json"
            audit_path = root / "audit.json"
            scheduler = PersistentScheduler(database)
            scheduler.submit(
                [
                    RolloutJob("scenario_a", 0, "model", "config"),
                    RolloutJob("scenario_a", 1, "model", "config"),
                    RolloutJob("scenario_b", 0, "model", "config"),
                    RolloutJob("scenario_b", 1, "model", "config"),
                ]
            )
            scheduler.run(QualityWorker(), max_retries=0)
            audit_path.write_text(
                json.dumps(
                    {
                        "scenario_reports": [
                            {
                                "scenario_id": "scenario_a",
                                "rollouts": 2,
                                "high_quality": 2,
                                "closed_loop_rate": 1.0,
                                "multi_step_complex_rate": 1.0,
                            },
                            {
                                "scenario_id": "scenario_b",
                                "rollouts": 2,
                                "high_quality": 0,
                                "closed_loop_rate": 1.0,
                                "multi_step_complex_rate": 1.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "select-scale-candidates",
                        "--database",
                        str(database),
                        "--audit",
                        str(audit_path),
                        "--min-high-quality-rate",
                        "0.5",
                        "--output",
                        str(output_path),
                    ]
                )

            printed = json.loads(output.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["candidates"], ["scenario_a"])

    def test_batch_enqueue_uses_scale_candidate_selection_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ScenarioRegistry(root / "registry")
            kept = _add_named_scenario(registry, "scenario_keep")
            _add_named_scenario(registry, "scenario_skip")
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps({"candidates": [kept.scenario_id]}),
                encoding="utf-8",
            )
            database = root / "jobs.sqlite3"
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "enqueue",
                        "--registry",
                        str(root / "registry"),
                        "--database",
                        str(database),
                        "--model",
                        "deepseek-v4-pro",
                        "--config-hash",
                        "config",
                        "--rollouts",
                        "3",
                        "--selection-file",
                        str(selection_path),
                    ]
                )

            rows = PersistentScheduler(database).rows()

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), {"pending": 3})
        self.assertEqual({row["scenario_id"] for row in rows}, {kept.scenario_id})

    def test_estimate_scale_run_uses_pilot_scenario_token_rates(self) -> None:
        queue_rows = [
            {"job_id": "job_1", "scenario_id": "scenario_a", "status": "pending"},
            {"job_id": "job_2", "scenario_id": "scenario_a", "status": "pending"},
            {"job_id": "job_3", "scenario_id": "scenario_b", "status": "pending"},
        ]
        pilot_rows = [
            {"scenario_id": "scenario_a", "status": "completed", "tokens": 100},
            {"scenario_id": "scenario_a", "status": "completed", "tokens": 300},
            {"scenario_id": "scenario_b", "status": "completed", "tokens": 50},
        ]

        estimate = estimate_scale_run(
            queue_rows,
            pilot_rows,
            shard_size=2,
            cost_per_million_tokens=2.0,
        )

        self.assertEqual(estimate["pending_jobs"], 3)
        self.assertEqual(estimate["estimated_tokens"], 450.0)
        self.assertEqual(estimate["estimated_cost"], 0.0009)
        self.assertEqual([shard["max_jobs"] for shard in estimate["shards"]], [2, 1])
        self.assertEqual(estimate["shards"][0]["job_ids"], ["job_1", "job_2"])

    def test_selected_job_ids_for_run_reads_estimate_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            estimate_path = Path(directory) / "estimate.json"
            estimate_path.write_text(
                json.dumps(
                    {
                        "shards": [
                            {"job_ids": ["job_b", "job_a"]},
                            {"job_ids": ["job_c"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            job_ids = _selected_job_ids_for_run(
                explicit_job_ids=["job_a"],
                job_id_file=estimate_path,
                shard_index=0,
            )

        self.assertEqual(job_ids, ["job_a", "job_b"])

    def test_selected_job_status_summarizes_explicit_jobs(self) -> None:
        rows = [
            {"job_id": "job_a", "status": "completed", "success": 1, "tokens": 10},
            {"job_id": "job_b", "status": "pending", "success": 0, "tokens": 0},
            {"job_id": "job_c", "status": "infrastructure_failed", "success": 0, "tokens": 3},
        ]

        status = selected_job_status(rows, ["job_a", "job_b", "job_missing"])

        self.assertEqual(status["selected_jobs"], 3)
        self.assertEqual(status["found_jobs"], 2)
        self.assertEqual(status["missing_job_ids"], ["job_missing"])
        self.assertEqual(status["status_counts"], {"completed": 1, "pending": 1})
        self.assertFalse(status["all_selected_terminal"])
        self.assertEqual(status["tokens"], 10)

    def test_planned_batch_run_selects_pending_jobs_without_mutation(self) -> None:
        rows = [
            {"job_id": "job_b", "status": "pending"},
            {"job_id": "job_a", "status": "completed"},
            {"job_id": "job_c", "status": "pending"},
        ]

        plan = planned_batch_run(rows, job_ids=["job_a", "job_c", "job_missing"], max_jobs=1)

        self.assertEqual(plan["requested_job_count"], 3)
        self.assertEqual(plan["selected_job_count"], 2)
        self.assertEqual(plan["missing_job_count"], 1)
        self.assertEqual(plan["missing_job_ids"], ["job_missing"])
        self.assertEqual(plan["runnable_job_count"], 1)
        self.assertEqual(plan["status_counts"], {"completed": 1, "pending": 1})
        self.assertEqual(plan["job_ids"], ["job_c"])

    def test_batch_run_dry_run_selects_estimate_shard_without_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "queue.sqlite3"
            scheduler = PersistentScheduler(database)
            jobs = [
                RolloutJob("scenario_a", 0, "deepseek-v4-pro", "scale"),
                RolloutJob("scenario_b", 0, "deepseek-v4-pro", "scale"),
            ]
            scheduler.submit(jobs)
            estimate_path = root / "estimate.json"
            estimate_path.write_text(
                json.dumps({"shards": [{"job_ids": [jobs[1].job_id]}]}),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "run",
                        "--registry",
                        str(root / "missing-registry"),
                        "--database",
                        str(database),
                        "--config",
                        str(root / "missing-config.json"),
                        "--trace-directory",
                        str(root / "traces"),
                        "--job-id-file",
                        str(estimate_path),
                        "--shard-index",
                        "0",
                        "--dry-run",
                    ]
                )

            printed = json.loads(output.getvalue())
            rows = PersistentScheduler(database).rows()

        self.assertEqual(exit_code, 0)
        self.assertTrue(printed["dry_run"])
        self.assertEqual(printed["requested_job_count"], 1)
        self.assertEqual(printed["selected_job_count"], 1)
        self.assertEqual(printed["missing_job_count"], 0)
        self.assertEqual(printed["runnable_job_count"], 1)
        self.assertEqual(printed["job_ids"], [jobs[1].job_id])
        self.assertEqual({row["status"] for row in rows}, {"pending"})

    def test_existing_completed_trace_can_be_recovered_as_rollout_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "job_existing.jsonl"
            _write_recoverable_trace(trace_path)

            outcome = _rollout_outcome_from_existing_trace(trace_path)

        self.assertTrue(outcome.trace_id)
        self.assertTrue(outcome.success)
        self.assertFalse(outcome.infrastructure_failure)
        self.assertEqual(outcome.metrics["verifier_hidden_command_passed"], 1.0)
        self.assertEqual(outcome.metrics["verifier_all_non_agent_passed"], 1.0)
        self.assertEqual(outcome.metrics["tool_calls"], 1.0)
        self.assertEqual(outcome.tokens, 123)

    def test_batch_estimate_scale_cli_writes_estimate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = PersistentScheduler(root / "queue.sqlite3")
            pilot = PersistentScheduler(root / "pilot.sqlite3")
            queue.submit(
                [
                    RolloutJob("scenario_a", 0, "model", "scale"),
                    RolloutJob("scenario_a", 1, "model", "scale"),
                ]
            )
            pilot.submit([RolloutJob("scenario_a", 0, "model", "pilot")])
            pilot.run(
                _StaticWorker(
                    RolloutOutcome(
                        trace_id="trace_a",
                        success=True,
                        tokens=123,
                    )
                ),
                max_retries=0,
            )
            output_path = root / "estimate.json"
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "estimate-scale",
                        "--database",
                        str(root / "queue.sqlite3"),
                        "--pilot-database",
                        str(root / "pilot.sqlite3"),
                        "--output",
                        str(output_path),
                        "--shard-size",
                        "1",
                    ]
                )

            printed = json.loads(output.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["pending_jobs"], 2)
        self.assertEqual(written["estimated_tokens"], 246.0)
        self.assertEqual(len(written["shards"]), 2)

    def test_batch_shard_status_cli_writes_selected_job_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = PersistentScheduler(root / "queue.sqlite3")
            jobs = [
                RolloutJob("scenario_a", 0, "model", "scale"),
                RolloutJob("scenario_a", 1, "model", "scale"),
            ]
            queue.submit(jobs)
            estimate_path = root / "estimate.json"
            estimate_path.write_text(
                json.dumps(
                    {
                        "shards": [
                            {
                                "shard_index": 0,
                                "job_ids": [jobs[0].job_id],
                                "estimated_tokens": 123.0,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "status.json"
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "shard-status",
                        "--database",
                        str(root / "queue.sqlite3"),
                        "--job-id-file",
                        str(estimate_path),
                        "--shard-index",
                        "0",
                        "--output",
                        str(output_path),
                    ]
                )

            printed = json.loads(output.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["selected_jobs"], 1)
        self.assertEqual(written["status_counts"], {"pending": 1})
        self.assertEqual(written["estimate"]["estimated_tokens"], 123.0)

    def test_scale_continuation_decision_requires_terminal_quality(self) -> None:
        status = {"missing_jobs": 0, "all_selected_terminal": True, "selected_jobs": 3}
        report = {
            "rollouts": 3,
            "unique_traces": 3,
            "success_rate": 2 / 3,
            "infrastructure_failures": 0,
            "average_metrics": {"verifier_hidden_command_passed": 2 / 3},
        }
        audit = {
            "completed_jobs_reviewed": 3,
            "high_quality_rate": 2 / 3,
            "closed_loop_rate": 1.0,
            "multi_step_complex_rate": 1.0,
        }

        passed = scale_continuation_decision(
            report,
            status,
            audit=audit,
            min_high_quality_rate=0.5,
            min_closed_loop_rate=0.8,
            min_multi_step_complex_rate=0.8,
        )
        held = scale_continuation_decision(
            report,
            {**status, "all_selected_terminal": False},
        )
        audit_held = scale_continuation_decision(
            report,
            status,
            audit={**audit, "high_quality_rate": 1 / 3},
            min_high_quality_rate=0.5,
        )

        self.assertEqual(passed["decision"], "continue")
        self.assertEqual(passed["observed"]["high_quality_rate"], 2 / 3)
        self.assertEqual(held["decision"], "hold")
        self.assertIn("all_selected_terminal", held["failures"])
        self.assertEqual(audit_held["decision"], "hold")
        self.assertIn("min_high_quality_rate", audit_held["failures"])

    def test_batch_decide_continuation_cli_writes_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "quality.json"
            status_path = root / "status.json"
            audit_path = root / "audit.json"
            output_path = root / "decision.json"
            report_path.write_text(
                json.dumps(
                    {
                        "rollouts": 3,
                        "unique_traces": 3,
                        "success_rate": 2 / 3,
                        "infrastructure_failures": 0,
                        "average_metrics": {"verifier_hidden_command_passed": 2 / 3},
                    }
                ),
                encoding="utf-8",
            )
            status_path.write_text(
                json.dumps(
                    {
                        "selected_jobs": 3,
                        "found_jobs": 3,
                        "missing_jobs": 0,
                        "all_selected_terminal": True,
                    }
                ),
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "completed_jobs_reviewed": 3,
                        "high_quality_rate": 1 / 3,
                        "closed_loop_rate": 1.0,
                        "multi_step_complex_rate": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "decide-continuation",
                        "--report",
                        str(report_path),
                        "--status",
                        str(status_path),
                        "--audit",
                        str(audit_path),
                        "--min-high-quality-rate",
                        "0.5",
                        "--output",
                        str(output_path),
                    ]
                )

            printed = json.loads(output.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["decision"], "hold")
        self.assertIn("min_high_quality_rate", written["failures"])

    def test_scale_readiness_summary_marks_clean_pending_shard_ready_to_run(self) -> None:
        selection = {"candidates": ["scenario_a", "scenario_b"]}
        estimate = {
            "pending_jobs": 40,
            "scenario_count": 2,
            "estimated_tokens": 1000.0,
            "estimated_cost": 0.2,
            "shards": [{"shard_index": 0}, {"shard_index": 1}],
        }
        status = {
            "selected_jobs": 20,
            "found_jobs": 20,
            "missing_jobs": 0,
            "status_counts": {"pending": 20},
            "all_selected_terminal": False,
            "estimate": {
                "shard_index": 0,
                "max_jobs": 20,
                "estimated_tokens": 500.0,
                "estimated_cost": 0.1,
            },
        }
        audit = {
            "completed_jobs_reviewed": 0,
            "high_quality_rate": 0.0,
            "closed_loop_rate": 0.0,
            "multi_step_complex_rate": 0.0,
        }
        decision = {
            "decision": "hold",
            "failures": [
                "all_selected_terminal",
                "min_success_rate",
                "min_unique_traces",
                "min_hidden_command_pass_rate",
                "min_high_quality_rate",
                "min_closed_loop_rate",
                "min_multi_step_complex_rate",
            ],
            "thresholds": {"min_high_quality_rate": 0.5},
        }

        readiness = scale_readiness_summary(
            selection=selection,
            estimate=estimate,
            status=status,
            audit=audit,
            decision=decision,
        )

        self.assertTrue(readiness["ready"]["pre_run_ready"])
        self.assertFalse(readiness["ready"]["continuation_ready"])
        self.assertEqual(
            readiness["ready"]["next_action"],
            "run_selected_shard_after_spend_approval",
        )
        self.assertEqual(readiness["candidate_count"], 2)
        self.assertEqual(readiness["selected_shard"]["estimated_tokens"], 500.0)

    def test_scale_readiness_summary_marks_completed_shard_ready_to_continue(self) -> None:
        readiness = scale_readiness_summary(
            selection={"candidates": ["scenario_a"]},
            estimate={"pending_jobs": 20, "scenario_count": 1, "shards": [{"shard_index": 0}]},
            status={
                "selected_jobs": 20,
                "found_jobs": 20,
                "missing_jobs": 0,
                "status_counts": {"completed": 20},
                "all_selected_terminal": True,
                "estimate": {"shard_index": 0, "max_jobs": 20},
            },
            audit={
                "completed_jobs_reviewed": 20,
                "high_quality_rate": 0.6,
                "closed_loop_rate": 0.9,
                "multi_step_complex_rate": 0.9,
            },
            decision={"decision": "continue", "failures": [], "thresholds": {}},
        )

        self.assertFalse(readiness["ready"]["pre_run_ready"])
        self.assertTrue(readiness["ready"]["continuation_ready"])
        self.assertEqual(readiness["ready"]["next_action"], "continue_next_shard")

    def test_batch_scale_readiness_cli_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "selection.json"
            estimate_path = root / "estimate.json"
            status_path = root / "status.json"
            audit_path = root / "audit.json"
            decision_path = root / "decision.json"
            output_path = root / "readiness.json"
            selection_path.write_text(json.dumps({"candidates": ["scenario_a"]}), encoding="utf-8")
            estimate_path.write_text(
                json.dumps(
                    {
                        "pending_jobs": 1,
                        "scenario_count": 1,
                        "estimated_tokens": 123.0,
                        "shards": [{"shard_index": 0}],
                    }
                ),
                encoding="utf-8",
            )
            status_path.write_text(
                json.dumps(
                    {
                        "selected_jobs": 1,
                        "found_jobs": 1,
                        "missing_jobs": 0,
                        "status_counts": {"pending": 1},
                        "all_selected_terminal": False,
                    }
                ),
                encoding="utf-8",
            )
            audit_path.write_text(json.dumps({"completed_jobs_reviewed": 0}), encoding="utf-8")
            decision_path.write_text(
                json.dumps({"decision": "hold", "failures": ["all_selected_terminal"]}),
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(
                    [
                        "batch",
                        "scale-readiness",
                        "--selection",
                        str(selection_path),
                        "--estimate",
                        str(estimate_path),
                        "--status",
                        str(status_path),
                        "--audit",
                        str(audit_path),
                        "--decision",
                        str(decision_path),
                        "--output",
                        str(output_path),
                    ]
                )

            printed = json.loads(output.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertTrue(written["ready"]["pre_run_ready"])

    def test_budget_stops_processing_additional_jobs(self) -> None:
        class ExpensiveWorker:
            def run(self, job):
                return RolloutOutcome(trace_id=f"trace_{job.job_id}", tokens=10)

        with tempfile.TemporaryDirectory() as directory:
            scheduler = PersistentScheduler(Path(directory) / "jobs.sqlite3")
            scheduler.submit([RolloutJob(f"s{index}", 0, "m", "c") for index in range(3)])
            summary = scheduler.run(
                ExpensiveWorker(),
                max_workers=1,
                budget=RunBudget(max_tokens=10),
            )
        self.assertGreaterEqual(summary["tokens"], 10)
        self.assertLess(summary["processed"], 3)

    def test_exponential_backoff_and_worker_health(self) -> None:
        calls = []
        delays = []

        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("transient")
            return "ok"

        result = retry_with_backoff(operation, retries=3, initial_delay=0.5, sleep=delays.append)
        self.assertEqual(result, "ok")
        self.assertEqual(delays, [0.5, 1.0])
        self.assertTrue(worker_health(lambda: {"sandbox": "ready"})["healthy"])
        gates = ResourceGates(models=1, sandboxes=1, artifacts=1)
        self.assertTrue(gates.model.acquire(blocking=False))
        self.assertFalse(gates.model.acquire(blocking=False))
        gates.model.release()


def _write_trace(
    path: Path,
    *,
    job_id: str,
    scenario_id: str,
    success: bool,
    termination_reason: str,
) -> None:
    session_id = f"session_{job_id}"
    events = [
        {
            "event_type": "session_started",
            "payload": {
                "scenario_id": scenario_id,
                "public_task": {"query": "Fix the failing parser test."},
            },
        },
        {"event_type": "user_message", "payload": {"content": "Fix the failing parser test."}},
    ]
    tool_names = [
        "search_files",
        "read_file",
        "read_file",
        "apply_patch",
        "run_command",
        "git_diff",
        "git_status",
        "run_command",
    ]
    for index, tool_name in enumerate(tool_names):
        events.extend(
            [
                {
                    "event_type": "model_response",
                    "payload": {
                        "content": "" if index < len(tool_names) - 1 else "Implemented fix.",
                        "tool_calls": [{"function": {"name": tool_name}}],
                    },
                },
                {
                    "event_type": "tool_requested",
                    "payload": {"name": tool_name, "call_id": f"call_{index}"},
                },
                {
                    "event_type": "tool_finished",
                    "payload": {
                        "name": tool_name,
                        "call_id": f"call_{index}",
                        "output": {"exit_code": 0, "stdout": "ok"},
                    },
                },
            ]
        )
    for verifier in ("hidden_test_patch", "hidden_command", "agent_termination"):
        events.append(
            {
                "event_type": "verification_result",
                "payload": {"verifier": verifier, "passed": success, "score": float(success)},
            }
        )
    events.append(
        {
            "event_type": "session_finished",
            "payload": {"success": success, "termination_reason": termination_reason},
        }
    )
    with path.open("w", encoding="utf-8") as handle:
        for sequence, event in enumerate(events):
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "session_id": session_id,
                        "sequence": sequence,
                        "event_id": f"event_{sequence}",
                        **event,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _write_recoverable_trace(path: Path) -> None:
    events = [
        {
            "event_type": "session_started",
            "payload": {
                "scenario_instance_id": "instance_a",
                "initial_state_hash": "state_before",
            },
        },
        {
            "event_type": "model_response",
            "payload": {
                "message_id": "assistant_1",
                "content": "I will run the test.",
                "usage": {"total_tokens": 123},
            },
        },
        {
            "event_type": "tool_requested",
            "payload": {"call_id": "call_1", "name": "run_command", "arguments": {}},
        },
        {
            "event_type": "verification_result",
            "payload": {"verifier": "hidden_test_patch", "passed": True, "score": 1.0},
        },
        {
            "event_type": "verification_result",
            "payload": {"verifier": "hidden_command", "passed": True, "score": 1.0},
        },
        {
            "event_type": "verification_result",
            "payload": {"verifier": "agent_termination", "passed": True, "score": 1.0},
        },
        {
            "event_type": "session_finished",
            "payload": {
                "success": True,
                "termination_reason": "success",
                "final_state_hash": "state_after",
            },
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for sequence, event in enumerate(events):
            handle.write(
                json.dumps(
                    {
                        "schema_version": 1,
                        "session_id": "session_recoverable",
                        "sequence": sequence,
                        **event,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _quality_row(
    scenario_id: str,
    *,
    success: bool,
    metrics: dict[str, float],
    status: str = "completed",
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "trace_id": f"trace_{scenario_id}_{int(success)}",
        "success": int(success),
        "tokens": int(metrics.get("tokens", 100.0)),
        "metrics": metrics,
        "status": status,
    }


def _add_named_scenario(registry: ScenarioRegistry, scenario_id: str) -> Scenario:
    scenario = Scenario(
        QuerySeed(PublicTaskContext(f"Repair {scenario_id}.")),
        EnvironmentSpec(name="fixture", version="1"),
        scenario_id=scenario_id,
    )
    registry.add_scenario(scenario)
    return scenario


class _StaticWorker:
    def __init__(self, outcome: RolloutOutcome) -> None:
        self.outcome = outcome

    def run(self, job):
        del job
        return self.outcome


if __name__ == "__main__":
    unittest.main()
