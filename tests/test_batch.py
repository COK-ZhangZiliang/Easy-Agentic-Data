import json
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.batch import (
    JsonCallCache,
    PersistentScheduler,
    ResourceGates,
    RolloutJob,
    RolloutOutcome,
    RunBudget,
    enqueue_human_review,
    load_human_reviews,
    quality_report,
    retry_with_backoff,
    worker_health,
    write_release_manifest,
)


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
                        "metrics": {"simulator_error_rate": 0.0, "goal_alignment": 1.0},
                    },
                    {
                        "trace_id": "trace_a",
                        "scenario_id": "scenario_a",
                        "success": 0,
                        "reward": 0.0,
                        "status": "failed",
                        "goal_type": "refund",
                        "metrics": {"simulator_error_rate": 0.5, "goal_alignment": 0.5},
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
        self.assertEqual(report["episode_reward_mean"], 0.5)
        self.assertEqual(report["success_by_goal_type"]["refund"], 0.5)
        self.assertGreater(report["in_group_reward_std_mean"], 0.0)
        self.assertEqual(report["average_metrics"]["goal_alignment"], 0.75)
        self.assertEqual(len(review_lines), 1)
        self.assertEqual(reviews[0]["trace_id"], "trace_a")

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


if __name__ == "__main__":
    unittest.main()
