from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


@dataclass(frozen=True)
class RolloutJob:
    scenario_id: str
    rollout_index: int
    model: str
    config_hash: str
    job_id: str = ""

    def __post_init__(self) -> None:
        if not self.job_id:
            payload = f"{self.scenario_id}:{self.rollout_index}:{self.model}:{self.config_hash}"
            object.__setattr__(
                self, "job_id", f"job_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"
            )


@dataclass(frozen=True)
class RolloutOutcome:
    trace_id: str = ""
    success: bool = False
    infrastructure_failure: bool = False
    tokens: int = 0
    cost: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""


class RolloutWorker(Protocol):
    def run(self, job: RolloutJob) -> RolloutOutcome: ...


@dataclass(frozen=True)
class RunBudget:
    max_seconds: float = 3600.0
    max_tokens: int = 1_000_000
    max_cost: float = 100.0


class PersistentScheduler:
    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def submit(self, jobs: Iterable[RolloutJob]) -> None:
        with self._connect() as connection:
            for job in jobs:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO jobs
                    (job_id, scenario_id, rollout_index, model, config_hash, status, attempts)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        job.job_id,
                        job.scenario_id,
                        job.rollout_index,
                        job.model,
                        job.config_hash,
                        JobStatus.PENDING.value,
                    ),
                )

    def recover_interrupted(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ? WHERE status = ?",
                (JobStatus.PENDING.value, JobStatus.RUNNING.value),
            )

    def run(
        self,
        worker: RolloutWorker,
        *,
        max_workers: int = 1,
        max_retries: int = 2,
        budget: RunBudget | None = None,
        max_jobs: int | None = None,
    ) -> dict[str, Any]:
        budget = budget or RunBudget()
        started = time.monotonic()
        totals = {"tokens": 0, "cost": 0.0, "processed": 0}
        self.recover_interrupted()
        pending = self._pending_jobs()
        if max_jobs is not None:
            pending = pending[:max_jobs]

        def execute(job: RolloutJob) -> tuple[RolloutJob, RolloutOutcome]:
            self._mark_running(job.job_id)
            try:
                outcome = worker.run(job)
            except Exception as exc:
                outcome = RolloutOutcome(
                    infrastructure_failure=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
            return job, outcome

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(execute, job): job for job in pending}
            for future in as_completed(futures):
                job, outcome = future.result()
                totals["processed"] += 1
                totals["tokens"] += outcome.tokens
                totals["cost"] += outcome.cost
                self._finish(job, outcome, max_retries)
                if (
                    time.monotonic() - started >= budget.max_seconds
                    or totals["tokens"] >= budget.max_tokens
                    or totals["cost"] >= budget.max_cost
                ):
                    for pending_future in futures:
                        pending_future.cancel()
                    break
        return {**totals, "status_counts": self.status_counts()}

    def status_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        return {row["status"]: row["count"] for row in rows}

    def completed_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY job_id",
                (JobStatus.COMPLETED.value,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _pending_jobs(self) -> list[RolloutJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY job_id",
                (JobStatus.PENDING.value,),
            ).fetchall()
        return [
            RolloutJob(
                row["scenario_id"],
                row["rollout_index"],
                row["model"],
                row["config_hash"],
                row["job_id"],
            )
            for row in rows
        ]

    def _mark_running(self, job_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, attempts = attempts + 1 WHERE job_id = ?",
                (JobStatus.RUNNING.value, job_id),
            )

    def _finish(self, job: RolloutJob, outcome: RolloutOutcome, max_retries: int) -> None:
        with self._lock, self._connect() as connection:
            attempts = connection.execute(
                "SELECT attempts FROM jobs WHERE job_id = ?", (job.job_id,)
            ).fetchone()["attempts"]
            if outcome.infrastructure_failure and attempts <= max_retries:
                status = JobStatus.PENDING
            elif outcome.infrastructure_failure:
                status = JobStatus.INFRASTRUCTURE_FAILED
            elif outcome.trace_id:
                status = JobStatus.COMPLETED
            else:
                status = JobStatus.FAILED
            connection.execute(
                """
                UPDATE jobs SET status = ?, trace_id = ?, success = ?, tokens = ?,
                cost = ?, metrics = ?, error = ? WHERE job_id = ?
                """,
                (
                    status.value,
                    outcome.trace_id,
                    int(outcome.success),
                    outcome.tokens,
                    outcome.cost,
                    json.dumps(outcome.metrics, sort_keys=True),
                    outcome.error,
                    job.job_id,
                ),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, scenario_id TEXT, rollout_index INTEGER,
                    model TEXT, config_hash TEXT, status TEXT, attempts INTEGER,
                    trace_id TEXT DEFAULT '', success INTEGER DEFAULT 0,
                    tokens INTEGER DEFAULT 0, cost REAL DEFAULT 0,
                    metrics TEXT DEFAULT '{}', error TEXT DEFAULT ''
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


class RateLimiter:
    def __init__(self, calls_per_second: float) -> None:
        self.interval = 1.0 / calls_per_second
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            if delay:
                time.sleep(delay)
            self._next = max(now, self._next) + self.interval


class ResourceGates:
    """Independent concurrency gates for model, sandbox, and artifact operations."""

    def __init__(self, *, models: int = 1, sandboxes: int = 1, artifacts: int = 1) -> None:
        self.model = threading.BoundedSemaphore(models)
        self.sandbox = threading.BoundedSemaphore(sandboxes)
        self.artifact = threading.BoundedSemaphore(artifacts)


def retry_with_backoff(
    operation: Callable[[], Any],
    *,
    retries: int = 3,
    initial_delay: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    delay = initial_delay
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception:
            if attempt >= retries:
                raise
            sleep(delay)
            delay *= 2


class JsonCallCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.values = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        )

    def get_or_compute(self, key: dict[str, Any], compute: Callable[[], Any]) -> Any:
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest not in self.values:
            self.values[digest] = compute()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.values, sort_keys=True), encoding="utf-8")
        return self.values[digest]


def endpoint_health(url: str, timeout_seconds: float = 2.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return {"healthy": 200 <= response.status < 300, "status": response.status}
    except Exception as exc:
        return {"healthy": False, "error": f"{type(exc).__name__}: {exc}"}


def worker_health(check: Callable[[], Any]) -> dict[str, Any]:
    try:
        detail = check()
        return {"healthy": True, "detail": detail}
    except Exception as exc:
        return {"healthy": False, "error": f"{type(exc).__name__}: {exc}"}


def write_release_manifest(path: str | Path, **versions: Any) -> None:
    required = {"scenarios", "models", "prompts", "tools", "images", "evaluators", "exporters"}
    missing = required - versions.keys()
    if missing:
        raise ValueError(f"Missing manifest version groups: {sorted(missing)}")
    Path(path).write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def quality_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    items = list(rows)
    successes = sum(int(row.get("success", 0)) for row in items)
    rewards = [_row_reward(row) for row in items]
    traces = [row.get("trace_id", "") for row in items if row.get("trace_id")]
    metrics = []
    for row in items:
        value = row.get("metrics", {})
        if isinstance(value, str):
            value = json.loads(value or "{}")
        metrics.append(value)
    metric_keys = {
        "tool_calls",
        "tool_errors",
        "user_turns",
        "turns",
        "tokens",
        "policy_denials",
        "turn_reward_total",
        "turn_reward_mean",
        "simulator_error_rate",
        "goal_alignment",
    }
    aggregates = {
        key: sum(float(item.get(key, 0.0)) for item in metrics) / len(metrics) if metrics else 0.0
        for key in metric_keys
    }
    reward_groups: dict[str, list[float]] = {}
    success_by_goal_type: dict[str, dict[str, float]] = {}
    for row, reward in zip(items, rewards, strict=False):
        scenario_id = str(row.get("scenario_id") or row.get("prompt_group") or "default")
        reward_groups.setdefault(scenario_id, []).append(reward)
        goal_type = str(row.get("goal_type") or row.get("category") or "")
        if goal_type:
            bucket = success_by_goal_type.setdefault(goal_type, {"successes": 0.0, "rollouts": 0.0})
            bucket["successes"] += float(int(row.get("success", 0)))
            bucket["rollouts"] += 1.0
    goal_success_rates = {
        key: value["successes"] / value["rollouts"] if value["rollouts"] else 0.0
        for key, value in success_by_goal_type.items()
    }
    group_stds = [_std(values) for values in reward_groups.values() if len(values) > 1]
    return {
        "rollouts": len(items),
        "successes": successes,
        "success_rate": successes / len(items) if items else 0.0,
        "episode_reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "episode_reward_std": _std(rewards),
        "in_group_reward_std_mean": sum(group_stds) / len(group_stds) if group_stds else 0.0,
        "low_information_groups": sorted(
            key for key, values in reward_groups.items() if len(values) > 1 and _std(values) == 0.0
        ),
        "success_by_goal_type": goal_success_rates,
        "unique_traces": len(set(traces)),
        "duplicate_traces": len(traces) - len(set(traces)),
        "infrastructure_failures": sum(
            row.get("status") == JobStatus.INFRASTRUCTURE_FAILED.value for row in items
        ),
        "average_metrics": aggregates,
    }


def enqueue_human_review(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_human_reviews(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _row_reward(row: dict[str, Any]) -> float:
    if "reward" in row:
        return float(row["reward"])
    return float(int(row.get("success", 0)))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5
