from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
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
class ConsumedUsageTotals:
    """Absolute, cumulative resources accounted to one rollout job."""

    tokens: int
    cost: float
    elapsed_ms: float

    def __post_init__(self) -> None:
        _validate_consumed_usage_totals(self)


@dataclass(frozen=True)
class RolloutOutcome:
    trace_id: str = ""
    success: bool = False
    infrastructure_failure: bool = False
    tokens: int = 0
    cost: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""
    absolute_consumed_usage: ConsumedUsageTotals | None = None


class RolloutWorker(Protocol):
    def run(self, job: RolloutJob) -> RolloutOutcome: ...


@dataclass(frozen=True)
class RunBudget:
    max_seconds: float = 3600.0
    max_tokens: int = 1_000_000
    max_cost: float = 100.0
    max_job_seconds: float = 0.0
    max_job_tokens: int = 0
    max_job_cost: float = 0.0

    def __post_init__(self) -> None:
        for name in ("max_seconds", "max_cost", "max_job_seconds", "max_job_cost"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("max_tokens", "max_job_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


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
        job_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        budget = budget or RunBudget()
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        started = time.monotonic()
        totals = {
            "tokens": 0,
            "cost": 0.0,
            "elapsed_seconds": 0.0,
            "processed": 0,
        }
        self.recover_interrupted()
        pending = self._pending_jobs()
        if job_ids is not None:
            selected_job_ids = set(job_ids)
            pending = [job for job in pending if job.job_id in selected_job_ids]
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

        cursor = 0
        budget_exhausted = False
        reservation_stop_reasons: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while cursor < len(pending):
                stop_reasons = _budget_stop_reasons(totals, budget, started)
                if stop_reasons:
                    budget_exhausted = True
                    break
                batch: list[RolloutJob] = []
                reserved = {"tokens": 0, "cost": 0.0, "seconds": 0.0}
                while cursor < len(pending) and len(batch) < max_workers:
                    reservation_stop_reasons = _job_reservation_stop_reasons(
                        totals,
                        reserved,
                        budget,
                        started,
                    )
                    if reservation_stop_reasons:
                        budget_exhausted = True
                        break
                    batch.append(pending[cursor])
                    cursor += 1
                    reserved["tokens"] += budget.max_job_tokens
                    reserved["cost"] += budget.max_job_cost
                    reserved["seconds"] += budget.max_job_seconds
                if not batch:
                    break
                futures = [executor.submit(execute, job) for job in batch]
                for future in as_completed(futures):
                    job, outcome = future.result()
                    totals["processed"] += 1
                    totals["tokens"] += outcome.tokens
                    totals["cost"] += outcome.cost
                    totals["elapsed_seconds"] += _outcome_elapsed_seconds(outcome)
                    self._finish(job, outcome, max_retries)
        return {
            **totals,
            "budget_exhausted": budget_exhausted,
            "budget_stop_reasons": sorted(
                set(_budget_stop_reasons(totals, budget, started))
                | set(reservation_stop_reasons)
            ),
            "status_counts": self.status_counts(),
        }

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

    def rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM jobs ORDER BY job_id").fetchall()
        return [dict(row) for row in rows]

    def reconcile_consumed_usage(
        self,
        totals_by_job_id: Mapping[str, ConsumedUsageTotals],
    ) -> None:
        """Atomically replace accounted usage with monotonic absolute ledger totals."""

        totals = dict(totals_by_job_id)
        for job_id, usage in totals.items():
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("Consumed usage job IDs must be non-empty strings")
            _validate_consumed_usage_totals(usage, context=f"job {job_id!r}")
        if not totals:
            return

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT job_id, consumed_tokens, consumed_cost, consumed_elapsed_ms
                FROM jobs
                """
            ).fetchall()
            current_by_job_id = {str(row["job_id"]): row for row in rows}
            unknown_job_ids = sorted(set(totals) - current_by_job_id.keys())
            if unknown_job_ids:
                raise ValueError(f"Unknown rollout job IDs: {unknown_job_ids}")

            for job_id, usage in sorted(totals.items()):
                _require_monotonic_consumed_usage(
                    current_by_job_id[job_id],
                    usage,
                    context=f"job {job_id!r}",
                )
            connection.executemany(
                """
                UPDATE jobs SET consumed_tokens = ?, consumed_cost = ?,
                consumed_elapsed_ms = ? WHERE job_id = ?
                """,
                [
                    (usage.tokens, usage.cost, usage.elapsed_ms, job_id)
                    for job_id, usage in sorted(totals.items())
                ],
            )

    def reconcile_interrupted_outcome(
        self,
        job: RolloutJob,
        outcome: RolloutOutcome,
        *,
        max_retries: int,
    ) -> None:
        """Apply a durable worker outcome to a still-running row without a new attempt."""

        if outcome.absolute_consumed_usage is None:
            raise ValueError("Interrupted outcome reconciliation requires absolute usage")
        self._finish(job, outcome, max_retries, require_running=True)

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

    def _finish(
        self,
        job: RolloutJob,
        outcome: RolloutOutcome,
        max_retries: int,
        *,
        require_running: bool = False,
    ) -> None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, attempts, consumed_tokens, consumed_cost, consumed_elapsed_ms
                FROM jobs WHERE job_id = ?
                """,
                (job.job_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Unknown rollout job ID: {job.job_id}")
            if require_running and row["status"] != JobStatus.RUNNING.value:
                raise ValueError("Interrupted outcome can only reconcile a running job")
            attempts = row["attempts"]
            if outcome.infrastructure_failure and attempts <= max_retries:
                status = JobStatus.PENDING
            elif outcome.infrastructure_failure:
                status = JobStatus.INFRASTRUCTURE_FAILED
            elif outcome.trace_id:
                status = JobStatus.COMPLETED
            else:
                status = JobStatus.FAILED
            if outcome.absolute_consumed_usage is None:
                consumed_tokens = int(row["consumed_tokens"] or 0) + outcome.tokens
                consumed_cost = float(row["consumed_cost"] or 0.0) + outcome.cost
                consumed_elapsed_ms = float(row["consumed_elapsed_ms"] or 0.0) + (
                    _outcome_elapsed_seconds(outcome) * 1000
                )
            else:
                usage = outcome.absolute_consumed_usage
                _validate_consumed_usage_totals(usage, context=f"job {job.job_id!r}")
                _require_monotonic_consumed_usage(
                    row,
                    usage,
                    context=f"job {job.job_id!r}",
                )
                consumed_tokens = usage.tokens
                consumed_cost = usage.cost
                consumed_elapsed_ms = usage.elapsed_ms
            connection.execute(
                """
                UPDATE jobs SET status = ?, trace_id = ?, success = ?, tokens = ?,
                cost = ?, metrics = ?, error = ?, consumed_tokens = ?,
                consumed_cost = ?, consumed_elapsed_ms = ? WHERE job_id = ?
                """,
                (
                    status.value,
                    outcome.trace_id,
                    int(outcome.success),
                    outcome.tokens,
                    outcome.cost,
                    json.dumps(outcome.metrics, sort_keys=True),
                    outcome.error,
                    consumed_tokens,
                    consumed_cost,
                    consumed_elapsed_ms,
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
                    metrics TEXT DEFAULT '{}', error TEXT DEFAULT '',
                    consumed_tokens INTEGER DEFAULT 0,
                    consumed_cost REAL DEFAULT 0,
                    consumed_elapsed_ms REAL DEFAULT 0
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            additions = {
                "consumed_tokens": "INTEGER DEFAULT 0",
                "consumed_cost": "REAL DEFAULT 0",
                "consumed_elapsed_ms": "REAL DEFAULT 0",
            }
            added = []
            for name, declaration in additions.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {declaration}")
                    added.append(name)
            if added:
                for row in connection.execute(
                    "SELECT job_id, tokens, cost, metrics FROM jobs"
                ).fetchall():
                    try:
                        metrics = json.loads(row["metrics"] or "{}")
                    except (TypeError, json.JSONDecodeError):
                        metrics = {}
                    elapsed_ms = (
                        float(metrics.get("elapsed_ms", 0.0) or 0.0)
                        if isinstance(metrics, dict)
                        else 0.0
                    )
                    connection.execute(
                        """
                        UPDATE jobs SET consumed_tokens = ?, consumed_cost = ?,
                        consumed_elapsed_ms = ? WHERE job_id = ?
                        """,
                        (row["tokens"] or 0, row["cost"] or 0.0, elapsed_ms, row["job_id"]),
                    )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


def _outcome_elapsed_seconds(outcome: RolloutOutcome) -> float:
    value = outcome.metrics.get("elapsed_ms", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    amount = float(value) / 1000
    return amount if math.isfinite(amount) and amount >= 0 else 0.0


def _validate_consumed_usage_totals(
    usage: ConsumedUsageTotals,
    *,
    context: str = "consumed usage",
) -> None:
    if not isinstance(usage, ConsumedUsageTotals):
        raise ValueError(f"{context} must be ConsumedUsageTotals")
    if isinstance(usage.tokens, bool) or not isinstance(usage.tokens, int):
        raise ValueError(f"{context} tokens must be a non-negative integer")
    if usage.tokens < 0:
        raise ValueError(f"{context} tokens must be a non-negative integer")
    for name in ("cost", "elapsed_ms"):
        value = getattr(usage, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context} {name} must be finite and non-negative")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{context} {name} must be finite and non-negative")


def _require_monotonic_consumed_usage(
    current: sqlite3.Row,
    proposed: ConsumedUsageTotals,
    *,
    context: str,
) -> None:
    current_values = {
        "tokens": int(current["consumed_tokens"] or 0),
        "cost": float(current["consumed_cost"] or 0.0),
        "elapsed_ms": float(current["consumed_elapsed_ms"] or 0.0),
    }
    proposed_values = {
        "tokens": proposed.tokens,
        "cost": proposed.cost,
        "elapsed_ms": proposed.elapsed_ms,
    }
    decreased = [
        name
        for name, current_value in current_values.items()
        if proposed_values[name] < current_value
    ]
    if decreased:
        raise ValueError(
            f"{context} absolute consumed usage cannot decrease fields: {decreased}"
        )


def _budget_stop_reasons(
    totals: dict[str, int | float],
    budget: RunBudget,
    started: float,
) -> list[str]:
    reasons = []
    if totals["tokens"] >= budget.max_tokens:
        reasons.append("tokens")
    if totals["cost"] >= budget.max_cost:
        reasons.append("cost")
    if totals["elapsed_seconds"] >= budget.max_seconds:
        reasons.append("aggregate_seconds")
    if time.monotonic() - started >= budget.max_seconds:
        reasons.append("wall_seconds")
    return reasons


def _job_reservation_stop_reasons(
    totals: dict[str, int | float],
    reserved: dict[str, int | float],
    budget: RunBudget,
    started: float,
) -> list[str]:
    reasons = []
    if budget.max_job_tokens and (
        totals["tokens"] + reserved["tokens"] + budget.max_job_tokens
        > budget.max_tokens
    ):
        reasons.append("job_token_reservation")
    if budget.max_job_cost and (
        totals["cost"] + reserved["cost"] + budget.max_job_cost > budget.max_cost
    ):
        reasons.append("job_cost_reservation")
    if budget.max_job_seconds:
        reserved_seconds = reserved["seconds"] + budget.max_job_seconds
        if totals["elapsed_seconds"] + reserved_seconds > budget.max_seconds:
            reasons.append("job_aggregate_seconds_reservation")
        if time.monotonic() - started + reserved_seconds > budget.max_seconds:
            reasons.append("job_wall_seconds_reservation")
    return reasons


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
    scenario_ids = {str(row.get("scenario_id") or "") for row in items if row.get("scenario_id")}
    metrics = []
    for row in items:
        value = row.get("metrics", {})
        if isinstance(value, str):
            value = json.loads(value or "{}")
        metrics.append(value)
    metric_keys = {
        key
        for item in metrics
        for key, value in item.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
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
    pass_groups = [
        any(int(row.get("success", 0)) for row in group)
        for group in _rows_by_key(items, "scenario_id").values()
    ]
    return {
        "rollouts": len(items),
        "scenario_count": len(scenario_ids),
        "successes": successes,
        "success_rate": successes / len(items) if items else 0.0,
        "pass_at_observed_k": (
            sum(1 for passed in pass_groups if passed) / len(pass_groups) if pass_groups else 0.0
        ),
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


def audit_trace_logic(
    rows: Iterable[dict[str, Any]],
    trace_directory: str | Path,
    *,
    job_ids: Iterable[str] | None = None,
    include_items: bool = True,
) -> dict[str, Any]:
    """Review completed trace files for logical flow, completeness, and complexity."""

    trace_root = Path(trace_directory)
    items = []
    selected_ids = set(job_ids or [])
    all_rows = list(rows)
    rows_by_job_id = {str(row.get("job_id") or ""): row for row in all_rows}
    completed_rows = [
        row
        for row in all_rows
        if row.get("status") == JobStatus.COMPLETED.value
        and (not selected_ids or str(row.get("job_id") or "") in selected_ids)
    ]
    missing_trace_jobs = []
    trace_load_errors = []
    for row in completed_rows:
        job_id = str(row.get("job_id") or "")
        trace_path = trace_root / f"{job_id}.jsonl"
        if not trace_path.exists():
            missing_trace_jobs.append(job_id)
            continue
        try:
            events = _load_trace_events(trace_path)
        except Exception as exc:
            trace_load_errors.append(
                {"job_id": job_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        items.append(_audit_trace_item(row, trace_path, events))

    extra_trace_files = []
    if not selected_ids and trace_root.exists():
        extra_trace_files = [
            str(path)
            for path in sorted(trace_root.glob("*.jsonl"))
            if path.stem not in rows_by_job_id
        ]
    verdict_counts = Counter(str(item["verdict"]) for item in items)
    termination_counts = Counter(str(item["termination_reason"]) for item in items)
    failure_counts = Counter(
        reason for item in items for reason in item.get("failure_reasons", [])
    )
    scenario_reports = _trace_audit_scenario_reports(items)
    summary: dict[str, Any] = {
        "audit_version": 1,
        "trace_directory": str(trace_root),
        "database_rows": len(all_rows),
        "database_status_counts": dict(Counter(str(row.get("status") or "") for row in all_rows)),
        "selected_job_count": len(selected_ids) if selected_ids else len(completed_rows),
        "completed_jobs_reviewed": len(items),
        "missing_trace_jobs": missing_trace_jobs,
        "trace_load_errors": trace_load_errors,
        "extra_trace_files": extra_trace_files,
        "verdict_counts": dict(verdict_counts),
        "high_quality_rate": _ratio(verdict_counts.get("high_quality", 0), len(items)),
        "strict_success_rate": _ratio(
            sum(1 for item in items if item.get("complete_verified")), len(items)
        ),
        "closed_loop_rate": _ratio(
            sum(1 for item in items if item.get("closed_loop")), len(items)
        ),
        "multi_step_complex_rate": _ratio(
            sum(1 for item in items if item.get("multi_step_complex")), len(items)
        ),
        "coherent_order_rate": _ratio(
            sum(1 for item in items if item.get("coherent_read_patch_test_order")),
            len(items),
        ),
        "termination_counts": dict(termination_counts),
        "failure_reason_counts": dict(failure_counts),
        "tool_calls": _numeric_summary([int(item["tool_calls"]) for item in items]),
        "model_messages": _numeric_summary([int(item["model_messages"]) for item in items]),
        "scenario_reports": scenario_reports,
    }
    if include_items:
        summary["items"] = items
    return summary


def scenario_quality_report(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rollout quality by scenario for scale-up decisions."""

    reports = []
    for scenario_id, group in sorted(_rows_by_key(list(rows), "scenario_id").items()):
        metrics = [_row_metrics(row) for row in group]
        successes = sum(int(row.get("success", 0)) for row in group)
        infrastructure_failures = sum(
            row.get("status") == JobStatus.INFRASTRUCTURE_FAILED.value for row in group
        )
        report = {
            "scenario_id": scenario_id,
            "rollouts": len(group),
            "successes": successes,
            "success_rate": successes / len(group) if group else 0.0,
            "hidden_command_pass_rate": _average_metric(
                metrics, "verifier_hidden_command_passed"
            ),
            "all_non_agent_pass_rate": _average_metric(
                metrics, "verifier_all_non_agent_passed"
            ),
            "agent_stop_rate": _average_metric(metrics, "verifier_agent_termination_passed"),
            "infrastructure_failures": infrastructure_failures,
            "infrastructure_failure_rate": (
                infrastructure_failures / len(group) if group else 0.0
            ),
            "average_tool_calls": _average_metric(metrics, "tool_calls"),
            "average_tokens": sum(int(row.get("tokens", 0) or 0) for row in group) / len(group)
            if group
            else 0.0,
        }
        reports.append(report)
    return reports


def select_scale_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    audit: dict[str, Any] | None = None,
    min_rollouts: int = 2,
    min_success_rate: float = 0.5,
    min_hidden_command_pass_rate: float = 0.5,
    min_all_non_agent_pass_rate: float = 0.5,
    min_agent_stop_rate: float = 0.0,
    min_high_quality_rate: float = 0.0,
    min_closed_loop_rate: float = 0.0,
    min_multi_step_complex_rate: float = 0.0,
    max_infrastructure_failure_rate: float = 0.0,
    min_average_tool_calls: float = 6.0,
) -> dict[str, Any]:
    """Select scenario groups that are strong enough for a larger synthesis run."""

    thresholds = {
        "min_rollouts": min_rollouts,
        "min_success_rate": min_success_rate,
        "min_hidden_command_pass_rate": min_hidden_command_pass_rate,
        "min_all_non_agent_pass_rate": min_all_non_agent_pass_rate,
        "min_agent_stop_rate": min_agent_stop_rate,
        "min_high_quality_rate": min_high_quality_rate,
        "min_closed_loop_rate": min_closed_loop_rate,
        "min_multi_step_complex_rate": min_multi_step_complex_rate,
        "max_infrastructure_failure_rate": max_infrastructure_failure_rate,
        "min_average_tool_calls": min_average_tool_calls,
    }
    reports = []
    candidates = []
    audit_reports = _audit_reports_by_scenario(audit)
    for report in scenario_quality_report(rows):
        report = {**report, **audit_reports.get(str(report["scenario_id"]), {})}
        failures = _scale_candidate_failures(report, thresholds)
        annotated = {**report, "scale_candidate": not failures, "candidate_failures": failures}
        reports.append(annotated)
        if not failures:
            candidates.append(report["scenario_id"])
    return {"thresholds": thresholds, "candidates": candidates, "scenario_reports": reports}


def estimate_scale_run(
    queue_rows: Iterable[dict[str, Any]],
    pilot_rows: Iterable[dict[str, Any]],
    *,
    shard_size: int = 20,
    cost_per_million_tokens: float = 0.0,
) -> dict[str, Any]:
    """Estimate pending scale-up cost from observed pilot token usage."""

    pending = [
        row for row in queue_rows if row.get("status") in {JobStatus.PENDING.value, "running"}
    ]
    observations = [
        row
        for row in pilot_rows
        if row.get("status") not in {JobStatus.PENDING.value, "running"}
        and int(row.get("tokens", 0) or 0) > 0
    ]
    observed_tokens = [int(row.get("tokens", 0) or 0) for row in observations]
    fallback_tokens = sum(observed_tokens) / len(observed_tokens) if observed_tokens else 0.0
    tokens_by_scenario: dict[str, list[int]] = {}
    for row in observations:
        tokens_by_scenario.setdefault(str(row.get("scenario_id") or "default"), []).append(
            int(row.get("tokens", 0) or 0)
        )
    token_estimates = {
        scenario_id: sum(values) / len(values) for scenario_id, values in tokens_by_scenario.items()
    }
    scenario_counts: dict[str, int] = {}
    scenario_tokens: dict[str, float] = {}
    for row in pending:
        scenario_id = str(row.get("scenario_id") or "default")
        estimate = token_estimates.get(scenario_id, fallback_tokens)
        scenario_counts[scenario_id] = scenario_counts.get(scenario_id, 0) + 1
        scenario_tokens[scenario_id] = scenario_tokens.get(scenario_id, 0.0) + estimate
    scenario_estimates = [
        {
            "scenario_id": scenario_id,
            "pending_jobs": scenario_counts[scenario_id],
            "observed_average_tokens": token_estimates.get(scenario_id, fallback_tokens),
            "estimated_tokens": scenario_tokens[scenario_id],
        }
        for scenario_id in sorted(scenario_counts)
    ]
    shards = []
    ordered = sorted(pending, key=lambda row: str(row.get("job_id") or ""))
    for index in range(0, len(ordered), max(1, shard_size)):
        shard_rows = ordered[index : index + max(1, shard_size)]
        estimated_tokens = sum(
            token_estimates.get(str(row.get("scenario_id") or "default"), fallback_tokens)
            for row in shard_rows
        )
        shards.append(
            {
                "shard_index": len(shards),
                "max_jobs": len(shard_rows),
                "first_job_id": shard_rows[0].get("job_id", "") if shard_rows else "",
                "last_job_id": shard_rows[-1].get("job_id", "") if shard_rows else "",
                "job_ids": [row.get("job_id", "") for row in shard_rows],
                "estimated_tokens": estimated_tokens,
                "estimated_cost": estimated_tokens / 1_000_000 * cost_per_million_tokens,
            }
        )
    total_tokens = sum(item["estimated_tokens"] for item in scenario_estimates)
    return {
        "pending_jobs": len(pending),
        "scenario_count": len(scenario_estimates),
        "pilot_observations": len(observations),
        "fallback_average_tokens": fallback_tokens,
        "estimated_tokens": total_tokens,
        "cost_per_million_tokens": cost_per_million_tokens,
        "estimated_cost": total_tokens / 1_000_000 * cost_per_million_tokens,
        "scenario_estimates": scenario_estimates,
        "shards": shards,
    }


def selected_job_status(rows: Iterable[dict[str, Any]], job_ids: Iterable[str]) -> dict[str, Any]:
    """Summarize scheduler state for an explicit set of jobs."""

    selected = list(dict.fromkeys(job_ids))
    row_by_id = {str(row.get("job_id") or ""): row for row in rows}
    found = [row_by_id[job_id] for job_id in selected if job_id in row_by_id]
    missing = [job_id for job_id in selected if job_id not in row_by_id]
    status_counts: dict[str, int] = {}
    for row in found:
        status = str(row.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
    successes = sum(int(row.get("success", 0) or 0) for row in found)
    tokens = sum(int(row.get("tokens", 0) or 0) for row in found)
    return {
        "selected_jobs": len(selected),
        "found_jobs": len(found),
        "missing_jobs": len(missing),
        "missing_job_ids": missing,
        "status_counts": status_counts,
        "completed_jobs": status_counts.get(JobStatus.COMPLETED.value, 0),
        "successes": successes,
        "success_rate": successes / len(found) if found else 0.0,
        "tokens": tokens,
        "all_selected_terminal": bool(found)
        and not missing
        and all(
            str(row.get("status") or "")
            in {
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.INFRASTRUCTURE_FAILED.value,
            }
            for row in found
        ),
    }


def planned_batch_run(
    rows: Iterable[dict[str, Any]],
    *,
    job_ids: Iterable[str] | None = None,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    """Describe which pending jobs a batch run would start without mutating state."""

    items = list(rows)
    requested_ids = list(dict.fromkeys(job_ids or []))
    selected_ids = set(requested_ids)
    row_by_id = {str(row.get("job_id") or ""): row for row in items}
    selected = [
        row
        for row in items
        if not selected_ids or str(row.get("job_id") or "") in selected_ids
    ]
    missing = [job_id for job_id in requested_ids if job_id not in row_by_id]
    runnable = [
        row for row in selected if str(row.get("status") or "") == JobStatus.PENDING.value
    ]
    runnable = sorted(runnable, key=lambda row: str(row.get("job_id") or ""))
    if max_jobs is not None:
        runnable = runnable[:max_jobs]
    status_counts: dict[str, int] = {}
    for row in selected:
        status = str(row.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "requested_job_count": len(requested_ids) if requested_ids else len(items),
        "selected_job_count": len(selected),
        "missing_job_count": len(missing),
        "missing_job_ids": missing,
        "runnable_job_count": len(runnable),
        "status_counts": status_counts,
        "job_ids": [str(row.get("job_id") or "") for row in runnable],
    }


def scale_continuation_decision(
    report: dict[str, Any],
    status: dict[str, Any],
    *,
    audit: dict[str, Any] | None = None,
    min_success_rate: float = 0.3,
    min_unique_traces: int = 1,
    min_hidden_command_pass_rate: float = 0.4,
    min_high_quality_rate: float = 0.0,
    min_closed_loop_rate: float = 0.0,
    min_multi_step_complex_rate: float = 0.0,
    max_infrastructure_failures: int = 0,
) -> dict[str, Any]:
    """Decide whether a completed shard is good enough to continue scaling."""

    thresholds = {
        "min_success_rate": min_success_rate,
        "min_unique_traces": min_unique_traces,
        "min_hidden_command_pass_rate": min_hidden_command_pass_rate,
        "min_high_quality_rate": min_high_quality_rate,
        "min_closed_loop_rate": min_closed_loop_rate,
        "min_multi_step_complex_rate": min_multi_step_complex_rate,
        "max_infrastructure_failures": max_infrastructure_failures,
    }
    failures = []
    if int(status.get("missing_jobs", 0) or 0) != 0:
        failures.append("missing_jobs")
    if not bool(status.get("all_selected_terminal", False)):
        failures.append("all_selected_terminal")
    if int(report.get("infrastructure_failures", 0) or 0) > max_infrastructure_failures:
        failures.append("max_infrastructure_failures")
    if float(report.get("success_rate", 0.0) or 0.0) < min_success_rate:
        failures.append("min_success_rate")
    if int(report.get("unique_traces", 0) or 0) < min_unique_traces:
        failures.append("min_unique_traces")
    average_metrics = report.get("average_metrics", {})
    if not isinstance(average_metrics, dict):
        average_metrics = {}
    if (
        float(average_metrics.get("verifier_hidden_command_passed", 0.0) or 0.0)
        < min_hidden_command_pass_rate
    ):
        failures.append("min_hidden_command_pass_rate")
    audit = audit or {}
    if float(audit.get("high_quality_rate", 0.0) or 0.0) < min_high_quality_rate:
        failures.append("min_high_quality_rate")
    if float(audit.get("closed_loop_rate", 0.0) or 0.0) < min_closed_loop_rate:
        failures.append("min_closed_loop_rate")
    if (
        float(audit.get("multi_step_complex_rate", 0.0) or 0.0)
        < min_multi_step_complex_rate
    ):
        failures.append("min_multi_step_complex_rate")
    return {
        "decision": "continue" if not failures else "hold",
        "failures": failures,
        "thresholds": thresholds,
        "observed": {
            "selected_jobs": status.get("selected_jobs", 0),
            "found_jobs": status.get("found_jobs", 0),
            "missing_jobs": status.get("missing_jobs", 0),
            "all_selected_terminal": status.get("all_selected_terminal", False),
            "rollouts": report.get("rollouts", 0),
            "unique_traces": report.get("unique_traces", 0),
            "success_rate": report.get("success_rate", 0.0),
            "infrastructure_failures": report.get("infrastructure_failures", 0),
            "hidden_command_pass_rate": average_metrics.get(
                "verifier_hidden_command_passed", 0.0
            ),
            "audit_completed_jobs_reviewed": audit.get("completed_jobs_reviewed", 0),
            "high_quality_rate": audit.get("high_quality_rate", 0.0),
            "closed_loop_rate": audit.get("closed_loop_rate", 0.0),
            "multi_step_complex_rate": audit.get("multi_step_complex_rate", 0.0),
        },
    }


def scale_readiness_summary(
    *,
    selection: dict[str, Any],
    estimate: dict[str, Any],
    status: dict[str, Any],
    audit: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Combine scale-up planning artifacts into one reviewable readiness record."""

    candidates = selection.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    shards = estimate.get("shards", [])
    if not isinstance(shards, list):
        shards = []
    status_counts = status.get("status_counts", {})
    if not isinstance(status_counts, dict):
        status_counts = {}
    selected_jobs = int(status.get("selected_jobs", 0) or 0)
    pending_jobs = int(status_counts.get(JobStatus.PENDING.value, 0) or 0)
    running_jobs = int(status_counts.get(JobStatus.RUNNING.value, 0) or 0)
    completed_jobs = int(status_counts.get(JobStatus.COMPLETED.value, 0) or 0)
    failed_jobs = int(status_counts.get(JobStatus.FAILED.value, 0) or 0)
    infrastructure_failed_jobs = int(
        status_counts.get(JobStatus.INFRASTRUCTURE_FAILED.value, 0) or 0
    )
    terminal_jobs = completed_jobs + failed_jobs + infrastructure_failed_jobs
    decision_failures = decision.get("failures", [])
    if not isinstance(decision_failures, list):
        decision_failures = []
    expected_pre_run_failures = {
        "all_selected_terminal",
        "min_success_rate",
        "min_unique_traces",
        "min_hidden_command_pass_rate",
        "min_high_quality_rate",
        "min_closed_loop_rate",
        "min_multi_step_complex_rate",
    }
    missing_jobs = int(status.get("missing_jobs", 0) or 0)
    found_jobs = int(status.get("found_jobs", 0) or 0)
    all_pending = (
        selected_jobs > 0
        and found_jobs == selected_jobs
        and missing_jobs == 0
        and pending_jobs == selected_jobs
        and running_jobs == 0
        and terminal_jobs == 0
    )
    pre_run_ready = all_pending and set(str(item) for item in decision_failures).issubset(
        expected_pre_run_failures
    )
    continuation_ready = decision.get("decision") == "continue"
    if continuation_ready:
        next_action = "continue_next_shard"
    elif pre_run_ready:
        next_action = "run_selected_shard_after_spend_approval"
    else:
        next_action = "hold_and_inspect_failures"
    selected_estimate = status.get("estimate", {})
    if not isinstance(selected_estimate, dict):
        selected_estimate = {}
    summary = {
        "readiness_version": 1,
        "candidate_count": len(candidates),
        "candidate_scenario_ids": [str(item) for item in candidates],
        "queue": {
            "pending_jobs": int(estimate.get("pending_jobs", 0) or 0),
            "scenario_count": int(estimate.get("scenario_count", 0) or 0),
            "estimated_tokens": float(estimate.get("estimated_tokens", 0.0) or 0.0),
            "estimated_cost": float(estimate.get("estimated_cost", 0.0) or 0.0),
            "shard_count": len(shards),
        },
        "selected_shard": {
            "shard_index": selected_estimate.get("shard_index"),
            "max_jobs": selected_estimate.get("max_jobs", selected_jobs),
            "estimated_tokens": float(selected_estimate.get("estimated_tokens", 0.0) or 0.0),
            "estimated_cost": float(selected_estimate.get("estimated_cost", 0.0) or 0.0),
        },
        "status": {
            "selected_jobs": selected_jobs,
            "found_jobs": found_jobs,
            "missing_jobs": missing_jobs,
            "status_counts": status_counts,
            "all_selected_terminal": bool(status.get("all_selected_terminal", False)),
        },
        "audit": {
            "completed_jobs_reviewed": int(audit.get("completed_jobs_reviewed", 0) or 0),
            "high_quality_rate": float(audit.get("high_quality_rate", 0.0) or 0.0),
            "closed_loop_rate": float(audit.get("closed_loop_rate", 0.0) or 0.0),
            "multi_step_complex_rate": float(
                audit.get("multi_step_complex_rate", 0.0) or 0.0
            ),
            "coherent_order_rate": float(audit.get("coherent_order_rate", 0.0) or 0.0),
        },
        "decision": {
            "decision": decision.get("decision", "hold"),
            "failures": [str(item) for item in decision_failures],
            "thresholds": decision.get("thresholds", {}),
        },
        "ready": {
            "pre_run_ready": pre_run_ready,
            "continuation_ready": continuation_ready,
            "next_action": next_action,
        },
    }
    return summary


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


def _load_trace_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
    return events


def _audit_trace_item(
    row: dict[str, Any],
    trace_path: Path,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    tools = []
    tool_errors = []
    verification: dict[str, dict[str, Any]] = {}
    final = ""
    query = ""
    scenario_id = str(row.get("scenario_id") or "")
    termination_reason = ""
    session_success = False
    model_messages = 0
    contentful_model_messages = 0
    first_read = None
    first_patch = None
    first_test = None
    for index, event in enumerate(events):
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        if event_type == "session_started":
            public_task = payload.get("public_task", {})
            if isinstance(public_task, dict):
                query = str(public_task.get("query") or "")
            scenario_id = str(payload.get("scenario_id") or scenario_id)
        elif event_type == "tool_requested":
            name = str(payload.get("name") or "")
            tools.append(name)
            if name == "read_file" and first_read is None:
                first_read = index
            elif name == "apply_patch" and first_patch is None:
                first_patch = index
            elif name == "run_command" and first_test is None:
                first_test = index
        elif event_type == "tool_finished":
            if _tool_event_failed(payload):
                tool_errors.append(
                    {
                        "sequence": event.get("sequence"),
                        "name": payload.get("name", ""),
                    }
                )
        elif event_type == "model_response":
            model_messages += 1
            content = str(payload.get("content") or "")
            if content:
                contentful_model_messages += 1
                final = content
        elif event_type == "verification_result":
            verifier = str(payload.get("verifier") or "")
            verification[verifier] = {
                "passed": bool(payload.get("passed", False)),
                "reason": payload.get("reason", ""),
                "score": payload.get("score", 0.0),
            }
        elif event_type == "session_finished":
            session_success = bool(payload.get("success", False))
            termination_reason = str(payload.get("termination_reason") or "")

    unique_tools = sorted(set(tool for tool in tools if tool))
    has_read = "read_file" in tools
    has_search = "search_files" in tools
    has_patch = "apply_patch" in tools
    has_test = "run_command" in tools
    has_diff_or_status = "git_diff" in tools or "git_status" in tools
    coherent_order = (
        first_read is not None
        and first_patch is not None
        and first_test is not None
        and first_read < first_patch < first_test
    )
    closed_loop = has_read and has_patch and has_test and has_diff_or_status
    multi_step_complex = len(tools) >= 8 and model_messages >= 6 and len(unique_tools) >= 4
    complete_verified = bool(row.get("success")) and session_success and all(
        verification.get(name, {}).get("passed") is True
        for name in ("hidden_test_patch", "hidden_command", "agent_termination")
    )
    failure_reasons = _trace_failure_reasons(
        has_patch=has_patch,
        has_test=has_test,
        final=final,
        coherent_order=coherent_order,
        verification=verification,
        termination_reason=termination_reason,
        tool_error_count=len(tool_errors),
    )
    if complete_verified and closed_loop and multi_step_complex and coherent_order:
        verdict = "high_quality"
    elif closed_loop and coherent_order and verification.get("hidden_test_patch", {}).get(
        "passed"
    ) is True:
        verdict = "usable_but_failed_gate"
    elif has_read and (has_search or has_test) and len(tools) >= 8:
        verdict = "incomplete_exploration"
    else:
        verdict = "low_quality"
    return {
        "job_id": str(row.get("job_id") or ""),
        "scenario_id": scenario_id,
        "trace_file": str(trace_path),
        "db_success": bool(row.get("success")),
        "session_success": session_success,
        "termination_reason": termination_reason,
        "event_count": len(events),
        "model_messages": model_messages,
        "contentful_model_messages": contentful_model_messages,
        "tool_calls": len(tools),
        "unique_tools": unique_tools,
        "tool_error_count": len(tool_errors),
        "query_chars": len(query),
        "final_chars": len(final),
        "has_read": has_read,
        "has_search": has_search,
        "has_patch": has_patch,
        "has_test_command": has_test,
        "has_diff_or_status": has_diff_or_status,
        "coherent_read_patch_test_order": coherent_order,
        "closed_loop": closed_loop,
        "multi_step_complex": multi_step_complex,
        "complete_verified": complete_verified,
        "verdict": verdict,
        "failure_reasons": failure_reasons,
        "verification": verification,
        "final_summary_preview": final[:500],
    }


def _tool_event_failed(payload: dict[str, Any]) -> bool:
    if payload.get("error") is not None:
        return True
    output = payload.get("output")
    if not isinstance(output, dict):
        return False
    if output.get("status") == "error":
        return True
    exit_code = output.get("exit_code")
    return isinstance(exit_code, int) and exit_code != 0


def _trace_failure_reasons(
    *,
    has_patch: bool,
    has_test: bool,
    final: str,
    coherent_order: bool,
    verification: dict[str, dict[str, Any]],
    termination_reason: str,
    tool_error_count: int,
) -> list[str]:
    failures = []
    if not has_patch:
        failures.append("no_patch")
    if not has_test:
        failures.append("no_test_command")
    if not final:
        failures.append("no_final_summary")
    if not coherent_order:
        failures.append("weak_read_patch_test_order")
    for name in ("hidden_test_patch", "hidden_command", "agent_termination"):
        if verification.get(name, {}).get("passed") is False:
            failures.append(f"{name}_failed")
    if termination_reason and termination_reason != "success":
        failures.append(f"termination_{termination_reason}")
    if tool_error_count:
        failures.append("tool_errors_observed")
    return failures


def _trace_audit_scenario_reports(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for scenario_id, group in sorted(_rows_by_key(items, "scenario_id").items()):
        reports.append(
            {
                "scenario_id": scenario_id,
                "rollouts": len(group),
                "high_quality": sum(item["verdict"] == "high_quality" for item in group),
                "usable_but_failed_gate": sum(
                    item["verdict"] == "usable_but_failed_gate" for item in group
                ),
                "successes": sum(bool(item["complete_verified"]) for item in group),
                "success_rate": _ratio(
                    sum(bool(item["complete_verified"]) for item in group), len(group)
                ),
                "closed_loop_rate": _ratio(
                    sum(bool(item["closed_loop"]) for item in group), len(group)
                ),
                "multi_step_complex_rate": _ratio(
                    sum(bool(item["multi_step_complex"]) for item in group), len(group)
                ),
                "average_tool_calls": _average_number(group, "tool_calls"),
                "average_model_messages": _average_number(group, "model_messages"),
                "verdicts": dict(Counter(str(item["verdict"]) for item in group)),
            }
        )
    return reports


def _numeric_summary(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "max": ordered[-1],
    }


def _average_number(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows) if rows else 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _row_reward(row: dict[str, Any]) -> float:
    if "reward" in row:
        return float(row["reward"])
    return float(int(row.get("success", 0)))


def _row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metrics", {})
    if isinstance(value, str):
        value = json.loads(value or "{}")
    return value if isinstance(value, dict) else {}


def _average_metric(metrics: list[dict[str, Any]], key: str) -> float:
    if not metrics:
        return 0.0
    return sum(float(item.get(key, 0.0)) for item in metrics) / len(metrics)


def _scale_candidate_failures(report: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    failures = []
    checks = {
        "min_rollouts": report["rollouts"] >= thresholds["min_rollouts"],
        "min_success_rate": report["success_rate"] >= thresholds["min_success_rate"],
        "min_hidden_command_pass_rate": report["hidden_command_pass_rate"]
        >= thresholds["min_hidden_command_pass_rate"],
        "min_all_non_agent_pass_rate": report["all_non_agent_pass_rate"]
        >= thresholds["min_all_non_agent_pass_rate"],
        "min_agent_stop_rate": report["agent_stop_rate"] >= thresholds["min_agent_stop_rate"],
        "min_high_quality_rate": report.get("high_quality_rate", 0.0)
        >= thresholds["min_high_quality_rate"],
        "min_closed_loop_rate": report.get("closed_loop_rate", 0.0)
        >= thresholds["min_closed_loop_rate"],
        "min_multi_step_complex_rate": report.get("multi_step_complex_rate", 0.0)
        >= thresholds["min_multi_step_complex_rate"],
        "max_infrastructure_failure_rate": report["infrastructure_failure_rate"]
        <= thresholds["max_infrastructure_failure_rate"],
        "min_average_tool_calls": report["average_tool_calls"]
        >= thresholds["min_average_tool_calls"],
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    return failures


def _audit_reports_by_scenario(audit: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not audit:
        return {}
    reports = audit.get("scenario_reports", [])
    if not isinstance(reports, list):
        return {}
    by_scenario = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        scenario_id = str(report.get("scenario_id") or "")
        rollouts = int(report.get("rollouts", 0) or 0)
        if not scenario_id or rollouts <= 0:
            continue
        high_quality = int(report.get("high_quality", 0) or 0)
        closed_loop = float(
            report.get("closed_loop_rate", _scenario_item_rate(audit, scenario_id, "closed_loop"))
            or 0.0
        )
        multi_step = float(
            report.get(
                "multi_step_complex_rate",
                _scenario_item_rate(audit, scenario_id, "multi_step_complex"),
            )
            or 0.0
        )
        by_scenario[scenario_id] = {
            "audit_rollouts": rollouts,
            "high_quality": high_quality,
            "high_quality_rate": high_quality / rollouts,
            "closed_loop_rate": closed_loop,
            "multi_step_complex_rate": multi_step,
        }
    return by_scenario


def _scenario_item_rate(audit: dict[str, Any], scenario_id: str, key: str) -> float:
    items = audit.get("items", [])
    if not isinstance(items, list):
        return 0.0
    matched = [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("scenario_id") or "") == scenario_id
    ]
    if not matched:
        return 0.0
    return sum(1 for item in matched if bool(item.get(key))) / len(matched)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5


def _rows_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key) or "default"), []).append(row)
    return grouped
