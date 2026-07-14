from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from easy_agentic_data.batch import ConsumedUsageTotals, RolloutOutcome
from easy_agentic_data.models import utc_now
from easy_agentic_data.pilot_contract import PilotRunContract, canonical_sha256
from easy_agentic_data.traces import load_trace

PILOT_USAGE_LEDGER_SCHEMA = "easy_agentic_data.pilot_usage_ledger.v1"
LEDGER_DIRECTORY_NAME = ".pilot-usage-ledger"


class UnknownProviderUsageError(RuntimeError):
    """A provider call may have incurred cost but has no completed usage receipt."""


@dataclass(frozen=True)
class PilotJobUsageState:
    job_id: str
    totals: ConsumedUsageTotals
    attempt_count: int
    call_count: int
    latest_outcome: RolloutOutcome | None
    record_sha256s: tuple[str, ...]
    provider_response_ids: tuple[str, ...]

    def to_evidence(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tokens": self.totals.tokens,
            "cost_usd": _decimal_text(Decimal(str(self.totals.cost))),
            "elapsed_ms": self.totals.elapsed_ms,
            "attempt_count": self.attempt_count,
            "call_count": self.call_count,
            "record_set_sha256": canonical_sha256(list(self.record_sha256s)),
            "provider_response_ids_sha256": canonical_sha256(list(self.provider_response_ids)),
        }


@dataclass(frozen=True)
class PilotUsageLedgerAudit:
    contract_id: str
    jobs: Mapping[str, PilotJobUsageState]
    ledger_sha256: str

    @property
    def consumed_totals(self) -> dict[str, ConsumedUsageTotals]:
        return {job_id: state.totals for job_id, state in self.jobs.items()}

    def to_evidence(self) -> dict[str, Any]:
        job_evidence = [self.jobs[job_id].to_evidence() for job_id in sorted(self.jobs)]
        return {
            "contract_id": self.contract_id,
            "job_count": len(job_evidence),
            "attempt_count": sum(item["attempt_count"] for item in job_evidence),
            "call_count": sum(item["call_count"] for item in job_evidence),
            "ledger_sha256": self.ledger_sha256,
            "jobs_sha256": canonical_sha256(job_evidence),
        }


class PilotUsageAttempt:
    """Durable append-only call journal for one scheduler attempt."""

    def __init__(
        self,
        trace_directory: str | Path,
        *,
        contract_id: str,
        job_id: str,
        attempt_id: str | None = None,
        recovery_elapsed_floor_ms: float | None = None,
    ) -> None:
        if not contract_id.startswith("pilot_"):
            raise ValueError("Usage ledger contract_id must be a pilot ID")
        if not job_id.startswith("rollout_"):
            raise ValueError("Usage ledger job_id must be a rollout ID")
        _require_safe_component(job_id, "job_id")
        if attempt_id is not None:
            if not attempt_id.startswith("attempt_"):
                raise ValueError("Usage ledger attempt_id must start with attempt_")
            _require_safe_component(attempt_id, "attempt_id")
        self.contract_id = contract_id
        self.job_id = job_id
        job_root = _prepare_job_root(Path(trace_directory), job_id)
        self.attempt_id = attempt_id or _new_attempt_id(job_root)
        if not self.attempt_id.startswith("attempt_"):
            raise ValueError("Usage ledger attempt_id must start with attempt_")
        _require_safe_component(self.attempt_id, "attempt_id")
        self.directory = job_root / self.attempt_id
        self.directory.mkdir(exist_ok=False)
        _require_safe_directory(self.directory, "usage attempt directory")
        _fsync_directory(self.directory.parent)
        _fsync_directory(self.directory.parent.parent)
        self._started: dict[int, str] = {}
        self._completed: dict[int, str] = {}
        self._terminal = False
        marker = {
            "schema": PILOT_USAGE_LEDGER_SCHEMA,
            "kind": "attempt_started",
            "contract_id": self.contract_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "started_at": utc_now(),
        }
        if recovery_elapsed_floor_ms is not None:
            marker["recovery_elapsed_floor_ms"] = _finite_nonnegative_number(
                recovery_elapsed_floor_ms,
                "recovery_elapsed_floor_ms",
            )
        _write_record(self.directory, "attempt", marker)

    def call_started(self, observed: Mapping[str, Any]) -> None:
        if self._terminal:
            raise RuntimeError("Cannot start a provider call after attempt terminal")
        call_index = _nonnegative_integer(observed.get("call_index"), "call_index")
        if call_index != len(self._started) or call_index in self._started:
            raise ValueError("Provider call indexes must be sequential and unique")
        response_format = observed.get("response_format")
        material = {
            "schema": PILOT_USAGE_LEDGER_SCHEMA,
            "kind": "call_started",
            "contract_id": self.contract_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "call_index": call_index,
            "started_at": _required_text(observed, "started_at"),
            "model": _required_text(observed, "model"),
            "prompt_hash": _required_sha256(observed.get("prompt_hash"), "prompt_hash"),
            "message_count": _nonnegative_integer(observed.get("message_count"), "message_count"),
            "tool_count": _nonnegative_integer(observed.get("tool_count"), "tool_count"),
            "temperature": _finite_number(observed.get("temperature"), "temperature"),
            "max_tokens": _positive_integer(observed.get("max_tokens"), "max_tokens"),
            "response_format_sha256": canonical_sha256(response_format),
        }
        record = _write_record(
            self.directory,
            f"call-{call_index:06d}.started",
            material,
        )
        self._started[call_index] = record["record_sha256"]

    def call_completed(self, observed: Mapping[str, Any]) -> None:
        if self._terminal:
            raise RuntimeError("Cannot complete a provider call after attempt terminal")
        call_index = _nonnegative_integer(observed.get("call_index"), "call_index")
        if call_index not in self._started or call_index in self._completed:
            raise ValueError("Completed provider call has no unique started marker")
        usage = observed.get("usage")
        if not isinstance(usage, Mapping) or not usage:
            raise ValueError("Completed provider call must contain non-empty usage")
        identity = observed.get("provider_response_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("Completed provider call lacks provider response identity")
        material = {
            "schema": PILOT_USAGE_LEDGER_SCHEMA,
            "kind": "call_completed",
            "contract_id": self.contract_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "call_index": call_index,
            "started_record_sha256": self._started[call_index],
            "completed_at": utc_now(),
            "response_model": _required_text(observed, "response_model"),
            "usage": _canonical_usage(usage),
            "retry_count": _nonnegative_integer(observed.get("retry_count"), "retry_count"),
            "latency_ms": _finite_nonnegative_number(observed.get("latency_ms"), "latency_ms"),
            "provider_response_identity": dict(identity),
            "provider_response_identity_sha256": _required_sha256(
                observed.get("provider_response_identity_sha256"),
                "provider_response_identity_sha256",
            ),
            "provider_response_sha256": _required_sha256(
                observed.get("provider_response_sha256"),
                "provider_response_sha256",
            ),
        }
        if material["provider_response_identity_sha256"] != canonical_sha256(
            material["provider_response_identity"]
        ):
            raise ValueError("Provider response identity hash mismatch")
        record = _write_record(
            self.directory,
            f"call-{call_index:06d}.completed",
            material,
        )
        self._completed[call_index] = record["record_sha256"]

    def usage(self) -> dict[str, Any]:
        if set(self._started) != set(self._completed):
            raise UnknownProviderUsageError(
                "Provider usage is unknown because a started call has no completed receipt"
            )
        values = [
            _read_record_by_sha(
                self.directory,
                f"call-{index:06d}.completed",
                self._completed[index],
            )["usage"]
            for index in sorted(self._completed)
        ]
        return _aggregate_usage(values)

    def finalize(
        self,
        outcome: RolloutOutcome,
        *,
        elapsed_ms: float,
        finished_at: str | None = None,
    ) -> dict[str, Any]:
        if self._terminal:
            raise RuntimeError("Usage attempt already has a terminal record")
        self.usage()
        elapsed = _finite_nonnegative_number(elapsed_ms, "elapsed_ms")
        record = _write_terminal_record(
            self.directory,
            contract_id=self.contract_id,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            finished_at=finished_at or utc_now(),
            elapsed_ms=elapsed,
            started_hashes=[self._started[index] for index in sorted(self._started)],
            completed_hashes=[self._completed[index] for index in sorted(self._completed)],
            outcome=outcome,
        )
        self._terminal = True
        return record


def recover_running_pilot_usage_attempt(
    contract: PilotRunContract,
    row: Mapping[str, Any],
    trace_directory: str | Path,
    *,
    recovered_at: str | None = None,
) -> bool:
    """Durably fail a provably interrupted running attempt without guessing usage."""

    if row.get("status") != "running":
        return False
    job_id = _required_text(row, "job_id")
    if job_id not in {assignment.job_id for assignment in contract.rollouts}:
        raise ValueError("Usage recovery job is outside the pilot contract")
    scheduler_attempts = _nonnegative_integer(row.get("attempts", 0), "scheduler attempts")
    if scheduler_attempts == 0:
        raise UnknownProviderUsageError("Running pilot job has no scheduler admission")

    trace_root = Path(trace_directory)
    canonical_paths = _canonical_artifact_paths(trace_root, job_id)
    canonical_trace = canonical_paths[0]
    _reject_unsafe_path(canonical_trace, "canonical trace")
    if _path_lexists(canonical_trace):
        return False

    attempt_directories = _attempt_directories(trace_root, job_id)
    if scheduler_attempts not in {len(attempt_directories), len(attempt_directories) + 1}:
        raise UnknownProviderUsageError(
            f"Scheduler attempt count cannot be safely recovered for {job_id}"
        )

    inspected: list[tuple[Path, dict[str, Any]]] = []
    for directory in attempt_directories:
        inspected.append(
            (
                directory,
                _inspect_attempt(
                    contract,
                    job_id,
                    directory,
                    require_terminal=False,
                ),
            )
        )
    incomplete = [item for item in inspected if item[1]["terminal"] is None]

    if scheduler_attempts == len(attempt_directories) + 1:
        if incomplete:
            raise UnknownProviderUsageError(
                "Scheduler admission gap coexists with an incomplete usage attempt"
            )
        _require_no_canonical_artifacts(canonical_paths)
        attempt = PilotUsageAttempt(
            trace_root,
            contract_id=contract.contract_id,
            job_id=job_id,
            recovery_elapsed_floor_ms=contract.budgets.max_agent_seconds * 1000,
        )
        attempt.finalize(
            RolloutOutcome(
                infrastructure_failure=True,
                error=(
                    "Scheduler admission was interrupted before the durable usage attempt marker"
                ),
            ),
            # There is no durable admission timestamp. Charge the full reserved
            # per-attempt wall-time budget so recovery cannot undercount time.
            elapsed_ms=contract.budgets.max_agent_seconds * 1000,
            finished_at=recovered_at,
        )
        return True

    if not incomplete:
        return False
    if len(incomplete) != 1 or incomplete[0][0] != attempt_directories[-1]:
        raise UnknownProviderUsageError(
            "Only the latest usage attempt can be recovered without a terminal"
        )
    _require_no_canonical_artifacts(canonical_paths)
    directory, state = incomplete[0]
    finished_at = recovered_at or utc_now()
    elapsed_ms = _conservative_recovery_elapsed_ms(state, finished_at)
    outcome = RolloutOutcome(
        infrastructure_failure=True,
        tokens=state["tokens"],
        cost=float(state["cost"]),
        error="Usage attempt was interrupted after durable provider receipts",
    )
    _write_terminal_record(
        directory,
        contract_id=contract.contract_id,
        job_id=job_id,
        attempt_id=directory.name,
        finished_at=finished_at,
        elapsed_ms=elapsed_ms,
        started_hashes=state["started_hashes"],
        completed_hashes=state["completed_hashes"],
        outcome=outcome,
        cost_usd=state["cost"],
    )
    _audit_attempt(contract, job_id, directory)
    return True


def audit_pilot_usage_ledger(
    contract: PilotRunContract,
    rows: Sequence[Mapping[str, Any]],
    trace_directory: str | Path,
    *,
    require_database_match: bool = False,
) -> PilotUsageLedgerAudit:
    """Validate all immutable attempt records and recompute absolute job usage."""

    root = Path(trace_directory)
    ledger_root = root / LEDGER_DIRECTORY_NAME
    rows_by_id = {str(row.get("job_id") or ""): row for row in rows}
    expected_ids = {assignment.job_id for assignment in contract.rollouts}
    if set(rows_by_id) != expected_ids:
        raise ValueError("Pilot usage ledger rows do not match the contract job set")
    if _path_lexists(root):
        _require_safe_directory(root, "trace directory")
    if _path_lexists(ledger_root):
        _require_safe_directory(ledger_root, "usage ledger directory")
        entries = list(ledger_root.iterdir())
        unsafe = [path.name for path in entries if path.is_symlink() or not path.is_dir()]
        if unsafe:
            raise ValueError(f"Usage ledger contains unsafe job paths: {unsafe}")
        extra = sorted(path.name for path in entries if path.name not in expected_ids)
        if extra:
            raise ValueError(f"Usage ledger contains jobs outside the contract: {extra}")

    states: dict[str, PilotJobUsageState] = {}
    for assignment in contract.rollouts:
        state = _audit_job(
            contract,
            assignment.job_id,
            rows_by_id[assignment.job_id],
            root,
        )
        if require_database_match:
            _require_database_usage_match(rows_by_id[assignment.job_id], state.totals)
        states[assignment.job_id] = state
    provider_response_ids = [
        response_id for state in states.values() for response_id in state.provider_response_ids
    ]
    if len(provider_response_ids) != len(set(provider_response_ids)):
        raise ValueError("Provider response IDs must be globally unique in the usage ledger")
    evidence = [states[job_id].to_evidence() for job_id in sorted(states)]
    return PilotUsageLedgerAudit(
        contract_id=contract.contract_id,
        jobs=states,
        ledger_sha256=canonical_sha256(evidence),
    )


def load_pilot_job_usage(
    contract: PilotRunContract,
    job_id: str,
    trace_directory: str | Path,
) -> PilotJobUsageState:
    """Recompute one job after its terminal record is durable."""

    if job_id not in {assignment.job_id for assignment in contract.rollouts}:
        raise ValueError("Usage ledger job is outside the pilot contract")
    return _audit_job(contract, job_id, None, Path(trace_directory))


def _audit_job(
    contract: PilotRunContract,
    job_id: str,
    row: Mapping[str, Any] | None,
    trace_root: Path,
) -> PilotJobUsageState:
    attempt_directories = _attempt_directories(trace_root, job_id)
    if row is not None:
        scheduler_attempts = _nonnegative_integer(row.get("attempts", 0), "scheduler attempts")
        if scheduler_attempts != len(attempt_directories):
            raise UnknownProviderUsageError(
                f"Scheduler attempt count differs from terminal usage ledger for {job_id}"
            )

    total_tokens = 0
    total_cost = Decimal("0")
    total_elapsed_ms = Decimal("0")
    total_calls = 0
    record_hashes: list[str] = []
    provider_response_ids: list[str] = []
    terminals: list[tuple[str, dict[str, Any], dict[str, Any], Decimal]] = []
    for attempt_directory in attempt_directories:
        attempt = _audit_attempt(contract, job_id, attempt_directory)
        total_tokens += attempt["tokens"]
        total_cost += attempt["cost"]
        total_elapsed_ms += Decimal(str(attempt["terminal"]["elapsed_ms"]))
        total_calls += attempt["call_count"]
        record_hashes.extend(attempt["record_sha256s"])
        provider_response_ids.extend(attempt["provider_response_ids"])
        terminals.append(
            (
                attempt_directory.name,
                attempt["terminal"],
                attempt["usage"],
                attempt["cost"],
            )
        )

    canonical_trace = trace_root / f"{job_id}.jsonl"
    _reject_unsafe_path(canonical_trace, "canonical trace")
    traced_terminals = [item for item in terminals if item[1]["trace_id"]]
    if _path_lexists(canonical_trace):
        if not canonical_trace.is_file():
            raise ValueError("Canonical trace path is not a regular file")
        trace = load_trace(canonical_trace, tolerate_truncated=False)
        matching_terminals = [
            item for item in traced_terminals if item[1]["trace_id"] == trace.trace_id
        ]
        if len(matching_terminals) != 1:
            raise ValueError("Canonical rollout must bind exactly one terminal usage attempt")
        attempt_id, terminal, attempt_usage, attempt_cost = matching_terminals[0]
        evidence_path = canonical_trace.parent / "run-evidence" / f"{job_id}.json"
        _reject_unsafe_path(evidence_path, "canonical run evidence")
        evidence = _read_json(evidence_path)
        if terminal["trace_id"] != trace.trace_id or evidence.get("trace_id") != trace.trace_id:
            raise ValueError("Canonical trace does not match terminal usage ledger")
        evidence_usage = evidence.get("usage")
        if not isinstance(evidence_usage, Mapping) or dict(evidence_usage) != attempt_usage:
            raise ValueError("Canonical run evidence usage differs from terminal usage ledger")
        evidence_cost = _nonnegative_decimal(evidence.get("cost"), "evidence cost")
        if evidence_cost != attempt_cost:
            raise ValueError("Canonical run evidence cost differs from terminal usage ledger")
        if evidence.get("usage_attempt_id") != attempt_id:
            raise ValueError("Canonical run evidence usage attempt lineage mismatch")
        if row is not None and row.get("status") == "completed":
            if (
                _nonnegative_integer(row.get("tokens", 0), "scheduler tokens")
                != terminal["attempt_tokens"]
            ):
                raise ValueError("Canonical scheduler tokens differ from terminal usage ledger")
            if _nonnegative_decimal(row.get("cost", 0), "scheduler cost") != attempt_cost:
                raise ValueError("Canonical scheduler cost differs from terminal usage ledger")
    latest_outcome = None
    if terminals:
        latest_terminal = sorted(terminals, key=lambda item: item[0])[-1][1]
        latest_trace_id = latest_terminal["trace_id"]
        latest_success = latest_terminal["success"]
        latest_infrastructure_failure = latest_terminal["infrastructure_failure"]
        latest_error = latest_terminal["error"]
        if latest_trace_id and not canonical_trace.is_file():
            latest_trace_id = ""
            latest_success = False
            latest_infrastructure_failure = True
            latest_error = "Validated staged rollout was not canonically published"
        latest_outcome = RolloutOutcome(
            trace_id=latest_trace_id,
            success=latest_success,
            infrastructure_failure=latest_infrastructure_failure,
            tokens=latest_terminal["attempt_tokens"],
            cost=float(Decimal(latest_terminal["attempt_cost_usd"])),
            metrics={"elapsed_ms": latest_terminal["elapsed_ms"]},
            error=latest_error,
            absolute_consumed_usage=ConsumedUsageTotals(
                total_tokens,
                float(total_cost),
                float(total_elapsed_ms),
            ),
        )
    return PilotJobUsageState(
        job_id=job_id,
        totals=ConsumedUsageTotals(
            total_tokens,
            float(total_cost),
            float(total_elapsed_ms),
        ),
        attempt_count=len(attempt_directories),
        call_count=total_calls,
        latest_outcome=latest_outcome,
        record_sha256s=tuple(sorted(record_hashes)),
        provider_response_ids=tuple(sorted(provider_response_ids)),
    )


def _audit_attempt(
    contract: PilotRunContract,
    job_id: str,
    directory: Path,
) -> dict[str, Any]:
    state = _inspect_attempt(contract, job_id, directory, require_terminal=True)
    if state["terminal"] is None:  # pragma: no cover - guaranteed by the inspector
        raise AssertionError("Required terminal disappeared during usage audit")
    return state


def _inspect_attempt(
    contract: PilotRunContract,
    job_id: str,
    directory: Path,
    *,
    require_terminal: bool,
) -> dict[str, Any]:
    _require_safe_directory(directory, "usage attempt directory")
    _require_safe_component(directory.name, "attempt_id")
    if not directory.name.startswith("attempt_"):
        raise ValueError("Usage ledger attempt directory has an invalid name")
    entries = list(directory.iterdir())
    if any(path.name.startswith(".") and path.name.endswith(".tmp") for path in entries):
        raise ValueError("Usage ledger contains an incomplete temporary record")
    unexpected_files = [
        path.name
        for path in entries
        if not path.is_file() or path.is_symlink() or path.suffix != ".json"
    ]
    if unexpected_files:
        raise ValueError(f"Usage ledger attempt contains unsafe files: {unexpected_files}")
    record_entries = [(path, _read_record(path)) for path in sorted(entries)]
    records = [record for _, record in record_entries]
    allowed_kinds = {
        "attempt_started",
        "call_started",
        "call_completed",
        "attempt_terminal",
    }
    if any(record.get("kind") not in allowed_kinds for record in records):
        raise ValueError("Usage ledger contains an unknown record kind")
    for path, record in record_entries:
        kind = record["kind"]
        if kind == "attempt_started":
            expected_prefix = "attempt"
        elif kind == "attempt_terminal":
            expected_prefix = "terminal"
        else:
            index = _nonnegative_integer(record.get("call_index"), "call_index")
            suffix = "started" if kind == "call_started" else "completed"
            expected_prefix = f"call-{index:06d}.{suffix}"
        expected_name = f"{expected_prefix}.{record['record_sha256']}.json"
        if path.name != expected_name:
            raise ValueError("Usage ledger record filename prefix mismatch")
    markers = [record for record in records if record.get("kind") == "attempt_started"]
    terminals = [record for record in records if record.get("kind") == "attempt_terminal"]
    if len(markers) != 1:
        raise UnknownProviderUsageError(
            f"Usage ledger attempt {directory.name} has no unique admission marker"
        )
    if len(terminals) > 1 or (require_terminal and not terminals):
        raise UnknownProviderUsageError(
            f"Usage ledger attempt {directory.name} has no unique terminal record"
        )
    for record in records:
        if (
            record.get("schema") != PILOT_USAGE_LEDGER_SCHEMA
            or record.get("contract_id") != contract.contract_id
            or record.get("job_id") != job_id
            or record.get("attempt_id") != directory.name
        ):
            raise ValueError("Usage ledger record lineage mismatch")

    marker = markers[0]
    marker_started_at = _parse_utc_timestamp(
        _required_text(marker, "started_at"), "attempt started_at"
    )
    recovery_elapsed_floor_ms = _finite_nonnegative_number(
        marker.get("recovery_elapsed_floor_ms", 0),
        "recovery_elapsed_floor_ms",
    )
    started = _records_by_call_index(records, "call_started")
    completed = _records_by_call_index(records, "call_completed")
    if set(started) != set(completed):
        raise UnknownProviderUsageError(
            "Provider usage is unknown because a started call has no completed receipt"
        )
    usage_values = []
    completed_hashes = []
    started_hashes = []
    provider_response_ids = []
    completed_times = []
    for index in sorted(started):
        if completed[index].get("started_record_sha256") != started[index]["record_sha256"]:
            raise ValueError("Completed provider call is not bound to its started record")
        completed_at = _parse_utc_timestamp(
            _required_text(completed[index], "completed_at"), "call completed_at"
        )
        if completed_at < marker_started_at:
            raise ValueError("Provider completion precedes the attempt admission marker")
        completed_times.append(completed_at)
        _finite_nonnegative_number(completed[index].get("latency_ms"), "call latency_ms")
        usage_values.append(completed[index].get("usage"))
        started_hashes.append(started[index]["record_sha256"])
        completed_hashes.append(completed[index]["record_sha256"])
        identity = completed[index].get("provider_response_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("Completed provider call lacks response identity")
        response_id = _required_text(identity, "id")
        if completed[index].get("provider_response_identity_sha256") != canonical_sha256(
            dict(identity)
        ):
            raise ValueError("Completed provider response identity hash mismatch")
        provider_response_ids.append(response_id)
        if _nonnegative_integer(completed[index].get("retry_count"), "retry_count"):
            raise UnknownProviderUsageError(
                "Provider adapter retries cannot be reconciled to per-request usage"
            )
    usage = _aggregate_usage(usage_values)
    if usage:
        usage_cost = contract.pricing.calculate_cost(usage)
        tokens = usage_cost.total_tokens
        cost = usage_cost.cost_usd
    else:
        tokens = 0
        cost = Decimal("0")
    terminal = terminals[0] if terminals else None
    if terminal is not None:
        if (
            terminal.get("started_call_record_sha256s") != started_hashes
            or terminal.get("completed_call_record_sha256s") != completed_hashes
        ):
            raise ValueError("Attempt terminal does not bind the complete provider call set")
        if (
            terminal.get("attempt_tokens") != tokens
            or Decimal(_required_text(terminal, "attempt_cost_usd")) != cost
        ):
            raise ValueError("Attempt terminal usage totals do not match completed calls")
        terminal_finished_at = _parse_utc_timestamp(
            _required_text(terminal, "finished_at"), "terminal finished_at"
        )
        if terminal_finished_at < marker_started_at or any(
            completed_at > terminal_finished_at for completed_at in completed_times
        ):
            raise ValueError("Attempt terminal timestamp does not contain its receipts")
        _finite_nonnegative_number(terminal.get("elapsed_ms"), "terminal elapsed_ms")
        _strict_bool(terminal.get("success"), "terminal success")
        _strict_bool(
            terminal.get("infrastructure_failure"),
            "terminal infrastructure_failure",
        )
        if not isinstance(terminal.get("trace_id"), str) or not isinstance(
            terminal.get("error"), str
        ):
            raise ValueError("Attempt terminal outcome fields are invalid")
    return {
        "marker": marker,
        "terminal": terminal,
        "usage": usage,
        "tokens": tokens,
        "cost": cost,
        "call_count": len(completed),
        "record_sha256s": sorted(record["record_sha256"] for record in records),
        "provider_response_ids": provider_response_ids,
        "started_hashes": started_hashes,
        "completed_hashes": completed_hashes,
        "completed_records": [completed[index] for index in sorted(completed)],
        "recovery_elapsed_floor_ms": recovery_elapsed_floor_ms,
    }


def _records_by_call_index(
    records: Sequence[Mapping[str, Any]],
    kind: str,
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if record.get("kind") != kind:
            continue
        index = _nonnegative_integer(record.get("call_index"), "call_index")
        if index in result:
            raise ValueError(f"Usage ledger contains duplicate {kind} call indexes")
        result[index] = record
    if sorted(result) != list(range(len(result))):
        raise ValueError(f"Usage ledger {kind} call indexes are not contiguous")
    return result


def _write_terminal_record(
    directory: Path,
    *,
    contract_id: str,
    job_id: str,
    attempt_id: str,
    finished_at: str,
    elapsed_ms: float,
    started_hashes: Sequence[str],
    completed_hashes: Sequence[str],
    outcome: RolloutOutcome,
    cost_usd: Decimal | None = None,
) -> dict[str, Any]:
    _parse_utc_timestamp(finished_at, "terminal finished_at")
    material = {
        "schema": PILOT_USAGE_LEDGER_SCHEMA,
        "kind": "attempt_terminal",
        "contract_id": contract_id,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "finished_at": finished_at,
        "elapsed_ms": _finite_nonnegative_number(elapsed_ms, "elapsed_ms"),
        "started_call_record_sha256s": list(started_hashes),
        "completed_call_record_sha256s": list(completed_hashes),
        "trace_id": outcome.trace_id,
        "success": bool(outcome.success),
        "infrastructure_failure": bool(outcome.infrastructure_failure),
        "attempt_tokens": _nonnegative_integer(outcome.tokens, "attempt_tokens"),
        "attempt_cost_usd": _decimal_text(
            _nonnegative_decimal(
                outcome.cost if cost_usd is None else cost_usd,
                "attempt_cost_usd",
            )
        ),
        "error": str(outcome.error),
    }
    return _write_record(directory, "terminal", material)


def _write_record(directory: Path, prefix: str, material: Mapping[str, Any]) -> dict[str, Any]:
    _require_safe_directory(directory, "usage record directory")
    digest = canonical_sha256(material)
    record = {**material, "record_sha256": digest}
    destination = directory / f"{prefix}.{digest}.json"
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if _path_lexists(destination):
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("Content-addressed usage record path is unsafe")
        if destination.read_bytes() != encoded:
            raise ValueError("Content-addressed usage record path contains different bytes")
        return record
    temporary = directory / f".{prefix}.{time.time_ns()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return record


def _read_record(path: Path) -> dict[str, Any]:
    record = _read_json(path)
    digest = _required_sha256(record.get("record_sha256"), "record_sha256")
    material = {key: value for key, value in record.items() if key != "record_sha256"}
    if canonical_sha256(material) != digest or not path.name.endswith(f".{digest}.json"):
        raise ValueError("Usage ledger content hash or filename mismatch")
    return record


def _read_record_by_sha(directory: Path, prefix: str, digest: str) -> dict[str, Any]:
    return _read_record(directory / f"{prefix}.{digest}.json")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular non-symlink JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _prepare_job_root(trace_root: Path, job_id: str) -> Path:
    _require_safe_component(job_id, "job_id")
    if _path_lexists(trace_root):
        _require_safe_directory(trace_root, "trace directory")
    else:
        trace_root.mkdir(parents=True)
        _require_safe_directory(trace_root, "trace directory")
    ledger_root = trace_root / LEDGER_DIRECTORY_NAME
    if _path_lexists(ledger_root):
        _require_safe_directory(ledger_root, "usage ledger directory")
    else:
        ledger_root.mkdir()
        _require_safe_directory(ledger_root, "usage ledger directory")
        _fsync_directory(trace_root)
    job_root = ledger_root / job_id
    if _path_lexists(job_root):
        _require_safe_directory(job_root, "usage ledger job directory")
    else:
        job_root.mkdir()
        _require_safe_directory(job_root, "usage ledger job directory")
        _fsync_directory(ledger_root)
    return job_root


def _attempt_directories(trace_root: Path, job_id: str) -> list[Path]:
    _require_safe_component(job_id, "job_id")
    if not _path_lexists(trace_root):
        return []
    _require_safe_directory(trace_root, "trace directory")
    ledger_root = trace_root / LEDGER_DIRECTORY_NAME
    if not _path_lexists(ledger_root):
        return []
    _require_safe_directory(ledger_root, "usage ledger directory")
    job_root = ledger_root / job_id
    if not _path_lexists(job_root):
        return []
    _require_safe_directory(job_root, "usage ledger job directory")
    entries = list(job_root.iterdir())
    unsafe = []
    for path in entries:
        try:
            _require_safe_component(path.name, "attempt_id")
        except ValueError:
            unsafe.append(path.name)
            continue
        if path.is_symlink() or not path.is_dir() or not path.name.startswith("attempt_"):
            unsafe.append(path.name)
    if unsafe:
        raise ValueError(f"Usage ledger job contains unsafe attempt paths: {unsafe}")
    return sorted(entries)


def _canonical_artifact_paths(trace_root: Path, job_id: str) -> tuple[Path, ...]:
    return (
        trace_root / f"{job_id}.jsonl",
        trace_root / "candidate-patches" / f"{job_id}.patch",
        trace_root / "private-evaluations" / f"{job_id}.json",
        trace_root / "run-evidence" / f"{job_id}.json",
    )


def _require_no_canonical_artifacts(paths: Sequence[Path]) -> None:
    present = []
    for path in paths:
        _reject_unsafe_path(path, "canonical artifact")
        if _path_lexists(path):
            present.append(str(path))
    if present:
        raise UnknownProviderUsageError(
            "Interrupted usage attempt has canonical artifact residue and cannot be recovered"
        )


def _conservative_recovery_elapsed_ms(
    state: Mapping[str, Any],
    finished_at: str,
) -> float:
    marker = state.get("marker")
    if not isinstance(marker, Mapping):
        raise ValueError("Usage recovery state lacks an admission marker")
    started = _parse_utc_timestamp(_required_text(marker, "started_at"), "attempt started_at")
    finished = _parse_utc_timestamp(finished_at, "terminal finished_at")
    if finished < started:
        raise ValueError("Usage recovery time precedes the durable admission marker")
    completed_records = state.get("completed_records")
    if not isinstance(completed_records, Sequence):
        raise ValueError("Usage recovery state lacks completed call receipts")
    latency_ms = Decimal("0")
    for record in completed_records:
        if not isinstance(record, Mapping):
            raise ValueError("Usage recovery has an invalid completed call receipt")
        completed_at = _parse_utc_timestamp(
            _required_text(record, "completed_at"), "call completed_at"
        )
        if completed_at > finished:
            raise ValueError("Usage recovery time precedes a completed provider receipt")
        latency_ms += Decimal(
            str(_finite_nonnegative_number(record.get("latency_ms"), "call latency_ms"))
        )
    wall_ms = Decimal(str((finished - started).total_seconds())) * Decimal("1000")
    elapsed_floor_ms = Decimal(
        str(
            _finite_nonnegative_number(
                state.get("recovery_elapsed_floor_ms", 0),
                "recovery_elapsed_floor_ms",
            )
        )
    )
    return float(max(wall_ms, latency_ms, elapsed_floor_ms))


def _new_attempt_id(job_root: Path) -> str:
    value = time.time_ns()
    while (job_root / f"attempt_{value}").exists():
        value += 1
    return f"attempt_{value}"


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _reject_unsafe_path(path: Path, name: str) -> None:
    if _path_lexists(path.parent):
        _require_safe_directory(path.parent, f"{name} parent directory")
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    if _path_lexists(path) and not path.is_file():
        raise ValueError(f"{name} must be a regular file when present")


def _require_safe_directory(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be a regular non-symlink directory")


def _require_safe_component(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
    ):
        raise ValueError(f"{name} must be a safe path component")


def _parse_utc_timestamp(value: str, name: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _aggregate_usage(values: Sequence[Any]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for value in values:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("Completed provider usage must be a non-empty object")
        _merge_usage(aggregate, value)
    return dict(sorted(aggregate.items()))


def _canonical_usage(value: Mapping[str, Any]) -> dict[str, Any]:
    return _aggregate_usage([value])


def _merge_usage(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if isinstance(raw_value, bool):
            raise ValueError(f"Usage value must be a non-negative integer: {key}")
        if isinstance(raw_value, int):
            if raw_value < 0:
                raise ValueError(f"Usage value must be a non-negative integer: {key}")
            prior = target.get(key, 0)
            if not isinstance(prior, int):
                raise ValueError(f"Conflicting usage shape: {key}")
            target[key] = prior + raw_value
        elif isinstance(raw_value, Mapping):
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Conflicting usage shape: {key}")
            _merge_usage(nested, raw_value)
        else:
            raise ValueError(f"Usage value must be a non-negative integer: {key}")


def _require_database_usage_match(
    row: Mapping[str, Any],
    totals: ConsumedUsageTotals,
) -> None:
    if _nonnegative_integer(row.get("consumed_tokens", 0), "consumed_tokens") != totals.tokens:
        raise ValueError("Scheduler consumed_tokens does not match immutable usage ledger")
    if not math.isclose(
        float(_nonnegative_decimal(row.get("consumed_cost", 0), "consumed_cost")),
        totals.cost,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Scheduler consumed_cost does not match immutable usage ledger")
    if not math.isclose(
        _finite_nonnegative_number(row.get("consumed_elapsed_ms", 0), "consumed_elapsed_ms"),
        totals.elapsed_ms,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Scheduler consumed_elapsed_ms does not match immutable usage ledger")


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonnegative_decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a decimal number")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    return "0" if normalized == 0 else text


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
