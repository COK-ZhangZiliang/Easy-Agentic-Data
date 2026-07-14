from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

EXPECTED_TRACE_COUNT = 40
REVIEW_SAMPLE_SIZE = 20
MIN_ACCEPTABLE_RATE = 0.90

REVIEW_QUEUE_SCHEMA_VERSION = "easy_agentic_data.trajectory_review_queue.v2"
REVIEW_DECISION_SCHEMA_VERSION = "easy_agentic_data.trajectory_review_decision.v1"
QUARANTINE_SET_SCHEMA_VERSION = "easy_agentic_data.trajectory_quarantine_set.v2"
REVIEW_GATE_SCHEMA_VERSION = "easy_agentic_data.trajectory_review_gate.v3"

_REVIEW_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "trace_id",
        "reviewer_alias",
        "timestamp",
        "verdict",
        "issue_codes",
        "notes",
        "quarantine",
    }
)
_REVIEW_QUEUE_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "source_trace_count",
        "source_sha256",
        "sample_size",
        "sample_sha256",
        "coverage",
        "items",
        "queue_sha256",
    }
)
_REVIEW_QUEUE_ITEM_FIELDS = frozenset(
    {
        "queue_index",
        "contract_id",
        "job_id",
        "trace_path",
        "trace_id",
        "scenario_id",
        "repository",
        "language",
        "success",
        "termination_reason",
        "risk_score",
        "risk_reasons",
        "summary_sha256",
    }
)
_REVIEW_GATE_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "queue_sha256",
        "decisions",
        "decision_sha256",
        "required_decision_count",
        "decision_count",
        "unique_decision_count",
        "verdict_counts",
        "acceptable_count",
        "acceptable_rate",
        "min_acceptable_rate",
        "critical_count",
        "critical_unquarantined_trace_ids",
        "duplicate_trace_ids",
        "unexpected_trace_ids",
        "unresolved_trace_ids",
        "validation_errors",
        "quarantined_trace_ids",
        "quarantine_sha256",
        "passed",
        "review_gate_sha256",
    }
)

_VALID_VERDICTS = frozenset({"acceptable", "minor", "critical"})
_ISSUE_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")
_CRITICAL_RISK_SCORE = 1_000
_RISK_WEIGHTS = {
    "hidden_content_leak": 1_000,
    "hard_verifier_bypass": 1_000,
    "schema_invalid": 500,
    "replay_invalid": 500,
    "success_not_reproduced": 500,
    "infrastructure_failure": 300,
    "duplicate_trace": 100,
    "unsuccessful": 75,
    "risky_termination": 40,
}
_NON_RISKY_TERMINATIONS = frozenset({"completed", "final_answer", "success"})


@dataclass(frozen=True)
class ReviewDecision:
    """A human-authored trajectory review decision.

    This type validates and canonicalizes supplied decisions. It never infers a verdict or
    reviewer identity from trajectory evidence.
    """

    trace_id: str
    reviewer_alias: str
    timestamp: str
    verdict: str
    issue_codes: tuple[str, ...] = ()
    notes: str = ""
    quarantine: bool = False

    def __post_init__(self) -> None:
        trace_id = _required_text(self.trace_id, "trace_id")
        reviewer_alias = _required_text(self.reviewer_alias, "reviewer_alias")
        timestamp = _canonical_timestamp(self.timestamp)
        verdict = _required_text(self.verdict, "verdict").lower()
        if verdict not in _VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(_VALID_VERDICTS)}")
        if (
            isinstance(self.issue_codes, (str, bytes))
            or not isinstance(self.issue_codes, Sequence)
            or not all(isinstance(code, str) for code in self.issue_codes)
        ):
            raise ValueError("issue_codes must be a sequence of issue codes")
        codes = tuple(code.strip().lower() for code in self.issue_codes)
        if len(set(codes)) != len(codes):
            raise ValueError("issue_codes must be unique")
        if any(not _ISSUE_CODE_PATTERN.fullmatch(code) for code in codes):
            raise ValueError("issue_codes contains an invalid issue code")
        if verdict != "acceptable" and not codes:
            raise ValueError(f"{verdict} verdict requires at least one issue code")
        if not isinstance(self.quarantine, bool):
            raise ValueError("quarantine must be a boolean")

        object.__setattr__(self, "trace_id", trace_id)
        object.__setattr__(self, "reviewer_alias", reviewer_alias)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "issue_codes", tuple(sorted(codes)))
        object.__setattr__(self, "notes", _canonical_notes(self.notes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REVIEW_DECISION_SCHEMA_VERSION,
            "trace_id": self.trace_id,
            "reviewer_alias": self.reviewer_alias,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "issue_codes": list(self.issue_codes),
            "notes": self.notes,
            "quarantine": self.quarantine,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReviewDecision:
        if not isinstance(value, Mapping):
            raise ValueError("review decision must be an object")
        _require_exact_fields(value, _REVIEW_DECISION_FIELDS, "review decision")
        schema_version = value.get("schema_version")
        if schema_version != REVIEW_DECISION_SCHEMA_VERSION:
            raise ValueError(f"unsupported review decision schema: {schema_version!r}")
        issue_codes = value.get("issue_codes")
        if not isinstance(issue_codes, list) or not all(
            isinstance(code, str) for code in issue_codes
        ):
            raise ValueError("issue_codes must be a list of strings")
        return cls(
            trace_id=value.get("trace_id"),
            reviewer_alias=value.get("reviewer_alias"),
            timestamp=value.get("timestamp"),
            verdict=value.get("verdict"),
            issue_codes=tuple(issue_codes),
            notes=value.get("notes"),
            quarantine=value.get("quarantine"),
        )


def build_trajectory_review_queue(
    summaries: Sequence[Mapping[str, Any]],
    *,
    expected_trace_count: int = EXPECTED_TRACE_COUNT,
    sample_size: int = REVIEW_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Build a deterministic risk-prioritized, stratified human-review queue."""

    if len(summaries) != expected_trace_count:
        raise ValueError(f"review source must contain exactly {expected_trace_count} summaries")
    if sample_size <= 0 or sample_size > expected_trace_count:
        raise ValueError("sample_size must be positive and no greater than expected_trace_count")

    normalized = [_normalize_summary(summary) for summary in summaries]
    contract_ids = {summary["contract_id"] for summary in normalized}
    if len(contract_ids) != 1:
        raise ValueError("review source summaries must bind one contract_id")
    contract_id = next(iter(contract_ids))
    trace_ids = [summary["trace_id"] for summary in normalized]
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("review source summaries must have a unique trace_id")
    job_ids = [summary["job_id"] for summary in normalized]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("review source summaries must have a unique job_id")
    normalized.sort(key=lambda summary: summary["trace_id"])

    selected = _select_summaries(normalized, sample_size)
    queue_order = sorted(
        selected,
        key=lambda summary: (
            -summary["risk_score"],
            summary["scenario_id"],
            summary["trace_id"],
        ),
    )
    items = [
        {
            "queue_index": index,
            **summary,
            "summary_sha256": _stable_sha256(summary),
        }
        for index, summary in enumerate(queue_order)
    ]
    source_sha256 = _stable_sha256(normalized)
    sample_sha256 = _sample_sha256(item["trace_id"] for item in items)
    material = {
        "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
        "contract_id": contract_id,
        "source_trace_count": len(normalized),
        "source_sha256": source_sha256,
        "sample_size": len(items),
        "sample_sha256": sample_sha256,
        "coverage": _coverage_summary(normalized, selected),
        "items": items,
    }
    return {**material, "queue_sha256": _stable_sha256(material)}


def validate_review_decisions(
    queue: Mapping[str, Any],
    decisions: Sequence[ReviewDecision | Mapping[str, Any]],
) -> tuple[ReviewDecision, ...]:
    """Strictly validate the final set of 20 human decisions for a review queue."""

    queue_items = _validated_queue_items(queue)
    expected_count = len(queue_items)
    if expected_count != REVIEW_SAMPLE_SIZE:
        raise ValueError(f"review queue must contain exactly {REVIEW_SAMPLE_SIZE} items")
    parsed = tuple(_coerce_decision(decision) for decision in decisions)
    if len(parsed) != expected_count:
        raise ValueError(f"review decisions must contain exactly {expected_count} entries")
    decision_ids = [decision.trace_id for decision in parsed]
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("review decisions must have a unique trace_id")
    expected_ids = {str(item["trace_id"]) for item in queue_items}
    actual_ids = set(decision_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"review decisions do not match queue; missing={missing}, unexpected={unexpected}"
        )
    return tuple(sorted(parsed, key=lambda decision: decision.trace_id))


def build_quarantine_set(
    queue: Mapping[str, Any],
    decisions: Sequence[ReviewDecision | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a stable set from explicit human quarantine decisions only."""

    queue_items = _validated_queue_items(queue)
    expected_ids = {str(item["trace_id"]) for item in queue_items}
    parsed = tuple(_coerce_decision(decision) for decision in decisions)
    decision_ids = [decision.trace_id for decision in parsed]
    if len(set(decision_ids)) != len(decision_ids):
        raise ValueError("review decisions must have a unique trace_id")
    unexpected = sorted(set(decision_ids) - expected_ids)
    if unexpected:
        raise ValueError(f"review decisions contain unexpected trace IDs: {unexpected}")
    trace_ids = sorted(decision.trace_id for decision in parsed if decision.quarantine)
    material = {
        "schema_version": QUARANTINE_SET_SCHEMA_VERSION,
        "contract_id": str(queue["contract_id"]),
        "queue_sha256": str(queue["queue_sha256"]),
        "decision_count": len(parsed),
        "trace_ids": trace_ids,
    }
    return {**material, "quarantine_sha256": _stable_sha256(material)}


def summarize_review_gate(
    queue: Mapping[str, Any],
    decisions: Sequence[ReviewDecision | Mapping[str, Any]],
    *,
    min_acceptable_rate: float = MIN_ACCEPTABLE_RATE,
) -> dict[str, Any]:
    """Summarize M2 human-review progress without manufacturing missing decisions."""

    if (
        isinstance(min_acceptable_rate, bool)
        or not isinstance(min_acceptable_rate, (int, float))
        or not 0.0 <= min_acceptable_rate <= 1.0
    ):
        raise ValueError("min_acceptable_rate must be between zero and one")
    queue_items = _validated_queue_items(queue)
    expected_ids = {str(item["trace_id"]) for item in queue_items}
    parsed = tuple(_coerce_decision(decision) for decision in decisions)
    grouped: dict[str, list[ReviewDecision]] = {}
    for decision in parsed:
        grouped.setdefault(decision.trace_id, []).append(decision)

    duplicate_ids = sorted(trace_id for trace_id, values in grouped.items() if len(values) > 1)
    unexpected_ids = sorted(set(grouped) - expected_ids)
    recognized: dict[str, ReviewDecision] = {}
    for trace_id in sorted(set(grouped) & expected_ids):
        recognized[trace_id] = min(
            grouped[trace_id], key=lambda decision: _canonical_json(decision.to_dict())
        )
    missing_ids = sorted(expected_ids - set(recognized))
    critical_unquarantined = sorted(
        decision.trace_id
        for decision in recognized.values()
        if decision.verdict == "critical" and not decision.quarantine
    )
    explicitly_unresolved = sorted(
        decision.trace_id
        for decision in recognized.values()
        if any(
            code == "unresolved" or code.startswith("unresolved.")
            for code in decision.issue_codes
        )
    )
    unresolved_ids = sorted(set(missing_ids + critical_unquarantined + explicitly_unresolved))

    verdict_counts = Counter(decision.verdict for decision in recognized.values())
    expected_count = len(queue_items)
    acceptable_count = verdict_counts["acceptable"]
    acceptable_rate = acceptable_count / expected_count if expected_count else 0.0
    validation_errors = []
    if expected_count != REVIEW_SAMPLE_SIZE:
        validation_errors.append("queue_sample_size_not_exact")
    if len(parsed) != expected_count:
        validation_errors.append("decision_count_not_exact")
    if duplicate_ids:
        validation_errors.append("duplicate_decisions")
    if unexpected_ids:
        validation_errors.append("unexpected_decisions")

    recognized_decisions = tuple(recognized[trace_id] for trace_id in sorted(recognized))
    quarantine = build_quarantine_set(queue, recognized_decisions)
    decision_records = [decision.to_dict() for decision in recognized_decisions]
    decision_sha256 = _stable_sha256(decision_records)
    passed = (
        not validation_errors
        and not unresolved_ids
        and acceptable_rate >= min_acceptable_rate
        and len(recognized) == expected_count
    )
    material = {
        "schema_version": REVIEW_GATE_SCHEMA_VERSION,
        "contract_id": str(queue["contract_id"]),
        "queue_sha256": str(queue["queue_sha256"]),
        "decisions": decision_records,
        "decision_sha256": decision_sha256,
        "required_decision_count": expected_count,
        "decision_count": len(parsed),
        "unique_decision_count": len(grouped),
        "verdict_counts": {
            verdict: verdict_counts.get(verdict, 0) for verdict in sorted(_VALID_VERDICTS)
        },
        "acceptable_count": acceptable_count,
        "acceptable_rate": acceptable_rate,
        "min_acceptable_rate": min_acceptable_rate,
        "critical_count": verdict_counts["critical"],
        "critical_unquarantined_trace_ids": critical_unquarantined,
        "duplicate_trace_ids": duplicate_ids,
        "unexpected_trace_ids": unexpected_ids,
        "unresolved_trace_ids": unresolved_ids,
        "validation_errors": validation_errors,
        "quarantined_trace_ids": quarantine["trace_ids"],
        "quarantine_sha256": quarantine["quarantine_sha256"],
        "passed": passed,
    }
    return {**material, "review_gate_sha256": _stable_sha256(material)}


def validate_review_gate(
    queue: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a complete, passing human-review gate against its exact queue.

    The gate is self-contained: it carries the canonical human-authored decisions and all
    derived fields. Validation re-parses those records and recomputes every field rather than
    trusting stored verdict counts, quarantine metadata, rates, or pass flags.
    """

    queue_items = _validated_queue_items(queue)
    if len(queue_items) != REVIEW_SAMPLE_SIZE:
        raise ValueError(f"review queue must contain exactly {REVIEW_SAMPLE_SIZE} items")
    if not isinstance(gate, Mapping):
        raise ValueError("review gate must be an object")
    _require_exact_fields(gate, _REVIEW_GATE_FIELDS, "review gate")
    if gate.get("schema_version") != REVIEW_GATE_SCHEMA_VERSION:
        raise ValueError("unsupported trajectory review gate schema")
    if gate.get("contract_id") != queue.get("contract_id"):
        raise ValueError("trajectory review gate contract binding mismatch")
    if gate.get("queue_sha256") != queue.get("queue_sha256"):
        raise ValueError("trajectory review gate queue binding mismatch")

    declared_hash = gate.get("review_gate_sha256")
    if not _is_sha256(declared_hash):
        raise ValueError("trajectory review gate hash must be a SHA-256 digest")
    supplied_material = {
        key: value for key, value in gate.items() if key != "review_gate_sha256"
    }
    if declared_hash != _stable_sha256(supplied_material):
        raise ValueError("trajectory review gate hash mismatch")

    decision_values = gate.get("decisions")
    if not isinstance(decision_values, list) or not all(
        isinstance(value, Mapping) for value in decision_values
    ):
        raise ValueError("review gate decisions must be a list of objects")
    if len(decision_values) != REVIEW_SAMPLE_SIZE:
        raise ValueError(
            f"review gate must embed exactly {REVIEW_SAMPLE_SIZE} decisions"
        )
    decisions = validate_review_decisions(queue, decision_values)
    canonical_records = [decision.to_dict() for decision in decisions]
    if _canonical_json(decision_values) != _canonical_json(canonical_records):
        raise ValueError("review gate decisions are not canonically serialized")

    recomputed = summarize_review_gate(
        queue,
        decisions,
        min_acceptable_rate=MIN_ACCEPTABLE_RATE,
    )
    if _canonical_json(gate) != _canonical_json(recomputed):
        raise ValueError("trajectory review gate derived fields mismatch")
    if recomputed["acceptable_rate"] < MIN_ACCEPTABLE_RATE:
        raise ValueError("trajectory review acceptable rate is below 90 percent")
    if recomputed["critical_unquarantined_trace_ids"]:
        raise ValueError("critical review decisions must be quarantined")
    if recomputed["unresolved_trace_ids"]:
        raise ValueError("trajectory review gate contains unresolved decisions")
    if recomputed["validation_errors"]:
        raise ValueError("trajectory review gate contains validation errors")
    if recomputed["passed"] is not True:
        raise ValueError("trajectory review gate did not pass")
    return recomputed


def _normalize_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    contract_id = _required_mapping_text(summary, "contract_id")
    job_id = _required_mapping_text(summary, "job_id")
    trace_path = _canonical_trace_path(summary.get("trace_path"), job_id)
    trace_id = _required_mapping_text(summary, "trace_id")
    scenario_id = _required_mapping_text(summary, "scenario_id")
    repository = _required_mapping_text(summary, "repository", "repository_slug", "repo")
    language = _required_mapping_text(summary, "language").lower()
    success = _strict_bool(summary.get("success"), "success")
    termination_reason = _required_mapping_text(summary, "termination_reason").lower()
    reasons = _risk_reasons(summary, success, termination_reason)
    declared_score = summary.get("risk_score", 0)
    if isinstance(declared_score, bool) or not isinstance(declared_score, int):
        raise ValueError(f"risk_score for {trace_id} must be a non-negative integer")
    if declared_score < 0:
        raise ValueError(f"risk_score for {trace_id} must be a non-negative integer")
    risk_score = declared_score + sum(_RISK_WEIGHTS.get(reason, 10) for reason in reasons)
    return {
        "contract_id": contract_id,
        "job_id": job_id,
        "trace_path": trace_path,
        "trace_id": trace_id,
        "scenario_id": scenario_id,
        "repository": repository,
        "language": language,
        "success": success,
        "termination_reason": termination_reason,
        "risk_score": risk_score,
        "risk_reasons": reasons,
    }


def _risk_reasons(
    summary: Mapping[str, Any], success: bool, termination_reason: str
) -> list[str]:
    flags = summary.get("risk_flags", ())
    if isinstance(flags, (str, bytes)) or not isinstance(flags, Sequence):
        raise ValueError("risk_flags must be a sequence of strings")
    reasons = {_risk_code(flag) for flag in flags}
    true_fields = {
        "hidden_content_leak": "hidden_content_leak",
        "hard_verifier_bypass": "hard_verifier_bypass",
        "infrastructure_failure": "infrastructure_failure",
        "duplicate": "duplicate_trace",
    }
    for field_name, reason in true_fields.items():
        if field_name in summary and _strict_bool(summary[field_name], field_name):
            reasons.add(reason)
    false_fields = {
        "schema_valid": "schema_invalid",
        "replay_valid": "replay_invalid",
    }
    for field_name, reason in false_fields.items():
        if field_name in summary and not _strict_bool(summary[field_name], field_name):
            reasons.add(reason)
    reproduction_fields = ("success_reproduced", "clean_reset_verified")
    if success and any(
        field_name in summary and not _strict_bool(summary[field_name], field_name)
        for field_name in reproduction_fields
    ):
        reasons.add("success_not_reproduced")
    if not success:
        reasons.add("unsuccessful")
    if termination_reason not in _NON_RISKY_TERMINATIONS:
        reasons.add("risky_termination")
    return sorted(reasons)


def _select_summaries(
    summaries: Sequence[dict[str, Any]], sample_size: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    critical = sorted(
        (summary for summary in summaries if summary["risk_score"] >= _CRITICAL_RISK_SCORE),
        key=lambda summary: (-summary["risk_score"], summary["trace_id"]),
    )
    for summary in critical[:sample_size]:
        selected.append(summary)
        selected_ids.add(summary["trace_id"])

    while len(selected) < sample_size:
        candidates = [summary for summary in summaries if summary["trace_id"] not in selected_ids]
        if not candidates:
            raise ValueError("not enough unique trace summaries to build review sample")
        covered = {
            field_name: {summary[field_name] for summary in selected}
            for field_name in (
                "scenario_id",
                "repository",
                "language",
                "success",
                "termination_reason",
            )
        }

        _, chosen_id = min(
            (_selection_rank(summary, covered), summary["trace_id"])
            for summary in candidates
        )
        chosen = next(summary for summary in candidates if summary["trace_id"] == chosen_id)
        selected.append(chosen)
        selected_ids.add(chosen["trace_id"])
    return selected


def _selection_rank(
    summary: Mapping[str, Any], covered: Mapping[str, set[Any]]
) -> tuple[Any, ...]:
    return (
        -(summary["scenario_id"] not in covered["scenario_id"]),
        -(summary["repository"] not in covered["repository"]),
        -(summary["language"] not in covered["language"]),
        -(summary["success"] not in covered["success"]),
        -(summary["termination_reason"] not in covered["termination_reason"]),
        -summary["risk_score"],
        summary["trace_id"],
    )


def _coverage_summary(
    source: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    fields = {
        "scenarios": "scenario_id",
        "repositories": "repository",
        "languages": "language",
        "success": "success",
        "termination_reasons": "termination_reason",
    }
    coverage = {}
    for label, field_name in fields.items():
        available = sorted({item[field_name] for item in source}, key=_sort_key)
        covered = sorted({item[field_name] for item in selected}, key=_sort_key)
        coverage[label] = {
            "available_count": len(available),
            "covered_count": len(covered),
            "complete": covered == available,
            "values": covered,
        }
    return coverage


def _validated_queue_items(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(queue, Mapping):
        raise ValueError("trajectory review queue must be an object")
    _require_exact_fields(queue, _REVIEW_QUEUE_FIELDS, "trajectory review queue")
    if queue.get("schema_version") != REVIEW_QUEUE_SCHEMA_VERSION:
        raise ValueError("unsupported trajectory review queue schema")
    contract_id = _required_mapping_text(queue, "contract_id")
    if contract_id != queue["contract_id"]:
        raise ValueError("trajectory review queue contract_id is not canonical")
    declared_hash = queue.get("queue_sha256")
    if not _is_sha256(declared_hash):
        raise ValueError("trajectory review queue hash must be a SHA-256 digest")
    material = {key: value for key, value in queue.items() if key != "queue_sha256"}
    if declared_hash != _stable_sha256(material):
        raise ValueError("trajectory review queue hash mismatch")
    source_trace_count = queue.get("source_trace_count")
    sample_size = queue.get("sample_size")
    if (
        isinstance(source_trace_count, bool)
        or not isinstance(source_trace_count, int)
        or source_trace_count <= 0
    ):
        raise ValueError("trajectory review source_trace_count must be a positive integer")
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size <= 0
        or sample_size > source_trace_count
    ):
        raise ValueError("trajectory review sample_size must be a valid positive integer")
    if not _is_sha256(queue.get("source_sha256")):
        raise ValueError("trajectory review source hash must be a SHA-256 digest")
    if not _is_sha256(queue.get("sample_sha256")):
        raise ValueError("trajectory review sample hash must be a SHA-256 digest")
    if not isinstance(queue.get("coverage"), Mapping):
        raise ValueError("trajectory review coverage must be an object")
    items = queue.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("trajectory review queue items must be a list of objects")
    if len(items) != sample_size:
        raise ValueError("trajectory review queue item count does not match sample_size")
    for index, item in enumerate(items):
        _require_exact_fields(item, _REVIEW_QUEUE_ITEM_FIELDS, "review queue item")
        if not isinstance(item.get("queue_index"), int) or isinstance(
            item.get("queue_index"), bool
        ) or item.get("queue_index") != index:
            raise ValueError("trajectory review queue indexes must be contiguous integers")
        for field_name in (
            "contract_id",
            "job_id",
            "trace_path",
            "trace_id",
            "scenario_id",
            "repository",
            "language",
            "termination_reason",
        ):
            text = _required_mapping_text(item, field_name)
            if text != item[field_name]:
                raise ValueError(f"review queue item {field_name} is not canonical")
        if item["contract_id"] != contract_id:
            raise ValueError("review queue item contract binding mismatch")
        if item["trace_path"] != _canonical_trace_path(
            item["trace_path"], item["job_id"]
        ):
            raise ValueError("review queue item trace_path is not canonical")
        if item["language"] != str(item["language"]).lower():
            raise ValueError("review queue item language is not canonical")
        if item["termination_reason"] != str(item["termination_reason"]).lower():
            raise ValueError("review queue item termination_reason is not canonical")
        if not isinstance(item.get("success"), bool):
            raise ValueError("review queue item success must be a boolean")
        risk_score = item.get("risk_score")
        if isinstance(risk_score, bool) or not isinstance(risk_score, int) or risk_score < 0:
            raise ValueError("review queue item risk_score must be a non-negative integer")
        risk_reasons = item.get("risk_reasons")
        if not isinstance(risk_reasons, list) or not all(
            isinstance(reason, str) for reason in risk_reasons
        ):
            raise ValueError("review queue item risk_reasons must be a list of strings")
        if risk_reasons != sorted(set(risk_reasons)) or any(
            reason != _risk_code(reason) for reason in risk_reasons
        ):
            raise ValueError("review queue item risk_reasons are not canonical")
        summary_hash = item.get("summary_sha256")
        summary_material = {
            key: value
            for key, value in item.items()
            if key not in {"queue_index", "summary_sha256"}
        }
        if not _is_sha256(summary_hash) or summary_hash != _stable_sha256(summary_material):
            raise ValueError("trajectory review queue item summary hash mismatch")
    trace_ids = [str(item.get("trace_id") or "") for item in items]
    if any(not trace_id for trace_id in trace_ids) or len(set(trace_ids)) != len(trace_ids):
        raise ValueError("trajectory review queue items must have unique trace IDs")
    job_ids = [str(item.get("job_id") or "") for item in items]
    if any(not job_id for job_id in job_ids) or len(set(job_ids)) != len(job_ids):
        raise ValueError("trajectory review queue items must have unique job IDs")
    if queue.get("sample_sha256") != _sample_sha256(trace_ids):
        raise ValueError("trajectory review sample hash mismatch")
    return items


def _coerce_decision(value: ReviewDecision | Mapping[str, Any]) -> ReviewDecision:
    if isinstance(value, ReviewDecision):
        return value
    if isinstance(value, Mapping):
        return ReviewDecision.from_dict(value)
    raise ValueError("review decision must be a ReviewDecision or mapping")


def _sample_sha256(trace_ids: Sequence[str] | Any) -> str:
    return _stable_sha256(
        {
            "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
            "trace_ids": sorted(str(trace_id) for trace_id in trace_ids),
        }
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_mapping_text(value: Mapping[str, Any], *field_names: str) -> str:
    for field_name in field_names:
        candidate = value.get(field_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError(f"trace summary requires non-empty field {field_names[0]!r}")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _canonical_trace_path(value: Any, job_id: str) -> str:
    trace_path = _required_text(value, "trace_path")
    parsed = PurePosixPath(trace_path)
    expected = f"{job_id}.jsonl"
    if (
        parsed.is_absolute()
        or parsed.parts != (expected,)
        or parsed.as_posix() != trace_path
    ):
        raise ValueError(
            "trace_path must be the safe canonical relative path "
            f"{expected!r}"
        )
    return trace_path


def _strict_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected, key=repr)
    raise ValueError(f"{label} fields mismatch; missing={missing}, unexpected={unexpected}")


def _risk_code(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("risk_flags must contain non-empty strings")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", normalized):
        raise ValueError(f"invalid risk flag: {value!r}")
    return normalized


def _canonical_timestamp(value: Any) -> str:
    text = _required_text(value, "timestamp")
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_notes(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("notes must be a string")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _sort_key(value: Any) -> str:
    return _canonical_json(value)
