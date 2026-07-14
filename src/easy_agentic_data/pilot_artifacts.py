from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from easy_agentic_data.agent import DEFAULT_SYSTEM_PROMPT
from easy_agentic_data.coding_tools import SCHEMAS, CodingToolRuntime
from easy_agentic_data.evaluation import (
    EvaluationReport,
    apply_agent_termination,
    contamination_findings,
    evaluation_result_metrics,
    public_evaluation_result,
)
from easy_agentic_data.llm.observability import validate_observed_prompt_lineage
from easy_agentic_data.models import stable_id, utc_now
from easy_agentic_data.pilot_contract import PilotRolloutAssignment, PilotRunContract
from easy_agentic_data.pilot_usage_ledger import audit_pilot_usage_ledger
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_rollouts import (
    _VALIDATION_RECEIPT_AUTHORITY,
    RolloutArtifactPaths,
    RolloutValidationReceipt,
    deterministic_evaluators,
    safe_error_message,
    verify_candidate_from_clean_reset,
)
from easy_agentic_data.sandbox import MemorySandbox
from easy_agentic_data.trace_exporters import (
    analysis_record,
    trace_to_rl_episode,
    trace_to_sft,
    traces_to_preference,
)
from easy_agentic_data.traces import EventType, TerminationReason, Trace, load_trace, replay_trace
from easy_agentic_data.trajectory_review import (
    build_trajectory_review_queue,
    validate_review_gate,
)

PILOT_EXPORT_SCHEMA = "easy_agentic_data.pilot_exports.v1"
PILOT_REPRODUCTION_SCHEMA = "easy_agentic_data.pilot_reproduction.v1"
PILOT_QUALITY_SCHEMA = "easy_agentic_data.pilot_quality_report.v1"


@dataclass(frozen=True)
class PilotTraceArtifact:
    assignment: PilotRolloutAssignment
    row: dict[str, Any]
    trace: Trace
    report: EvaluationReport
    run_evidence: dict[str, Any]
    candidate_patch: str
    validation_receipt: RolloutValidationReceipt

    @property
    def trace_id(self) -> str:
        return self.trace.trace_id


def load_pilot_trace_artifacts(
    contract: PilotRunContract,
    registry: ScenarioRegistry,
    rows: Iterable[Mapping[str, Any]],
    trace_directory: str | Path,
    *,
    require_complete: bool = True,
) -> list[PilotTraceArtifact]:
    """Load and cross-check canonical traces, private reports, patches, and scheduler rows."""

    row_items = [dict(row) for row in rows]
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in row_items:
        job_id = str(row.get("job_id") or "")
        if not job_id:
            raise ValueError("Pilot scheduler rows require job_id")
        if job_id in rows_by_id:
            raise ValueError(f"Duplicate pilot scheduler row: {job_id}")
        rows_by_id[job_id] = row
    expected_ids = {assignment.job_id for assignment in contract.rollouts}
    unexpected = sorted(set(rows_by_id) - expected_ids)
    if unexpected:
        raise ValueError(f"Scheduler contains jobs outside the pilot contract: {unexpected}")

    root = Path(trace_directory)
    artifacts: list[PilotTraceArtifact] = []
    missing: list[str] = []
    for assignment in contract.rollouts:
        row = rows_by_id.get(assignment.job_id)
        if row is None or row.get("status") != "completed":
            missing.append(assignment.job_id)
            continue
        trace_path = root / f"{assignment.job_id}.jsonl"
        if not trace_path.exists():
            missing.append(assignment.job_id)
            continue
        artifacts.append(
            validate_pilot_rollout_artifact(
                contract,
                registry,
                assignment,
                row,
                trace_path,
            )
        )
    if require_complete and missing:
        raise ValueError(f"Pilot is missing {len(missing)} completed canonical trajectories")
    provider_response_ids = [
        _required_text(identity, "id")
        for artifact in artifacts
        for identity in _provider_response_identities(artifact.run_evidence)
    ]
    if len(provider_response_ids) != len(set(provider_response_ids)):
        raise ValueError("Pilot provider response IDs must be globally unique")
    return artifacts


def _provider_response_identities(
    evidence: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    calls = evidence.get("observed_calls")
    if not isinstance(calls, list):
        raise ValueError("Run evidence observed_calls must be a list")
    identities: list[Mapping[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            raise ValueError("Run evidence observed_calls must contain objects")
        identity = call.get("provider_response_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("Observed call lacks provider response identity")
        identities.append(identity)
    return identities


def validate_pilot_rollout_artifact(
    contract: PilotRunContract,
    registry: ScenarioRegistry,
    assignment: PilotRolloutAssignment,
    row: Mapping[str, Any],
    trace_path: str | Path,
    *,
    artifact_paths: RolloutArtifactPaths | None = None,
) -> PilotTraceArtifact:
    """Strictly validate one staged or canonical pilot artifact set."""

    _validate_scheduler_row(contract, assignment, row)
    path = Path(trace_path)
    paths = artifact_paths or RolloutArtifactPaths.for_trace(path)
    initial_artifact_sha256 = _validated_rollout_artifact_hashes(
        paths,
        expected_trace_path=path,
        job_id=assignment.job_id,
    )
    trace = load_trace(path, tolerate_truncated=False)
    replay = replay_trace(trace)
    scenario = registry.get_scenario(assignment.scenario_id)
    start = next(
        (event for event in trace.events if event.event_type is EventType.SESSION_STARTED),
        None,
    )
    if start is None:
        raise ValueError(f"Trace has no session start: {assignment.job_id}")
    initial_state_hash = _required_text(start.payload, "initial_state_hash")
    instance = registry.materialize(
        assignment.scenario_id,
        random_seed=assignment.random_seed,
        initial_state_hash=initial_state_hash,
    )
    expected_start_payload = {
        "scenario_instance_id": instance.instance_id,
        "scenario_id": instance.scenario_id,
        "environment_id": instance.environment_id,
        "initial_state_hash": instance.initial_state_hash,
        "public_task": instance.public_task.to_dict(),
        "random_seed": instance.random_seed,
        "parameters": instance.parameters,
    }
    if start.payload != expected_start_payload:
        raise ValueError(f"Trace scenario instance does not match registry: {assignment.job_id}")
    if str(row.get("trace_id") or "") != trace.trace_id:
        raise ValueError(f"Scheduler trace ID mismatch for {assignment.job_id}")
    private_payload = _read_json(paths.private_evaluation)
    _validate_content_id(
        private_payload,
        id_field="private_evaluation_id",
        prefix="private_evaluation",
        artifact_name=f"private evaluation for {assignment.job_id}",
    )
    if private_payload.get("schema") != "easy_agentic_data.private_evaluation.v1":
        raise ValueError(f"Unsupported private evaluation for {assignment.job_id}")
    if private_payload.get("trace_id") != trace.trace_id:
        raise ValueError(f"Private evaluation trace mismatch for {assignment.job_id}")
    report_value = private_payload.get("report")
    if not isinstance(report_value, dict):
        raise ValueError(f"Invalid private evaluation for {assignment.job_id}")
    report = EvaluationReport.from_dict(report_value)
    if report.to_dict() != report_value:
        raise ValueError(f"Private evaluation is not canonical: {assignment.job_id}")
    evidence = _read_json(paths.run_evidence)
    candidate_patch = paths.candidate_patch.read_text(encoding="utf-8")
    if private_payload.get("candidate_patch_sha256") != _sha256_text(candidate_patch):
        raise ValueError(f"Private candidate patch hash mismatch for {assignment.job_id}")
    if private_payload.get("clean_reset") is not True:
        raise ValueError(f"Private evaluation was not clean-reset for {assignment.job_id}")
    _validate_rollout_lineage(
        contract,
        assignment,
        trace,
        replay.state.success,
        report,
        evidence,
        candidate_patch,
        row,
        scenario,
    )
    final_artifact_sha256 = _validated_rollout_artifact_hashes(
        paths,
        expected_trace_path=path,
        job_id=assignment.job_id,
    )
    if final_artifact_sha256 != initial_artifact_sha256:
        raise ValueError(f"Rollout artifacts changed during strict validation: {assignment.job_id}")
    validation_receipt = RolloutValidationReceipt(
        contract_id=contract.contract_id,
        job_id=assignment.job_id,
        trace_id=trace.trace_id,
        artifact_sha256=initial_artifact_sha256,
        _authority=_VALIDATION_RECEIPT_AUTHORITY,
    )
    return PilotTraceArtifact(
        assignment=assignment,
        row=dict(row),
        trace=trace,
        report=report,
        run_evidence=evidence,
        candidate_patch=candidate_patch,
        validation_receipt=validation_receipt,
    )


def _validated_rollout_artifact_hashes(
    paths: RolloutArtifactPaths,
    *,
    expected_trace_path: Path,
    job_id: str,
) -> dict[str, str]:
    """Require the exact artifact layout and hash regular files contained by its root."""

    expected_paths = RolloutArtifactPaths.for_trace(expected_trace_path)
    if paths != expected_paths:
        raise ValueError(f"Rollout artifact paths do not match the trace: {job_id}")
    root_path = expected_trace_path.parent
    try:
        root_metadata = root_path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Rollout artifact root is missing: {job_id}") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"Rollout artifact root must be a non-symlink directory: {job_id}")
    try:
        artifact_root = root_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Rollout artifact root is missing: {job_id}") from exc
    resolved_root_metadata = artifact_root.stat()
    if (
        resolved_root_metadata.st_dev,
        resolved_root_metadata.st_ino,
    ) != (root_metadata.st_dev, root_metadata.st_ino):
        raise ValueError(f"Rollout artifact root changed during validation: {job_id}")
    artifact_paths = {
        "trace": paths.trace,
        "candidate_patch": paths.candidate_patch,
        "private_evaluation": paths.private_evaluation,
        "run_evidence": paths.run_evidence,
    }
    hashes: dict[str, str] = {}
    for name, artifact_path in artifact_paths.items():
        try:
            metadata = artifact_path.lstat()
        except FileNotFoundError as exc:
            raise ValueError(
                f"Missing rollout artifact for {job_id}: {artifact_path.name}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"Rollout artifact must be a regular non-symlink file: {artifact_path.name}"
            )
        try:
            resolved = artifact_path.resolve(strict=True)
            resolved.relative_to(artifact_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"Rollout artifact resolves outside its root: {artifact_path.name}"
            ) from exc
        hashes[name] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def reproduce_successful_trajectories(
    contract: PilotRunContract,
    registry: ScenarioRegistry,
    rows: Iterable[Mapping[str, Any]],
    trace_directory: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Rerun every successful candidate from its patch in another clean workspace reset."""

    artifacts = load_pilot_trace_artifacts(contract, registry, rows, trace_directory)
    output_path = Path(output)
    private_root = output_path.parent / "private-reproductions"
    items: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not artifact.report.success:
            continue
        assignment = artifact.assignment
        evidence = artifact.run_evidence
        item: dict[str, Any] = {
            "job_id": assignment.job_id,
            "trace_id": artifact.trace_id,
            "scenario_id": assignment.scenario_id,
            "random_seed": assignment.random_seed,
            "candidate_patch_sha256": _sha256_text(artifact.candidate_patch),
        }
        try:
            initial_hash = _required_text(evidence, "initial_state_hash")
            candidate_hash = _required_text(evidence, "candidate_state_hash")
            instance = registry.materialize(
                assignment.scenario_id,
                random_seed=assignment.random_seed,
                initial_state_hash=initial_hash,
            )
            if instance.instance_id != artifact.report.scenario_instance_id:
                raise ValueError("Reproduction scenario instance lineage does not match")
            clean = verify_candidate_from_clean_reset(
                registry.get_scenario(assignment.scenario_id),
                instance,
                artifact.trace,
                artifact.candidate_patch,
                expected_initial_state_hash=initial_hash,
                expected_candidate_state_hash=candidate_hash,
                diagnostics=artifact.report.metrics,
                turn_rewards=artifact.report.turn_rewards,
            )
            termination = replay_trace(artifact.trace).state.termination_reason
            if termination is None:
                raise ValueError("Successful trace has no termination reason")
            rerun_report = apply_agent_termination(
                clean.report,
                TerminationReason(termination),
            )
            signature_matches = _report_signature(rerun_report) == _report_signature(
                artifact.report
            )
            reproduced = rerun_report.success and signature_matches
            private_payload = {
                "schema": "easy_agentic_data.private_reproduction.v1",
                "job_id": assignment.job_id,
                "trace_id": artifact.trace_id,
                "report": rerun_report.to_dict(),
            }
            private_payload["private_reproduction_id"] = stable_id(
                "private_reproduction",
                private_payload,
            )
            _write_json_atomic(private_root / f"{assignment.job_id}.json", private_payload)
            item.update(
                {
                    "reproduced": reproduced,
                    "signature_matches": signature_matches,
                    "infrastructure_failure": rerun_report.infrastructure_failure,
                    "private_reproduction_sha256": _sha256_json(private_payload),
                    "results": [
                        public_evaluation_result(result) for result in rerun_report.results
                    ],
                }
            )
        except Exception as exc:
            item.update(
                {
                    "reproduced": False,
                    "signature_matches": False,
                    "infrastructure_failure": True,
                    "error": safe_error_message(exc),
                    "private_reproduction_sha256": "",
                    "results": [],
                }
            )
        item["reproduction_id"] = stable_id("reproduction", item)
        items.append(item)
    items.sort(key=lambda item: item["job_id"])
    material = {
        "schema": PILOT_REPRODUCTION_SCHEMA,
        "contract_id": contract.contract_id,
        "required_success_count": sum(artifact.report.success for artifact in artifacts),
        "reproduction_count": len(items),
        "reproduced_count": sum(bool(item["reproduced"]) for item in items),
        "all_successes_reproduced": all(bool(item["reproduced"]) for item in items),
        "items": items,
    }
    result = {**material, "reproduction_sha256": _sha256_json(material)}
    _write_json_atomic(output_path, result)
    return result


def write_pilot_exports(
    contract: PilotRunContract,
    registry: ScenarioRegistry,
    rows: Iterable[Mapping[str, Any]],
    trace_directory: str | Path,
    output_directory: str | Path,
    *,
    reproduction: Mapping[str, Any],
    private_reproduction_directory: str | Path,
    quarantined_trace_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Write immutable derived views from the exact evaluated pilot traces."""

    artifacts = load_pilot_trace_artifacts(contract, registry, rows, trace_directory)
    reproduced = validate_pilot_reproduction(
        contract,
        artifacts,
        reproduction,
        private_reproduction_directory,
    )
    if isinstance(quarantined_trace_ids, (str, bytes)):
        raise ValueError("Quarantined trace IDs must be an iterable of strings")
    quarantined = set(quarantined_trace_ids)
    if any(not isinstance(trace_id, str) or not trace_id for trace_id in quarantined):
        raise ValueError("Quarantined trace IDs must be non-empty strings")
    known_trace_ids = {artifact.trace_id for artifact in artifacts}
    unknown_quarantine = sorted(quarantined - known_trace_ids)
    if unknown_quarantine:
        raise ValueError(f"Quarantine contains unknown traces: {unknown_quarantine}")

    records, skip_reasons = _build_pilot_export_records(
        artifacts,
        reproduced,
        quarantined,
    )
    output_root = Path(output_directory)
    files: dict[str, dict[str, Any]] = {}
    for name, values in records.items():
        path = output_root / f"{name}.jsonl"
        _write_jsonl_atomic(path, values)
        files[name] = {
            "path": path.name,
            "count": len(values),
            "sha256": _sha256_bytes(path.read_bytes()),
        }
    material = {
        "schema": PILOT_EXPORT_SCHEMA,
        "contract_id": contract.contract_id,
        "source_trace_count": len(artifacts),
        "source_trace_ids_sha256": _sha256_json(
            sorted(artifact.trace_id for artifact in artifacts)
        ),
        "quarantined_trace_ids": sorted(quarantined),
        "reproduction_sha256": _required_sha256(
            reproduction.get("reproduction_sha256"),
            "reproduction_sha256",
        ),
        "files": files,
        "skip_reasons": skip_reasons,
        "gates": {
            "analysis_covers_all_traces": len(records["analysis"]) == len(artifacts),
            "sft_hard_verified_and_reproduced": all(
                item["trace_id"] in reproduced and reproduced[item["trace_id"]]
                for item in records["sft"]
            ),
            "preference_positive_margin": all(item["margin"] > 0 for item in records["preference"]),
        },
    }
    manifest = {**material, "export_manifest_id": stable_id("pilot_exports", material)}
    _write_json_atomic(output_root / "manifest.json", manifest)
    return manifest


def validate_pilot_reproduction(
    contract: PilotRunContract,
    artifacts: Sequence[PilotTraceArtifact],
    value: Mapping[str, Any],
    private_reproduction_directory: str | Path,
) -> dict[str, bool]:
    """Validate that a reproduction artifact covers the exact successful pilot traces."""

    expected_top_level = {
        "schema",
        "contract_id",
        "required_success_count",
        "reproduction_count",
        "reproduced_count",
        "all_successes_reproduced",
        "items",
        "reproduction_sha256",
    }
    if set(value) != expected_top_level:
        raise ValueError("Reproduction artifact fields do not match the schema")
    if value.get("schema") != PILOT_REPRODUCTION_SCHEMA:
        raise ValueError("Unsupported pilot reproduction schema")
    if value.get("contract_id") != contract.contract_id:
        raise ValueError("Reproduction artifact belongs to another pilot contract")
    material = {key: item for key, item in value.items() if key != "reproduction_sha256"}
    if value.get("reproduction_sha256") != _sha256_json(material):
        raise ValueError("Reproduction artifact content hash mismatch")

    successful = {artifact.trace_id: artifact for artifact in artifacts if artifact.report.success}
    private_root = Path(private_reproduction_directory)
    items = value.get("items")
    if not isinstance(items, list):
        raise ValueError("Reproduction items must be a list")
    reproduced: dict[str, bool] = {}
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("Reproduction item must be an object")
        item = dict(raw_item)
        allowed = {
            "job_id",
            "trace_id",
            "scenario_id",
            "random_seed",
            "candidate_patch_sha256",
            "reproduced",
            "signature_matches",
            "infrastructure_failure",
            "private_reproduction_sha256",
            "results",
            "reproduction_id",
            "error",
        }
        if not set(item).issubset(allowed) or set(item) - {"error"} != allowed - {"error"}:
            raise ValueError("Reproduction item fields do not match the schema")
        trace_id = _required_text(item, "trace_id")
        if trace_id in reproduced:
            raise ValueError("Reproduction trace IDs must be unique")
        artifact = successful.get(trace_id)
        if artifact is None:
            raise ValueError("Reproduction contains a non-success or unknown trace")
        expected_lineage = {
            "job_id": artifact.assignment.job_id,
            "scenario_id": artifact.assignment.scenario_id,
            "random_seed": artifact.assignment.random_seed,
            "candidate_patch_sha256": _sha256_text(artifact.candidate_patch),
        }
        invalid = [key for key, expected in expected_lineage.items() if item.get(key) != expected]
        if invalid:
            raise ValueError(f"Reproduction lineage mismatch: {invalid}")
        declared_id = _required_text(item, "reproduction_id")
        id_material = {
            key: item_value for key, item_value in item.items() if key != "reproduction_id"
        }
        if declared_id != stable_id("reproduction", id_material):
            raise ValueError("Reproduction item content ID mismatch")
        was_reproduced = _strict_bool(item.get("reproduced"), "reproduced")
        signature_matches = _strict_bool(item.get("signature_matches"), "signature_matches")
        infrastructure_failure = _strict_bool(
            item.get("infrastructure_failure"),
            "infrastructure_failure",
        )
        results = item.get("results")
        if not isinstance(results, list) or not all(
            isinstance(result, Mapping) for result in results
        ):
            raise ValueError("Reproduction results must be a list of objects")
        for result in results:
            _validate_public_result_shape(result)
        private_sha256 = item.get("private_reproduction_sha256")
        if private_sha256 not in {"", None}:
            declared_private_sha = _required_sha256(
                private_sha256,
                "private_reproduction_sha256",
            )
            private_path = private_root / f"{artifact.assignment.job_id}.json"
            if not private_path.is_file():
                raise ValueError(f"Missing private reproduction: {private_path.name}")
            private_payload = _read_json(private_path)
            _validate_content_id(
                private_payload,
                id_field="private_reproduction_id",
                prefix="private_reproduction",
                artifact_name=f"private reproduction for {artifact.assignment.job_id}",
            )
            if private_payload.get("schema") != "easy_agentic_data.private_reproduction.v1":
                raise ValueError("Unsupported private reproduction schema")
            if (
                private_payload.get("job_id") != artifact.assignment.job_id
                or private_payload.get("trace_id") != artifact.trace_id
            ):
                raise ValueError("Private reproduction lineage mismatch")
            if _sha256_json(private_payload) != declared_private_sha:
                raise ValueError("Private reproduction content hash mismatch")
            rerun_value = private_payload.get("report")
            if not isinstance(rerun_value, dict):
                raise ValueError("Private reproduction report must be an object")
            rerun_report = EvaluationReport.from_dict(rerun_value)
            if rerun_report.to_dict() != rerun_value:
                raise ValueError("Private reproduction report is not canonical")
            actual_signature_matches = _report_signature(rerun_report) == _report_signature(
                artifact.report
            )
            actual_reproduced = rerun_report.success and actual_signature_matches
            if (
                signature_matches != actual_signature_matches
                or infrastructure_failure != rerun_report.infrastructure_failure
                or was_reproduced != actual_reproduced
                or results != [public_evaluation_result(result) for result in rerun_report.results]
            ):
                raise ValueError("Reproduction result does not match its private rerun report")
        else:
            if was_reproduced or signature_matches or not infrastructure_failure or results:
                raise ValueError("Reproduction without a private rerun must be an infra failure")
        if was_reproduced:
            if "error" in item:
                raise ValueError("A reproduced success must not contain an error")
        elif "error" in item and not isinstance(item["error"], str):
            raise ValueError("Reproduction error must be a string")
        reproduced[trace_id] = was_reproduced

    if set(reproduced) != set(successful):
        missing = sorted(set(successful) - set(reproduced))
        unexpected = sorted(set(reproduced) - set(successful))
        raise ValueError(
            f"Reproduction success set mismatch: missing={missing} unexpected={unexpected}"
        )
    expected_count = len(successful)
    reproduced_count = sum(reproduced.values())
    declared_counts = {
        "required_success_count": expected_count,
        "reproduction_count": len(items),
        "reproduced_count": reproduced_count,
    }
    invalid_counts = [
        key for key, expected in declared_counts.items() if value.get(key) != expected
    ]
    if invalid_counts:
        raise ValueError(f"Reproduction counts are inconsistent: {invalid_counts}")
    declared_all = _strict_bool(
        value.get("all_successes_reproduced"),
        "all_successes_reproduced",
    )
    if declared_all != all(reproduced.values()):
        raise ValueError("Reproduction aggregate status is inconsistent")
    return reproduced


def _reverify_reproduction_from_clean_reset(
    contract: PilotRunContract,
    registry: ScenarioRegistry,
    rows: Iterable[Mapping[str, Any]],
    trace_directory: str | Path,
    artifacts: Sequence[PilotTraceArtifact],
    declared_reproduction: Mapping[str, Any],
    declared_private_directory: str | Path,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Execute and validate a fresh reproduction without trusting declared rerun files."""

    declared_map = validate_pilot_reproduction(
        contract,
        artifacts,
        declared_reproduction,
        declared_private_directory,
    )
    with tempfile.TemporaryDirectory(prefix="ead-pilot-quality-reverify-") as directory:
        root = Path(directory)
        output_path = root / "reproduction.json"
        returned = reproduce_successful_trajectories(
            contract,
            registry,
            rows,
            trace_directory,
            output_path,
        )
        if not output_path.is_file():
            raise ValueError("Independent clean-reset reproduction did not persist its artifact")
        fresh_reproduction = _read_json(output_path)
        if returned != fresh_reproduction:
            raise ValueError(
                "Independent clean-reset reproduction return value does not match its artifact"
            )
        fresh_map = validate_pilot_reproduction(
            contract,
            artifacts,
            fresh_reproduction,
            root / "private-reproductions",
        )

    if fresh_map != declared_map or fresh_reproduction != dict(declared_reproduction):
        raise ValueError(
            "Independent clean-reset reproduction does not match the declared reproduction"
        )
    return fresh_reproduction, fresh_map


def _reproduction_reverification_evidence(
    declared_reproduction: Mapping[str, Any],
    fresh_reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    """Build public hash evidence for an independently executed reproduction."""

    if dict(declared_reproduction) != dict(fresh_reproduction):
        raise ValueError(
            "Independent clean-reset reproduction does not match the declared reproduction"
        )
    items = fresh_reproduction.get("items")
    if not isinstance(items, list):
        raise ValueError("Independent reproduction items must be a list")
    private_hashes: list[list[str]] = []
    trace_ids: list[str] = []
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            raise ValueError("Independent reproduction item must be an object")
        job_id = _required_text(raw_item, "job_id")
        trace_ids.append(_required_text(raw_item, "trace_id"))
        raw_private_sha = raw_item.get("private_reproduction_sha256")
        private_sha = (
            ""
            if raw_private_sha in {"", None}
            else _required_sha256(
                raw_private_sha,
                "private_reproduction_sha256",
            )
        )
        private_hashes.append(
            [
                job_id,
                private_sha,
            ]
        )
    declared_sha = _required_sha256(
        declared_reproduction.get("reproduction_sha256"),
        "reproduction_sha256",
    )
    fresh_sha = _required_sha256(
        fresh_reproduction.get("reproduction_sha256"),
        "reproduction_sha256",
    )
    if fresh_sha != declared_sha:
        raise ValueError("Independent reproduction content hash does not match the declaration")
    return {
        "independent_execution": True,
        "fresh_private_artifacts_validated": bool(private_hashes)
        and all(value for _, value in private_hashes),
        "semantic_match": True,
        "content_hash_match": True,
        "executed_success_count": len(items),
        "successful_trace_ids_sha256": _sha256_json(sorted(trace_ids)),
        "private_reproduction_hashes_sha256": _sha256_json(sorted(private_hashes)),
        "declared_reproduction_sha256": declared_sha,
        "fresh_reproduction_sha256": fresh_sha,
    }


def validate_pilot_export_manifest(
    contract: PilotRunContract,
    artifacts: Sequence[PilotTraceArtifact],
    reproduction: Mapping[str, Any],
    private_reproduction_directory: str | Path,
    manifest: Mapping[str, Any],
    output_directory: str | Path,
) -> dict[str, Any]:
    """Validate export bytes and manifest semantics against their canonical source artifacts."""

    reproduced = validate_pilot_reproduction(
        contract,
        artifacts,
        reproduction,
        private_reproduction_directory,
    )
    raw_quarantined = manifest.get("quarantined_trace_ids")
    if not isinstance(raw_quarantined, list) or not all(
        isinstance(trace_id, str) and trace_id for trace_id in raw_quarantined
    ):
        raise ValueError("Export quarantine must be a list of non-empty trace IDs")
    if raw_quarantined != sorted(set(raw_quarantined)):
        raise ValueError("Export quarantine trace IDs must be sorted and unique")
    quarantined = set(raw_quarantined)
    known = {artifact.trace_id for artifact in artifacts}
    unknown = sorted(quarantined - known)
    if unknown:
        raise ValueError(f"Export quarantine contains unknown traces: {unknown}")
    records, skip_reasons = _build_pilot_export_records(
        artifacts,
        reproduced,
        quarantined,
    )
    root = Path(output_directory)
    files: dict[str, dict[str, Any]] = {}
    for name, expected_records in records.items():
        path = root / f"{name}.jsonl"
        if not path.is_file():
            raise ValueError(f"Missing pilot export file: {path.name}")
        expected_bytes = _jsonl_bytes(expected_records)
        actual_bytes = path.read_bytes()
        if actual_bytes != expected_bytes:
            raise ValueError(f"Pilot export content does not match source traces: {path.name}")
        files[name] = {
            "path": path.name,
            "count": len(expected_records),
            "sha256": _sha256_bytes(actual_bytes),
        }
    expected_gates = {
        "analysis_covers_all_traces": len(records["analysis"]) == len(artifacts),
        "sft_hard_verified_and_reproduced": all(
            reproduced.get(item["trace_id"]) is True for item in records["sft"]
        ),
        "preference_positive_margin": all(item["margin"] > 0 for item in records["preference"]),
    }
    material = {
        "schema": PILOT_EXPORT_SCHEMA,
        "contract_id": contract.contract_id,
        "source_trace_count": len(artifacts),
        "source_trace_ids_sha256": _sha256_json(
            sorted(artifact.trace_id for artifact in artifacts)
        ),
        "quarantined_trace_ids": sorted(quarantined),
        "reproduction_sha256": _required_sha256(
            reproduction.get("reproduction_sha256"),
            "reproduction_sha256",
        ),
        "files": files,
        "skip_reasons": skip_reasons,
        "gates": expected_gates,
    }
    expected_manifest = {
        **material,
        "export_manifest_id": stable_id("pilot_exports", material),
    }
    if dict(manifest) != expected_manifest:
        raise ValueError("Pilot export manifest does not match its source files and lineage")
    return expected_manifest


def _build_pilot_export_records(
    artifacts: Sequence[PilotTraceArtifact],
    reproduced: Mapping[str, bool],
    quarantined: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    analysis: list[dict[str, Any]] = []
    rl: list[dict[str, Any]] = []
    sft: list[dict[str, Any]] = []
    preference: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    by_scenario: dict[str, list[PilotTraceArtifact]] = {}
    for artifact in artifacts:
        by_scenario.setdefault(artifact.assignment.scenario_id, []).append(artifact)
        record = analysis_record(artifact.trace, artifact.report)
        record.update(
            {
                "job_id": artifact.assignment.job_id,
                "scenario_id": artifact.assignment.scenario_id,
                "rollout_index": artifact.assignment.rollout_index,
                "random_seed": artifact.assignment.random_seed,
                "quarantined": artifact.trace_id in quarantined,
                "clean_reset_reproduced": reproduced.get(artifact.trace_id, False),
            }
        )
        analysis.append(record)
        if artifact.trace_id in quarantined:
            skip_reasons["quarantined"] += 1
            continue
        if artifact.report.infrastructure_failure:
            skip_reasons["infrastructure_failure"] += 1
            continue
        rl.append(trace_to_rl_episode(artifact.trace, artifact.report))
        if artifact.report.success:
            if reproduced.get(artifact.trace_id) is True:
                sft.append(trace_to_sft(artifact.trace, artifact.report))
            else:
                skip_reasons["success_not_reproduced"] += 1

    for scenario_id, group in sorted(by_scenario.items()):
        if len(group) != 2:
            raise ValueError(f"Scenario {scenario_id} does not have exactly two trajectories")
        eligible = [
            item
            for item in group
            if item.trace_id not in quarantined and not item.report.infrastructure_failure
        ]
        if len(eligible) != 2:
            skip_reasons["preference_ineligible"] += 1
            continue
        chosen, rejected = sorted(
            eligible,
            key=lambda item: (item.report.reward, item.trace_id),
            reverse=True,
        )
        if chosen.report.reward <= rejected.report.reward:
            skip_reasons["preference_tie"] += 1
            continue
        preference.append(
            traces_to_preference(
                chosen.trace,
                chosen.report,
                rejected.trace,
                rejected.report,
            )
        )

    return (
        {
            "analysis": sorted(analysis, key=lambda item: item["trace_id"]),
            "rl": sorted(rl, key=lambda item: item["trace_id"]),
            "sft": sorted(sft, key=lambda item: item["trace_id"]),
            "preference": sorted(preference, key=lambda item: item["id"]),
        },
        dict(sorted(skip_reasons.items())),
    )


def _expected_positive_margin_scenario_count(
    artifacts: Sequence[PilotTraceArtifact],
    quarantined: set[str],
) -> int:
    """Independently count scenarios eligible for one positive-margin preference pair."""

    by_scenario: dict[str, list[PilotTraceArtifact]] = {}
    for artifact in artifacts:
        by_scenario.setdefault(artifact.assignment.scenario_id, []).append(artifact)
    count = 0
    for scenario_id, group in by_scenario.items():
        if len(group) != 2:
            raise ValueError(f"Scenario {scenario_id} does not have exactly two trajectories")
        eligible = [
            artifact
            for artifact in group
            if artifact.trace_id not in quarantined and not artifact.report.infrastructure_failure
        ]
        if len(eligible) == 2 and len({artifact.report.reward for artifact in eligible}) == 2:
            count += 1
    return count


def build_pilot_quality_report(
    contract: PilotRunContract,
    registry: ScenarioRegistry,
    rows: Iterable[Mapping[str, Any]],
    trace_directory: str | Path,
    *,
    reproduction: Mapping[str, Any] | None = None,
    private_reproduction_directory: str | Path | None = None,
    export_manifest: Mapping[str, Any] | None = None,
    export_directory: str | Path | None = None,
    review_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure every M2 exit gate and emit review-ready trajectory summaries."""

    row_items = [dict(row) for row in rows]
    usage_ledger_audit = audit_pilot_usage_ledger(
        contract,
        row_items,
        trace_directory,
        require_database_match=True,
    )
    artifacts = load_pilot_trace_artifacts(
        contract,
        registry,
        row_items,
        trace_directory,
        require_complete=False,
    )
    rows_by_id = {str(row.get("job_id") or ""): row for row in row_items}
    expected_ids = {assignment.job_id for assignment in contract.rollouts}
    infrastructure_jobs = sorted(
        job_id
        for job_id in expected_ids
        if rows_by_id.get(job_id, {}).get("status") == "infrastructure_failed"
    )
    reproduction_map: dict[str, bool] = {}
    reproduction_reverification: dict[str, Any] = {
        "independent_execution": False,
        "fresh_private_artifacts_validated": False,
        "semantic_match": False,
        "content_hash_match": False,
        "executed_success_count": 0,
        "successful_trace_ids_sha256": _sha256_json([]),
        "private_reproduction_hashes_sha256": _sha256_json([]),
        "declared_reproduction_sha256": "",
        "fresh_reproduction_sha256": "",
    }
    if reproduction is not None:
        fresh_reproduction, reproduction_map = _reverify_reproduction_from_clean_reset(
            contract,
            registry,
            row_items,
            trace_directory,
            artifacts,
            reproduction,
            _required_path(
                private_reproduction_directory,
                "private_reproduction_directory",
            ),
        )
        reproduction_reverification = _reproduction_reverification_evidence(
            reproduction,
            fresh_reproduction,
        )
    termination_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    rewards: Counter[str] = Counter()
    trace_ids: list[str] = []
    semantic_hashes: list[str] = []
    leak_trace_ids: list[str] = []
    hard_bypass_trace_ids: list[str] = []
    unreproduced_successes: list[str] = []
    agent_token_budget_violation_ids: list[str] = []
    agent_time_budget_violation_ids: list[str] = []
    total_tokens = sum(state.totals.tokens for state in usage_ledger_audit.jobs.values())
    total_cost = sum(
        (Decimal(str(state.totals.cost)) for state in usage_ledger_audit.jobs.values()),
        Decimal("0"),
    )
    total_seconds = sum(
        (
            Decimal(str(state.totals.elapsed_ms)) / Decimal("1000")
            for state in usage_ledger_audit.jobs.values()
        ),
        Decimal("0"),
    )
    review_summaries: list[dict[str, Any]] = []

    for artifact in artifacts:
        replay = replay_trace(artifact.trace)
        termination = replay.state.termination_reason or "missing"
        termination_counts[termination] += 1
        rewards[str(artifact.report.reward)] += 1
        trace_ids.append(artifact.trace_id)
        semantic_hashes.append(_semantic_trace_sha256(artifact.trace))
        for event in artifact.trace.events:
            if event.event_type is EventType.TOOL_REQUESTED:
                tool_counts[str(event.payload.get("name") or "unknown")] += 1
        initial_hash = _trace_initial_hash(artifact.trace)
        instance = registry.materialize(
            artifact.assignment.scenario_id,
            random_seed=artifact.assignment.random_seed,
            initial_state_hash=initial_hash,
        )
        if contamination_findings(artifact.trace.path, instance):
            leak_trace_ids.append(artifact.trace_id)
        expected_evaluators = [
            evaluator.name for evaluator in deterministic_evaluators(instance, artifact.trace)
        ] + ["agent_termination"]
        if _hard_verifier_bypassed(artifact, expected_evaluators):
            hard_bypass_trace_ids.append(artifact.trace_id)
        if artifact.report.success and not reproduction_map.get(artifact.trace_id, False):
            unreproduced_successes.append(artifact.trace_id)
        if artifact.report.metrics["tokens"] > contract.budgets.max_agent_tokens:
            agent_token_budget_violation_ids.append(artifact.trace_id)
        if artifact.report.metrics["agent_elapsed_ms"] > contract.budgets.max_agent_seconds * 1000:
            agent_time_budget_violation_ids.append(artifact.trace_id)
        scenario = registry.get_scenario(artifact.assignment.scenario_id)
        repository = str(scenario.environment.metadata.get("repository") or "unknown")
        language = str(scenario.environment.metadata.get("language") or "unknown").lower()
        risk_flags = []
        if artifact.trace_id in leak_trace_ids:
            risk_flags.append("hidden_content_leak")
        if artifact.trace_id in hard_bypass_trace_ids:
            risk_flags.append("hard_verifier_bypass")
        if artifact.trace_id in unreproduced_successes:
            risk_flags.append("success_not_reproduced")
        review_summaries.append(
            {
                "contract_id": contract.contract_id,
                "job_id": artifact.assignment.job_id,
                "trace_path": f"{artifact.assignment.job_id}.jsonl",
                "trace_id": artifact.trace_id,
                "scenario_id": artifact.assignment.scenario_id,
                "repository": repository,
                "language": language,
                "success": artifact.report.success,
                "termination_reason": termination,
                "infrastructure_failure": artifact.report.infrastructure_failure,
                "schema_valid": True,
                "replay_valid": True,
                "success_reproduced": reproduction_map.get(artifact.trace_id, False),
                "hidden_content_leak": artifact.trace_id in leak_trace_ids,
                "hard_verifier_bypass": artifact.trace_id in hard_bypass_trace_ids,
                "duplicate": False,
                "risk_flags": risk_flags,
            }
        )
    duplicate_trace_count = len(trace_ids) - len(set(trace_ids))
    semantic_duplicate_count = len(semantic_hashes) - len(set(semantic_hashes))
    semantic_counts = Counter(semantic_hashes)
    for summary, semantic_hash in zip(review_summaries, semantic_hashes, strict=False):
        summary["duplicate"] = semantic_counts[semantic_hash] > 1
    expected_count = len(contract.rollouts)
    infra_count = len(infrastructure_jobs)
    infra_rate = infra_count / expected_count
    success_count = sum(artifact.report.success for artifact in artifacts)
    validated_review: dict[str, Any] | None = None
    review_queue: dict[str, Any] | None = None
    if review_gate is not None:
        if len(review_summaries) != expected_count:
            raise ValueError("A review gate cannot be bound before all pilot traces exist")
        review_queue = build_trajectory_review_queue(review_summaries)
        validated_review = validate_review_gate(review_queue, review_gate)

    validated_export: dict[str, Any] | None = None
    if export_manifest is not None:
        if reproduction is None:
            raise ValueError("An export manifest requires a validated reproduction artifact")
        if export_directory is None:
            raise ValueError("An export manifest requires its export directory for byte validation")
        validated_export = validate_pilot_export_manifest(
            contract,
            artifacts,
            reproduction,
            _required_path(
                private_reproduction_directory,
                "private_reproduction_directory",
            ),
            export_manifest,
            export_directory,
        )
    quarantine_bound = (
        validated_review is not None
        and validated_export is not None
        and validated_export["quarantined_trace_ids"] == validated_review["quarantined_trace_ids"]
    )
    reproduction_complete = (
        reproduction is not None
        and len(reproduction_map) == success_count
        and all(reproduction_map.values())
    )
    export_gates = validated_export["gates"] if validated_export is not None else {}
    export_record_counts = {
        name: int(validated_export["files"][name]["count"]) if validated_export is not None else 0
        for name in ("analysis", "rl", "sft", "preference")
    }
    export_skip_reasons = (
        dict(validated_export["skip_reasons"]) if validated_export is not None else {}
    )
    expected_preference_count = (
        _expected_positive_margin_scenario_count(
            artifacts,
            set(validated_export["quarantined_trace_ids"]),
        )
        if validated_export is not None
        else 0
    )
    reproduction_counts_match = (
        reproduction is not None
        and reproduction.get("required_success_count") == success_count
        and reproduction.get("reproduction_count") == success_count
        and reproduction.get("reproduced_count") == success_count
        and reproduction_reverification["executed_success_count"] == success_count
    )
    within_tokens = total_tokens <= contract.budgets.max_total_tokens
    within_cost = total_cost <= contract.budgets.max_total_cost_usd
    within_seconds = total_seconds <= Decimal(str(contract.budgets.max_total_seconds))
    gates = {
        "exact_40_canonical_traces": len(artifacts) == expected_count == 40,
        "all_schema_valid_and_replayable": len(artifacts) == expected_count,
        "infrastructure_failure_rate_at_most_5_percent": infra_rate <= 0.05,
        "zero_hidden_content_leaks": not leak_trace_ids,
        "zero_hard_verifier_bypasses": not hard_bypass_trace_ids,
        "independent_clean_reset_reproduction_verified": all(
            reproduction_reverification[field]
            for field in (
                "independent_execution",
                "fresh_private_artifacts_validated",
                "semantic_match",
                "content_hash_match",
            )
        ),
        "all_successes_clean_reset_reproduced": reproduction_complete
        and not unreproduced_successes,
        "reproduction_counts_match_successes": reproduction_counts_match,
        "minimum_successes_met": success_count >= contract.quality_targets.minimum_successes,
        "all_agent_token_budgets_respected": not agent_token_budget_violation_ids,
        "all_agent_time_budgets_respected": not agent_time_budget_violation_ids,
        "within_total_token_budget": within_tokens,
        "within_total_cost_budget": within_cost,
        "within_total_time_budget": within_seconds,
        "immutable_usage_ledger_reconciled": True,
        "sft_hard_verified_and_reproduced": export_gates.get("sft_hard_verified_and_reproduced")
        is True,
        "minimum_sft_records_met": export_record_counts["sft"]
        >= contract.quality_targets.minimum_sft,
        "minimum_rl_records_met": export_record_counts["rl"] >= contract.quality_targets.minimum_rl,
        "minimum_preference_records_met": export_record_counts["preference"]
        >= contract.quality_targets.minimum_preference,
        "preference_count_matches_positive_margin_eligibility": export_manifest is not None
        and export_record_counts["preference"] == expected_preference_count,
        "export_counts_and_skip_reasons_recomputed": validated_export is not None,
        "preference_positive_margin": export_gates.get("preference_positive_margin") is True,
        "human_review_passed": validated_review is not None,
        "review_quarantine_matches_exports": quarantine_bound,
    }
    material: dict[str, Any] = {
        "schema": PILOT_QUALITY_SCHEMA,
        "contract_id": contract.contract_id,
        "expected_trajectories": expected_count,
        "canonical_trajectories": len(artifacts),
        "successes": success_count,
        "success_rate": success_count / len(artifacts) if artifacts else 0.0,
        "infrastructure_failures": infra_count,
        "infrastructure_failure_rate": infra_rate,
        "infrastructure_job_ids": infrastructure_jobs,
        "termination_counts": dict(sorted(termination_counts.items())),
        "tool_use": {
            "total_calls": sum(tool_counts.values()),
            "by_tool": dict(sorted(tool_counts.items())),
        },
        "reward_counts": dict(sorted(rewards.items())),
        "quality_targets": contract.quality_targets.to_dict(),
        "export_record_counts": export_record_counts,
        "eligible_positive_margin_scenario_count": expected_preference_count,
        "export_skip_reasons": export_skip_reasons,
        "leak_trace_ids": sorted(leak_trace_ids),
        "hard_bypass_trace_ids": sorted(hard_bypass_trace_ids),
        "unreproduced_success_trace_ids": sorted(unreproduced_successes),
        "agent_token_budget_violation_trace_ids": sorted(agent_token_budget_violation_ids),
        "agent_time_budget_violation_trace_ids": sorted(agent_time_budget_violation_ids),
        "duplicate_trace_count": duplicate_trace_count,
        "semantic_duplicate_count": semantic_duplicate_count,
        "usage": {
            "total_tokens": total_tokens,
            "total_cost_usd": format(total_cost, "f"),
            "total_seconds": format(total_seconds, "f"),
            "max_total_tokens": contract.budgets.max_total_tokens,
            "max_total_cost_usd": format(contract.budgets.max_total_cost_usd, "f"),
            "max_total_seconds": contract.budgets.max_total_seconds,
            "pricing_sha256": contract.pricing.pricing_sha256,
        },
        "usage_ledger": usage_ledger_audit.to_evidence(),
        "reproduction_sha256": (
            str(reproduction.get("reproduction_sha256")) if reproduction is not None else ""
        ),
        "reproduction_reverification": reproduction_reverification,
        "export_manifest_id": (
            str(validated_export["export_manifest_id"]) if validated_export is not None else ""
        ),
        "review_queue_sha256": (
            str(review_queue["queue_sha256"]) if review_queue is not None else ""
        ),
        "review_gate_sha256": (
            str(validated_review["review_gate_sha256"]) if validated_review is not None else ""
        ),
        "review_summaries": sorted(
            review_summaries,
            key=lambda item: (item["scenario_id"], item["trace_id"]),
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "generated_at": utc_now(),
    }
    material["report_id"] = stable_id(
        "pilot_quality",
        {key: value for key, value in material.items() if key != "generated_at"},
    )
    return material


def _validate_scheduler_row(
    contract: PilotRunContract,
    assignment: PilotRolloutAssignment,
    row: Mapping[str, Any],
) -> None:
    expected = {
        "job_id": assignment.job_id,
        "scenario_id": assignment.scenario_id,
        "rollout_index": assignment.rollout_index,
        "model": contract.provider.model,
        "config_hash": contract.contract_id,
        "status": "completed",
    }
    invalid = [key for key, value in expected.items() if row.get(key) != value]
    if invalid:
        raise ValueError(f"Scheduler row does not match pilot contract: {invalid}")


def _validate_rollout_lineage(
    contract: PilotRunContract,
    assignment: PilotRolloutAssignment,
    trace: Trace,
    replay_success: bool | None,
    report: EvaluationReport,
    evidence: Mapping[str, Any],
    candidate_patch: str,
    row: Mapping[str, Any],
    scenario: Any,
) -> None:
    start = next(
        (event for event in trace.events if event.event_type is EventType.SESSION_STARTED),
        None,
    )
    if start is None:
        raise ValueError(f"Trace has no session start: {assignment.job_id}")
    expected_start = {
        "scenario_id": assignment.scenario_id,
        "random_seed": assignment.random_seed,
    }
    if any(start.payload.get(key) != value for key, value in expected_start.items()):
        raise ValueError(f"Trace assignment lineage mismatch: {assignment.job_id}")
    if start.payload.get("scenario_instance_id") != report.scenario_instance_id:
        raise ValueError(f"Trace evaluation lineage mismatch: {assignment.job_id}")
    if replay_success is not report.success:
        raise ValueError(f"Trace success does not match evaluation: {assignment.job_id}")
    expected_public_results = []
    for result in report.results:
        public = public_evaluation_result(result)
        expected_public_results.append({"verifier": public.pop("evaluator"), **public})
    trace_results = [
        event.payload for event in trace.events if event.event_type is EventType.VERIFICATION_RESULT
    ]
    if trace_results != expected_public_results:
        raise ValueError(f"Public and private verifier results differ: {assignment.job_id}")
    report_infrastructure = any(result.infrastructure_failure for result in report.results)
    expected_success = bool(report.results) and all(result.passed for result in report.results)
    expected_success = expected_success and not report_infrastructure
    if (
        report.infrastructure_failure != report_infrastructure
        or report.success != expected_success
        or report.reward != int(report.success)
    ):
        raise ValueError(f"Private evaluation aggregate is inconsistent: {assignment.job_id}")

    _validate_content_id(
        evidence,
        id_field="evidence_id",
        prefix="run_evidence",
        artifact_name=f"run evidence for {assignment.job_id}",
    )
    if evidence.get("schema") != "easy_agentic_data.registry_rollout_evidence.v1":
        raise ValueError(f"Unsupported run evidence for {assignment.job_id}")
    corpus_binding = next(
        item for item in contract.corpus.scenarios if item.scenario_id == assignment.scenario_id
    )
    finished = next(
        (
            event
            for event in reversed(trace.events)
            if event.event_type is EventType.SESSION_FINISHED
        ),
        None,
    )
    if finished is None:
        raise ValueError(f"Trace has no session finish: {assignment.job_id}")
    evidence_expected = {
        "run_contract_id": contract.contract_id,
        "provider_binding_sha256": contract.provider.config_sha256,
        "trace_id": trace.trace_id,
        "scenario_id": assignment.scenario_id,
        "scenario_instance_id": report.scenario_instance_id,
        "environment_id": corpus_binding.environment_id,
        "image_digest": scenario.environment.image_digest,
        "random_seed": assignment.random_seed,
        "candidate_patch_sha256": _sha256_text(candidate_patch),
        "initial_state_hash": start.payload.get("initial_state_hash"),
        "candidate_state_hash": finished.payload.get("final_state_hash"),
        "success": report.success,
        "infrastructure_failure": report.infrastructure_failure,
        "reward": report.reward,
        "prompt_sha256": contract.versions.prompt_sha256,
        "contract_versions": contract.versions.to_dict(),
        "provider_config": contract.provider.to_dict(),
        "tool_schema_sha256": _scenario_tool_schema_sha256(scenario),
    }
    invalid = [key for key, value in evidence_expected.items() if evidence.get(key) != value]
    if invalid:
        raise ValueError(f"Run evidence does not match pilot lineage: {invalid}")
    if evidence.get("clean_reset_verification") is not True:
        raise ValueError(f"Run evidence lacks clean-reset verification: {assignment.job_id}")
    usage_attempt_id = _required_text(evidence, "usage_attempt_id")
    if not usage_attempt_id.startswith("attempt_"):
        raise ValueError(f"Run evidence usage attempt is invalid: {assignment.job_id}")
    if evidence.get("provider_runtime_sha256") != _sha256_json(contract.provider.to_dict()):
        raise ValueError(f"Run evidence provider hash mismatch: {assignment.job_id}")

    expected_budgets = {
        "max_turns": contract.budgets.max_agent_turns,
        "max_tool_calls": contract.budgets.max_agent_tool_calls,
        "max_tokens": contract.budgets.max_agent_tokens,
        "max_seconds": contract.budgets.max_agent_seconds,
        "malformed_tool_retries": contract.budgets.malformed_tool_retries,
    }
    if evidence.get("budgets") != expected_budgets:
        raise ValueError(f"Run evidence budgets mismatch: {assignment.job_id}")
    evaluator_names = [result.evaluator for result in report.results]
    if evidence.get("evaluator_names") != evaluator_names or evidence.get(
        "evaluator_set_sha256"
    ) != _sha256_json(evaluator_names):
        raise ValueError(f"Run evidence evaluator lineage mismatch: {assignment.job_id}")

    usage = evidence.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError(f"Run evidence usage is invalid: {assignment.job_id}")
    trace_usage = _aggregate_trace_usage(trace)
    if dict(usage) != trace_usage:
        raise ValueError(f"Run evidence usage does not match model events: {assignment.job_id}")
    usage_cost = contract.pricing.calculate_cost(usage)
    declared_agent_tokens = _finite_nonnegative_number(report.metrics.get("tokens"), "agent tokens")
    if declared_agent_tokens != usage_cost.total_tokens:
        raise ValueError(f"Run evidence agent token count mismatch: {assignment.job_id}")
    agent_elapsed_ms = _finite_nonnegative_number(
        report.metrics.get("agent_elapsed_ms"), "agent_elapsed_ms"
    )
    evidence_cost = _nonnegative_decimal(evidence.get("cost"), "run evidence cost")
    if evidence_cost != usage_cost.cost_usd:
        raise ValueError(f"Run evidence cost mismatch: {assignment.job_id}")
    retry_count = _validate_observed_calls(
        contract,
        trace,
        evidence.get("observed_calls"),
        _scenario_tool_schemas(scenario),
    )
    if evidence.get("retry_count") != retry_count:
        raise ValueError(f"Run evidence retry count mismatch: {assignment.job_id}")
    model_turns = sum(event.event_type is EventType.MODEL_RESPONSE for event in trace.events)
    declared_turns = _nonnegative_integer(evidence.get("turns"), "turns")
    tool_calls = sum(event.event_type is EventType.TOOL_FINISHED for event in trace.events)
    declared_tool_calls = _nonnegative_integer(evidence.get("tool_calls"), "tool_calls")
    if not model_turns <= declared_turns <= model_turns + 1 or declared_tool_calls != tool_calls:
        raise ValueError(f"Run evidence agent counters mismatch: {assignment.job_id}")
    if report.metrics.get("turns") != float(declared_turns) or report.metrics.get(
        "tool_calls"
    ) != float(declared_tool_calls):
        raise ValueError(f"Run evidence diagnostics mismatch: {assignment.job_id}")

    agent_termination = [
        result for result in report.results if result.evaluator == "agent_termination"
    ]
    if len(agent_termination) != 1 or agent_termination[0].evidence.get(
        "termination_reason"
    ) != evidence.get("termination_reason"):
        raise ValueError(f"Run evidence termination mismatch: {assignment.job_id}")
    termination_reason = str(evidence.get("termination_reason") or "")
    if declared_agent_tokens > contract.budgets.max_agent_tokens:
        raise ValueError(f"Agent token budget was exceeded: {assignment.job_id}")
    if (
        agent_elapsed_ms > contract.budgets.max_agent_seconds * 1000
        and termination_reason != TerminationReason.TIMEOUT.value
    ):
        raise ValueError(f"Agent time budget overrun was not terminated: {assignment.job_id}")
    elapsed_ms = _finite_nonnegative_number(evidence.get("elapsed_ms"), "elapsed_ms")
    expected_metrics = {
        **report.metrics,
        **evaluation_result_metrics(report),
        "elapsed_ms": elapsed_ms,
    }
    row_metrics = _row_metrics(row)
    row_expected = {
        "trace_id": trace.trace_id,
        "success": int(report.success),
        "tokens": usage_cost.total_tokens,
    }
    row_invalid = [key for key, value in row_expected.items() if row.get(key) != value]
    if row_invalid or row_metrics != expected_metrics:
        raise ValueError(
            f"Scheduler outcome does not match run evidence: {row_invalid or ['metrics']}"
        )
    row_cost = _nonnegative_decimal(row.get("cost"), "scheduler cost")
    if row_cost != usage_cost.cost_usd:
        raise ValueError(f"Scheduler cost does not match run evidence: {assignment.job_id}")


def _scenario_tool_schema_sha256(scenario: Any) -> str:
    return _sha256_json(_scenario_tool_schemas(scenario))


def _scenario_tool_schemas(scenario: Any) -> list[dict[str, Any]]:
    policy = ToolPolicy(
        scenario.environment.capability_packs or SCHEMAS.keys(),
        network_enabled=scenario.environment.network_policy != "disabled",
    )
    return CodingToolRuntime(MemorySandbox(), policy).schemas()


def _hard_verifier_bypassed(
    artifact: PilotTraceArtifact,
    expected_evaluators: Sequence[str],
) -> bool:
    results = artifact.report.results
    actual_names = [result.evaluator for result in results]
    if actual_names != list(expected_evaluators):
        return True
    if not artifact.report.success:
        return False
    return any(not result.passed or result.infrastructure_failure for result in results)


def _report_signature(report: EvaluationReport) -> list[tuple[str, bool, float, bool]]:
    return [
        (result.evaluator, result.passed, result.score, result.infrastructure_failure)
        for result in report.results
    ]


def _validate_content_id(
    value: Mapping[str, Any],
    *,
    id_field: str,
    prefix: str,
    artifact_name: str,
) -> None:
    declared = value.get(id_field)
    material = {key: item for key, item in value.items() if key != id_field}
    if declared != stable_id(prefix, material):
        raise ValueError(f"Content ID mismatch for {artifact_name}")


def _validate_public_result_shape(value: Mapping[str, Any]) -> None:
    expected_fields = {
        "evaluator",
        "passed",
        "score",
        "reason",
        "reason_sha256",
        "evidence",
        "infrastructure_failure",
    }
    if set(value) != expected_fields:
        raise ValueError("Public reproduction result fields do not match the schema")
    _required_text(value, "evaluator")
    _strict_bool(value.get("passed"), "passed")
    _finite_nonnegative_number(value.get("score"), "score")
    if value.get("reason") not in {
        "Evaluator passed",
        "Evaluator failed",
        "Evaluator infrastructure failure",
    }:
        raise ValueError("Public evaluator reason is not redacted")
    _required_sha256(value.get("reason_sha256"), "reason_sha256")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Public evaluator evidence must be an object")
    allowed_evidence = {
        "evidence_sha256",
        "field_count",
        "exit_code",
        "stdout_sha256",
        "stdout_bytes",
        "stderr_sha256",
        "stderr_bytes",
    }
    if not set(evidence).issubset(allowed_evidence):
        raise ValueError("Public evaluator evidence contains a private field")
    _required_sha256(evidence.get("evidence_sha256"), "evidence_sha256")
    _nonnegative_integer(evidence.get("field_count"), "field_count")
    if "exit_code" in evidence:
        _nonnegative_integer(evidence["exit_code"], "exit_code", allow_negative=True)
    for field_name in ("stdout_sha256", "stderr_sha256"):
        if field_name in evidence:
            _required_sha256(evidence[field_name], field_name)
    for field_name in ("stdout_bytes", "stderr_bytes"):
        if field_name in evidence:
            _nonnegative_integer(evidence[field_name], field_name)
    _strict_bool(value.get("infrastructure_failure"), "infrastructure_failure")


def _public_result_signature(
    results: Sequence[Mapping[str, Any]],
) -> list[tuple[str, bool, float, bool]]:
    return [
        (
            _required_text(result, "evaluator"),
            _strict_bool(result.get("passed"), "passed"),
            _finite_nonnegative_number(result.get("score"), "score"),
            _strict_bool(
                result.get("infrastructure_failure"),
                "infrastructure_failure",
            ),
        )
        for result in results
    ]


def _aggregate_trace_usage(trace: Trace) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    model_event_count = 0
    for event in trace.events:
        if event.event_type is not EventType.MODEL_RESPONSE:
            continue
        model_event_count += 1
        value = event.payload.get("usage")
        if not isinstance(value, Mapping) or not value:
            raise ValueError("Every model response must record non-empty token usage")
        _merge_usage(usage, value)
    if model_event_count == 0 or not usage:
        raise ValueError("Canonical pilot traces must contain observed model usage")
    return dict(sorted(usage.items()))


def _merge_usage(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    for raw_key, raw_amount in value.items():
        key = str(raw_key)
        if isinstance(raw_amount, int) and not isinstance(raw_amount, bool):
            if raw_amount < 0:
                raise ValueError(f"Token usage cannot be negative: {key}")
            existing = target.get(key, 0)
            if isinstance(existing, bool) or not isinstance(existing, int):
                raise ValueError(f"Conflicting token usage shape: {key}")
            target[key] = existing + raw_amount
        elif isinstance(raw_amount, Mapping):
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Conflicting token usage shape: {key}")
            _merge_usage(nested, raw_amount)
        else:
            raise ValueError(f"Token usage must contain non-negative integers: {key}")


def _validate_observed_calls(
    contract: PilotRunContract,
    trace: Trace,
    value: Any,
    tool_schemas: list[dict[str, Any]],
) -> int:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError("Observed model calls must be a list of objects")
    validate_observed_prompt_lineage(
        value,
        trace,
        DEFAULT_SYSTEM_PROMPT,
        tool_schemas,
    )
    model_events = [event for event in trace.events if event.event_type is EventType.MODEL_RESPONSE]
    if len(value) != len(model_events):
        raise ValueError("Observed model calls do not match trace model responses")
    total_retries = 0
    consumed_tokens = 0
    provider_response_ids: set[str] = set()
    allowed_response_models = {
        contract.provider.model,
        *contract.provider.response_model_aliases,
    }
    for index, (raw_record, event) in enumerate(zip(value, model_events, strict=True)):
        record = dict(raw_record)
        if record.get("call_index") != index or record.get("status") != "completed":
            raise ValueError("Observed model call order or status is invalid")
        if record.get("model") != contract.provider.model:
            raise ValueError("Observed model request does not match the pilot contract")
        response_model = _required_text(record, "response_model")
        if response_model not in allowed_response_models:
            raise ValueError("Observed response model is not frozen in the pilot contract")
        if event.payload.get("model") != response_model:
            raise ValueError("Observed response model does not match the public trace")
        provider_response_id = _validate_provider_response_identity(
            record,
            response_model=response_model,
        )
        if provider_response_id in provider_response_ids:
            raise ValueError("Provider response IDs must be unique within a rollout")
        provider_response_ids.add(provider_response_id)
        if record.get("usage") != event.payload.get("usage"):
            raise ValueError("Observed call usage does not match the public trace")
        _required_sha256(record.get("prompt_hash"), "prompt_hash")
        _required_text(record, "started_at")
        _nonnegative_integer(record.get("message_count"), "message_count")
        _nonnegative_integer(record.get("tool_count"), "tool_count")
        if record.get("temperature") != contract.provider.temperature:
            raise ValueError("Observed temperature does not match the pilot contract")
        remaining_tokens = contract.budgets.max_agent_tokens - consumed_tokens
        prompt_token_upper_bound = _nonnegative_integer(
            record.get("prompt_token_upper_bound"),
            "prompt_token_upper_bound",
        )
        if prompt_token_upper_bound == 0:
            raise ValueError("prompt_token_upper_bound must be positive")
        remaining_output_tokens = remaining_tokens - prompt_token_upper_bound
        if remaining_output_tokens <= 0:
            raise ValueError("Observed model call exceeded the declared call budget")
        expected_max_tokens = min(
            contract.provider.max_tokens,
            remaining_output_tokens,
        )
        if record.get("max_tokens") != expected_max_tokens:
            raise ValueError("Observed max_tokens does not match the pilot contract")
        call_usage = event.payload.get("usage")
        if not isinstance(call_usage, Mapping):
            raise ValueError("Observed call usage must be an object")
        call_cost = contract.pricing.calculate_cost(call_usage)
        if call_cost.input_tokens > prompt_token_upper_bound:
            raise ValueError("Observed input usage exceeded its pre-request token bound")
        if call_cost.output_tokens > expected_max_tokens:
            raise ValueError("Observed output usage exceeded the requested token bound")
        consumed_tokens += call_cost.total_tokens
        if consumed_tokens > contract.budgets.max_agent_tokens:
            raise ValueError("Observed usage exceeded the declared agent token budget")
        retries = _nonnegative_integer(record.get("retry_count"), "retry_count")
        total_retries += retries
        _finite_nonnegative_number(record.get("latency_ms"), "latency_ms")
    return total_retries


def _validate_provider_response_identity(
    record: Mapping[str, Any],
    *,
    response_model: str,
) -> str:
    identity = record.get("provider_response_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("Observed call lacks provider response identity")
    required_fields = {"id", "created", "object", "model"}
    allowed_fields = required_fields | {"system_fingerprint"}
    if not required_fields.issubset(identity) or not set(identity).issubset(allowed_fields):
        raise ValueError("Provider response identity fields do not match the schema")
    response_id = _required_text(identity, "id")
    created = identity.get("created")
    _nonnegative_integer(created, "provider response created")
    if identity.get("object") != "chat.completion":
        raise ValueError("Provider response object must be chat.completion")
    if identity.get("model") != response_model:
        raise ValueError("Provider response identity model does not match the response")
    if "system_fingerprint" in identity:
        fingerprint = identity["system_fingerprint"]
        if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
            raise ValueError("Provider system_fingerprint must be a non-empty string or null")
    identity_sha256 = _required_sha256(
        record.get("provider_response_identity_sha256"),
        "provider_response_identity_sha256",
    )
    if identity_sha256 != _sha256_json(dict(identity)):
        raise ValueError("Provider response identity hash mismatch")
    _required_sha256(
        record.get("provider_response_sha256"),
        "provider_response_sha256",
    )
    return response_id


def _scheduler_usage_totals(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, Decimal, Decimal]:
    total_tokens = 0
    total_cost = Decimal("0")
    total_seconds = Decimal("0")
    for row in rows:
        token_value = row.get("consumed_tokens", row.get("tokens", 0))
        cost_value = row.get("consumed_cost", row.get("cost", 0))
        total_tokens += _nonnegative_integer(token_value, "scheduler consumed_tokens")
        total_cost += _nonnegative_decimal(cost_value, "scheduler consumed_cost")
        metrics = _row_metrics(row)
        elapsed_value = row.get(
            "consumed_elapsed_ms",
            metrics.get("elapsed_ms", 0.0),
        )
        elapsed_ms = _finite_nonnegative_number(elapsed_value, "consumed_elapsed_ms")
        total_seconds += Decimal(str(elapsed_ms)) / Decimal("1000")
    return total_tokens, total_cost, total_seconds


def _row_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    value = row.get("metrics", {})
    if isinstance(value, str):
        try:
            value = json.loads(value or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Scheduler metrics must contain valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Scheduler metrics must be an object")
    result: dict[str, float] = {}
    for raw_key, raw_amount in value.items():
        if not isinstance(raw_key, str):
            raise ValueError("Scheduler metric names must be strings")
        result[raw_key] = _finite_nonnegative_number(raw_amount, raw_key)
    return result


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _nonnegative_integer(
    value: Any,
    field_name: str,
    *,
    allow_negative: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if not allow_negative and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _finite_nonnegative_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


def _nonnegative_decimal(value: Any, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a decimal number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a decimal number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


def _required_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _semantic_trace_sha256(trace: Trace) -> str:
    payload = []
    for event in trace.events:
        if event.event_type not in {
            EventType.USER_MESSAGE,
            EventType.MODEL_RESPONSE,
            EventType.TOOL_REQUESTED,
            EventType.TOOL_FINISHED,
            EventType.TOOL_MESSAGE,
        }:
            continue
        item = dict(event.payload)
        item.pop("message_id", None)
        item.pop("call_id", None)
        item.pop("usage", None)
        item.pop("model", None)
        payload.append({"event_type": event.event_type.value, "payload": item})
    return _sha256_json(payload)


def _trace_initial_hash(trace: Trace) -> str:
    for event in trace.events:
        if event.event_type is EventType.SESSION_STARTED:
            return _required_text(event.payload, "initial_state_hash")
    raise ValueError("Trace has no initial state hash")


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _required_path(value: str | Path | None, field_name: str) -> Path:
    if value is None or not str(value):
        raise ValueError(f"{field_name} is required")
    return Path(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    _write_text_atomic(path, _jsonl_bytes(values).decode("utf-8"))


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n" for value in values
    ).encode("utf-8")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_text(encoded)
