from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol

from easy_agentic_data.sandbox import Sandbox
from easy_agentic_data.scenarios import ScenarioInstance
from easy_agentic_data.traces import EventType, TerminationReason, Trace, TraceRecorder


@dataclass(frozen=True)
class EvaluationEvidence:
    evaluator: str
    passed: bool
    score: float
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    infrastructure_failure: bool = False


@dataclass(frozen=True)
class EvaluationReport:
    scenario_instance_id: str
    results: List[EvaluationEvidence]
    success: bool
    reward: int
    infrastructure_failure: bool
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeterministicEvaluator(Protocol):
    name: str

    def evaluate(self, sandbox: Sandbox, instance: ScenarioInstance) -> EvaluationEvidence: ...


class HiddenCommandEvaluator:
    name = "hidden_command"

    def __init__(self, command: List[str]) -> None:
        self.command = command

    def evaluate(self, sandbox: Sandbox, instance: ScenarioInstance) -> EvaluationEvidence:
        del instance
        try:
            result = sandbox.execute(self.command)
            return EvaluationEvidence(
                self.name,
                result.exit_code == 0,
                1.0 if result.exit_code == 0 else 0.0,
                "Hidden command passed" if result.exit_code == 0 else "Hidden command failed",
                {"exit_code": result.exit_code, "stdout": result.stdout, "stderr": result.stderr},
            )
        except Exception as exc:
            return EvaluationEvidence(
                self.name, False, 0.0, f"Evaluator infrastructure failure: {exc}",
                infrastructure_failure=True,
            )


class RequiredStateEvaluator:
    name = "required_state"

    def evaluate(self, sandbox: Sandbox, instance: ScenarioInstance) -> EvaluationEvidence:
        requirements = instance.hidden_evaluator.required_state
        failures = []
        for path, expected in requirements.get("file_equals", {}).items():
            try:
                actual = sandbox.read(path)
            except Exception:
                actual = None
            if actual != expected:
                failures.append(path)
        for path, fragment in requirements.get("file_contains", {}).items():
            try:
                actual = sandbox.read(path)
            except Exception:
                actual = ""
            if fragment not in actual:
                failures.append(path)
        return EvaluationEvidence(
            self.name, not failures, 1.0 if not failures else 0.0,
            "Required state reached" if not failures else f"Required state missing: {sorted(set(failures))}",
        )


class ForbiddenStateEvaluator:
    name = "forbidden_state"

    def evaluate(self, sandbox: Sandbox, instance: ScenarioInstance) -> EvaluationEvidence:
        violations = []
        for path, expected in instance.hidden_evaluator.forbidden_state.get("file_equals", {}).items():
            try:
                actual = sandbox.read(path)
            except Exception:
                actual = None
            if actual != expected:
                violations.append(path)
        return EvaluationEvidence(
            self.name, not violations, 1.0 if not violations else 0.0,
            "No forbidden state changes" if not violations else f"Forbidden changes: {violations}",
        )


class EvaluationSuite:
    def __init__(self, evaluators: Iterable[DeterministicEvaluator]) -> None:
        self.evaluators = list(evaluators)

    def evaluate(
        self,
        sandbox: Sandbox,
        instance: ScenarioInstance,
        *,
        diagnostics: Dict[str, float] | None = None,
    ) -> EvaluationReport:
        results = [evaluator.evaluate(sandbox, instance) for evaluator in self.evaluators]
        infrastructure_failure = any(result.infrastructure_failure for result in results)
        success = bool(results) and all(result.passed for result in results) and not infrastructure_failure
        return EvaluationReport(
            instance.instance_id,
            results,
            success,
            1 if success else 0,
            infrastructure_failure,
            diagnostics or {},
        )


def finalize_evaluation_trace(
    recorder: TraceRecorder,
    report: EvaluationReport,
    *,
    final_state_hash: str,
) -> None:
    for result in report.results:
        recorder.record(
            EventType.VERIFICATION_RESULT,
            {
                "verifier": result.evaluator,
                "passed": result.passed,
                "score": result.score,
                "reason": result.reason,
                "evidence": result.evidence,
                "infrastructure_failure": result.infrastructure_failure,
            },
        )
    recorder.record(
        EventType.SESSION_FINISHED,
        {
            "termination_reason": (
                TerminationReason.INFRASTRUCTURE_FAILURE.value
                if report.infrastructure_failure
                else TerminationReason.SUCCESS.value
                if report.success
                else TerminationReason.AGENT_STOP.value
            ),
            "final_state_hash": final_state_hash,
            "success": report.success,
        },
    )


def contamination_findings(trace_path: str | Path, instance: ScenarioInstance) -> List[str]:
    content = Path(trace_path).read_text(encoding="utf-8")
    candidates = (
        instance.hidden_evaluator.hidden_tests
        + instance.hidden_evaluator.reference_artifacts
        + ([instance.hidden_evaluator.reference_answer] if instance.hidden_evaluator.reference_answer else [])
    )
    return [value for value in candidates if value and value in content]


def pass_at_k(rewards: Iterable[int]) -> Dict[str, float]:
    values = list(rewards)
    successes = sum(1 for value in values if value > 0)
    return {
        "rollouts": float(len(values)),
        "successes": float(successes),
        "pass_at_k": 1.0 if successes else 0.0,
        "success_rate": successes / len(values) if values else 0.0,
    }


def trace_policy_evidence(trace: Trace) -> EvaluationEvidence:
    denied = [
        event.payload
        for event in trace.events
        if event.event_type is EventType.POLICY_DECISION
        and event.payload.get("decision") != "allow"
    ]
    return EvaluationEvidence(
        "policy_integrity",
        not denied,
        1.0 if not denied else 0.0,
        "No policy violations" if not denied else f"Policy denied {len(denied)} action(s)",
        {"denied": denied},
    )


def workspace_summary(sandbox: Sandbox) -> Dict[str, Any]:
    hashes = {}
    for path in sandbox.list_files():
        try:
            hashes[path] = hashlib.sha256(sandbox.read(path).encode("utf-8")).hexdigest()
        except Exception:
            continue
    return {
        "state_hash": sandbox.state_hash(),
        "file_hashes": hashes,
        "diff": sandbox.diff(),
    }


def rank_reports(reports: Iterable[EvaluationReport]) -> List[EvaluationReport]:
    """Rank by deterministic reward, then by diagnostic efficiency only."""

    return sorted(
        reports,
        key=lambda report: (
            report.reward,
            -report.metrics.get("tool_errors", 0.0),
            -report.metrics.get("turns", 0.0),
            -report.metrics.get("tokens", 0.0),
            -report.metrics.get("latency_ms", 0.0),
            -report.metrics.get("patch_bytes", 0.0),
        ),
        reverse=True,
    )
