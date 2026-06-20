from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from easy_agentic_data.sandbox import Sandbox
from easy_agentic_data.scenarios import ScenarioInstance
from easy_agentic_data.traces import EventType, TerminationReason, Trace, TraceRecorder


@dataclass(frozen=True)
class EvaluationEvidence:
    evaluator: str
    passed: bool
    score: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    infrastructure_failure: bool = False


@dataclass(frozen=True)
class TurnRewardEvidence:
    turn_index: int
    event_id: str
    kind: str
    action_type: str
    reward: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    scenario_instance_id: str
    results: list[EvaluationEvidence]
    success: bool
    reward: int
    infrastructure_failure: bool
    metrics: dict[str, float]
    turn_rewards: list[TurnRewardEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicEvaluator(Protocol):
    name: str

    def evaluate(self, sandbox: Sandbox, instance: ScenarioInstance) -> EvaluationEvidence: ...


class HiddenCommandEvaluator:
    name = "hidden_command"

    def __init__(self, command: list[str]) -> None:
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
                self.name,
                False,
                0.0,
                f"Evaluator infrastructure failure: {exc}",
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
            self.name,
            not failures,
            1.0 if not failures else 0.0,
            "Required state reached"
            if not failures
            else f"Required state missing: {sorted(set(failures))}",
        )


class ForbiddenStateEvaluator:
    name = "forbidden_state"

    def evaluate(self, sandbox: Sandbox, instance: ScenarioInstance) -> EvaluationEvidence:
        violations = []
        for path, expected in instance.hidden_evaluator.forbidden_state.get(
            "file_equals", {}
        ).items():
            try:
                actual = sandbox.read(path)
            except Exception:
                actual = None
            if actual != expected:
                violations.append(path)
        return EvaluationEvidence(
            self.name,
            not violations,
            1.0 if not violations else 0.0,
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
        diagnostics: dict[str, float] | None = None,
        turn_rewards: Iterable[TurnRewardEvidence] | None = None,
    ) -> EvaluationReport:
        turn_reward_items = list(turn_rewards or [])
        metrics = dict(diagnostics or {})
        metrics.update(turn_reward_metrics(turn_reward_items))
        results = [evaluator.evaluate(sandbox, instance) for evaluator in self.evaluators]
        infrastructure_failure = any(result.infrastructure_failure for result in results)
        success = (
            bool(results)
            and all(result.passed for result in results)
            and not infrastructure_failure
        )
        return EvaluationReport(
            instance.instance_id,
            results,
            success,
            1 if success else 0,
            infrastructure_failure,
            metrics,
            turn_reward_items,
        )


def apply_agent_termination(
    report: EvaluationReport,
    termination_reason: TerminationReason,
) -> EvaluationReport:
    """Fail the outcome if the agent stopped for a budget, policy, or infrastructure reason."""

    passed = termination_reason in {TerminationReason.AGENT_STOP, TerminationReason.SUCCESS}
    evidence = EvaluationEvidence(
        "agent_termination",
        passed,
        1.0 if passed else 0.0,
        (
            "Agent completed normally"
            if passed
            else f"Agent terminated before completion: {termination_reason.value}"
        ),
        {"termination_reason": termination_reason.value},
        infrastructure_failure=termination_reason is TerminationReason.INFRASTRUCTURE_FAILURE,
    )
    results = [*report.results, evidence]
    infrastructure_failure = report.infrastructure_failure or evidence.infrastructure_failure
    success = report.success and evidence.passed and not infrastructure_failure
    return EvaluationReport(
        report.scenario_instance_id,
        results,
        success,
        1 if success else 0,
        infrastructure_failure,
        report.metrics,
        report.turn_rewards,
    )


def finalize_evaluation_trace(
    recorder: TraceRecorder,
    report: EvaluationReport,
    *,
    final_state_hash: str,
    termination_reason: TerminationReason | None = None,
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
                else termination_reason.value
                if termination_reason is not None and not report.success
                else TerminationReason.SUCCESS.value
                if report.success
                else TerminationReason.AGENT_STOP.value
            ),
            "final_state_hash": final_state_hash,
            "success": report.success,
        },
    )


def contamination_findings(trace_path: str | Path, instance: ScenarioInstance) -> list[str]:
    content = Path(trace_path).read_text(encoding="utf-8")
    candidates = (
        instance.hidden_evaluator.hidden_tests
        + instance.hidden_evaluator.reference_artifacts
        + (
            [instance.hidden_evaluator.reference_answer]
            if instance.hidden_evaluator.reference_answer
            else []
        )
    )
    return [value for value in candidates if value and value in content]


def pass_at_k(rewards: Iterable[int]) -> dict[str, float]:
    values = list(rewards)
    successes = sum(1 for value in values if value > 0)
    return {
        "rollouts": float(len(values)),
        "successes": float(successes),
        "pass_at_k": 1.0 if successes else 0.0,
        "success_rate": successes / len(values) if values else 0.0,
    }


def derive_turn_rewards(
    trace: Trace,
    instance: ScenarioInstance | None = None,
) -> list[TurnRewardEvidence]:
    rewards: list[TurnRewardEvidence] = []
    turn_index = -1
    requested_tools: dict[str, str] = {}
    requested_arguments: dict[str, dict[str, Any]] = {}
    for event in trace.events:
        payload = event.payload
        if event.event_type is EventType.MODEL_RESPONSE:
            turn_index += 1
        elif event.event_type is EventType.TOOL_REQUESTED:
            call_id = payload["call_id"]
            name = payload["name"]
            arguments = payload.get("arguments", {})
            requested_tools[call_id] = name
            requested_arguments[call_id] = arguments if isinstance(arguments, dict) else {}
            if name == "ask_user":
                reward = _ask_user_reward(
                    str(requested_arguments[call_id].get("question", "")),
                    instance,
                )
                rewards.append(
                    TurnRewardEvidence(
                        turn_index=max(turn_index, 0),
                        event_id=event.event_id,
                        kind="information_gathering",
                        action_type="ask_user",
                        reward=reward,
                        reason=(
                            "Asked for relevant hidden user information"
                            if reward > 0
                            else "Asked the simulated user for information"
                        ),
                        evidence={"call_id": call_id, "tool": name},
                    )
                )
        elif event.event_type is EventType.POLICY_DECISION:
            if payload["decision"] != "allow":
                rewards.append(
                    TurnRewardEvidence(
                        turn_index=max(turn_index, 0),
                        event_id=event.event_id,
                        kind="policy",
                        action_type=requested_tools.get(payload["call_id"], "tool_call"),
                        reward=-1.0,
                        reason="Tool action was denied by policy",
                        evidence={"call_id": payload["call_id"], "decision": payload["decision"]},
                    )
                )
        elif event.event_type is EventType.TOOL_FINISHED:
            call_id = payload["call_id"]
            action_type = requested_tools.get(call_id, "tool_call")
            status = payload["status"]
            rewards.append(
                TurnRewardEvidence(
                    turn_index=max(turn_index, 0),
                    event_id=event.event_id,
                    kind="tool_execution",
                    action_type=action_type,
                    reward=0.1 if status == "completed" else -0.1,
                    reason="Tool execution completed" if status == "completed" else "Tool failed",
                    evidence={"call_id": call_id, "tool": action_type, "status": status},
                )
            )
    return rewards


def turn_reward_metrics(turn_rewards: Iterable[TurnRewardEvidence]) -> dict[str, float]:
    rewards = [item.reward for item in turn_rewards]
    positives = sum(1 for value in rewards if value > 0)
    negatives = sum(1 for value in rewards if value < 0)
    return {
        "turn_reward_total": sum(rewards),
        "turn_reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
        "positive_turn_rewards": float(positives),
        "negative_turn_rewards": float(negatives),
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


def workspace_summary(sandbox: Sandbox) -> dict[str, Any]:
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


def rank_reports(reports: Iterable[EvaluationReport]) -> list[EvaluationReport]:
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


def _ask_user_reward(question: str, instance: ScenarioInstance | None) -> float:
    if instance is None:
        return 0.2
    known_facts = instance.hidden_user.known_facts
    lowered = question.lower()
    for key in known_facts:
        if key.lower() in lowered or key.lower().replace("_", " ") in lowered:
            return 0.2
    return 0.0
