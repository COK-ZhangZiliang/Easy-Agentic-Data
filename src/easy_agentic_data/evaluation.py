from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from easy_agentic_data.sandbox import Sandbox
from easy_agentic_data.scenarios import ScenarioInstance, json_payload_contains_string
from easy_agentic_data.traces import (
    EventType,
    TerminationReason,
    Trace,
    TraceRecorder,
    load_trace,
)

_PUBLIC_EVALUATOR_NAMES = frozenset(
    {
        "agent_termination",
        "forbidden_state",
        "hidden_command",
        "hidden_test_patch",
        "policy_integrity",
        "required_state",
        "trace_quality",
    }
)


@dataclass(frozen=True)
class EvaluationEvidence:
    evaluator: str
    passed: bool
    score: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    infrastructure_failure: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationEvidence:
        data = _require_json_object(value, "evaluation evidence")
        return cls(
            evaluator=_require_string(data.get("evaluator"), "evaluation evidence.evaluator"),
            passed=_require_bool(data.get("passed"), "evaluation evidence.passed"),
            score=_require_finite_number(data.get("score"), "evaluation evidence.score"),
            reason=_require_string(data.get("reason"), "evaluation evidence.reason"),
            evidence=_require_json_object(
                data.get("evidence", {}), "evaluation evidence.evidence"
            ),
            infrastructure_failure=_require_bool(
                data.get("infrastructure_failure", False),
                "evaluation evidence.infrastructure_failure",
            ),
        )


@dataclass(frozen=True)
class TurnRewardEvidence:
    turn_index: int
    event_id: str
    kind: str
    action_type: str
    reward: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TurnRewardEvidence:
        data = _require_json_object(value, "turn reward evidence")
        turn_index = _require_integer(
            data.get("turn_index"), "turn reward evidence.turn_index"
        )
        if turn_index < 0:
            raise ValueError("turn reward evidence.turn_index must be non-negative")
        return cls(
            turn_index=turn_index,
            event_id=_require_string(data.get("event_id"), "turn reward evidence.event_id"),
            kind=_require_string(data.get("kind"), "turn reward evidence.kind"),
            action_type=_require_string(
                data.get("action_type"), "turn reward evidence.action_type"
            ),
            reward=_require_finite_number(data.get("reward"), "turn reward evidence.reward"),
            reason=_require_string(data.get("reason"), "turn reward evidence.reason"),
            evidence=_require_json_object(
                data.get("evidence", {}), "turn reward evidence.evidence"
            ),
        )


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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvaluationReport:
        data = _require_json_object(value, "evaluation report")
        results = data.get("results")
        turn_rewards = data.get("turn_rewards", [])
        metrics = data.get("metrics", {})
        if not isinstance(results, list) or not isinstance(turn_rewards, list):
            raise ValueError("Evaluation results and turn rewards must be lists")
        if not isinstance(metrics, dict):
            raise ValueError("evaluation report.metrics must be an object")
        parsed_results = []
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                raise ValueError(f"evaluation report.results[{index}] must be an object")
            parsed_results.append(EvaluationEvidence.from_dict(item))
        parsed_turn_rewards = []
        for index, item in enumerate(turn_rewards):
            if not isinstance(item, dict):
                raise ValueError(f"evaluation report.turn_rewards[{index}] must be an object")
            parsed_turn_rewards.append(TurnRewardEvidence.from_dict(item))
        parsed_metrics: dict[str, float] = {}
        for key, amount in metrics.items():
            if not isinstance(key, str):
                raise ValueError("evaluation report.metrics keys must be strings")
            parsed_metrics[key] = _require_finite_number(
                amount, f"evaluation report.metrics[{key!r}]"
            )
        return cls(
            scenario_instance_id=_require_string(
                data.get("scenario_instance_id"), "evaluation report.scenario_instance_id"
            ),
            results=parsed_results,
            success=_require_bool(data.get("success"), "evaluation report.success"),
            reward=_require_integer(data.get("reward"), "evaluation report.reward"),
            infrastructure_failure=_require_bool(
                data.get("infrastructure_failure"),
                "evaluation report.infrastructure_failure",
            ),
            metrics=parsed_metrics,
            turn_rewards=parsed_turn_rewards,
        )


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_finite_number(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _require_json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    _validate_json_value(value, field_name, set())
    return dict(value)


def _validate_json_value(value: Any, field_name: str, active: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, (dict, list)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{field_name} contains a recursive value")
        active.add(identity)
        try:
            if isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{field_name} contains a non-string object key")
                    _validate_json_value(item, f"{field_name}.{key}", active)
            else:
                for index, item in enumerate(value):
                    _validate_json_value(item, f"{field_name}[{index}]", active)
        finally:
            active.remove(identity)
        return
    raise ValueError(f"{field_name} contains a non-JSON value")


def public_evaluation_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a public, content-bound summary without evaluator-private values."""

    safe_evidence = _require_json_object(evidence, "evaluation evidence.evidence")
    encoded = json.dumps(
        safe_evidence,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    summary: dict[str, Any] = {
        "evidence_sha256": hashlib.sha256(encoded).hexdigest(),
        "field_count": len(safe_evidence),
    }
    exit_code = safe_evidence.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        summary["exit_code"] = exit_code
    for stream in ("stdout", "stderr"):
        value = safe_evidence.get(stream)
        if isinstance(value, str):
            stream_bytes = value.encode("utf-8")
            summary[f"{stream}_sha256"] = hashlib.sha256(stream_bytes).hexdigest()
            summary[f"{stream}_bytes"] = len(stream_bytes)
    return summary


def public_evaluation_result(result: EvaluationEvidence) -> dict[str, Any]:
    """Serialize verifier evidence for public traces and derived analysis records."""

    evaluator = _require_string(result.evaluator, "evaluation evidence.evaluator")
    reason_value = _require_string(result.reason, "evaluation evidence.reason")
    passed = _require_bool(result.passed, "evaluation evidence.passed")
    infrastructure_failure = _require_bool(
        result.infrastructure_failure,
        "evaluation evidence.infrastructure_failure",
    )
    score = _require_finite_number(result.score, "evaluation evidence.score")
    public_evaluator = (
        evaluator
        if evaluator in _PUBLIC_EVALUATOR_NAMES
        else f"custom_evaluator_{hashlib.sha256(evaluator.encode('utf-8')).hexdigest()[:16]}"
    )
    reason_bytes = reason_value.encode("utf-8")
    if infrastructure_failure:
        reason = "Evaluator infrastructure failure"
    elif passed:
        reason = "Evaluator passed"
    else:
        reason = "Evaluator failed"
    return {
        "evaluator": public_evaluator,
        "passed": passed,
        "score": score,
        "reason": reason,
        "reason_sha256": hashlib.sha256(reason_bytes).hexdigest(),
        "evidence": public_evaluation_evidence(result.evidence),
        "infrastructure_failure": infrastructure_failure,
    }


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


class HiddenTestPatchEvaluator:
    name = "hidden_test_patch"

    def evaluate(self, sandbox: Sandbox, instance: ScenarioInstance) -> EvaluationEvidence:
        patch = instance.hidden_evaluator.metadata.get("test_patch", "")
        if not isinstance(patch, str) or not patch.strip():
            return EvaluationEvidence(self.name, True, 1.0, "No hidden test patch configured")
        try:
            sandbox.write(".ead_hidden_test.patch", patch)
            result = sandbox.execute(["git", "apply", ".ead_hidden_test.patch"])
            return EvaluationEvidence(
                self.name,
                result.exit_code == 0,
                1.0 if result.exit_code == 0 else 0.0,
                "Hidden test patch applied"
                if result.exit_code == 0
                else "Hidden test patch failed to apply",
                {
                    "exit_code": result.exit_code,
                    "stdout": _redact_hidden_values(result.stdout, [patch]),
                    "stderr": _redact_hidden_values(result.stderr, [patch]),
                },
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


class TraceRequirementEvaluator:
    name = "trace_quality"

    def __init__(
        self,
        trace: Trace | None,
        *,
        retrieval_requirements: Iterable[str] = (),
        trace_quality_rubric: Iterable[str] = (),
    ) -> None:
        self.trace = trace
        self.retrieval_requirements = [item for item in retrieval_requirements if item]
        self.trace_quality_rubric = [item for item in trace_quality_rubric if item]

    def evaluate(self, sandbox: Sandbox, instance: ScenarioInstance) -> EvaluationEvidence:
        del sandbox, instance
        if self.trace is None:
            return EvaluationEvidence(
                self.name,
                False,
                0.0,
                "Trace-quality evaluator requires a recorded trace",
                infrastructure_failure=True,
            )
        observable = _observable_trace_text(self.trace)
        inspected_tools = _inspection_tools(self.trace)
        final_answer = _final_assistant_content(self.trace)
        missing = [
            requirement
            for requirement in self.retrieval_requirements
            if not _trace_requirement_satisfied(requirement, observable)
        ]
        passed = (
            bool(final_answer.strip())
            and not missing
            and (not self.retrieval_requirements or bool(inspected_tools))
        )
        reason = (
            "Trace retrieval and final answer requirements passed"
            if passed
            else "Trace retrieval or final answer requirements failed"
        )
        return EvaluationEvidence(
            self.name,
            passed,
            1.0 if passed else 0.0,
            reason,
            {
                "missing_retrieval_requirements": missing,
                "inspected_tools": inspected_tools,
                "final_answer_present": bool(final_answer.strip()),
                "rubric_items": self.trace_quality_rubric,
            },
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


def evaluation_result_metrics(report: EvaluationReport) -> dict[str, float]:
    """Summarize verifier outcomes as numeric metrics for batch aggregation."""

    metrics: dict[str, float] = {}
    non_agent_results = []
    for result in report.results:
        name = _metric_safe_name(result.evaluator)
        metrics[f"verifier_{name}_passed"] = 1.0 if result.passed else 0.0
        if result.evaluator != "agent_termination":
            non_agent_results.append(result)
    metrics["verifier_all_non_agent_passed"] = (
        1.0
        if non_agent_results
        and all(result.passed and not result.infrastructure_failure for result in non_agent_results)
        else 0.0
    )
    return metrics


def finalize_evaluation_trace(
    recorder: TraceRecorder,
    report: EvaluationReport,
    *,
    final_state_hash: str,
    termination_reason: TerminationReason | None = None,
) -> None:
    for result in report.results:
        public_result = public_evaluation_result(result)
        recorder.record(
            EventType.VERIFICATION_RESULT,
            {
                "verifier": public_result.pop("evaluator"),
                **public_result,
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
    trace = load_trace(trace_path, tolerate_truncated=False)
    candidates = instance.sensitive_strings()
    return [
        value
        for value in candidates
        if any(json_payload_contains_string(event.payload, value) for event in trace.events)
    ]


def _metric_safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _redact_hidden_values(text: str, hidden_values: Iterable[str]) -> str:
    redacted = text
    for value in hidden_values:
        if value:
            redacted = redacted.replace(value, "[redacted hidden context]")
    return redacted


def _observable_trace_text(trace: Trace) -> str:
    fragments: list[str] = []
    for event in trace.events:
        if event.event_type in {
            EventType.USER_MESSAGE,
            EventType.MODEL_RESPONSE,
            EventType.TOOL_REQUESTED,
            EventType.TOOL_FINISHED,
        }:
            fragments.append(str(event.payload))
    return _normalize_observable_text("\n".join(fragments))


def _inspection_tools(trace: Trace) -> list[str]:
    tools = []
    for event in trace.events:
        if event.event_type is not EventType.TOOL_REQUESTED:
            continue
        name = str(event.payload.get("name") or "")
        if name in {"list_files", "read_file", "search_files", "git_diff", "git_status"}:
            tools.append(name)
    return sorted(set(tools))


def _final_assistant_content(trace: Trace) -> str:
    for event in reversed(trace.events):
        if event.event_type is EventType.MODEL_RESPONSE:
            content = event.payload.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _trace_requirement_satisfied(requirement: str, observable_text: str) -> bool:
    requirement_text = _normalize_observable_text(requirement)
    if requirement_text and requirement_text in observable_text:
        return True
    path_tokens = re.findall(r"(?:[\w.-]+/)+[\w.-]+", requirement)
    if path_tokens:
        return all(_normalize_observable_text(path) in observable_text for path in path_tokens)
    words = [word for word in re.findall(r"[a-zA-Z0-9_./-]+", requirement_text) if len(word) > 2]
    return bool(words) and all(word in observable_text for word in words)


def _normalize_observable_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


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
