from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from easy_agentic_data.evaluation import EvaluationReport, public_evaluation_result
from easy_agentic_data.traces import (
    EventType,
    ReplayResult,
    TerminationReason,
    Trace,
    replay_trace,
)

_HARD_VERIFIERS = frozenset(
    {
        "hidden_command",
        "required_state",
        "forbidden_state",
    }
)
_PUBLIC_METRICS = frozenset(
    {
        "turns",
        "tool_calls",
        "tokens",
        "latency_ms",
        "elapsed_ms",
        "patch_bytes",
        "tool_errors",
        "user_turns",
        "turn_reward_total",
        "turn_reward_mean",
        "positive_turn_rewards",
        "negative_turn_rewards",
        "clarifications",
        "corrections",
        "refusals",
        "confirmations",
        "contradictions",
        "early_stops",
        "diversity",
        "goal_components_total",
        "goal_components_satisfied",
        "goal_alignment",
        "disclosure_violations",
        "unavailable_fact_requests",
        "unavailable_fact_leaks",
        "critical_simulator_errors",
        "simulator_error_rate",
        "verifier_all_non_agent_passed",
    }
)
_PUBLIC_VERIFIER_METRICS = frozenset(
    f"verifier_{name}_passed"
    for name in {
        *_HARD_VERIFIERS,
        "hidden_test_patch",
        "agent_termination",
        "policy_integrity",
    }
)
_PUBLIC_TURN_REWARD_KINDS = frozenset(
    {"information_gathering", "policy", "tool_execution"}
)


def trace_to_sft(trace: Trace, report: EvaluationReport) -> dict[str, Any]:
    if not report.success:
        raise ValueError("Only successful traces can be exported for SFT")
    if _report_has_infrastructure_failure(report):
        raise ValueError("SFT cannot include an infrastructure failure")
    hard_results = [result for result in report.results if result.evaluator in _HARD_VERIFIERS]
    if (
        not hard_results
        or not all(result.passed and result.score > 0 for result in hard_results)
        or not all(result.passed and not result.infrastructure_failure for result in report.results)
    ):
        raise ValueError(
            "SFT requires at least one non-agent hard verifier and all hard verifiers to pass"
        )
    replay = _require_complete_report_trace(trace, report)
    if replay.state.success is not True:
        raise ValueError("SFT requires a trace finalized as successful")
    return {
        "id": f"sft_{trace.trace_id}",
        "trace_id": trace.trace_id,
        "scenario_instance_id": report.scenario_instance_id,
        "messages": replay.state.messages,
        "reward": report.reward,
    }


def traces_to_preference(
    chosen: Trace,
    chosen_report: EvaluationReport,
    rejected: Trace,
    rejected_report: EvaluationReport,
) -> dict[str, Any]:
    if chosen.trace_id == rejected.trace_id:
        raise ValueError("Preference pairs require two distinct traces")
    if _report_has_infrastructure_failure(chosen_report) or _report_has_infrastructure_failure(
        rejected_report
    ):
        raise ValueError("Preference pairs cannot include infrastructure failures")
    chosen_scenario_id, chosen_instance_id = _trace_lineage(chosen)
    rejected_scenario_id, rejected_instance_id = _trace_lineage(rejected)
    if not chosen_scenario_id or chosen_scenario_id != rejected_scenario_id:
        raise ValueError("Preference pairs must come from the same scenario")
    if chosen_instance_id != chosen_report.scenario_instance_id:
        raise ValueError("Chosen trace and evaluation report instance lineage do not match")
    if rejected_instance_id != rejected_report.scenario_instance_id:
        raise ValueError("Rejected trace and evaluation report instance lineage do not match")
    _require_complete_report_trace(chosen, chosen_report)
    _require_complete_report_trace(rejected, rejected_report)
    margin = chosen_report.reward - rejected_report.reward
    if margin <= 0:
        raise ValueError("Preference pairs require a positive deterministic reward margin")
    return {
        "id": f"preference_{chosen.trace_id}_{rejected.trace_id}",
        "scenario_id": chosen_scenario_id,
        "scenario_instance_id": chosen_report.scenario_instance_id,
        "chosen_scenario_instance_id": chosen_report.scenario_instance_id,
        "rejected_scenario_instance_id": rejected_report.scenario_instance_id,
        "chosen_trace_id": chosen.trace_id,
        "rejected_trace_id": rejected.trace_id,
        "chosen": replay_trace(chosen).state.messages,
        "rejected": replay_trace(rejected).state.messages,
        "margin": margin,
    }


def trace_to_rl_episode(trace: Trace, report: EvaluationReport) -> dict[str, Any]:
    if _report_has_infrastructure_failure(report):
        raise ValueError("RL episodes cannot include infrastructure failures")
    replay = _require_complete_report_trace(trace, report)
    if replay.state.termination_reason == TerminationReason.INFRASTRUCTURE_FAILURE.value:
        raise ValueError("RL episodes cannot include infrastructure failures")
    public_turn_rewards = _public_turn_rewards(trace, report)
    steps: list[dict[str, Any]] = []
    turn_rewards_by_event = {reward["event_id"]: reward for reward in public_turn_rewards}
    final_assistant_step = -1
    for event in trace.events:
        step = _event_to_episode_step(event)
        if step is None:
            continue
        turn_reward = turn_rewards_by_event.get(event.event_id)
        reward_components = {
            "turn": turn_reward["reward"] if turn_reward is not None else 0.0,
            "outcome": 0.0,
        }
        step["reward_components"] = reward_components
        step["reward"] = sum(reward_components.values())
        if turn_reward is not None:
            step["reward_reason"] = turn_reward["reason"]
            step["reward_reason_sha256"] = turn_reward["reason_sha256"]
            step["reward_kind"] = turn_reward["kind"]
        if event.event_type is EventType.MODEL_RESPONSE:
            final_assistant_step = len(steps)
        steps.append(step)
    if steps:
        target = final_assistant_step if final_assistant_step >= 0 else len(steps) - 1
        steps[target]["reward_components"]["outcome"] = float(report.reward)
        steps[target]["reward"] = sum(steps[target]["reward_components"].values())
    return {
        "id": f"episode_{trace.trace_id}",
        "trace_id": trace.trace_id,
        "scenario_instance_id": report.scenario_instance_id,
        "schema": "easy_agentic_data.rl_episode.v1",
        "steps": steps,
        "rewards": {
            "outcome": float(report.reward),
            "turn": public_turn_rewards,
        },
        "termination_reason": replay.state.termination_reason,
        "success": report.success,
    }


def analysis_record(trace: Trace, report: EvaluationReport) -> dict[str, Any]:
    _require_complete_report_trace(trace, report)
    metrics, metrics_sha256, omitted_metric_count = _public_metrics(report.metrics)
    return {
        "trace_id": trace.trace_id,
        "scenario_instance_id": report.scenario_instance_id,
        "success": report.success,
        "infrastructure_failure": _report_has_infrastructure_failure(report),
        "results": [public_evaluation_result(result) for result in report.results],
        "metrics": metrics,
        "metrics_sha256": metrics_sha256,
        "omitted_metric_count": omitted_metric_count,
    }


def _report_has_infrastructure_failure(report: EvaluationReport) -> bool:
    if not isinstance(report.infrastructure_failure, bool):
        raise ValueError("Evaluation report infrastructure_failure must be a boolean")
    result_failures = []
    for result in report.results:
        if not isinstance(result.infrastructure_failure, bool):
            raise ValueError("Evaluation result infrastructure_failure must be a boolean")
        result_failures.append(result.infrastructure_failure)
    return report.infrastructure_failure or any(result_failures)


def _trace_lineage(trace: Trace) -> tuple[str, str]:
    for event in trace.events:
        if event.event_type is EventType.SESSION_STARTED:
            return (
                str(event.payload.get("scenario_id") or ""),
                str(event.payload.get("scenario_instance_id") or ""),
            )
    return "", ""


def _require_report_trace_instance(trace: Trace, report: EvaluationReport) -> None:
    _, instance_id = _trace_lineage(trace)
    if not instance_id or instance_id != report.scenario_instance_id:
        raise ValueError("Trace and evaluation report instance lineage do not match")


def _require_complete_report_trace(
    trace: Trace,
    report: EvaluationReport,
) -> ReplayResult:
    if not isinstance(report.success, bool):
        raise ValueError("Evaluation report success must be a boolean")
    if not isinstance(report.reward, int) or isinstance(report.reward, bool):
        raise ValueError("Evaluation report reward must be an integer")
    if trace.truncated:
        raise ValueError("Export requires a complete, non-truncated trace")
    if not trace.events or trace.events[-1].event_type is not EventType.SESSION_FINISHED:
        raise ValueError("Export requires a finalized trace")
    _require_report_trace_instance(trace, report)
    terminal = trace.events[-1].payload
    if not isinstance(terminal.get("success"), bool):
        raise ValueError("Trace terminal success must be a boolean")
    if terminal["success"] is not report.success:
        raise ValueError("Trace and evaluation report success values do not match")
    expected_verifications = []
    for result in report.results:
        public_result = public_evaluation_result(result)
        expected_verifications.append(
            {"verifier": public_result.pop("evaluator"), **public_result}
        )
    actual_verifications = [
        event.payload
        for event in trace.events
        if event.event_type is EventType.VERIFICATION_RESULT
    ]
    if actual_verifications != expected_verifications:
        raise ValueError("Trace verification evidence does not match the evaluation report")
    return replay_trace(trace)


def _public_metrics(metrics: dict[str, float]) -> tuple[dict[str, float], str, int]:
    if not isinstance(metrics, dict):
        raise ValueError("Evaluation metrics must be an object")
    projected: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise ValueError("Evaluation metric names must be strings")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("Evaluation metric values must be finite numbers")
        try:
            amount = float(value)
        except OverflowError as exc:
            raise ValueError("Evaluation metric values must be finite numbers") from exc
        if not math.isfinite(amount):
            raise ValueError("Evaluation metric values must be finite numbers")
        if key in _PUBLIC_METRICS or key in _PUBLIC_VERIFIER_METRICS:
            projected[key] = amount
    return (
        dict(sorted(projected.items())),
        _sha256_json(metrics),
        len(metrics) - len(projected),
    )


def _public_turn_rewards(trace: Trace, report: EvaluationReport) -> list[dict[str, Any]]:
    event_ids = {event.event_id for event in trace.events}
    seen: set[str] = set()
    projected: list[dict[str, Any]] = []
    for reward in report.turn_rewards:
        if reward.event_id not in event_ids:
            raise ValueError("Turn reward references an event outside the trace")
        if reward.event_id in seen:
            raise ValueError("Turn reward event IDs must be unique")
        seen.add(reward.event_id)
        if (
            not isinstance(reward.turn_index, int)
            or isinstance(reward.turn_index, bool)
            or reward.turn_index < 0
        ):
            raise ValueError("Turn reward indexes must be non-negative integers")
        if not isinstance(reward.reward, (int, float)) or isinstance(reward.reward, bool):
            raise ValueError("Turn reward values must be finite numbers")
        try:
            amount = float(reward.reward)
        except OverflowError as exc:
            raise ValueError("Turn reward values must be finite numbers") from exc
        if not math.isfinite(amount):
            raise ValueError("Turn reward values must be finite numbers")
        if (
            not isinstance(reward.kind, str)
            or not isinstance(reward.reason, str)
            or not isinstance(reward.action_type, str)
        ):
            raise ValueError("Turn reward labels must be strings")
        if not isinstance(reward.evidence, dict):
            raise ValueError("Turn reward evidence must be an object")
        kind = reward.kind if reward.kind in _PUBLIC_TURN_REWARD_KINDS else "other"
        reason = (
            "Positive turn reward"
            if amount > 0
            else "Negative turn reward"
            if amount < 0
            else "Neutral turn reward"
        )
        projected.append(
            {
                "turn_index": reward.turn_index,
                "event_id": reward.event_id,
                "kind": kind,
                "action_type_sha256": _sha256_text(reward.action_type),
                "reward": amount,
                "reason": reason,
                "reason_sha256": _sha256_text(reward.reason),
                "evidence_sha256": _sha256_json(reward.evidence),
                "evidence_field_count": len(reward.evidence),
            }
        )
    return projected


def _sha256_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Private evaluation data must be finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _event_to_episode_step(event) -> dict[str, Any] | None:
    payload = event.payload
    if event.event_type is EventType.SYSTEM_MESSAGE:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "message_id": payload["message_id"],
            "role": "system",
            "step_type": "instruction",
            "content": payload["content"],
            "loss_mask": 0,
            "action_mask": 0,
        }
    if event.event_type is EventType.USER_MESSAGE:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "message_id": payload["message_id"],
            "role": "user",
            "step_type": "observation",
            "content": payload["content"],
            "loss_mask": 0,
            "action_mask": 0,
        }
    if event.event_type is EventType.MODEL_RESPONSE:
        step = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "message_id": payload["message_id"],
            "role": "assistant",
            "step_type": "action",
            "content": payload.get("content"),
            "tool_calls": payload.get("tool_calls", []),
            "action_type": _assistant_action_type(payload),
            "token_start": None,
            "token_end": None,
            "loss_mask": 1,
            "action_mask": 1,
        }
        if payload.get("reasoning_content") is not None:
            step["reasoning_content"] = payload["reasoning_content"]
        return step
    if event.event_type is EventType.TOOL_REQUESTED:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "role": "assistant",
            "step_type": "action_metadata",
            "action_type": payload["name"],
            "call_id": payload["call_id"],
            "arguments": payload["arguments"],
            "loss_mask": 0,
            "action_mask": 0,
        }
    if event.event_type is EventType.TOOL_FINISHED:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "role": "environment",
            "step_type": "execution_result",
            "call_id": payload["call_id"],
            "status": payload["status"],
            "output": payload.get("output"),
            "error": payload.get("error"),
            "loss_mask": 0,
            "action_mask": 0,
        }
    if event.event_type is EventType.TOOL_MESSAGE:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "message_id": payload["message_id"],
            "role": "tool",
            "step_type": "observation",
            "content": payload["content"],
            "name": payload["name"],
            "tool_call_id": payload["tool_call_id"],
            "loss_mask": 0,
            "action_mask": 0,
        }
    return None


def _assistant_action_type(payload: dict[str, Any]) -> str:
    tool_calls = payload.get("tool_calls") or []
    if tool_calls:
        names = []
        for call in tool_calls:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name") or call.get("name", "")
            names.append(str(name))
        if names and all(name == "ask_user" for name in names):
            return "ask_user"
        return "tool_call"
    content = str(payload.get("content") or "").lower()
    if "confirm" in content or "proceed" in content:
        return "confirm"
    if content.strip():
        return "answer"
    return "stop"
