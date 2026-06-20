from __future__ import annotations

from typing import Any

from easy_agentic_data.evaluation import EvaluationReport
from easy_agentic_data.traces import EventType, Trace, replay_trace


def trace_to_sft(trace: Trace, report: EvaluationReport) -> dict[str, Any]:
    if not report.success:
        raise ValueError("Only successful traces can be exported for SFT")
    replay = replay_trace(trace)
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
    margin = chosen_report.reward - rejected_report.reward
    if margin <= 0:
        raise ValueError("Preference pairs require a positive deterministic reward margin")
    return {
        "id": f"preference_{chosen.trace_id}_{rejected.trace_id}",
        "scenario_instance_id": chosen_report.scenario_instance_id,
        "chosen_trace_id": chosen.trace_id,
        "rejected_trace_id": rejected.trace_id,
        "chosen": replay_trace(chosen).state.messages,
        "rejected": replay_trace(rejected).state.messages,
        "margin": margin,
    }


def trace_to_rl_episode(trace: Trace, report: EvaluationReport) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    turn_rewards_by_event = {
        reward.event_id: reward for reward in getattr(report, "turn_rewards", [])
    }
    final_assistant_step = -1
    for event in trace.events:
        step = _event_to_episode_step(event)
        if step is None:
            continue
        turn_reward = turn_rewards_by_event.get(event.event_id)
        reward_components = {
            "turn": turn_reward.reward if turn_reward is not None else 0.0,
            "outcome": 0.0,
        }
        step["reward_components"] = reward_components
        step["reward"] = sum(reward_components.values())
        if turn_reward is not None:
            step["reward_reason"] = turn_reward.reason
            step["reward_kind"] = turn_reward.kind
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
            "turn": [reward.__dict__ for reward in getattr(report, "turn_rewards", [])],
        },
        "termination_reason": replay_trace(trace).state.termination_reason,
        "success": report.success,
    }


def analysis_record(trace: Trace, report: EvaluationReport) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "scenario_instance_id": report.scenario_instance_id,
        "success": report.success,
        "infrastructure_failure": report.infrastructure_failure,
        "results": [result.__dict__ for result in report.results],
        "metrics": report.metrics,
    }


def _event_to_episode_step(event) -> dict[str, Any] | None:
    payload = event.payload
    if event.event_type is EventType.USER_MESSAGE:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
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
            "role": "tool",
            "step_type": "observation",
            "call_id": payload["call_id"],
            "status": payload["status"],
            "output": payload.get("output"),
            "error": payload.get("error"),
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
