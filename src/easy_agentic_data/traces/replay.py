from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from easy_agentic_data.traces.events import EventType
from easy_agentic_data.traces.recorder import Trace, load_trace


@dataclass
class ReplayState:
    session_id: str = ""
    scenario_instance_id: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    policy_decisions: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    workspace_state_hash: str = ""
    termination_reason: str | None = None
    success: bool | None = None


@dataclass(frozen=True)
class ReplayResult:
    trace_id: str
    event_count: int
    truncated: bool
    terminal_state_hash: str
    state: ReplayState

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["state"] = asdict(self.state)
        return value


def replay_trace(trace_or_path: Trace | str | Path) -> ReplayResult:
    """Reconstruct observable session state without invoking a model or tool."""

    trace = trace_or_path if isinstance(trace_or_path, Trace) else load_trace(trace_or_path)
    state = ReplayState(session_id=trace.session_id)

    for event in trace.events:
        payload = event.payload
        if event.event_type is EventType.SESSION_STARTED:
            state.scenario_instance_id = payload["scenario_instance_id"]
            state.workspace_state_hash = payload["initial_state_hash"]
        elif event.event_type is EventType.USER_MESSAGE:
            state.messages.append(
                {
                    "message_id": payload["message_id"],
                    "role": "user",
                    "content": payload["content"],
                }
            )
        elif event.event_type is EventType.MODEL_RESPONSE:
            message = {
                "message_id": payload["message_id"],
                "role": "assistant",
                "content": payload["content"],
                "tool_calls": payload.get("tool_calls", []),
            }
            if payload.get("reasoning_content") is not None:
                message["reasoning_content"] = payload["reasoning_content"]
            state.messages.append(message)
        elif event.event_type is EventType.TOOL_REQUESTED:
            state.tool_calls[payload["call_id"]] = {
                "name": payload["name"],
                "arguments": payload["arguments"],
                "status": "requested",
            }
        elif event.event_type is EventType.POLICY_DECISION:
            state.policy_decisions.append(dict(payload))
            call = state.tool_calls.setdefault(payload["call_id"], {})
            call["policy_decision"] = payload["decision"]
            call["policy_reason"] = payload["reason"]
        elif event.event_type is EventType.TOOL_STARTED:
            call = state.tool_calls.setdefault(payload["call_id"], {"name": payload["name"]})
            call["status"] = "started"
        elif event.event_type is EventType.TOOL_FINISHED:
            call = state.tool_calls.setdefault(payload["call_id"], {})
            call.update(
                {
                    "status": payload["status"],
                    "exit_code": payload.get("exit_code"),
                    "stdout_artifact": payload.get("stdout_artifact"),
                    "stderr_artifact": payload.get("stderr_artifact"),
                    "error": payload.get("error"),
                }
            )
        elif event.event_type is EventType.WORKSPACE_DIFF:
            if (
                state.workspace_state_hash
                and payload["before_state_hash"] != state.workspace_state_hash
            ):
                raise ValueError("workspace_diff before_state_hash does not match replay state")
            state.workspace_state_hash = payload["after_state_hash"]
        elif event.event_type is EventType.VERIFICATION_RESULT:
            state.verifications.append(dict(payload))
        elif event.event_type is EventType.SESSION_FINISHED:
            final_hash = payload["final_state_hash"]
            if state.workspace_state_hash and final_hash != state.workspace_state_hash:
                raise ValueError("session_finished final_state_hash does not match replay state")
            state.workspace_state_hash = final_hash
            state.termination_reason = payload["termination_reason"]
            state.success = bool(payload["success"])

    return ReplayResult(
        trace_id=trace.trace_id,
        event_count=len(trace.events),
        truncated=trace.truncated,
        terminal_state_hash=state.workspace_state_hash,
        state=state,
    )
