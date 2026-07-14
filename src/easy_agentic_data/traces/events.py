from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from easy_agentic_data.models import stable_id, utc_now

TRACE_SCHEMA_VERSION = 2


class EventType(str, Enum):
    SESSION_STARTED = "session_started"
    SYSTEM_MESSAGE = "system_message"
    USER_MESSAGE = "user_message"
    MODEL_RESPONSE = "model_response"
    TOOL_REQUESTED = "tool_requested"
    POLICY_DECISION = "policy_decision"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_MESSAGE = "tool_message"
    WORKSPACE_DIFF = "workspace_diff"
    VERIFICATION_RESULT = "verification_result"
    SESSION_FINISHED = "session_finished"


class TerminationReason(str, Enum):
    SUCCESS = "success"
    AGENT_STOP = "agent_stop"
    USER_STOP = "user_stop"
    POLICY_VIOLATION = "policy_violation"
    TIMEOUT = "timeout"
    TOKEN_BUDGET = "token_budget"
    TOOL_BUDGET = "tool_budget"
    MALFORMED_TOOL_CALLS = "malformed_tool_calls"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


_REQUIRED_PAYLOAD_FIELDS = {
    EventType.SESSION_STARTED: {"scenario_instance_id", "initial_state_hash"},
    EventType.SYSTEM_MESSAGE: {"message_id", "content"},
    EventType.USER_MESSAGE: {"message_id", "content"},
    EventType.MODEL_RESPONSE: {"message_id", "content"},
    EventType.TOOL_REQUESTED: {"call_id", "name", "arguments"},
    EventType.POLICY_DECISION: {"call_id", "decision", "reason"},
    EventType.TOOL_STARTED: {"call_id", "name"},
    EventType.TOOL_FINISHED: {"call_id", "status"},
    EventType.TOOL_MESSAGE: {"message_id", "name", "tool_call_id", "content"},
    EventType.WORKSPACE_DIFF: {"before_state_hash", "after_state_hash"},
    EventType.VERIFICATION_RESULT: {"verifier", "passed", "score"},
    EventType.SESSION_FINISHED: {"termination_reason", "final_state_hash", "success"},
}

_FORBIDDEN_PUBLIC_FIELDS = {
    "evaluation_payload",
    "evaluator_payload",
    "evaluator_state",
    "forbidden_state",
    "hidden_command",
    "hidden_commands",
    "hidden_user",
    "hidden_evaluator",
    "hidden_tests",
    "private_evaluation",
    "private_evaluator",
    "reference_answer",
    "reference_artifacts",
    "reference_patch",
    "reference_repair",
    "reference_repairs",
    "required_state",
    "rubric",
    "test_patch",
    "trace_quality_rubric",
    "unavailable_facts",
}


@dataclass
class TraceEvent:
    session_id: str
    sequence: int
    event_type: EventType
    payload: dict[str, Any]
    timestamp: str = field(default_factory=utc_now)
    schema_version: int = TRACE_SCHEMA_VERSION
    event_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported trace schema version {self.schema_version}; "
                f"expected {TRACE_SCHEMA_VERSION}"
            )
        if self.sequence < 0:
            raise ValueError("Trace event sequence cannot be negative")
        if not isinstance(self.event_type, EventType):
            self.event_type = EventType(self.event_type)
        forbidden = _find_forbidden_public_field(self.payload)
        if forbidden is not None:
            raise ValueError(f"Trace payload contains hidden field: {forbidden}")
        missing = _REQUIRED_PAYLOAD_FIELDS[self.event_type] - self.payload.keys()
        if missing:
            raise ValueError(
                f"Missing payload fields for {self.event_type.value}: {sorted(missing)}"
            )
        if self.event_type is EventType.SYSTEM_MESSAGE:
            _validate_system_message_payload(self.payload)
        if self.event_type is EventType.MODEL_RESPONSE:
            _validate_model_response_payload(self.payload)
        if self.event_type is EventType.TOOL_MESSAGE:
            _validate_tool_message_payload(self.payload)
        if self.event_type is EventType.SESSION_FINISHED:
            TerminationReason(self.payload["termination_reason"])
        expected_event_id = stable_id(
            "event",
            {
                "schema_version": self.schema_version,
                "session_id": self.session_id,
                "sequence": self.sequence,
                "event_type": self.event_type.value,
                "payload": self.payload,
            },
        )
        if self.event_id and self.event_id != expected_event_id:
            raise ValueError("Trace event_id does not match event content")
        self.event_id = expected_event_id

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TraceEvent:
        data = dict(value)
        version = data.get("schema_version")
        if version != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported trace schema version {version}; expected {TRACE_SCHEMA_VERSION}"
            )
        data["event_type"] = EventType(data["event_type"])
        return cls(**data)


def _find_forbidden_public_field(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = _normalized_field_name(key)
            if normalized_key in _FORBIDDEN_PUBLIC_FIELDS:
                return str(key)
            nested = _find_forbidden_public_field(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_forbidden_public_field(item)
            if nested is not None:
                return nested
    return None


def _normalized_field_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _validate_system_message_payload(payload: dict[str, Any]) -> None:
    message_id = payload["message_id"]
    if not isinstance(message_id, str) or not message_id:
        raise ValueError("system_message message_id must be a non-empty string")
    content = payload["content"]
    if not isinstance(content, str) or not content:
        raise ValueError("system_message content must be a non-empty string")


def _validate_model_response_payload(payload: dict[str, Any]) -> None:
    message_id = payload["message_id"]
    if not isinstance(message_id, str) or not message_id:
        raise ValueError("model_response message_id must be a non-empty string")
    content = payload["content"]
    if content is not None and not isinstance(content, str):
        raise ValueError("model_response content must be a string or null")
    tool_calls = payload.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise ValueError("model_response tool_calls must be a list")
    call_ids: set[str] = set()
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            raise ValueError(f"model_response tool_calls[{index}] must be an object")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"model_response tool_calls[{index}].id must be a non-empty string")
        if call_id in call_ids:
            raise ValueError(f"Duplicate assistant tool call id: {call_id}")
        call_ids.add(call_id)
        function = call.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"model_response tool_calls[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"model_response tool_calls[{index}].function.name must be a non-empty string"
            )


def _validate_tool_message_payload(payload: dict[str, Any]) -> None:
    for field_name in ("message_id", "tool_call_id"):
        value = payload[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"tool_message {field_name} must be a non-empty string")
    for field_name in ("name", "content"):
        if not isinstance(payload[field_name], str):
            raise ValueError(f"tool_message {field_name} must be a string")
