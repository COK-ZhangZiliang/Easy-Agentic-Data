from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from easy_agentic_data.models import stable_id, utc_now

TRACE_SCHEMA_VERSION = 1


class EventType(str, Enum):
    SESSION_STARTED = "session_started"
    USER_MESSAGE = "user_message"
    MODEL_RESPONSE = "model_response"
    TOOL_REQUESTED = "tool_requested"
    POLICY_DECISION = "policy_decision"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
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
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


_REQUIRED_PAYLOAD_FIELDS = {
    EventType.SESSION_STARTED: {"scenario_instance_id", "initial_state_hash"},
    EventType.USER_MESSAGE: {"message_id", "content"},
    EventType.MODEL_RESPONSE: {"message_id", "content"},
    EventType.TOOL_REQUESTED: {"call_id", "name", "arguments"},
    EventType.POLICY_DECISION: {"call_id", "decision", "reason"},
    EventType.TOOL_STARTED: {"call_id", "name"},
    EventType.TOOL_FINISHED: {"call_id", "status"},
    EventType.WORKSPACE_DIFF: {"before_state_hash", "after_state_hash"},
    EventType.VERIFICATION_RESULT: {"verifier", "passed", "score"},
    EventType.SESSION_FINISHED: {"termination_reason", "final_state_hash", "success"},
}

_FORBIDDEN_PUBLIC_FIELDS = {
    "hidden_user",
    "hidden_evaluator",
    "hidden_tests",
    "reference_answer",
    "reference_patch",
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
        if self.event_type is EventType.SESSION_FINISHED:
            TerminationReason(self.payload["termination_reason"])
        if not self.event_id:
            self.event_id = stable_id(
                "event",
                {
                    "schema_version": self.schema_version,
                    "session_id": self.session_id,
                    "sequence": self.sequence,
                    "event_type": self.event_type.value,
                    "payload": self.payload,
                },
            )

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
            if key in _FORBIDDEN_PUBLIC_FIELDS:
                return key
            nested = _find_forbidden_public_field(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_forbidden_public_field(item)
            if nested is not None:
                return nested
    return None
