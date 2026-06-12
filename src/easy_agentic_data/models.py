from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_api_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.name is not None:
            data["name"] = self.name
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass
class LLMResponse:
    message: Message
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    instruction: str
    category: str = "general"
    difficulty: int = 1
    constraints: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_id: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = stable_id(
                "task",
                {
                    "instruction": self.instruction,
                    "category": self.category,
                    "constraints": self.constraints,
                },
            )


@dataclass
class ToolEvent:
    call_id: str
    name: str
    arguments: Dict[str, Any]
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class Verification:
    verifier: str
    passed: bool
    score: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task: Task
    messages: List[Message]
    tool_events: List[ToolEvent] = field(default_factory=list)
    verifications: List[Verification] = field(default_factory=list)
    reward: float = 0.0
    status: str = "completed"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory_id: str = ""
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            self.trajectory_id = stable_id(
                "traj",
                {
                    "task_id": self.task.task_id,
                    "messages": [message.to_api_dict() for message in self.messages],
                    "rollout_index": self.metadata.get("rollout_index"),
                },
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreferencePair:
    task: Task
    chosen: Trajectory
    rejected: Trajectory
    margin: float
    pair_id: str = ""

    def __post_init__(self) -> None:
        if not self.pair_id:
            self.pair_id = stable_id(
                "pref",
                {
                    "task_id": self.task.task_id,
                    "chosen": self.chosen.trajectory_id,
                    "rejected": self.rejected.trajectory_id,
                },
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
