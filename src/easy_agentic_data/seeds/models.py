from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from easy_agentic_data.models import stable_id


@dataclass
class PublicTaskContext:
    """Information that may be shown to the agent and written into public traces."""

    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PublicTaskContext":
        return cls(**value)


@dataclass
class HiddenUserContext:
    """Private state available only to the simulated user."""

    goal: str = ""
    persona: str = ""
    known_facts: Dict[str, Any] = field(default_factory=dict)
    unavailable_facts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    patience_turns: int = 5
    interaction_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "HiddenUserContext":
        return cls(**value)


@dataclass
class QuerySeed:
    """Reusable query blueprint with a public prompt and isolated user state."""

    public: PublicTaskContext
    hidden_user: HiddenUserContext = field(default_factory=HiddenUserContext)
    category: str = "general"
    difficulty: int = 1
    provenance: str = ""
    license: str = ""
    split: str = "train"
    parent_seed_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed_id: str = ""

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "evaluation"}:
            raise ValueError(f"Unsupported query-seed split: {self.split}")
        self.difficulty = max(1, min(5, int(self.difficulty)))
        if not self.seed_id:
            content = asdict(self)
            content.pop("seed_id", None)
            self.seed_id = stable_id("seed", content)

    def to_dict(self, *, include_hidden: bool = True) -> Dict[str, Any]:
        value = asdict(self)
        if not include_hidden:
            value.pop("hidden_user", None)
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "QuerySeed":
        data = dict(value)
        data["public"] = PublicTaskContext.from_dict(data["public"])
        data["hidden_user"] = HiddenUserContext.from_dict(data.get("hidden_user", {}))
        return cls(**data)
