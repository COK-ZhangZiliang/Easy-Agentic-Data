from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from easy_agentic_data.models import stable_id

SUPPORTED_SEED_SPLITS = {
    "train",
    "validation",
    "evaluation",
    "dev",
    "eval_holdout",
    "quarantined",
}
NON_TRAIN_SEED_SPLITS = {"evaluation", "eval_holdout", "quarantined"}


@dataclass
class PublicTaskContext:
    """Information that may be shown to the agent and written into public traces."""

    query: str
    context: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PublicTaskContext:
        return cls(**value)


@dataclass
class HiddenUserContext:
    """Private state available only to the simulated user."""

    goal: str = ""
    persona: str = ""
    goal_components: dict[str, str] = field(default_factory=dict)
    known_facts: dict[str, Any] = field(default_factory=dict)
    unavailable_facts: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    patience_turns: int = 5
    interaction_policy: dict[str, Any] = field(default_factory=dict)
    disclosure_policy: dict[str, Any] = field(default_factory=dict)
    stop_conditions: list[str] = field(default_factory=list)
    business_knowledge_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HiddenUserContext:
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
    task_family: str = "general"
    source_method: str = "unspecified"
    train_eligible: bool = True
    contamination_tags: list[str] = field(default_factory=list)
    verifier_types: list[str] = field(default_factory=list)
    coverage_tags: list[str] = field(default_factory=list)
    parent_seed_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    seed_id: str = ""

    def __post_init__(self) -> None:
        if self.split not in SUPPORTED_SEED_SPLITS:
            raise ValueError(f"Unsupported query-seed split: {self.split}")
        self.difficulty = max(1, min(5, int(self.difficulty)))
        self.task_family = _normalize_label(self.task_family, default="general")
        self.source_method = _normalize_label(self.source_method, default="unspecified")
        self.contamination_tags = _normalize_label_list(self.contamination_tags)
        self.verifier_types = _normalize_label_list(self.verifier_types)
        self.coverage_tags = _normalize_label_list(self.coverage_tags)
        self.train_eligible = bool(self.train_eligible)
        if self.split in NON_TRAIN_SEED_SPLITS:
            self.train_eligible = False
        if not self.seed_id:
            content = asdict(self)
            content.pop("seed_id", None)
            self.seed_id = stable_id("seed", content)

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_hidden:
            value.pop("hidden_user", None)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QuerySeed:
        data = dict(value)
        data["public"] = PublicTaskContext.from_dict(data["public"])
        data["hidden_user"] = HiddenUserContext.from_dict(data.get("hidden_user", {}))
        return cls(**data)


def _normalize_label(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or default


def _normalize_label_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        items = [values]
    else:
        items = list(values)
    normalized = {_normalize_label(item, default="") for item in items}
    normalized.discard("")
    return sorted(normalized)
