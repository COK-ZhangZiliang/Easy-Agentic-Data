from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from easy_agentic_data.models import stable_id


@dataclass
class EnvironmentSpec:
    """Versioned recipe for creating a reproducible task environment."""

    name: str
    version: str
    description: str = ""
    image_digest: str = ""
    source_uri: str = ""
    source_revision: str = ""
    fixture_patch: str = ""
    working_directory: str = "/workspace"
    setup_commands: list[str] = field(default_factory=list)
    capability_packs: list[str] = field(default_factory=list)
    network_policy: str = "disabled"
    resource_limits: dict[str, Any] = field(default_factory=dict)
    health_check: list[str] = field(default_factory=list)
    reset_strategy: str = "recreate"
    evaluator_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    environment_id: str = ""

    def __post_init__(self) -> None:
        _reject_secret_metadata(self.metadata)
        if not self.environment_id:
            content = asdict(self)
            content.pop("environment_id", None)
            self.environment_id = stable_id("env", content)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EnvironmentSpec:
        return cls(**value)


def _reject_secret_metadata(metadata: dict[str, Any]) -> None:
    forbidden = {"api_key", "apikey", "password", "secret", "token", "credential"}
    for key in metadata:
        normalized = key.lower().replace("-", "_")
        if any(term in normalized for term in forbidden):
            raise ValueError(f"Environment metadata cannot contain secret-like field: {key}")
