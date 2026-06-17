from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    decision: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


class ToolPolicy:
    def __init__(self, allowed_tools: Iterable[str], *, network_enabled: bool = False) -> None:
        self.allowed_tools = set(allowed_tools)
        self.network_enabled = network_enabled

    def evaluate(self, name: str, arguments: dict[str, Any]) -> PolicyDecision:
        if name not in self.allowed_tools:
            return PolicyDecision("deny", f"Tool is not enabled: {name}")
        encoded = repr(arguments)
        if ".." in encoded or "/etc/" in encoded or "/var/run/docker.sock" in encoded:
            return PolicyDecision("deny", "Arguments reference a forbidden host path")
        if name == "run_command" and not self.network_enabled:
            command = " ".join(arguments.get("command", []))
            if any(term in command for term in ("curl ", "wget ", "http://", "https://")):
                return PolicyDecision("deny", "Network access is disabled")
        return PolicyDecision("allow", "Allowed by scenario capability policy")
