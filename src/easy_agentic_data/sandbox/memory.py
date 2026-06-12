from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Dict, List

from .base import CommandResult, SandboxLimits


class MemorySandbox:
    """Deterministic test sandbox. Production rollouts must use an isolated process backend."""

    def __init__(
        self,
        files: Dict[str, str] | None = None,
        commands: Dict[str, Callable[["MemorySandbox"], CommandResult] | CommandResult] | None = None,
        limits: SandboxLimits | None = None,
    ) -> None:
        self.initial_files = dict(files or {})
        self.files = dict(self.initial_files)
        self.commands = dict(commands or {})
        self.limits = limits or SandboxLimits()
        self.snapshots: Dict[str, Dict[str, str]] = {}
        self.created = False

    def create(self) -> None:
        self.files = dict(self.initial_files)
        self.created = True

    def execute(self, command: List[str], *, timeout_seconds: float | None = None) -> CommandResult:
        del timeout_seconds
        self._require_created()
        key = " ".join(command)
        result = self.commands.get(key)
        if result is None:
            return CommandResult(127, "", f"Unsupported test command: {key}", 0.0)
        return result(self) if callable(result) else result

    def read(self, path: str) -> str:
        self._require_created()
        normalized = _safe_path(path)
        if normalized not in self.files:
            raise FileNotFoundError(normalized)
        return self.files[normalized]

    def write(self, path: str, content: str) -> None:
        self._require_created()
        normalized = _safe_path(path)
        projected = (
            sum(len(value.encode("utf-8")) for value in self.files.values())
            - len(self.files.get(normalized, "").encode("utf-8"))
            + len(content.encode("utf-8"))
        )
        if projected > self.limits.max_workspace_bytes:
            raise ValueError("Workspace size limit exceeded")
        self.files[normalized] = content

    def list_files(self, path: str = ".") -> List[str]:
        self._require_created()
        prefix = "" if path in {"", "."} else f"{_safe_path(path).rstrip('/')}/"
        return sorted(name for name in self.files if name.startswith(prefix))

    def diff(self) -> str:
        changed = []
        for path in sorted(set(self.initial_files) | set(self.files)):
            before = self.initial_files.get(path)
            after = self.files.get(path)
            if before != after:
                changed.append(f"--- {path}\n+++ {path}\n-{before or ''}\n+{after or ''}")
        return "\n".join(changed)

    def state_hash(self) -> str:
        payload = json.dumps(self.files, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def snapshot(self) -> str:
        snapshot_id = f"snapshot_{self.state_hash()}"
        self.snapshots[snapshot_id] = dict(self.files)
        return snapshot_id

    def restore(self, snapshot_id: str) -> None:
        self.files = dict(self.snapshots[snapshot_id])

    def destroy(self) -> None:
        self.created = False

    def _require_created(self) -> None:
        if not self.created:
            raise RuntimeError("Sandbox has not been created")


def _safe_path(path: str) -> str:
    normalized = path.replace("\\", "/").lstrip("./")
    if not normalized or normalized == ".":
        return "."
    if normalized.startswith("/") or ".." in normalized.split("/"):
        raise PermissionError(f"Path escapes sandbox workspace: {path}")
    return normalized
