from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 30.0
    max_output_bytes: int = 100_000
    max_workspace_bytes: int = 50_000_000
    memory: str = "1g"
    cpus: float = 1.0
    pids: int = 128


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    truncated: bool = False


class Sandbox(Protocol):
    def create(self) -> None: ...
    def execute(
        self, command: list[str], *, timeout_seconds: float | None = None
    ) -> CommandResult: ...
    def read(self, path: str) -> str: ...
    def write(self, path: str, content: str) -> None: ...
    def list_files(self, path: str = ".") -> list[str]: ...
    def diff(self) -> str: ...
    def state_hash(self) -> str: ...
    def snapshot(self) -> str: ...
    def restore(self, snapshot_id: str) -> None: ...
    def destroy(self) -> None: ...
