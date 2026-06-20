from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from easy_agentic_data.environments import is_immutable_image_reference

from .base import CommandResult, SandboxLimits


class DockerSandbox:
    """Rootless Docker backend using a named volume and deny-by-default networking."""

    def __init__(
        self,
        *,
        image_digest: str,
        source_directory: str | Path,
        limits: SandboxLimits | None = None,
        network_enabled: bool = False,
    ) -> None:
        if not is_immutable_image_reference(image_digest):
            raise ValueError("Docker images must be content-addressed by digest")
        self.image_digest = image_digest
        self.source_directory = Path(source_directory).resolve()
        self.limits = limits or SandboxLimits()
        self.network_enabled = network_enabled
        self.container_name = ""
        self.volume_name = ""

    def create(self) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("Docker is not installed")
        suffix = hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:12]
        self.container_name = f"ead-{suffix}"
        self.volume_name = f"ead-workspace-{suffix}"
        self._run_host(["docker", "volume", "create", self.volume_name])
        self._run_host(
            [
                "docker",
                "create",
                "--name",
                self.container_name,
                "--user",
                "65532:65532",
                "--read-only",
                "--network",
                "bridge" if self.network_enabled else "none",
                "--cpus",
                str(self.limits.cpus),
                "--memory",
                self.limits.memory,
                "--pids-limit",
                str(self.limits.pids),
                "--mount",
                f"type=volume,src={self.volume_name},dst=/workspace",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=64m",
                "-w",
                "/workspace",
                "--entrypoint",
                "sleep",
                self.image_digest,
                "infinity",
            ]
        )
        self._run_host(["docker", "start", self.container_name])
        self._run_host(
            ["docker", "cp", f"{self.source_directory}/.", f"{self.container_name}:/workspace"]
        )
        self._run_host(
            [
                "docker",
                "exec",
                "--user",
                "0:0",
                self.container_name,
                "chown",
                "-R",
                "65532:65532",
                "/workspace",
            ]
        )

    def execute(self, command: list[str], *, timeout_seconds: float | None = None) -> CommandResult:
        started = time.perf_counter()
        completed = self._run_host(
            ["docker", "exec", self.container_name, *command],
            check=False,
            timeout=timeout_seconds or self.limits.timeout_seconds,
        )
        stdout, out_cut = _bounded(completed.stdout, self.limits.max_output_bytes)
        stderr, err_cut = _bounded(completed.stderr, self.limits.max_output_bytes)
        self._check_workspace_size()
        return CommandResult(
            completed.returncode,
            stdout,
            stderr,
            (time.perf_counter() - started) * 1000,
            out_cut or err_cut,
        )

    def read(self, path: str) -> str:
        _safe_relative(path)
        return self.execute(["cat", f"/workspace/{path}"]).stdout

    def write(self, path: str, content: str) -> None:
        _safe_relative(path)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            self._run_host(
                ["docker", "cp", handle.name, f"{self.container_name}:/workspace/{path}"]
            )
        self._run_host(
            [
                "docker",
                "exec",
                "--user",
                "0:0",
                self.container_name,
                "chown",
                "65532:65532",
                f"/workspace/{path}",
            ]
        )
        self._run_host(
            [
                "docker",
                "exec",
                "--user",
                "0:0",
                self.container_name,
                "chmod",
                "0644",
                f"/workspace/{path}",
            ]
        )
        self._check_workspace_size()

    def list_files(self, path: str = ".") -> list[str]:
        _safe_relative(path)
        result = self.execute(["find", path, "-type", "f", "-print"])
        return sorted(line.removeprefix("./") for line in result.stdout.splitlines())

    def diff(self) -> str:
        return self.execute(["git", "diff", "--no-ext-diff"]).stdout

    def state_hash(self) -> str:
        result = self.execute(
            [
                "sh",
                "-lc",
                ("find . -path './.git' -prune -o -type f -print0 | sort -z | xargs -0 sha256sum"),
            ]
        )
        return hashlib.sha256(result.stdout.encode()).hexdigest()

    def snapshot(self) -> str:
        return self.state_hash()

    def restore(self, snapshot_id: str) -> None:
        del snapshot_id
        self.destroy()
        self.create()

    def destroy(self) -> None:
        if self.container_name:
            self._run_host(["docker", "rm", "-f", self.container_name], check=False)
        if self.volume_name:
            self._run_host(["docker", "volume", "rm", self.volume_name], check=False)

    def _check_workspace_size(self) -> None:
        completed = self._run_host(
            ["docker", "exec", self.container_name, "du", "-sk", "/workspace"],
            check=False,
        )
        if completed.returncode == 0:
            size_bytes = int(completed.stdout.split()[0]) * 1024
            if size_bytes > self.limits.max_workspace_bytes:
                raise RuntimeError("Sandbox workspace size limit exceeded")

    @staticmethod
    def _run_host(
        command: list[str], *, check: bool = True, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, text=True, capture_output=True, check=check, timeout=timeout)


def _safe_relative(path: str) -> None:
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise PermissionError(f"Path escapes sandbox workspace: {path}")


def _bounded(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="replace"), True
