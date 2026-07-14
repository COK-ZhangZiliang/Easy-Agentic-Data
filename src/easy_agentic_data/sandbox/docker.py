from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from easy_agentic_data.environments import is_immutable_image_reference

from .base import CommandResult, SandboxLimits

SANDBOX_PYTHONPATH = (
    "/workspace/.ead_prefix/lib/python3.9/site-packages:"
    "/workspace/.ead_prefix/lib/python3.11/site-packages:"
    "/workspace/src:/workspace/.ead_site:/workspace"
)


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
        self.baseline_commit = ""

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
        return self._execute(command, timeout_seconds=timeout_seconds, user=None)

    def execute_as_root(
        self, command: list[str], *, timeout_seconds: float | None = None
    ) -> CommandResult:
        return self._execute(command, timeout_seconds=timeout_seconds, user="0:0")

    def _execute(
        self,
        command: list[str],
        *,
        timeout_seconds: float | None = None,
        user: str | None = None,
    ) -> CommandResult:
        started = time.perf_counter()
        docker_command = [
            "docker",
            "exec",
            "--env",
            f"PYTHONPATH={SANDBOX_PYTHONPATH}",
            "--env",
            "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0",
            "--env",
            "MPLCONFIGDIR=/tmp/matplotlib",
        ]
        if user is not None:
            docker_command.extend(["--user", user])
        docker_command.extend([self.container_name, *command])
        completed = self._run_host(
            docker_command,
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

    def prepare_git_baseline(self) -> str:
        """Freeze the post-setup workspace so later diffs include every candidate change."""

        commands = [
            ["git", "init", "-q"],
            ["git", "config", "user.name", "Easy Agentic Data"],
            ["git", "config", "user.email", "ead@example.invalid"],
            ["git", "add", "--all", "--force"],
            ["git", "commit", "--allow-empty", "-qm", "ead-baseline"],
        ]
        for command in commands:
            result = self.execute(command)
            _require_success(result, command)
        revision = ["git", "rev-parse", "HEAD"]
        result = self.execute(revision)
        _require_success(result, revision)
        self.baseline_commit = result.stdout.strip()
        if not self.baseline_commit:
            raise RuntimeError("Git baseline commit could not be resolved")
        return self.state_hash()

    def candidate_patch(self) -> str:
        """Return a binary-capable patch including tracked, deleted, and new files."""

        intent = ["git", "add", "--intent-to-add", "--all", "--force"]
        _require_success(self.execute(intent), intent)
        baseline = self.baseline_commit or "HEAD"
        if self.container_name:
            output_path = "/tmp/ead-candidate.patch"
            command = [
                "git",
                "diff",
                "--binary",
                "--no-ext-diff",
                f"--output={output_path}",
                baseline,
                "--",
            ]
            try:
                result = self.execute(command)
                _require_success(result, command)
                completed = self._run_host(
                    ["docker", "exec", self.container_name, "cat", output_path]
                )
                return completed.stdout
            finally:
                self.execute(["rm", "-f", output_path])
        command = ["git", "diff", "--binary", "--no-ext-diff", baseline, "--"]
        result = self.execute(command)
        _require_success(result, command)
        if result.truncated:
            raise RuntimeError("Candidate patch output was truncated")
        return result.stdout

    def apply_candidate_patch(self, patch: str) -> str:
        """Apply an agent patch to a clean baseline and return the candidate state hash."""

        if not patch:
            return self.state_hash()
        patch_hash = hashlib.sha256(patch.encode("utf-8")).hexdigest()[:16]
        patch_path = f".ead_candidate_{patch_hash}.patch"
        self.write(patch_path, patch)
        try:
            check = ["git", "apply", "--check", "--binary", patch_path]
            _require_success(self.execute(check), check)
            apply = ["git", "apply", "--binary", patch_path]
            _require_success(self.execute(apply), apply)
        finally:
            self.execute(["rm", "-f", patch_path])
        return self.state_hash()

    def diff(self) -> str:
        return self.candidate_patch()

    def state_hash(self) -> str:
        result = self.execute(
            [
                "sh",
                "-lc",
                (
                    "set -e; "
                    "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
                    "index=/tmp/ead-state-index-$$; rm -f \"$index\"; "
                    "trap 'rm -f \"$index\"' EXIT; "
                    "GIT_INDEX_FILE=\"$index\" git add --all --force >/dev/null; "
                    "GIT_INDEX_FILE=\"$index\" git write-tree; "
                    "else "
                    "find . -path './.git' -prune -o -type f -print0 "
                    "| sort -z | xargs -0 sha256sum; "
                    "fi"
                ),
            ]
        )
        _require_success(result, ["workspace-state-hash"])
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


def _require_success(result: CommandResult, command: list[str]) -> None:
    if result.exit_code == 0:
        return
    detail = (result.stderr or result.stdout).strip()
    raise RuntimeError(
        f"Sandbox command failed (exit={result.exit_code}, command={command!r}): {detail}"
    )
