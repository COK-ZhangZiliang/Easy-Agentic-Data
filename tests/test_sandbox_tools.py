import unittest
from unittest.mock import patch

from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import CommandResult, DockerSandbox, MemorySandbox
from easy_agentic_data.sandbox.docker import _bounded


class SandboxToolTests(unittest.TestCase):
    def test_only_policy_allowed_tools_are_exposed_to_model(self) -> None:
        sandbox = MemorySandbox({"app.py": "value = 1\n"})
        sandbox.create()
        runtime = CodingToolRuntime(
            sandbox,
            ToolPolicy(["read_file", "run_command"]),
        )

        schemas = runtime.schemas()

        names = [schema["function"]["name"] for schema in schemas]
        self.assertEqual(names, ["read_file", "run_command"])
        command = next(schema for schema in schemas if schema["function"]["name"] == "run_command")
        self.assertEqual(
            command["function"]["parameters"]["properties"]["command"]["items"],
            {"type": "string"},
        )

    def setUp(self) -> None:
        self.sandbox = MemorySandbox(
            {"app.py": "def value():\n    return 1\n"},
            {"python -m test": CommandResult(0, "ok\n", "", 1.0)},
        )
        self.sandbox.create()
        self.runtime = CodingToolRuntime(
            self.sandbox,
            ToolPolicy(
                [
                    "list_files",
                    "read_file",
                    "search_files",
                    "apply_patch",
                    "run_command",
                    "git_status",
                    "git_diff",
                    "ask_user",
                ]
            ),
        )

    def test_patch_changes_state_and_restore_recovers_snapshot(self) -> None:
        initial = self.sandbox.state_hash()
        snapshot = self.sandbox.snapshot()
        result = self.runtime.execute(
            "apply_patch",
            {"path": "app.py", "old": "return 1", "new": "return 2"},
        )
        self.assertIsNone(result.error)
        self.assertNotEqual(self.sandbox.state_hash(), initial)
        self.sandbox.restore(snapshot)
        self.assertEqual(self.sandbox.state_hash(), initial)

    def test_policy_denies_network_and_path_escape(self) -> None:
        network = self.runtime.execute("run_command", {"command": ["curl", "https://example.com"]})
        install = self.runtime.execute("run_command", {"command": ["pip", "install", "-e", "."]})
        offline_install = self.runtime.execute(
            "run_command", {"command": ["pip", "install", "--no-deps", "-e", "."]}
        )
        escaped = self.runtime.execute("read_file", {"path": "../secret"})
        self.assertIn("Network access is disabled", network.error or "")
        self.assertIn("Package installation is blocked", install.error or "")
        self.assertNotIn("Package installation is blocked", offline_install.error or "")
        self.assertIn("forbidden host path", escaped.error or "")

    def test_adversarial_arguments_and_oversized_output_are_contained(self) -> None:
        socket = self.runtime.execute("read_file", {"path": "/var/run/docker.sock"})
        expansion = self.runtime.execute(
            "run_command", {"command": ["sh", "-lc", "cat /etc/passwd"]}
        )
        traversal = self.runtime.execute("read_file", {"path": "link/../../secret"})
        bounded, truncated = _bounded("x" * 100, 10)
        self.assertIn("forbidden host path", socket.error or "")
        self.assertIn("forbidden host path", expansion.error or "")
        self.assertIn("forbidden host path", traversal.error or "")
        self.assertTrue(truncated)
        self.assertEqual(len(bounded), 10)

    def test_schema_validation_rejects_extra_arguments(self) -> None:
        result = self.runtime.execute("read_file", {"path": "app.py", "extra": True})
        self.assertIn("Unexpected tool arguments", result.error or "")

    def test_search_files_skips_unreadable_files(self) -> None:
        sandbox = _UnreadableFileSandbox(
            {
                "app.py": "def value():\n    return 1\n",
                "assets/logo.png": "",
            }
        )
        sandbox.create()
        runtime = CodingToolRuntime(sandbox, ToolPolicy(["search_files"]))

        result = runtime.execute("search_files", {"query": "return"})

        self.assertIsNone(result.error)
        self.assertEqual(result.output["match_count"], 1)
        self.assertEqual(result.output["skipped_count"], 1)
        self.assertFalse(result.output["truncated"])

    def test_search_files_uses_sandbox_grep_when_available(self) -> None:
        sandbox = _GrepSandbox({"app.py": "unused\n"})
        sandbox.create()
        runtime = CodingToolRuntime(sandbox, ToolPolicy(["search_files"]))

        result = runtime.execute("search_files", {"query": "needle"})

        self.assertIsNone(result.error)
        self.assertEqual(result.output["matches"][0]["path"], "src/app.py")
        self.assertEqual(result.output["matches"][0]["line"], 7)
        self.assertEqual(result.output["match_count"], 1)

    def test_search_files_does_not_slow_fallback_after_real_grep_failure(self) -> None:
        sandbox = _FailingGrepSandbox({"app.py": "needle\n"})
        sandbox.create()
        runtime = CodingToolRuntime(sandbox, ToolPolicy(["search_files"]))

        result = runtime.execute("search_files", {"query": "needle"})

        self.assertIsNone(result.error)
        self.assertEqual(result.output["grep_exit_code"], 2)
        self.assertIn("bad grep", result.output["grep_error"])

    def test_file_tools_return_bounded_structured_outputs(self) -> None:
        files = {f"file_{index}.txt": "x" for index in range(510)}
        files["large.txt"] = "a" * 40_010
        files["lines.txt"] = "one\ntwo\nthree\nfour\n"
        sandbox = MemorySandbox(files)
        sandbox.create()
        runtime = CodingToolRuntime(sandbox, ToolPolicy(["list_files", "read_file"]))

        listed = runtime.execute("list_files", {"path": "."})
        read = runtime.execute("read_file", {"path": "large.txt"})
        sliced = runtime.execute("read_file", {"path": "lines.txt", "offset": 2, "limit": 2})

        self.assertEqual(listed.output["file_count"], 512)
        self.assertEqual(len(listed.output["files"]), 500)
        self.assertTrue(listed.output["truncated"])
        self.assertEqual(read.output["chars"], 40_010)
        self.assertEqual(len(read.output["content"]), 40_000)
        self.assertTrue(read.output["truncated"])
        self.assertEqual(sliced.output["content"], "two\nthree\n")

    def test_git_diff_tool_returns_bounded_structured_output(self) -> None:
        sandbox = MemorySandbox({"app.py": "before\n"})
        sandbox.create()
        sandbox.write("app.py", "x" * 50_000)
        runtime = CodingToolRuntime(sandbox, ToolPolicy(["git_diff"]))

        result = runtime.execute("git_diff", {})

        self.assertEqual(len(result.output["diff"]), 40_000)
        self.assertGreater(result.output["chars"], 40_000)
        self.assertTrue(result.output["truncated"])

    def test_read_file_rejects_invalid_slice_bounds(self) -> None:
        offset = self.runtime.execute("read_file", {"path": "app.py", "offset": 0})
        limit = self.runtime.execute("read_file", {"path": "app.py", "limit": 0})

        self.assertIn("offset must be >= 1", offset.error or "")
        self.assertIn("limit must be >= 1", limit.error or "")

    def test_docker_backend_requires_digest_and_uses_safe_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "content-addressed by digest"):
            DockerSandbox(image_digest="python:3.11", source_directory=".")
        sandbox = DockerSandbox(
            image_digest="python@sha256:" + "a" * 64,
            source_directory=".",
        )
        local_image = DockerSandbox(image_digest="sha256:" + "b" * 64, source_directory=".")
        calls = []

        def capture(command, **kwargs):
            calls.append(command)
            stdout = "1\t/workspace\n" if command[-3:] == ["du", "-sk", "/workspace"] else ""
            return type("Result", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(sandbox, "_run_host", side_effect=capture),
        ):
            sandbox.create()
        create = next(call for call in calls if call[:2] == ["docker", "create"])
        self.assertEqual(local_image.image_digest, "sha256:" + "b" * 64)
        self.assertIn("none", create)
        self.assertIn("--read-only", create)
        self.assertIn("--pids-limit", create)
        self.assertNotIn("/var/run/docker.sock", " ".join(create))

    def test_docker_root_setup_execution_is_explicit(self) -> None:
        sandbox = DockerSandbox(image_digest="sha256:" + "b" * 64, source_directory=".")
        sandbox.container_name = "container"
        calls = []

        def capture(command, **kwargs):
            calls.append(command)
            stdout = "1\t/workspace\n" if command[-3:] == ["du", "-sk", "/workspace"] else ""
            return type("Result", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

        with patch.object(sandbox, "_run_host", side_effect=capture):
            sandbox.execute_as_root(["python", "-m", "pip", "install", "--no-deps", "-e", "."])

        self.assertEqual(
            calls[0][:11],
            [
                "docker",
                "exec",
                "--env",
                (
                    "PYTHONPATH=/workspace/.ead_prefix/lib/python3.9/site-packages:"
                    "/workspace/.ead_prefix/lib/python3.11/site-packages:"
                    "/workspace/src:/workspace/.ead_site:/workspace"
                ),
                "--env",
                "SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0",
                "--env",
                "MPLCONFIGDIR=/tmp/matplotlib",
                "--user",
                "0:0",
                "container",
            ],
        )

    def test_docker_candidate_patch_includes_new_files_and_binary_changes(self) -> None:
        sandbox = DockerSandbox(image_digest="sha256:" + "b" * 64, source_directory=".")
        calls = []

        def execute(command, **kwargs):
            del kwargs
            calls.append(command)
            if command[:2] == ["git", "diff"]:
                return CommandResult(0, "binary candidate patch", "", 1.0)
            return CommandResult(0, "", "", 1.0)

        with patch.object(sandbox, "execute", side_effect=execute):
            value = sandbox.candidate_patch()

        self.assertEqual(value, "binary candidate patch")
        self.assertEqual(
            calls[0],
            ["git", "add", "--intent-to-add", "--all", "--force"],
        )
        self.assertEqual(
            calls[1],
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        )

    def test_docker_prepare_git_baseline_commits_post_setup_workspace(self) -> None:
        sandbox = DockerSandbox(image_digest="sha256:" + "b" * 64, source_directory=".")
        commands = []

        def execute(command, **kwargs):
            del kwargs
            commands.append(command)
            stdout = "a" * 40 + "\n" if command == ["git", "rev-parse", "HEAD"] else ""
            return CommandResult(0, stdout, "", 1.0)

        with (
            patch.object(sandbox, "execute", side_effect=execute),
            patch.object(sandbox, "state_hash", return_value="baseline_hash"),
        ):
            baseline = sandbox.prepare_git_baseline()

        self.assertEqual(baseline, "baseline_hash")
        self.assertEqual(commands[0], ["git", "init", "-q"])
        self.assertIn(
            ["git", "commit", "--allow-empty", "-qm", "ead-baseline"],
            commands,
        )
        self.assertEqual(commands[-1], ["git", "rev-parse", "HEAD"])


class _UnreadableFileSandbox(MemorySandbox):
    def read(self, path: str) -> str:
        if path == "assets/logo.png":
            raise UnicodeDecodeError("utf-8", b"\x89", 0, 1, "invalid start byte")
        return super().read(path)


class _GrepSandbox(MemorySandbox):
    def execute(self, command, *, timeout_seconds=None):
        del timeout_seconds
        if command[:4] == ["grep", "-R", "-n", "-I"]:
            return CommandResult(0, "./src/app.py:7:contains needle\n", "", 1.0)
        return super().execute(command)

    def read(self, path: str) -> str:
        raise AssertionError(f"grep-backed search should not read {path}")


class _FailingGrepSandbox(MemorySandbox):
    def execute(self, command, *, timeout_seconds=None):
        del timeout_seconds
        if command[:4] == ["grep", "-R", "-n", "-I"]:
            return CommandResult(2, "", "bad grep option\n", 1.0)
        return super().execute(command)

    def read(self, path: str) -> str:
        raise AssertionError(f"failing grep should not read {path}")


if __name__ == "__main__":
    unittest.main()
