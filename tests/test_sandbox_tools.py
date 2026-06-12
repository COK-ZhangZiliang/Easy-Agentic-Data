import unittest
from unittest.mock import patch

from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import CommandResult, DockerSandbox, MemorySandbox
from easy_agentic_data.sandbox.docker import _bounded


class SandboxToolTests(unittest.TestCase):
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
                    "list_files", "read_file", "search_files", "apply_patch",
                    "run_command", "git_status", "git_diff", "ask_user",
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
        network = self.runtime.execute(
            "run_command", {"command": ["curl", "https://example.com"]}
        )
        escaped = self.runtime.execute("read_file", {"path": "../secret"})
        self.assertIn("Network access is disabled", network.error or "")
        self.assertIn("forbidden host path", escaped.error or "")

    def test_adversarial_arguments_and_oversized_output_are_contained(self) -> None:
        socket = self.runtime.execute(
            "read_file", {"path": "/var/run/docker.sock"}
        )
        expansion = self.runtime.execute(
            "run_command", {"command": ["sh", "-lc", "cat /etc/passwd"]}
        )
        traversal = self.runtime.execute(
            "read_file", {"path": "link/../../secret"}
        )
        bounded, truncated = _bounded("x" * 100, 10)
        self.assertIn("forbidden host path", socket.error or "")
        self.assertIn("forbidden host path", expansion.error or "")
        self.assertIn("forbidden host path", traversal.error or "")
        self.assertTrue(truncated)
        self.assertEqual(len(bounded), 10)

    def test_schema_validation_rejects_extra_arguments(self) -> None:
        result = self.runtime.execute("read_file", {"path": "app.py", "extra": True})
        self.assertIn("Unexpected tool arguments", result.error or "")

    def test_docker_backend_requires_digest_and_uses_safe_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned by digest"):
            DockerSandbox(image_digest="python:3.11", source_directory=".")
        sandbox = DockerSandbox(
            image_digest="python@sha256:" + "a" * 64,
            source_directory=".",
        )
        calls = []

        def capture(command, **kwargs):
            calls.append(command)
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch("shutil.which", return_value="/usr/bin/docker"), patch.object(
            sandbox, "_run_host", side_effect=capture
        ):
            sandbox.create()
        create = next(call for call in calls if call[:2] == ["docker", "create"])
        self.assertIn("none", create)
        self.assertIn("--read-only", create)
        self.assertIn("--pids-limit", create)
        self.assertNotIn("/var/run/docker.sock", " ".join(create))


if __name__ == "__main__":
    unittest.main()
