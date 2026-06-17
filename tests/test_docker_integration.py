import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.agent import HeadlessAgent
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.models import LLMResponse, Message
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import DockerSandbox, SandboxLimits
from easy_agentic_data.scenarios import Scenario, ScenarioInstance
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed
from easy_agentic_data.traces import TraceRecorder, load_trace, replay_trace

IMAGE = "python@sha256:f417205fec4ccb0d5023fdb5ecb4c8eba31c1834f94dcbcd1a2e8325fa7a7b89"
GIT_IMAGE = "alpine/git@sha256:4a0e72d49596a1f5d3701aeedafdadc5c0da4062be4657c7bdc4017387f591cc"


@unittest.skipUnless(
    os.environ.get("EAD_RUN_DOCKER_TESTS") == "1",
    "Set EAD_RUN_DOCKER_TESTS=1 to run real Docker integration tests",
)
class DockerIntegrationTests(unittest.TestCase):
    def test_isolated_container_executes_and_resets_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "app.py").write_text("value = 1\n", encoding="utf-8")
            (source / "test_app.py").write_text(
                "import app\nassert app.value == 2\n",
                encoding="utf-8",
            )
            sandbox = DockerSandbox(
                image_digest=IMAGE,
                source_directory=source,
                limits=SandboxLimits(
                    timeout_seconds=15,
                    max_output_bytes=10_000,
                    max_workspace_bytes=1_000_000,
                    memory="256m",
                    cpus=0.5,
                    pids=32,
                ),
            )
            sandbox.create()
            first_container = sandbox.container_name
            try:
                initial_hash = sandbox.state_hash()
                uid = sandbox.execute(["id", "-u"])
                network = sandbox.execute(
                    [
                        "python",
                        "-c",
                        (
                            "import socket\n"
                            "s=socket.socket();s.settimeout(1)\n"
                            "try:s.connect(('1.1.1.1',53));print('connected');raise SystemExit(1)\n"
                            "except OSError:print('blocked')\n"
                        ),
                    ]
                )
                root_write = sandbox.execute(["sh", "-c", "touch /rootfs-write-test"])

                sandbox.write("app.py", "value = 2\n")
                test_result = sandbox.execute(["python", "test_app.py"])
                changed_hash = sandbox.state_hash()
                sandbox.restore(initial_hash)
                restored_hash = sandbox.state_hash()

                self.assertEqual(uid.stdout.strip(), "65532")
                self.assertEqual(network.exit_code, 0)
                self.assertIn("blocked", network.stdout)
                self.assertNotEqual(root_write.exit_code, 0)
                self.assertEqual(
                    test_result.exit_code,
                    0,
                    f"stdout={test_result.stdout!r} stderr={test_result.stderr!r}",
                )
                self.assertNotEqual(changed_hash, initial_hash)
                self.assertEqual(restored_hash, initial_hash)
                self.assertNotEqual(sandbox.container_name, first_container)

                inspect = sandbox._run_host(["docker", "inspect", sandbox.container_name])
                self.assertNotIn("/var/run/docker.sock", inspect.stdout)
                self.assertIn('"ReadonlyRootfs": true', inspect.stdout)
                self.assertIn('"NetworkMode": "none"', inspect.stdout)
            finally:
                sandbox.destroy()

    def test_headless_agent_repairs_git_fixture_inside_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "app.txt").write_text("value=1\n", encoding="utf-8")
            (source / "test.sh").write_text(
                "grep -q '^value=2$' app.txt\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
            subprocess.run(
                ["git", "-C", str(source), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
            sandbox = DockerSandbox(
                image_digest=GIT_IMAGE,
                source_directory=source,
                limits=SandboxLimits(memory="256m", cpus=0.5, pids=32),
            )
            sandbox.create()
            try:
                scenario = Scenario(
                    QuerySeed(PublicTaskContext("Change app.txt so test.sh passes.")),
                    EnvironmentSpec(name="docker-agent-fixture", version="1"),
                )
                instance = ScenarioInstance.materialize(
                    scenario,
                    random_seed=1,
                    initial_state_hash=sandbox.state_hash(),
                )
                tools = CodingToolRuntime(
                    sandbox,
                    ToolPolicy(["read_file", "apply_patch", "run_command", "git_diff"]),
                )
                trace_path = root / "agent-trace.jsonl"
                with TraceRecorder(
                    trace_path,
                    session_id="docker_agent",
                    scenario_instance=instance,
                ) as recorder:
                    result = HeadlessAgent(_DockerAgentClient(), tools).run(instance, recorder)
                trace = load_trace(trace_path)
                replay = replay_trace(trace)

                self.assertEqual(result.tool_calls, 4)
                self.assertEqual(sandbox.execute(["sh", "test.sh"]).exit_code, 0)
                self.assertIn("value=2", sandbox.read("app.txt"))
                self.assertIn("value=2", sandbox.diff())
                self.assertEqual(replay.terminal_state_hash, sandbox.state_hash())
            finally:
                sandbox.destroy()


class _DockerAgentClient:
    model = "docker-scripted-agent"

    def __init__(self) -> None:
        self.index = 0

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        script = [
            _tool("read", "read_file", {"path": "app.txt"}),
            _tool(
                "patch",
                "apply_patch",
                {"path": "app.txt", "old": "value=1", "new": "value=2"},
            ),
            _tool("test", "run_command", {"command": ["sh", "test.sh"]}),
            _tool("diff", "git_diff", {}),
            Message("assistant", "Updated app.txt and verified test.sh."),
        ]
        message = script[self.index]
        self.index += 1
        return LLMResponse(message, self.model, {"total_tokens": 10})


def _tool(call_id: str, name: str, arguments: dict) -> Message:
    return Message(
        "assistant",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


if __name__ == "__main__":
    unittest.main()
