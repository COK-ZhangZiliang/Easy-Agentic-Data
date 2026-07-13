import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.agent import AgentBudgets, HeadlessAgent
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.config import LLMConfig
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.llm.observability import ObservedLLMClient
from easy_agentic_data.llm.openai_compatible import OpenAICompatibleClient
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import DockerSandbox, SandboxLimits
from easy_agentic_data.scenarios import Scenario, ScenarioInstance
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed
from easy_agentic_data.traces import TraceRecorder, load_trace, replay_trace

GIT_IMAGE = "alpine/git@sha256:4a0e72d49596a1f5d3701aeedafdadc5c0da4062be4657c7bdc4017387f591cc"


def _live_config() -> LLMConfig:
    return LLMConfig(
        provider="openai_compatible",
        model=os.environ.get("EAD_LIVE_LLM_MODEL", "deepseek-v4-flash"),
        base_url=os.environ.get("EAD_LIVE_LLM_BASE_URL", "https://api.deepseek.com"),
        api_key_env="DEEPSEEK_API_KEY",
        timeout_seconds=120,
        temperature=0.0,
        max_tokens=2048,
        request_body={"thinking": {"type": "disabled"}},
    )


@unittest.skipUnless(
    os.environ.get("EAD_RUN_LIVE_LLM_TESTS") == "1",
    "Set EAD_RUN_LIVE_LLM_TESTS=1 to run paid live LLM integration tests",
)
class LiveLLMIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("EAD_RUN_DOCKER_TESTS") == "1",
        "Set EAD_RUN_DOCKER_TESTS=1 to run the live Docker agent test",
    )
    def test_real_model_repairs_docker_fixture(self) -> None:
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
                    QuerySeed(
                        PublicTaskContext(
                            "Change app.txt from value=1 to value=2. Run sh test.sh and inspect "
                            "the Git diff before finishing."
                        )
                    ),
                    EnvironmentSpec(name="live-docker-agent-fixture", version="1"),
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
                client = ObservedLLMClient(OpenAICompatibleClient(_live_config()))
                trace_path = root / "agent-trace.jsonl"
                with TraceRecorder(
                    trace_path,
                    session_id="live_docker_agent",
                    scenario_instance=instance,
                ) as recorder:
                    result = HeadlessAgent(
                        client,
                        tools,
                        budgets=AgentBudgets(max_turns=10, max_tool_calls=12),
                    ).run(instance, recorder)

                replay = replay_trace(load_trace(trace_path))
                self.assertEqual(sandbox.execute(["sh", "test.sh"]).exit_code, 0)
                self.assertIn("value=2", sandbox.read("app.txt"))
                self.assertIn("value=2", sandbox.diff())
                self.assertGreaterEqual(result.tool_calls, 3)
                self.assertTrue(result.final_answer)
                self.assertEqual(replay.terminal_state_hash, sandbox.state_hash())
            finally:
                sandbox.destroy()


if __name__ == "__main__":
    unittest.main()
