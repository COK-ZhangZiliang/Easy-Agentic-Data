import json
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.agent import AgentBudgets, HeadlessAgent
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.models import LLMResponse, Message
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import CommandResult, MemorySandbox
from easy_agentic_data.scenarios import Scenario, ScenarioInstance
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed
from easy_agentic_data.traces import TraceRecorder, load_trace, replay_trace


class ScriptedClient:
    model = "scripted-agent"

    def __init__(self) -> None:
        self.index = 0

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        script = [
            _tool("read_1", "read_file", {"path": "app.py"}),
            _tool(
                "patch_1",
                "apply_patch",
                {"path": "app.py", "old": "return 1", "new": "return 2"},
            ),
            _tool("test_1", "run_command", {"command": ["python", "-m", "test"]}),
            Message("assistant", "Fixed the implementation and verified the test."),
        ]
        message = script[self.index]
        self.index += 1
        return LLMResponse(message, self.model, {"total_tokens": 10})


class HeadlessAgentTests(unittest.TestCase):
    def test_agent_repairs_fixture_and_records_replayable_trace(self) -> None:
        sandbox = MemorySandbox(
            {"app.py": "def value():\n    return 1\n"},
            {
                "python -m test": lambda box: CommandResult(
                    0 if "return 2" in box.read("app.py") else 1,
                    "ok\n",
                    "",
                    1.0,
                )
            },
        )
        sandbox.create()
        tools = CodingToolRuntime(
            sandbox,
            ToolPolicy(["read_file", "apply_patch", "run_command"]),
        )
        instance = _instance(sandbox.state_hash())
        agent = HeadlessAgent(ScriptedClient(), tools, budgets=AgentBudgets(max_turns=6))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.jsonl"
            with TraceRecorder(path, session_id="session_agent", scenario_instance=instance) as recorder:
                result = agent.run(instance, recorder)
            replay = replay_trace(load_trace(path))

        self.assertIn("return 2", sandbox.read("app.py"))
        self.assertEqual(result.final_answer, "Fixed the implementation and verified the test.")
        self.assertEqual(result.tool_calls, 3)
        self.assertEqual(replay.terminal_state_hash, sandbox.state_hash())
        self.assertEqual(replay.state.tool_calls["test_1"]["status"], "completed")

    def test_malformed_tool_calls_stop_after_retry_limit(self) -> None:
        class BrokenClient:
            model = "broken"

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                return LLMResponse(
                    Message(
                        "assistant",
                        tool_calls=[
                            {
                                "id": "bad",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{"},
                            }
                        ],
                    ),
                    self.model,
                )

        sandbox = MemorySandbox({"app.py": "x"})
        sandbox.create()
        tools = CodingToolRuntime(sandbox, ToolPolicy(["read_file"]))
        instance = _instance(sandbox.state_hash())
        agent = HeadlessAgent(
            BrokenClient(),
            tools,
            budgets=AgentBudgets(max_turns=5, malformed_tool_retries=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            with TraceRecorder(
                Path(directory) / "bad.jsonl",
                session_id="session_bad",
                scenario_instance=instance,
            ) as recorder:
                result = agent.run(instance, recorder)
        self.assertEqual(result.termination_reason.value, "infrastructure_failure")


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


def _instance(state_hash: str) -> ScenarioInstance:
    scenario = Scenario(
        QuerySeed(PublicTaskContext("Repair app.py so the test passes.")),
        EnvironmentSpec(name="memory-fixture", version="1"),
    )
    return ScenarioInstance.materialize(
        scenario,
        random_seed=1,
        initial_state_hash=state_hash,
    )


if __name__ == "__main__":
    unittest.main()
