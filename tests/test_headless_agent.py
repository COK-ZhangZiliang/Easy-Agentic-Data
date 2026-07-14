import json
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.agent import DEFAULT_SYSTEM_PROMPT, AgentBudgets, HeadlessAgent
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.llm import (
    ObservedLLMClient,
    prompt_hash,
    prompt_token_upper_bound,
    trace_prompt_fingerprints,
    validate_observed_prompt_lineage,
)
from easy_agentic_data.models import LLMResponse, Message
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import CommandResult, MemorySandbox
from easy_agentic_data.scenarios import Scenario, ScenarioInstance
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed
from easy_agentic_data.traces import EventType, TraceRecorder, load_trace, replay_trace


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
    def test_prompt_tokens_are_reserved_before_provider_execution(self) -> None:
        class BudgetClient:
            model = "budget-client"
            max_tokens = 2048

            def __init__(self) -> None:
                self.requested_max_tokens = None

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools
                self.requested_max_tokens = kwargs.get("max_tokens")
                return LLMResponse(
                    Message("assistant", "This response crosses the budget."),
                    self.model,
                    {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
                )

        client = BudgetClient()
        sandbox = MemorySandbox({"app.py": "x"})
        sandbox.create()
        instance = _instance(sandbox.state_hash())
        agent = HeadlessAgent(
            client,
            CodingToolRuntime(sandbox, ToolPolicy(["read_file"])),
            budgets=AgentBudgets(max_tokens=10),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.jsonl"
            with TraceRecorder(
                path,
                session_id="session_budget",
                scenario_instance=instance,
            ) as recorder:
                result = agent.run(instance, recorder)
            trace = load_trace(path)

        self.assertIsNone(client.requested_max_tokens)
        self.assertEqual(result.tokens, 0)
        self.assertEqual(result.termination_reason.value, "token_budget")
        self.assertEqual(
            sum(event.event_type.value == "model_response" for event in trace.events),
            0,
        )

    def test_provider_usage_cannot_exceed_pre_request_token_bound(self) -> None:
        class ViolatingClient:
            model = "violating-client"
            max_tokens = 4

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                return LLMResponse(
                    Message("assistant", "Too much output."),
                    self.model,
                    {
                        "prompt_tokens": 1,
                        "completion_tokens": 5,
                        "total_tokens": 6,
                    },
                )

        sandbox = MemorySandbox({"app.py": "x"})
        sandbox.create()
        instance = _instance(sandbox.state_hash())
        agent = HeadlessAgent(
            ViolatingClient(),
            CodingToolRuntime(sandbox, ToolPolicy(["read_file"])),
            budgets=AgentBudgets(max_tokens=10_000),
        )
        with tempfile.TemporaryDirectory() as directory:
            with TraceRecorder(
                Path(directory) / "provider-overrun.jsonl",
                session_id="session_provider_overrun",
                scenario_instance=instance,
            ) as recorder, self.assertRaisesRegex(
                ValueError,
                "output usage exceeded",
            ):
                agent.run(instance, recorder)

    def test_provider_output_limit_excludes_reserved_prompt_tokens(self) -> None:
        class RecordingClient:
            model = "recording-client"
            max_tokens = 10_000

            def __init__(self) -> None:
                self.requested_max_tokens = None

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools
                self.requested_max_tokens = kwargs["max_tokens"]
                return LLMResponse(
                    Message("assistant", "Done."),
                    self.model,
                    {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                )

        sandbox = MemorySandbox({"app.py": "x"})
        sandbox.create()
        instance = _instance(sandbox.state_hash())
        tools = CodingToolRuntime(sandbox, ToolPolicy(["read_file"]))
        prompt_bound = prompt_token_upper_bound(
            [
                Message("system", DEFAULT_SYSTEM_PROMPT),
                Message("user", instance.public_task.query),
            ],
            tools.schemas(),
        )
        client = RecordingClient()
        agent = HeadlessAgent(
            client,
            tools,
            budgets=AgentBudgets(max_tokens=prompt_bound + 25),
        )
        with tempfile.TemporaryDirectory() as directory:
            with TraceRecorder(
                Path(directory) / "reserved-prompt.jsonl",
                session_id="session_reserved_prompt",
                scenario_instance=instance,
            ) as recorder:
                agent.run(instance, recorder)

        self.assertEqual(client.requested_max_tokens, 25)

    def test_agent_token_budget_uses_prompt_and_completion_fallback(self) -> None:
        class UsageClient:
            model = "usage-client"

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                return LLMResponse(
                    Message("assistant", "Done."),
                    self.model,
                    {"prompt_tokens": 7, "completion_tokens": 5},
                )

        sandbox = MemorySandbox({"app.py": "x"})
        sandbox.create()
        instance = _instance(sandbox.state_hash())
        agent = HeadlessAgent(
            UsageClient(),
            CodingToolRuntime(sandbox, ToolPolicy(["read_file"])),
        )
        with tempfile.TemporaryDirectory() as directory:
            with TraceRecorder(
                Path(directory) / "usage.jsonl",
                session_id="session_usage",
                scenario_instance=instance,
            ) as recorder:
                result = agent.run(instance, recorder)

        self.assertEqual(result.tokens, 12)

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
            with TraceRecorder(
                path, session_id="session_agent", scenario_instance=instance
            ) as recorder:
                result = agent.run(instance, recorder)
            trace = load_trace(path)
            replay = replay_trace(trace)

        self.assertIn("return 2", sandbox.read("app.py"))
        self.assertEqual(result.final_answer, "Fixed the implementation and verified the test.")
        self.assertEqual(result.tool_calls, 3)
        self.assertEqual(replay.terminal_state_hash, sandbox.state_hash())
        self.assertEqual(replay.state.tool_calls["test_1"]["status"], "completed")
        self.assertEqual(
            [message["role"] for message in replay.state.messages],
            [
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "assistant",
            ],
        )
        self.assertEqual(replay.state.messages[0]["content"], DEFAULT_SYSTEM_PROMPT)
        first_tool_message = replay.state.messages[3]
        self.assertEqual(first_tool_message["name"], "read_file")
        self.assertEqual(first_tool_message["tool_call_id"], "read_1")
        self.assertEqual(json.loads(first_tool_message["content"])["ok"], True)

    def test_malformed_tool_calls_stop_after_retry_limit(self) -> None:
        class BrokenClient:
            model = "broken"

            def __init__(self) -> None:
                self.index = 0

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                response = LLMResponse(
                    Message(
                        "assistant",
                        tool_calls=[
                            {
                                "id": f"bad_{self.index}",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{"},
                            }
                        ],
                    ),
                    self.model,
                )
                self.index += 1
                return response

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
            path = Path(directory) / "bad.jsonl"
            with TraceRecorder(
                path,
                session_id="session_bad",
                scenario_instance=instance,
            ) as recorder:
                result = agent.run(instance, recorder)
            trace = load_trace(path)
        self.assertEqual(result.termination_reason.value, "malformed_tool_calls")
        tool_messages = [
            event for event in trace.events if event.event_type is EventType.TOOL_MESSAGE
        ]
        self.assertEqual(len(tool_messages), 2)
        self.assertTrue(
            all("Invalid tool arguments" in event.payload["content"] for event in tool_messages)
        )

    def test_policy_denial_records_the_exact_tool_role_message(self) -> None:
        class DeniedClient:
            model = "denied"

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                return LLMResponse(
                    _tool(
                        "denied_1",
                        "run_command",
                        {"command": ["curl", "https://example.invalid"]},
                    ),
                    self.model,
                    {"total_tokens": 5},
                )

        sandbox = MemorySandbox({"app.py": "x"})
        sandbox.create()
        instance = _instance(sandbox.state_hash())
        tools = CodingToolRuntime(sandbox, ToolPolicy(["run_command"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "denied.jsonl"
            with TraceRecorder(
                path,
                session_id="session_denied",
                scenario_instance=instance,
            ) as recorder:
                result = HeadlessAgent(DeniedClient(), tools).run(instance, recorder)
            trace = load_trace(path)
            replay = replay_trace(trace)

        self.assertEqual(result.termination_reason.value, "policy_violation")
        self.assertEqual(
            [message["role"] for message in replay.state.messages],
            ["system", "user", "assistant", "tool"],
        )
        tool_message = replay.state.messages[-1]
        self.assertEqual(tool_message["tool_call_id"], "denied_1")
        self.assertEqual(json.loads(tool_message["content"])["ok"], False)
        self.assertIn("Network access is disabled", tool_message["content"])

    def test_terminal_policy_denial_closes_every_parallel_tool_call(self) -> None:
        class MultiCallClient:
            model = "multi-call-denied"

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                return LLMResponse(
                    Message(
                        "assistant",
                        tool_calls=[
                            {
                                "id": "denied_1",
                                "type": "function",
                                "function": {
                                    "name": "run_command",
                                    "arguments": json.dumps(
                                        {"command": ["curl", "https://example.invalid"]}
                                    ),
                                },
                            },
                            {
                                "id": "cancelled_2",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "app.py"}),
                                },
                            },
                        ],
                    ),
                    self.model,
                    {"total_tokens": 5},
                )

        sandbox = MemorySandbox({"app.py": "x"})
        sandbox.create()
        instance = _instance(sandbox.state_hash())
        tools = CodingToolRuntime(sandbox, ToolPolicy(["run_command", "read_file"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parallel-denied.jsonl"
            with TraceRecorder(
                path,
                session_id="session_parallel_denied",
                scenario_instance=instance,
            ) as recorder:
                result = HeadlessAgent(MultiCallClient(), tools).run(instance, recorder)
            replay = replay_trace(load_trace(path, tolerate_truncated=False))

        self.assertEqual(result.termination_reason.value, "policy_violation")
        tool_messages = [
            message for message in replay.state.messages if message["role"] == "tool"
        ]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["denied_1", "cancelled_2"],
        )
        self.assertIn("policy violation", tool_messages[1]["content"])

    def test_ask_user_order_and_prompt_hash_round_trip(self) -> None:
        class AskUserClient:
            model = "ask-user"

            def __init__(self) -> None:
                self.index = 0

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                script = [
                    _tool("ask_1", "ask_user", {"question": "Which value?"}),
                    Message("assistant", "Used the supplied value."),
                ]
                message = script[self.index]
                self.index += 1
                return LLMResponse(message, self.model, {"total_tokens": 5})

        sandbox = MemorySandbox({"app.py": "x"})
        sandbox.create()
        instance = _instance(sandbox.state_hash())
        tools = CodingToolRuntime(sandbox, ToolPolicy(["ask_user"]))
        schemas = tools.schemas()
        observed = ObservedLLMClient(AskUserClient())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ask-user.jsonl"
            with TraceRecorder(
                path,
                session_id="session_ask_user",
                scenario_instance=instance,
            ) as recorder:
                HeadlessAgent(observed, tools).run(
                    instance,
                    recorder,
                    ask_user=lambda question: "2" if question == "Which value?" else None,
                )
            trace = load_trace(path)
            replay = replay_trace(trace)

        self.assertEqual(
            [message["role"] for message in replay.state.messages],
            ["system", "user", "assistant", "tool", "user", "assistant"],
        )
        self.assertEqual(json.loads(replay.state.messages[3]["content"]), {"answer": "2"})
        reconstructed = [
            _message_from_replay(message) for message in replay.state.messages[:-1]
        ]
        self.assertEqual(observed.records[1]["message_count"], len(reconstructed))
        self.assertEqual(observed.records[1]["prompt_hash"], prompt_hash(reconstructed, schemas))
        self.assertEqual(
            observed.records[1]["prompt_token_upper_bound"],
            prompt_token_upper_bound(reconstructed, schemas),
        )
        fingerprints = trace_prompt_fingerprints(trace, DEFAULT_SYSTEM_PROMPT, schemas)
        self.assertEqual(len(fingerprints), 2)
        self.assertEqual(observed.records[1]["prompt_hash"], fingerprints[1]["prompt_hash"])
        validate_observed_prompt_lineage(
            observed.records,
            trace,
            DEFAULT_SYSTEM_PROMPT,
            schemas,
        )
        tampered = [dict(record) for record in observed.records]
        tampered[0]["prompt_token_upper_bound"] += 1
        with self.assertRaisesRegex(ValueError, "prompt_token_upper_bound"):
            validate_observed_prompt_lineage(
                tampered,
                trace,
                DEFAULT_SYSTEM_PROMPT,
                schemas,
            )
        with self.assertRaisesRegex(ValueError, "does not match canonical trace"):
            trace_prompt_fingerprints(trace, "Different system prompt", schemas)


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


def _message_from_replay(value: dict) -> Message:
    return Message(
        value["role"],
        value.get("content"),
        name=value.get("name"),
        tool_call_id=value.get("tool_call_id"),
        tool_calls=value.get("tool_calls", []),
        reasoning_content=value.get("reasoning_content"),
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
