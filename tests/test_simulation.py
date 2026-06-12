import json
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.agent import HeadlessAgent
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.models import LLMResponse, Message
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import MemorySandbox
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance
from easy_agentic_data.seeds import HiddenUserContext, PublicTaskContext, QuerySeed
from easy_agentic_data.simulation import (
    LLMUserSimulator,
    RuleBasedUserSimulator,
    UserObservation,
    user_callback,
)
from easy_agentic_data.traces import TraceRecorder, load_trace


class SimulationTests(unittest.TestCase):
    def test_rule_based_user_reveals_known_fact_not_hidden_canary(self) -> None:
        instance = _instance()
        simulator = RuleBasedUserSimulator(instance)

        response = simulator.respond(
            UserObservation("What is the target value?", instance.public_task.query, 1)
        )

        self.assertEqual(response.content, "2")
        self.assertNotIn("EVALUATOR_CANARY", response.content or "")
        self.assertEqual(simulator.metrics.clarifications, 1)

    def test_agent_uses_ask_user_in_multi_turn_trace(self) -> None:
        class AskThenFinish:
            model = "ask-agent"

            def __init__(self):
                self.turn = 0

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                self.turn += 1
                if self.turn == 1:
                    return LLMResponse(
                        Message(
                            "assistant",
                            tool_calls=[
                                {
                                    "id": "ask_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ask_user",
                                        "arguments": json.dumps(
                                            {"question": "What is the target value?"}
                                        ),
                                    },
                                }
                            ],
                        ),
                        self.model,
                    )
                return LLMResponse(Message("assistant", "The target value is 2."), self.model)

        instance = _instance()
        sandbox = MemorySandbox({"app.py": "value = 1\n"})
        sandbox.create()
        tools = CodingToolRuntime(sandbox, ToolPolicy(["ask_user"]))
        simulator = RuleBasedUserSimulator(instance)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conversation.jsonl"
            with TraceRecorder(path, session_id="session_user", scenario_instance=instance) as recorder:
                result = HeadlessAgent(AskThenFinish(), tools).run(
                    instance,
                    recorder,
                    ask_user=user_callback(simulator, instance),
                )
            text = path.read_text(encoding="utf-8")
            trace = load_trace(path)

        self.assertIn("target value is 2", result.final_answer)
        self.assertIn('"content": "2"', text)
        self.assertNotIn("EVALUATOR_CANARY", text)
        self.assertGreaterEqual(len(trace.events), 7)

    def test_llm_simulator_prompt_excludes_evaluator_and_agent_prompt(self) -> None:
        class CaptureClient:
            model = "user-model"

            def __init__(self):
                self.messages = []

            def complete(self, messages, **kwargs):
                del kwargs
                self.messages = messages
                return LLMResponse(
                    Message(
                        "assistant",
                        json.dumps(
                            {"content": "2", "action": "clarify", "reason": "known fact"}
                        ),
                    ),
                    self.model,
                )

        client = CaptureClient()
        instance = _instance()
        simulator = LLMUserSimulator(client, instance)
        simulator.respond(UserObservation("What is the target value?", "Fix the value.", 1))
        prompt = repr([message.to_api_dict() for message in client.messages])
        self.assertNotIn("EVALUATOR_CANARY", prompt)
        self.assertNotIn("headless coding agent", prompt)


def _instance() -> ScenarioInstance:
    scenario = Scenario(
        QuerySeed(
            PublicTaskContext("Fix the configured value."),
            hidden_user=HiddenUserContext(
                goal="Help configure the requested public behavior.",
                known_facts={"target_value": 2},
                unavailable_facts=["deployment password"],
                patience_turns=3,
            ),
        ),
        EnvironmentSpec(name="user-fixture", version="1"),
        HiddenEvaluatorContext(reference_answer="EVALUATOR_CANARY"),
    )
    return ScenarioInstance.materialize(scenario, random_seed=1, initial_state_hash="state")


if __name__ == "__main__":
    unittest.main()
