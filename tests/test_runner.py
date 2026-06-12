import unittest

from easy_agentic_data.llm.mock import MockLLMClient
from easy_agentic_data.models import Task
from easy_agentic_data.runner import AgentRunner
from easy_agentic_data.tools import default_tool_registry


class RunnerTests(unittest.TestCase):
    def test_runner_executes_tool_and_finishes(self) -> None:
        task = Task(
            instruction="Use the calculator to add 2 and 4.",
            expected_tools=["calculator"],
        )
        trajectory = AgentRunner(MockLLMClient(), default_tool_registry(), max_turns=3).run(task)

        self.assertEqual(trajectory.status, "completed")
        self.assertEqual(trajectory.tool_events[0].output["result"], 6)
        self.assertIsNotNone(trajectory.messages[-1].content)
        self.assertIn("6", trajectory.messages[-1].content or "")


if __name__ == "__main__":
    unittest.main()
