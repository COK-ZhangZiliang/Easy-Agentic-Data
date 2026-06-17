import json
import unittest

from easy_agentic_data.generation import EvolTaskGenerator, SelfInstructTaskGenerator
from easy_agentic_data.models import LLMResponse, Message, Task


class CapturingClient:
    model = "capturing"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        return LLMResponse(
            Message("assistant", json.dumps(self.responses.pop(0))),
            self.model,
        )


class GenerationPromptTests(unittest.TestCase):
    def test_task_generation_uses_json_object_mode(self) -> None:
        client = CapturingClient(
            [
                {
                    "tasks": [
                        {
                            "instruction": "Add 2 and 3.",
                            "category": "calculation",
                            "difficulty": 1,
                            "constraints": [],
                            "expected_tools": ["calculator"],
                        }
                    ]
                }
            ]
        )

        tasks = SelfInstructTaskGenerator(client).generate(1, ["calculation"])

        self.assertEqual(len(tasks), 1)
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        prompt = "\n".join(message.content or "" for message in client.calls[0]["messages"])
        self.assertIn('"tasks"', prompt)
        self.assertIn("exactly the requested number", prompt)

    def test_task_generation_accepts_legacy_top_level_array(self) -> None:
        client = CapturingClient(
            [
                [
                    {
                        "instruction": "Add 2 and 3.",
                        "category": "calculation",
                        "difficulty": 1,
                    }
                ]
            ]
        )

        tasks = SelfInstructTaskGenerator(client).generate(1, ["calculation"])

        self.assertEqual(tasks[0].instruction, "Add 2 and 3.")

    def test_mock_count_marker_allows_sentence_punctuation(self) -> None:
        from easy_agentic_data.llm.mock import MockLLMClient

        tasks = SelfInstructTaskGenerator(MockLLMClient()).generate(
            2,
            ["calculation"],
        )

        self.assertEqual(len(tasks), 2)

    def test_task_generation_rejects_empty_batches(self) -> None:
        client = CapturingClient([{"tasks": []}])

        with self.assertRaisesRegex(ValueError, "empty tasks array"):
            SelfInstructTaskGenerator(client).generate(1, ["calculation"])

    def test_task_generation_rejects_non_object_tasks(self) -> None:
        client = CapturingClient([{"tasks": ["not-an-object"]}])

        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            SelfInstructTaskGenerator(client).generate(1, ["calculation"])

    def test_evolver_uses_json_mode_and_preserves_parent(self) -> None:
        client = CapturingClient(
            [
                {
                    "instruction": "Add 2 and 3, then state both operands.",
                    "category": "calculation",
                    "difficulty": 2,
                    "constraints": ["State both operands."],
                    "expected_tools": ["calculator"],
                }
            ]
        )
        source = Task("Add 2 and 3.", expected_tools=["calculator"])

        evolved = EvolTaskGenerator(client).evolve([source], 1)

        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(evolved[0].metadata["parent_task_id"], source.task_id)


if __name__ == "__main__":
    unittest.main()
