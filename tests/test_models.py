import unittest

from easy_agentic_data.models import Message, stable_id


class ModelTests(unittest.TestCase):
    def test_stable_id_is_order_independent(self) -> None:
        self.assertEqual(
            stable_id("item", {"a": 1, "b": 2}),
            stable_id("item", {"b": 2, "a": 1}),
        )

    def test_training_message_keeps_only_assistant_reasoning(self) -> None:
        user = Message("user", "Prompt", reasoning_content="private user note")
        assistant = Message("assistant", "Answer", reasoning_content="model reasoning")

        self.assertNotIn("reasoning_content", user.to_training_dict())
        self.assertEqual(assistant.to_training_dict()["reasoning_content"], "model reasoning")


if __name__ == "__main__":
    unittest.main()
