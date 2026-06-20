import unittest

from easy_agentic_data.exporters import preference_to_training, trajectory_to_sft
from easy_agentic_data.models import Message, PreferencePair, Task, Trajectory


class ModelTests(unittest.TestCase):
    def test_task_id_is_stable(self) -> None:
        left = Task(instruction="Do the thing", category="test")
        right = Task(instruction="Do the thing", category="test")
        self.assertEqual(left.task_id, right.task_id)

    def test_trajectory_id_depends_on_messages(self) -> None:
        task = Task(instruction="Answer")
        left = Trajectory(task, [Message("user", "A"), Message("assistant", "B")])
        right = Trajectory(task, [Message("user", "A"), Message("assistant", "C")])
        self.assertNotEqual(left.trajectory_id, right.trajectory_id)

    def test_trajectory_id_depends_on_assistant_reasoning(self) -> None:
        task = Task(instruction="Answer")
        left = Trajectory(
            task,
            [Message("user", "A"), Message("assistant", "B", reasoning_content="first")],
        )
        right = Trajectory(
            task,
            [Message("user", "A"), Message("assistant", "B", reasoning_content="second")],
        )

        self.assertNotEqual(left.trajectory_id, right.trajectory_id)

    def test_assistant_reasoning_is_preserved_in_training_records(self) -> None:
        task = Task(instruction="Answer")
        chosen = Trajectory(
            task,
            [
                Message("user", "Prompt", reasoning_content="ignored user reasoning"),
                Message("assistant", "B", reasoning_content="chosen reasoning"),
            ],
        )
        rejected = Trajectory(
            task,
            [Message("assistant", "C", reasoning_content="rejected reasoning")],
        )
        pair = PreferencePair(task, chosen, rejected, margin=1.0)

        serialized = chosen.to_dict()
        self.assertNotIn("reasoning_content", serialized["messages"][0])
        self.assertEqual(serialized["messages"][1]["reasoning_content"], "chosen reasoning")
        sft = trajectory_to_sft(chosen)
        self.assertEqual(sft["messages"][1]["reasoning_content"], "chosen reasoning")
        serialized_pair = pair.to_dict()
        self.assertEqual(
            serialized_pair["chosen"]["messages"][1]["reasoning_content"],
            "chosen reasoning",
        )
        self.assertEqual(
            serialized_pair["rejected"]["messages"][0]["reasoning_content"],
            "rejected reasoning",
        )
        preference = preference_to_training(pair)
        self.assertEqual(preference["chosen"][0]["reasoning_content"], "chosen reasoning")
        self.assertEqual(preference["rejected"][0]["reasoning_content"], "rejected reasoning")


if __name__ == "__main__":
    unittest.main()
