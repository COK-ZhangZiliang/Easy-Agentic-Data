import unittest

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

    def test_private_reasoning_is_excluded_from_training_records(self) -> None:
        task = Task(instruction="Answer")
        chosen = Trajectory(
            task,
            [Message("assistant", "B", reasoning_content="private chosen reasoning")],
        )
        rejected = Trajectory(
            task,
            [Message("assistant", "C", reasoning_content="private rejected reasoning")],
        )
        pair = PreferencePair(task, chosen, rejected, margin=1.0)

        self.assertNotIn("reasoning_content", chosen.to_dict()["messages"][0])
        serialized_pair = pair.to_dict()
        self.assertNotIn("reasoning_content", serialized_pair["chosen"]["messages"][0])
        self.assertNotIn("reasoning_content", serialized_pair["rejected"]["messages"][0])


if __name__ == "__main__":
    unittest.main()
