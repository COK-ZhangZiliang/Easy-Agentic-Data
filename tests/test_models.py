import unittest

from easy_agentic_data.models import Message, Task, Trajectory


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


if __name__ == "__main__":
    unittest.main()
