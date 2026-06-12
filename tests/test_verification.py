import unittest

from easy_agentic_data.models import Message, Task, ToolEvent, Trajectory
from easy_agentic_data.verification import (
    StructuralVerifier,
    ToolExecutionVerifier,
    VerificationSuite,
)


class VerificationTests(unittest.TestCase):
    def test_hard_failure_zeroes_reward(self) -> None:
        task = Task(instruction="Use calculator", expected_tools=["calculator"])
        trajectory = Trajectory(
            task=task,
            messages=[Message("user", "Use calculator"), Message("assistant", "Done")],
            tool_events=[
                ToolEvent(
                    call_id="call",
                    name="calculator",
                    arguments={},
                    error="bad arguments",
                )
            ],
        )
        suite = VerificationSuite([StructuralVerifier(), ToolExecutionVerifier()])

        suite.evaluate(trajectory)

        self.assertEqual(trajectory.reward, 0.0)
        self.assertFalse(trajectory.verifications[-1].passed)

    def test_verifier_exception_becomes_failed_result(self) -> None:
        class BrokenVerifier:
            name = "broken"

            def verify(self, trajectory: Trajectory):
                del trajectory
                raise ValueError("invalid judge response")

        task = Task(instruction="Answer")
        trajectory = Trajectory(
            task=task,
            messages=[Message("user", "Answer"), Message("assistant", "Done")],
        )

        VerificationSuite([BrokenVerifier()]).evaluate(trajectory)

        self.assertEqual(trajectory.reward, 0.0)
        self.assertFalse(trajectory.verifications[0].passed)
        self.assertIn("invalid judge response", trajectory.verifications[0].reason)


if __name__ == "__main__":
    unittest.main()
