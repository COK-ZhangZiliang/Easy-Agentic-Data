import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance
from easy_agentic_data.seeds import HiddenUserContext, PublicTaskContext, QuerySeed
from easy_agentic_data.traces import (
    EventType,
    LocalArtifactStore,
    TerminationReason,
    TraceEvent,
    TraceRecorder,
    load_trace,
    replay_trace,
)


class TraceContractTests(unittest.TestCase):
    def test_record_load_and_replay_reaches_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "trace.jsonl"
            instance = _instance()
            artifacts = LocalArtifactStore(root / "artifacts")
            stdout = artifacts.put_text("1 test passed\n")
            patch = artifacts.put_text("diff --git a/parser.py b/parser.py\n")

            _record_complete_trace(trace_path, instance, stdout.to_dict(), patch.to_dict())

            trace = load_trace(trace_path)
            result = replay_trace(trace)
            loaded_again = load_trace(trace_path)

            self.assertEqual(result.terminal_state_hash, "state_final")
            self.assertEqual(result.state.termination_reason, TerminationReason.SUCCESS.value)
            self.assertTrue(result.state.success)
            self.assertEqual(result.state.tool_calls["call_1"]["status"], "completed")
            self.assertEqual(result.event_count, 10)
            self.assertEqual(trace.trace_id, loaded_again.trace_id)
            self.assertEqual(artifacts.read_text(stdout), "1 test passed\n")

    def test_loader_recovers_complete_events_before_partial_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.jsonl"
            instance = _instance()
            with TraceRecorder(path, session_id="session_truncated") as recorder:
                recorder.start(instance)
                recorder.record(
                    EventType.USER_MESSAGE,
                    {"message_id": "user_1", "content": instance.public_task.query},
                )
            with path.open("ab") as handle:
                handle.write(b'{"schema_version": 1, "event_type": "model_')

            trace = load_trace(path)

            self.assertTrue(trace.truncated)
            self.assertEqual(len(trace.events), 2)
            with self.assertRaisesRegex(ValueError, "Invalid trace JSONL"):
                load_trace(path, tolerate_truncated=False)

    def test_unknown_schema_version_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.jsonl"
            event = TraceEvent(
                session_id="session_future",
                sequence=0,
                event_type=EventType.SESSION_STARTED,
                payload={
                    "scenario_instance_id": "instance_future",
                    "initial_state_hash": "state_initial",
                },
            ).to_dict()
            event["schema_version"] = 99
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported trace schema version 99"):
                load_trace(path)

    def test_recorder_rejects_hidden_context_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leak.jsonl"
            instance = _instance()
            with TraceRecorder(
                path,
                session_id="session_leak",
                scenario_instance=instance,
            ) as recorder:
                recorder.start(instance)
                with self.assertRaisesRegex(ValueError, "hidden context"):
                    recorder.record(
                        EventType.USER_MESSAGE,
                        {
                            "message_id": "user_2",
                            "content": "Leaked USER_CANARY_12345",
                        },
                    )

            trace = load_trace(path)
            self.assertEqual(len(trace.events), 1)

    def test_event_schema_rejects_hidden_context_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden field: hidden_tests"):
            TraceEvent(
                session_id="session_hidden_field",
                sequence=0,
                event_type=EventType.SESSION_STARTED,
                payload={
                    "scenario_instance_id": "instance_hidden_field",
                    "initial_state_hash": "state_initial",
                    "metadata": {"hidden_tests": ["do-not-expose"]},
                },
            )

    def test_recorder_never_overwrites_existing_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.jsonl"
            path.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                TraceRecorder(path, session_id="session_existing")

    def test_replay_cli_uses_only_recorded_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "trace.jsonl"
            instance = _instance()
            artifacts = LocalArtifactStore(root / "artifacts")
            stdout = artifacts.put_text("ok\n")
            patch = artifacts.put_text("patch\n")
            _record_complete_trace(path, instance, stdout.to_dict(), patch.to_dict())
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(["replay", "--trace", str(path)])

            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["terminal_state_hash"], "state_final")
            self.assertEqual(payload["event_count"], 10)


def _instance() -> ScenarioInstance:
    seed = QuerySeed(
        public=PublicTaskContext(query="Repair the failing parser test."),
        hidden_user=HiddenUserContext(goal="USER_CANARY_12345"),
        category="software_engineering",
    )
    environment = EnvironmentSpec(name="fixture", version="1")
    scenario = Scenario(
        query_seed=seed,
        environment=environment,
        hidden_evaluator=HiddenEvaluatorContext(
            reference_answer="EVALUATOR_CANARY_67890",
        ),
    )
    return ScenarioInstance.materialize(
        scenario,
        random_seed=7,
        initial_state_hash="state_initial",
    )


def _record_complete_trace(
    path: Path,
    instance: ScenarioInstance,
    stdout_artifact: dict,
    patch_artifact: dict,
) -> None:
    with TraceRecorder(
        path,
        session_id="session_fixture",
        scenario_instance=instance,
    ) as recorder:
        recorder.start(instance)
        recorder.record(
            EventType.USER_MESSAGE,
            {"message_id": "user_1", "content": instance.public_task.query},
        )
        recorder.record(
            EventType.MODEL_RESPONSE,
            {
                "message_id": "assistant_1",
                "content": None,
                "tool_calls": [{"id": "call_1", "name": "run_command"}],
            },
        )
        recorder.record(
            EventType.TOOL_REQUESTED,
            {
                "call_id": "call_1",
                "name": "run_command",
                "arguments": {"command": "python -m unittest"},
            },
        )
        recorder.record(
            EventType.POLICY_DECISION,
            {"call_id": "call_1", "decision": "allow", "reason": "fixture command"},
        )
        recorder.record(
            EventType.TOOL_STARTED,
            {"call_id": "call_1", "name": "run_command"},
        )
        recorder.record(
            EventType.TOOL_FINISHED,
            {
                "call_id": "call_1",
                "status": "completed",
                "exit_code": 0,
                "stdout_artifact": stdout_artifact,
            },
        )
        recorder.record(
            EventType.WORKSPACE_DIFF,
            {
                "before_state_hash": "state_initial",
                "after_state_hash": "state_final",
                "patch_artifact": patch_artifact,
            },
        )
        recorder.record(
            EventType.VERIFICATION_RESULT,
            {
                "verifier": "hidden_tests",
                "passed": True,
                "score": 1.0,
                "reason": "All tests passed",
            },
        )
        recorder.record(
            EventType.SESSION_FINISHED,
            {
                "termination_reason": TerminationReason.SUCCESS.value,
                "final_state_hash": "state_final",
                "success": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
