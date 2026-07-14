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
from easy_agentic_data.traces.events import TRACE_SCHEMA_VERSION

SYSTEM_PROMPT = "You are the trace-contract test agent."


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
            self.assertEqual(
                [message["role"] for message in result.state.messages],
                ["system", "user", "assistant", "tool"],
            )
            self.assertEqual(result.state.messages[0]["content"], SYSTEM_PROMPT)
            self.assertEqual(result.state.messages[-1]["tool_call_id"], "call_1")
            self.assertEqual(result.event_count, 12)
            self.assertEqual(trace.trace_id, loaded_again.trace_id)
            self.assertEqual(artifacts.read_text(stdout), "1 test passed\n")

    def test_loader_recovers_complete_events_before_partial_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.jsonl"
            instance = _instance()
            with TraceRecorder(path, session_id="session_truncated") as recorder:
                recorder.start(instance, system_prompt=SYSTEM_PROMPT)
                recorder.record(
                    EventType.USER_MESSAGE,
                    {"message_id": "user_1", "content": instance.public_task.query},
                )
            with path.open("ab") as handle:
                handle.write(b'{"schema_version": 1, "event_type": "model_')

            trace = load_trace(path)

            self.assertTrue(trace.truncated)
            self.assertEqual(len(trace.events), 3)
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

    def test_trace_v2_binds_system_message_and_rejects_legacy_v1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.jsonl"
            instance = _instance()
            with TraceRecorder(path, session_id="session_v2") as recorder:
                recorder.start(instance, system_prompt=SYSTEM_PROMPT)
            trace = load_trace(path)
            raw_events = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(TRACE_SCHEMA_VERSION, 2)
            self.assertTrue(all(item["schema_version"] == 2 for item in raw_events))
            self.assertEqual(trace.events[1].event_type, EventType.SYSTEM_MESSAGE)
            self.assertEqual(trace.events[1].payload["content"], SYSTEM_PROMPT)

            raw_events[0]["schema_version"] = 1
            path_v1 = Path(directory) / "legacy-v1.jsonl"
            path_v1.write_text(json.dumps(raw_events[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported trace schema version 1"):
                load_trace(path_v1)

    def test_loader_rejects_event_id_that_does_not_match_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.jsonl"
            event = TraceEvent(
                session_id="session_tampered",
                sequence=0,
                event_type=EventType.SESSION_STARTED,
                payload={
                    "scenario_instance_id": "instance_original",
                    "initial_state_hash": "state_initial",
                },
            ).to_dict()
            event["payload"]["scenario_instance_id"] = "instance_tampered"
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "event_id does not match"):
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
                recorder.start(instance, system_prompt=SYSTEM_PROMPT)
                with self.assertRaisesRegex(ValueError, "hidden context"):
                    recorder.record(
                        EventType.USER_MESSAGE,
                        {
                            "message_id": "user_2",
                            "content": "Leaked USER_CANARY_12345",
                        },
                    )
                with self.assertRaisesRegex(ValueError, "hidden context"):
                    recorder.record(
                        EventType.USER_MESSAGE,
                        {
                            "message_id": "user_3",
                            "content": "Leaked REQUIRED_STATE_CANARY_24680",
                        },
                    )

            trace = load_trace(path)
            self.assertEqual(len(trace.events), 2)

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

        for field_name in (
            "test_patch",
            "reference_artifacts",
            "required_state",
            "forbidden_state",
            "rubric",
            "trace_quality_rubric",
            "evaluator_payload",
            "Evaluation-Payload",
        ):
            with self.subTest(field_name=field_name), self.assertRaisesRegex(
                ValueError, "hidden field"
            ):
                TraceEvent(
                    session_id="session_private_field",
                    sequence=0,
                    event_type=EventType.SESSION_STARTED,
                    payload={
                        "scenario_instance_id": "instance_private_field",
                        "initial_state_hash": "state_initial",
                        "metadata": {field_name: "private evaluator material"},
                    },
                )

        public_projection = TraceEvent(
            session_id="session_public_projection",
            sequence=0,
            event_type=EventType.VERIFICATION_RESULT,
            payload={
                "verifier": "required_state",
                "passed": True,
                "score": 1.0,
                "reason": "Evaluator passed",
                "evidence": {"evidence_sha256": "0" * 64, "field_count": 1},
            },
        )
        self.assertEqual(public_projection.payload["verifier"], "required_state")

    def test_tool_message_causality_fails_closed(self) -> None:
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        model = (
            EventType.MODEL_RESPONSE,
            {"message_id": "assistant_0", "content": None, "tool_calls": [tool_call]},
        )
        tool = (
            EventType.TOOL_MESSAGE,
            {
                "message_id": "tool_0",
                "name": "read_file",
                "tool_call_id": "call_1",
                "content": "{}",
            },
        )
        finish = (
            EventType.SESSION_FINISHED,
            {
                "termination_reason": TerminationReason.AGENT_STOP.value,
                "final_state_hash": "state_initial",
                "success": False,
            },
        )
        cases = {
            "orphan": (
                [tool],
                "Orphan tool_message",
            ),
            "duplicate result": (
                [model, tool, tool],
                "Duplicate tool_message",
            ),
            "name mismatch": (
                [
                    model,
                    (
                        EventType.TOOL_MESSAGE,
                        {
                            "message_id": "tool_0",
                            "name": "run_command",
                            "tool_call_id": "call_1",
                            "content": "{}",
                        },
                    ),
                ],
                "name does not match",
            ),
            "missing result": (
                [model, finish],
                "session_finished cannot occur",
            ),
            "next model before result": (
                [
                    model,
                    (
                        EventType.MODEL_RESPONSE,
                        {"message_id": "assistant_1", "content": "done", "tool_calls": []},
                    ),
                ],
                "model_response cannot occur",
            ),
            "reused assistant call id": (
                [
                    model,
                    tool,
                    (
                        EventType.MODEL_RESPONSE,
                        {
                            "message_id": "assistant_1",
                            "content": None,
                            "tool_calls": [tool_call],
                        },
                    ),
                ],
                "Duplicate assistant tool call id",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for case_name, (events, expected_error) in cases.items():
                with self.subTest(case_name=case_name):
                    path = Path(directory) / f"{case_name.replace(' ', '-')}.jsonl"
                    with TraceRecorder(path, session_id=f"session_{case_name}") as recorder:
                        recorder.start(_instance(), system_prompt=SYSTEM_PROMPT)
                        for event_type, payload in events:
                            recorder.record(event_type, payload)
                    with self.assertRaisesRegex(ValueError, expected_error):
                        load_trace(path, tolerate_truncated=False)

    def test_duplicate_ids_inside_one_assistant_message_are_rejected(self) -> None:
        call = {
            "id": "call_duplicate",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
        with tempfile.TemporaryDirectory() as directory:
            with TraceRecorder(
                Path(directory) / "duplicate-call.jsonl",
                session_id="session_duplicate_call",
            ) as recorder:
                recorder.start(_instance(), system_prompt=SYSTEM_PROMPT)
                with self.assertRaisesRegex(ValueError, "Duplicate assistant tool call id"):
                    recorder.record(
                        EventType.MODEL_RESPONSE,
                        {
                            "message_id": "assistant_0",
                            "content": None,
                            "tool_calls": [call, call],
                        },
                    )

    def test_tool_message_schema_requires_complete_string_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing payload fields.*name"):
            TraceEvent(
                session_id="session_missing_tool_name",
                sequence=0,
                event_type=EventType.TOOL_MESSAGE,
                payload={
                    "message_id": "tool_0",
                    "tool_call_id": "call_0",
                    "content": "{}",
                },
            )
        with self.assertRaisesRegex(ValueError, "content must be a string"):
            TraceEvent(
                session_id="session_invalid_tool_content",
                sequence=0,
                event_type=EventType.TOOL_MESSAGE,
                payload={
                    "message_id": "tool_0",
                    "name": "read_file",
                    "tool_call_id": "call_0",
                    "content": {"ok": True},
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
            self.assertEqual(payload["event_count"], 12)


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
            required_state={
                "file_contains": {"src/parser.py": "REQUIRED_STATE_CANARY_24680"}
            },
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
        recorder.start(instance, system_prompt=SYSTEM_PROMPT)
        recorder.record(
            EventType.USER_MESSAGE,
            {"message_id": "user_1", "content": instance.public_task.query},
        )
        recorder.record(
            EventType.MODEL_RESPONSE,
            {
                "message_id": "assistant_1",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "run_command", "arguments": "{}"},
                    }
                ],
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
            EventType.TOOL_MESSAGE,
            {
                "message_id": "tool_1",
                "name": "run_command",
                "tool_call_id": "call_1",
                "content": json.dumps({"ok": True, "output": "1 test passed\n"}),
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
