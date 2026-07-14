import copy
import json
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.agent import DEFAULT_SYSTEM_PROMPT, HeadlessAgent
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.evaluation import (
    EvaluationEvidence,
    EvaluationReport,
    EvaluationSuite,
    HiddenCommandEvaluator,
    HiddenTestPatchEvaluator,
    RequiredStateEvaluator,
    TurnRewardEvidence,
    apply_agent_termination,
    contamination_findings,
    derive_turn_rewards,
    evaluation_result_metrics,
    finalize_evaluation_trace,
    pass_at_k,
    rank_reports,
    trace_policy_evidence,
    turn_reward_metrics,
    workspace_summary,
)
from easy_agentic_data.models import LLMResponse, Message
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import CommandResult, MemorySandbox
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed
from easy_agentic_data.trace_exporters import (
    analysis_record,
    trace_to_rl_episode,
    trace_to_sft,
    traces_to_preference,
)
from easy_agentic_data.traces import TraceRecorder, load_trace
from easy_agentic_data.traces.events import EventType, TerminationReason


class EvaluationExportTests(unittest.TestCase):
    def test_contamination_audit_detects_json_escaped_multiline_hidden_value(self) -> None:
        hidden_value = 'alpha line\n"quoted beta"\\gamma'
        scenario = Scenario(
            QuerySeed(PublicTaskContext("Inspect the workspace.")),
            EnvironmentSpec(name="contamination-fixture", version="1"),
            HiddenEvaluatorContext(
                required_state={"file_equals": {"private.txt": hidden_value}}
            ),
        )
        instance = ScenarioInstance.materialize(
            scenario,
            random_seed=17,
            initial_state_hash="initial",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "escaped-leak.jsonl"
            with TraceRecorder(
                path,
                session_id="session_escaped_leak",
                scenario_instance=instance,
            ) as recorder:
                recorder.start(instance, system_prompt=DEFAULT_SYSTEM_PROMPT)
                recorder.record(
                    EventType.USER_MESSAGE,
                    {"message_id": "message_leak", "content": hidden_value},
                )
                recorder.record(
                    EventType.SESSION_FINISHED,
                    {
                        "termination_reason": TerminationReason.AGENT_STOP.value,
                        "final_state_hash": "initial",
                        "success": False,
                    },
                )

            findings = contamination_findings(path, instance)

        self.assertEqual(findings, [hidden_value])

    def test_deterministic_evaluation_finalize_and_exports(self) -> None:
        sandbox = _sandbox()
        instance = _instance(sandbox.state_hash())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "success.jsonl"
            with TraceRecorder(
                path, session_id="session_success", scenario_instance=instance
            ) as recorder:
                HeadlessAgent(_PatchClient(), _tools(sandbox)).run(
                    instance, recorder, finalize=False
                )
                report = EvaluationSuite(
                    [
                        HiddenCommandEvaluator(["python", "-m", "hidden_test"]),
                        RequiredStateEvaluator(),
                    ]
                ).evaluate(sandbox, instance, diagnostics={"turns": 3, "tokens": 30})
                finalize_evaluation_trace(recorder, report, final_state_hash=sandbox.state_hash())
            trace = load_trace(path)

            self.assertTrue(report.success)
            self.assertEqual(report.reward, 1)
            self.assertFalse(contamination_findings(path, instance))
            self.assertTrue(trace_policy_evidence(trace).passed)
            self.assertEqual(workspace_summary(sandbox)["state_hash"], sandbox.state_hash())
            sft = trace_to_sft(trace, report)
            self.assertEqual(sft["trace_id"], trace.trace_id)
            self.assertTrue(
                any(
                    message.get("reasoning_content") == "patch succeeded"
                    for message in sft["messages"]
                )
            )
            self.assertEqual(
                [message["role"] for message in sft["messages"]],
                [
                    "system",
                    "user",
                    "assistant",
                    "tool",
                    "assistant",
                    "tool",
                    "assistant",
                ],
            )
            self.assertEqual(sft["messages"][0]["content"], DEFAULT_SYSTEM_PROMPT)
            sft_tool_messages = [
                message for message in sft["messages"] if message["role"] == "tool"
            ]
            self.assertEqual(
                [message["tool_call_id"] for message in sft_tool_messages],
                ["patch", "test"],
            )
            episode = trace_to_rl_episode(trace, report)
            self.assertEqual(episode["schema"], "easy_agentic_data.rl_episode.v1")
            self.assertEqual(episode["steps"][-1]["reward"], 1)
            self.assertEqual(episode["steps"][0]["loss_mask"], 0)
            self.assertEqual(episode["steps"][0]["role"], "system")
            self.assertEqual(episode["steps"][0]["content"], DEFAULT_SYSTEM_PROMPT)
            self.assertEqual(episode["steps"][-1]["action_type"], "answer")
            self.assertEqual(episode["steps"][-1]["reasoning_content"], "patch succeeded")
            self.assertEqual(
                [step["tool_call_id"] for step in episode["steps"] if step["role"] == "tool"],
                ["patch", "test"],
            )
            self.assertTrue(analysis_record(trace, report)["success"])

    def test_turn_rewards_attach_to_rl_episode_actions(self) -> None:
        sandbox = _sandbox()
        instance = _instance(sandbox.state_hash())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "turn_rewards.jsonl"
            with TraceRecorder(
                path, session_id="turn_rewards", scenario_instance=instance
            ) as recorder:
                HeadlessAgent(_PatchClient(), _tools(sandbox)).run(
                    instance, recorder, finalize=False
                )
                partial_trace = load_trace(path)
                turn_rewards = derive_turn_rewards(partial_trace, instance)
                report = EvaluationSuite([RequiredStateEvaluator()]).evaluate(
                    sandbox,
                    instance,
                    turn_rewards=turn_rewards,
                )
                finalize_evaluation_trace(recorder, report, final_state_hash=sandbox.state_hash())
            trace = load_trace(path)
            episode = trace_to_rl_episode(trace, report)

        self.assertGreater(turn_reward_metrics(turn_rewards)["turn_reward_total"], 0.0)
        self.assertGreater(report.metrics["positive_turn_rewards"], 0)
        self.assertTrue(
            any(step.get("reward_kind") == "tool_execution" for step in episode["steps"])
        )
        self.assertTrue(
            all(step["loss_mask"] == 0 for step in episode["steps"] if step["role"] != "assistant")
        )

    def test_private_metrics_and_turn_reward_details_are_not_exported(self) -> None:
        sandbox = _sandbox()
        instance = _instance(sandbox.state_hash())
        with tempfile.TemporaryDirectory() as directory:
            trace = _minimal_finished_trace(Path(directory) / "private-report.jsonl", instance)
            report = EvaluationReport(
                instance.instance_id,
                [EvaluationEvidence("required_state", True, 1.0, "fixture")],
                True,
                1,
                False,
                {"turns": 1.0, "PRIVATE_METRIC_CANARY_13579": 7.0},
                [
                    TurnRewardEvidence(
                        0,
                        trace.events[0].event_id,
                        "PRIVATE_KIND_CANARY_24680",
                        "PRIVATE_ACTION_CANARY_11223",
                        0.25,
                        "PRIVATE_REASON_CANARY_44556",
                        {"secret": "PRIVATE_EVIDENCE_CANARY_77889"},
                    )
                ],
            )

            analysis = analysis_record(trace, report)
            episode = trace_to_rl_episode(trace, report)
            encoded = json.dumps({"analysis": analysis, "episode": episode}, sort_keys=True)

        for private_value in (
            "PRIVATE_METRIC_CANARY_13579",
            "PRIVATE_KIND_CANARY_24680",
            "PRIVATE_ACTION_CANARY_11223",
            "PRIVATE_REASON_CANARY_44556",
            "PRIVATE_EVIDENCE_CANARY_77889",
        ):
            self.assertNotIn(private_value, encoded)
        self.assertEqual(analysis["metrics"], {"turns": 1.0})
        self.assertEqual(analysis["omitted_metric_count"], 1)
        self.assertRegex(analysis["metrics_sha256"], r"^[0-9a-f]{64}$")
        public_reward = episode["rewards"]["turn"][0]
        self.assertEqual(public_reward["kind"], "other")
        self.assertEqual(public_reward["reason"], "Positive turn reward")
        self.assertRegex(public_reward["reason_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(public_reward["evidence_sha256"], r"^[0-9a-f]{64}$")

    def test_evaluation_report_from_dict_rejects_coercions_and_invalid_json(self) -> None:
        valid = {
            "scenario_instance_id": "instance_valid",
            "results": [
                {
                    "evaluator": "required_state",
                    "passed": True,
                    "score": 1.0,
                    "reason": "passed",
                    "evidence": {"nested": [1, True, None]},
                    "infrastructure_failure": False,
                }
            ],
            "success": True,
            "reward": 1,
            "infrastructure_failure": False,
            "metrics": {"turns": 1.0},
            "turn_rewards": [
                {
                    "turn_index": 0,
                    "event_id": "event_valid",
                    "kind": "tool_execution",
                    "action_type": "run_command",
                    "reward": 0.1,
                    "reason": "completed",
                    "evidence": {"status": "completed"},
                }
            ],
        }
        restored = EvaluationReport.from_dict(valid)
        self.assertTrue(restored.success)
        self.assertEqual(restored.metrics, {"turns": 1.0})

        invalid_values = []
        for name, mutation in (
            ("report string boolean", lambda value: value.update(success="false")),
            (
                "result string boolean",
                lambda value: value["results"][0].update(passed="false"),
            ),
            (
                "infrastructure string boolean",
                lambda value: value.update(infrastructure_failure="false"),
            ),
            ("fractional outcome reward", lambda value: value.update(reward=0.5)),
            ("non-finite score", lambda value: value["results"][0].update(score=float("nan"))),
            ("non-object evidence", lambda value: value["results"][0].update(evidence=[])),
            (
                "non-finite nested evidence",
                lambda value: value["results"][0].update(evidence={"value": float("inf")}),
            ),
            ("non-object metrics", lambda value: value.update(metrics=[])),
            ("non-finite metric", lambda value: value.update(metrics={"turns": float("inf")})),
            (
                "non-object result",
                lambda value: value.update(results=[["not", "an", "object"]]),
            ),
            (
                "non-object turn reward evidence",
                lambda value: value["turn_rewards"][0].update(evidence=[]),
            ),
            (
                "non-finite turn reward",
                lambda value: value["turn_rewards"][0].update(reward=float("nan")),
            ),
        ):
            candidate = copy.deepcopy(valid)
            mutation(candidate)
            invalid_values.append((name, candidate))

        for name, candidate in invalid_values:
            with self.subTest(name=name), self.assertRaises(ValueError):
                EvaluationReport.from_dict(candidate)

    def test_preference_requires_positive_deterministic_margin(self) -> None:
        successful_sandbox = _sandbox()
        instance = _instance(successful_sandbox.state_hash())
        with tempfile.TemporaryDirectory() as directory:
            chosen_path = Path(directory) / "chosen.jsonl"
            with TraceRecorder(
                chosen_path, session_id="chosen", scenario_instance=instance
            ) as recorder:
                HeadlessAgent(_PatchClient(), _tools(successful_sandbox)).run(
                    instance, recorder, finalize=False
                )
                chosen_report = EvaluationSuite([RequiredStateEvaluator()]).evaluate(
                    successful_sandbox, instance
                )
                finalize_evaluation_trace(
                    recorder, chosen_report, final_state_hash=successful_sandbox.state_hash()
                )

            failed_sandbox = _sandbox()
            failed_path = Path(directory) / "failed.jsonl"
            with TraceRecorder(
                failed_path, session_id="failed", scenario_instance=instance
            ) as recorder:
                recorder.start(instance, system_prompt=DEFAULT_SYSTEM_PROMPT)
                failed_report = EvaluationSuite([RequiredStateEvaluator()]).evaluate(
                    failed_sandbox, instance
                )
                finalize_evaluation_trace(
                    recorder, failed_report, final_state_hash=failed_sandbox.state_hash()
                )
            pair = traces_to_preference(
                load_trace(chosen_path), chosen_report, load_trace(failed_path), failed_report
            )
            self.assertEqual(pair["margin"], 1)
            self.assertEqual(pair["scenario_id"], instance.scenario_id)
            self.assertEqual(
                [message["role"] for message in pair["chosen"]],
                [
                    "system",
                    "user",
                    "assistant",
                    "tool",
                    "assistant",
                    "tool",
                    "assistant",
                ],
            )
            self.assertEqual(
                pair["chosen_scenario_instance_id"], chosen_report.scenario_instance_id
            )
            self.assertEqual(
                pair["rejected_scenario_instance_id"], failed_report.scenario_instance_id
            )
            self.assertEqual(pass_at_k([1, 0, 0])["success_rate"], 1 / 3)

    def test_preference_rejects_cross_scenario_and_infrastructure_results(self) -> None:
        chosen_sandbox = _sandbox()
        chosen_sandbox.write("app.py", "value = 2\n")
        chosen_instance = _instance(chosen_sandbox.state_hash())
        rejected_sandbox = _sandbox()
        other_instance = _instance(rejected_sandbox.state_hash(), scenario_id="scenario_other")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chosen_trace = _minimal_finished_trace(root / "chosen.jsonl", chosen_instance)
            other_trace = _minimal_finished_trace(
                root / "other.jsonl", other_instance, success=False
            )
            chosen_report = EvaluationReport(
                chosen_instance.instance_id,
                [EvaluationEvidence("required_state", True, 1.0, "passed")],
                True,
                1,
                False,
                {},
            )
            rejected_report = EvaluationReport(
                other_instance.instance_id,
                [EvaluationEvidence("required_state", False, 0.0, "failed")],
                False,
                0,
                False,
                {},
            )

            with self.assertRaisesRegex(ValueError, "same scenario"):
                traces_to_preference(
                    chosen_trace,
                    chosen_report,
                    other_trace,
                    rejected_report,
                )

            same_scenario_trace = _minimal_finished_trace(
                root / "same-scenario.jsonl", chosen_instance, success=False
            )
            infrastructure_report = EvaluationReport(
                chosen_instance.instance_id,
                [
                    EvaluationEvidence(
                        "hidden_command",
                        False,
                        0.0,
                        "worker unavailable",
                        infrastructure_failure=True,
                    )
                ],
                False,
                0,
                True,
                {},
            )
            with self.assertRaisesRegex(ValueError, "infrastructure"):
                traces_to_preference(
                    chosen_trace,
                    chosen_report,
                    same_scenario_trace,
                    infrastructure_report,
                )

            with self.assertRaisesRegex(ValueError, "distinct traces"):
                traces_to_preference(
                    chosen_trace,
                    chosen_report,
                    chosen_trace,
                    rejected_report,
                )

    def test_sft_requires_consistent_non_agent_hard_verification(self) -> None:
        sandbox = _sandbox()
        instance = _instance(sandbox.state_hash())
        with tempfile.TemporaryDirectory() as directory:
            trace = _minimal_finished_trace(Path(directory) / "success.jsonl", instance)
            agent_only = EvaluationReport(
                instance.instance_id,
                [EvaluationEvidence("agent_termination", True, 1.0, "normal")],
                True,
                1,
                False,
                {},
            )
            inconsistent = EvaluationReport(
                instance.instance_id,
                [EvaluationEvidence("hidden_command", False, 0.0, "failed")],
                True,
                1,
                False,
                {},
            )
            infrastructure = EvaluationReport(
                instance.instance_id,
                [
                    EvaluationEvidence(
                        "hidden_command",
                        True,
                        1.0,
                        "passed",
                        infrastructure_failure=True,
                    )
                ],
                True,
                1,
                True,
                {},
            )
            fabricated_pass = EvaluationReport(
                instance.instance_id,
                [EvaluationEvidence("hidden_command", True, 1.0, "fabricated")],
                True,
                1,
                False,
                {},
            )
            patch_application_only = EvaluationReport(
                instance.instance_id,
                [EvaluationEvidence("hidden_test_patch", True, 1.0, "applied")],
                True,
                1,
                False,
                {},
            )

            with self.assertRaisesRegex(ValueError, "hard verifier"):
                trace_to_sft(trace, agent_only)
            with self.assertRaisesRegex(ValueError, "hard verifier"):
                trace_to_sft(trace, inconsistent)
            with self.assertRaisesRegex(ValueError, "infrastructure"):
                trace_to_sft(trace, infrastructure)
            with self.assertRaisesRegex(ValueError, "verification evidence"):
                trace_to_sft(trace, fabricated_pass)
            with self.assertRaisesRegex(ValueError, "hard verifier"):
                trace_to_sft(trace, patch_application_only)
            with self.assertRaisesRegex(ValueError, "infrastructure"):
                trace_to_rl_episode(trace, infrastructure)

    def test_public_verification_and_analysis_hash_private_evidence(self) -> None:
        private_stdout = "/private/evaluator/tests/test_hidden_oracle.py passed\n"
        private_stderr = "expected SECRET_HIDDEN_VALUE\n"
        sandbox = MemorySandbox(
            {"app.py": "value = 1\n"},
            {
                "python -m hidden_test": lambda box: CommandResult(
                    0,
                    private_stdout,
                    private_stderr,
                    1.0,
                )
            },
        )
        sandbox.create()
        instance = _instance(sandbox.state_hash())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "public.jsonl"
            with TraceRecorder(
                path, session_id="public_evidence", scenario_instance=instance
            ) as recorder:
                recorder.start(instance, system_prompt=DEFAULT_SYSTEM_PROMPT)
                report = EvaluationSuite(
                    [HiddenCommandEvaluator(["python", "-m", "hidden_test"])]
                ).evaluate(sandbox, instance)
                report.results.append(
                    EvaluationEvidence(
                        "required_state",
                        True,
                        1.0,
                        "Checked /private/evaluator/expected-state.json",
                        {"path": "/private/evaluator/expected-state.json"},
                    )
                )
                report.results.append(
                    EvaluationEvidence(
                        "PRIVATE_EVALUATOR_CANARY_99001",
                        True,
                        1.0,
                        "Private custom verifier passed",
                    )
                )
                finalize_evaluation_trace(recorder, report, final_state_hash=sandbox.state_hash())
            trace = load_trace(path)
            raw_trace = path.read_text(encoding="utf-8")
            verification = next(
                event for event in trace.events if event.event_type.value == "verification_result"
            )
            analysis = analysis_record(trace, report)
            raw_analysis = json.dumps(analysis, sort_keys=True)

        for private_value in (
            private_stdout,
            private_stderr,
            "/private/evaluator",
            "PRIVATE_EVALUATOR_CANARY_99001",
        ):
            self.assertNotIn(private_value, raw_trace)
            self.assertNotIn(private_value, raw_analysis)
        self.assertEqual(verification.payload["evidence"]["exit_code"], 0)
        self.assertEqual(verification.payload["evidence"]["stdout_bytes"], len(private_stdout))
        self.assertEqual(verification.payload["evidence"]["stderr_bytes"], len(private_stderr))
        self.assertRegex(verification.payload["evidence"]["evidence_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(verification.payload["evidence"]["stdout_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(verification.payload["evidence"]["stderr_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(analysis["results"][2]["evaluator"], r"^custom_evaluator_[0-9a-f]{16}$")
        self.assertNotIn("stdout", analysis["results"][0]["evidence"])
        self.assertNotIn("stderr", analysis["results"][0]["evidence"])

    def test_infrastructure_failure_is_not_task_failure_reward(self) -> None:
        class BrokenSandbox(MemorySandbox):
            def execute(self, command, *, timeout_seconds=None):
                del command, timeout_seconds
                raise RuntimeError("worker unavailable")

        sandbox = BrokenSandbox({"app.py": "value = 1\n"})
        sandbox.create()
        report = EvaluationSuite([HiddenCommandEvaluator(["hidden-test"])]).evaluate(
            sandbox, _instance(sandbox.state_hash())
        )
        self.assertTrue(report.infrastructure_failure)
        self.assertEqual(report.reward, 0)

    def test_hidden_test_patch_is_applied_before_hidden_commands(self) -> None:
        sandbox = MemorySandbox(
            {"app.py": "value = 2\n"},
            {
                "git apply .ead_hidden_test.patch": _apply_hidden_test_patch,
                "python -m hidden_test": lambda box: CommandResult(
                    0 if "assert value == 2" in box.read("tests/test_hidden.py") else 1,
                    "hidden\n",
                    "",
                    1.0,
                ),
            },
        )
        sandbox.create()
        scenario = Scenario(
            QuerySeed(PublicTaskContext("Set the value correctly.")),
            EnvironmentSpec(name="evaluation-fixture", version="1"),
            HiddenEvaluatorContext(
                hidden_tests=["python -m hidden_test"],
                metadata={
                    "test_patch": (
                        "diff --git a/tests/test_hidden.py b/tests/test_hidden.py\n"
                        "new file mode 100644\n"
                        "--- /dev/null\n"
                        "+++ b/tests/test_hidden.py\n"
                        "@@\n"
                        "+assert value == 2\n"
                    )
                },
            ),
        )
        instance = ScenarioInstance.materialize(
            scenario, random_seed=2, initial_state_hash=sandbox.state_hash()
        )

        report = EvaluationSuite(
            [HiddenTestPatchEvaluator(), HiddenCommandEvaluator(["python", "-m", "hidden_test"])]
        ).evaluate(sandbox, instance)

        self.assertTrue(report.success)
        self.assertEqual(
            [result.evaluator for result in report.results],
            ["hidden_test_patch", "hidden_command"],
        )

    def test_hidden_test_patch_evidence_redacts_patch_text(self) -> None:
        patch = (
            "diff --git a/tests/test_hidden.py b/tests/test_hidden.py\n"
            "--- /dev/null\n"
            "+++ b/tests/test_hidden.py\n"
            "@@\n"
            "+assert value == 2\n"
        )
        sandbox = MemorySandbox(
            {"app.py": "value = 1\n"},
            {
                "git apply .ead_hidden_test.patch": lambda box: CommandResult(
                    1, "", box.read(".ead_hidden_test.patch"), 1.0
                )
            },
        )
        sandbox.create()
        scenario = Scenario(
            QuerySeed(PublicTaskContext("Set the value correctly.")),
            EnvironmentSpec(name="evaluation-fixture", version="1"),
            HiddenEvaluatorContext(hidden_tests=[], metadata={"test_patch": patch}),
        )
        instance = ScenarioInstance.materialize(
            scenario, random_seed=3, initial_state_hash=sandbox.state_hash()
        )

        result = HiddenTestPatchEvaluator().evaluate(sandbox, instance)

        self.assertFalse(result.passed)
        self.assertFalse(result.infrastructure_failure)
        self.assertNotIn(patch, result.evidence["stderr"])
        self.assertIn("[redacted hidden context]", result.evidence["stderr"])

        report = EvaluationSuite([HiddenTestPatchEvaluator()]).evaluate(sandbox, instance)

        self.assertFalse(report.success)
        self.assertEqual(report.reward, 0)
        self.assertFalse(report.infrastructure_failure)

    def test_evaluation_result_metrics_summarize_verifier_passes(self) -> None:
        sandbox = _sandbox()
        sandbox.write("app.py", "value = 2\n")
        instance = _instance(sandbox.state_hash())

        report = EvaluationSuite(
            [
                HiddenCommandEvaluator(["python", "-m", "hidden_test"]),
                RequiredStateEvaluator(),
            ]
        ).evaluate(sandbox, instance)
        metrics = evaluation_result_metrics(report)

        self.assertEqual(metrics["verifier_hidden_command_passed"], 1.0)
        self.assertEqual(metrics["verifier_required_state_passed"], 1.0)
        self.assertEqual(metrics["verifier_all_non_agent_passed"], 1.0)

    def test_diagnostics_only_break_ties(self) -> None:
        sandbox = _sandbox()
        instance = _instance(sandbox.state_hash())
        suite = EvaluationSuite([RequiredStateEvaluator()])
        failed = suite.evaluate(sandbox, instance, diagnostics={"turns": 1})
        sandbox.write("app.py", "value = 2\n")
        slow_success = suite.evaluate(sandbox, instance, diagnostics={"turns": 5})
        fast_success = suite.evaluate(sandbox, instance, diagnostics={"turns": 2})
        ranked = rank_reports([failed, slow_success, fast_success])
        self.assertEqual([item.reward for item in ranked], [1, 1, 0])
        self.assertEqual(ranked[0].metrics["turns"], 2)

    def test_agent_termination_gate_prevents_budget_success(self) -> None:
        sandbox = _sandbox()
        instance = _instance(sandbox.state_hash())
        report = EvaluationSuite([RequiredStateEvaluator()]).evaluate(sandbox, instance)
        self.assertFalse(report.success)

        sandbox.write("app.py", "value = 2\n")
        successful = EvaluationSuite([RequiredStateEvaluator()]).evaluate(sandbox, instance)
        gated = apply_agent_termination(successful, TerminationReason.TOKEN_BUDGET)
        preserved = apply_agent_termination(successful, TerminationReason.AGENT_STOP)

        self.assertTrue(successful.success)
        self.assertFalse(gated.success)
        self.assertEqual(gated.reward, 0)
        self.assertEqual(gated.results[-1].evaluator, "agent_termination")
        self.assertTrue(preserved.success)


class _PatchClient:
    model = "patch-client"

    def __init__(self):
        self.index = 0

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        script = [
            _call(
                "patch", "apply_patch", {"path": "app.py", "old": "value = 1", "new": "value = 2"}
            ),
            _call("test", "run_command", {"command": ["python", "-m", "visible_test"]}),
            Message("assistant", "Done.", reasoning_content="patch succeeded"),
        ]
        message = script[self.index]
        self.index += 1
        return LLMResponse(message, self.model, {"total_tokens": 10})


def _call(call_id, name, arguments):
    return Message(
        "assistant",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


def _sandbox():
    sandbox = MemorySandbox(
        {"app.py": "value = 1\n", "protected.txt": "keep\n"},
        {
            "python -m visible_test": lambda box: CommandResult(
                0 if "value = 2" in box.read("app.py") else 1, "visible\n", "", 1.0
            ),
            "python -m hidden_test": lambda box: CommandResult(
                0 if "value = 2" in box.read("app.py") else 1, "hidden\n", "", 1.0
            ),
        },
    )
    sandbox.create()
    return sandbox


def _apply_hidden_test_patch(box):
    patch = box.read(".ead_hidden_test.patch")
    if "+assert value == 2" not in patch:
        return CommandResult(1, "", "missing expected patch content", 1.0)
    box.write("tests/test_hidden.py", "assert value == 2\n")
    return CommandResult(0, "", "", 1.0)


def _tools(sandbox):
    return CodingToolRuntime(sandbox, ToolPolicy(["apply_patch", "run_command"]))


def _minimal_finished_trace(path, instance, *, success=True):
    report = EvaluationReport(
        instance.instance_id,
        [EvaluationEvidence("required_state", success, 1.0 if success else 0.0, "fixture")],
        success,
        1 if success else 0,
        False,
        {},
    )
    with TraceRecorder(
        path,
        session_id=f"session_{path.stem}",
        scenario_instance=instance,
    ) as recorder:
        recorder.start(instance, system_prompt=DEFAULT_SYSTEM_PROMPT)
        finalize_evaluation_trace(recorder, report, final_state_hash=instance.initial_state_hash)
    return load_trace(path)


def _instance(initial_hash, *, scenario_id=""):
    scenario = Scenario(
        QuerySeed(
            PublicTaskContext(
                "Set the value correctly.",
                context={"expected_assignment": "value = 2"},
            )
        ),
        EnvironmentSpec(name="evaluation-fixture", version="1"),
        HiddenEvaluatorContext(
            hidden_tests=["python -m hidden_test"],
            required_state={"file_contains": {"app.py": "value = 2"}},
            forbidden_state={"file_equals": {"protected.txt": "keep\n"}},
        ),
        scenario_id=scenario_id,
    )
    return ScenarioInstance.materialize(scenario, random_seed=2, initial_state_hash=initial_hash)


if __name__ == "__main__":
    unittest.main()
