import json
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.agent import HeadlessAgent
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.evaluation import (
    EvaluationSuite,
    HiddenCommandEvaluator,
    RequiredStateEvaluator,
    contamination_findings,
    derive_turn_rewards,
    finalize_evaluation_trace,
    pass_at_k,
    rank_reports,
    turn_reward_metrics,
    trace_policy_evidence,
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


class EvaluationExportTests(unittest.TestCase):
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
            self.assertEqual(trace_to_sft(trace, report)["trace_id"], trace.trace_id)
            episode = trace_to_rl_episode(trace, report)
            self.assertEqual(episode["schema"], "easy_agentic_data.rl_episode.v1")
            self.assertEqual(episode["steps"][-1]["reward"], 1)
            self.assertEqual(episode["steps"][0]["loss_mask"], 0)
            self.assertEqual(episode["steps"][-1]["action_type"], "answer")
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
            all(
                step["loss_mask"] == 0
                for step in episode["steps"]
                if step["role"] != "assistant"
            )
        )

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
                recorder.start(instance)
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
            self.assertEqual(pass_at_k([1, 0, 0])["success_rate"], 1 / 3)

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
            Message("assistant", "Done."),
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


def _tools(sandbox):
    return CodingToolRuntime(sandbox, ToolPolicy(["apply_patch", "run_command"]))


def _instance(initial_hash):
    scenario = Scenario(
        QuerySeed(PublicTaskContext("Set the value correctly.")),
        EnvironmentSpec(name="evaluation-fixture", version="1"),
        HiddenEvaluatorContext(
            hidden_tests=["python -m hidden_test"],
            required_state={"file_contains": {"app.py": "value = 2"}},
            forbidden_state={"file_equals": {"protected.txt": "keep\n"}},
        ),
    )
    return ScenarioInstance.materialize(scenario, random_seed=2, initial_state_hash=initial_hash)


if __name__ == "__main__":
    unittest.main()
