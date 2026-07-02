import json
import shlex
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.agent import AgentBudgets, HeadlessAgent
from easy_agentic_data.cli import _deterministic_evaluators
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.evaluation import (
    EvaluationSuite,
    apply_agent_termination,
    derive_turn_rewards,
    finalize_evaluation_trace,
)
from easy_agentic_data.llm.base import LLMResponse
from easy_agentic_data.models import Message
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_sources import import_public_issue_pr_records
from easy_agentic_data.repository_synthetic import generate_repository_synthetic_scenarios
from easy_agentic_data.sandbox import CommandResult, MemorySandbox
from easy_agentic_data.scenario_decontamination import (
    audit_scenario_decontamination,
    scenarios_from_registry,
)
from easy_agentic_data.seed_library import (
    SUPPORTED_TASK_FAMILIES,
    SeedLibraryPolicy,
    audit_seed_library,
)
from easy_agentic_data.traces import TraceRecorder, load_trace, replay_trace

PINNED_IMAGE = "python@sha256:" + ("9" * 64)


class SeedLibraryRolloutTests(unittest.TestCase):
    def test_every_supported_family_has_registry_backed_smoke_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "registry"
            trace_root = Path(directory) / "traces"
            registry = ScenarioRegistry(root)

            import_public_issue_pr_records(
                registry,
                [_bug_repair_record()],
                source_format="public-issue",
                source_name="curated-public-issues",
            )
            generate_repository_synthetic_scenarios(
                registry,
                [_repository_synthesis_spec()],
                source_name="curated-repository-synthetic",
                task_families=sorted(SUPPORTED_TASK_FAMILIES - {"bug_repair"}),
                strict=True,
            )

            scenarios = scenarios_from_registry(registry)
            seeds = [scenario.query_seed for scenario in scenarios]
            self.assertEqual(
                {seed.task_family for seed in seeds},
                SUPPORTED_TASK_FAMILIES,
            )
            seed_audit = audit_seed_library(
                seeds,
                policy=SeedLibraryPolicy(
                    min_train_eligible=len(SUPPORTED_TASK_FAMILIES),
                    required_task_families=sorted(SUPPORTED_TASK_FAMILIES),
                ),
            )
            self.assertTrue(seed_audit.valid, [issue.code for issue in seed_audit.issues])
            scenario_audit = audit_scenario_decontamination(scenarios)
            self.assertTrue(
                scenario_audit.valid,
                [issue.code for issue in scenario_audit.issues],
            )

            outcomes = {}
            for scenario in scenarios:
                report, trace = _run_memory_registry_rollout(
                    registry,
                    scenario.scenario_id,
                    trace_root / f"{scenario.query_seed.task_family}.jsonl",
                )
                outcomes[scenario.query_seed.task_family] = report.success
                self.assertTrue(report.success, scenario.query_seed.task_family)
                self.assertTrue(replay_trace(trace).state.success)

            self.assertEqual(outcomes, {family: True for family in SUPPORTED_TASK_FAMILIES})


class _ScriptedFamilyClient:
    model = "scripted-family-agent"

    def __init__(self, hidden_command: str) -> None:
        self.index = 0
        self.hidden_command = hidden_command

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        script = [
            _tool("read_1", "read_file", {"path": "src/tool/parser.py"}),
            _tool(
                "patch_1",
                "apply_patch",
                {
                    "path": "src/tool/parser.py",
                    "old": "STATE = 'BROKEN'",
                    "new": "STATE = 'DONE'",
                },
            ),
            _tool(
                "test_1",
                "run_command",
                {"command": shlex.split(self.hidden_command)},
            ),
            _tool("diff_1", "git_diff", {}),
            Message(
                "assistant",
                "Completed the task after inspecting src/tool/parser.py and running validation.",
            ),
        ]
        message = script[self.index]
        self.index += 1
        return LLMResponse(message, self.model, {"total_tokens": 10})


def _run_memory_registry_rollout(
    registry: ScenarioRegistry,
    scenario_id: str,
    trace_path: Path,
):
    scenario = registry.get_scenario(scenario_id)
    hidden_commands = scenario.hidden_evaluator.hidden_tests
    if not hidden_commands:
        raise AssertionError(f"scenario has no executable hidden command: {scenario_id}")
    sandbox = MemorySandbox(
        {
            "src/tool/parser.py": "STATE = 'BROKEN'\n",
            "README.md": "Parser docs cite src/tool/parser.py.\n",
        },
        {command: _passes_when_done for command in hidden_commands},
    )
    sandbox.create()
    instance = registry.materialize(
        scenario_id,
        random_seed=7,
        initial_state_hash=sandbox.state_hash(),
    )
    tools = CodingToolRuntime(
        sandbox,
        ToolPolicy(["read_file", "apply_patch", "run_command", "git_diff"]),
    )
    client = _ScriptedFamilyClient(hidden_commands[0])
    with TraceRecorder(
        trace_path,
        session_id=f"session_{scenario_id}",
        scenario_instance=instance,
    ) as recorder:
        run_result = HeadlessAgent(
            client,
            tools,
            budgets=AgentBudgets(max_turns=8),
        ).run(instance, recorder, finalize=False)
        partial_trace = load_trace(trace_path)
        report = EvaluationSuite(
            _deterministic_evaluators(instance, partial_trace)
        ).evaluate(
            sandbox,
            instance,
            diagnostics={
                "turns": float(run_result.turns),
                "tool_calls": float(run_result.tool_calls),
                "tokens": float(run_result.tokens),
            },
            turn_rewards=derive_turn_rewards(partial_trace, instance),
        )
        report = apply_agent_termination(report, run_result.termination_reason)
        finalize_evaluation_trace(
            recorder,
            report,
            final_state_hash=sandbox.state_hash(),
            termination_reason=run_result.termination_reason,
        )
    return report, load_trace(trace_path)


def _passes_when_done(box: MemorySandbox) -> CommandResult:
    passed = "STATE = 'DONE'" in box.read("src/tool/parser.py")
    return CommandResult(
        0 if passed else 1,
        "ok\n" if passed else "",
        "" if passed else "parser state is not done\n",
        0.1,
    )


def _tool(call_id: str, name: str, arguments: dict) -> Message:
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


def _bug_repair_record() -> dict[str, object]:
    return {
        "id": "issue-100",
        "type": "issue",
        "repository": "example/tool",
        "source_uri": "https://github.com/example/tool.git",
        "source_revision": "1" * 40,
        "title": "Fix parser whitespace handling",
        "body": "The parser drops significant whitespace around quoted values.",
        "labels": ["bug", "parser"],
        "license": "MIT",
        "language": "Python",
        "image_digest": PINNED_IMAGE,
        "test_commands": ["python -m pytest tests/test_parser.py::test_whitespace"],
    }


def _repository_synthesis_spec() -> dict[str, object]:
    return {
        "repository": "example/tool",
        "source_uri": "https://github.com/example/tool.git",
        "source_revision": "2" * 40,
        "license": "MIT",
        "language": "Python",
        "image_digest": PINNED_IMAGE,
        "working_directory": "/workspace",
        "setup_commands": ["python -m pip install -e ."],
        "targets": [
            {
                "name": "parser",
                "paths": ["src/tool/parser.py", "tests/test_parser.py"],
                "test_commands": ["python -m pytest tests/test_parser.py"],
                "build_commands": ["python -m build"],
                "ci_commands": ["python -m pytest", "python -m build"],
                "doctest_commands": ["python -m doctest README.md"],
                "example_commands": ["python examples/parser_demo.py"],
                "benchmark_commands": ["python benchmarks/parser_bench.py --max-ms 50"],
                "adversarial_tests": ["python -m pytest tests/security/test_parser.py"],
                "migration_commands": ["python scripts/check_migration.py"],
                "required_state": {"file_contains": {"src/tool/parser.py": "DONE"}},
                "diff_constraints": ["do not rename the public Parser API"],
                "performance_threshold": {"max_ms": 50},
                "retrieval_requirements": ["src/tool/parser.py"],
                "trace_quality_rubric": ["cite src/tool/parser.py in the final answer"],
                "difficulty": 3,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
