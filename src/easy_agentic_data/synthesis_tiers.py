from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from easy_agentic_data.agent import AgentBudgets, HeadlessAgent
from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.evaluation import (
    EvaluationSuite,
    ForbiddenStateEvaluator,
    HiddenCommandEvaluator,
    RequiredStateEvaluator,
    apply_agent_termination,
    derive_turn_rewards,
    finalize_evaluation_trace,
)
from easy_agentic_data.models import LLMResponse, Message
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.sandbox import CommandResult, MemorySandbox
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance
from easy_agentic_data.seeds import HiddenUserContext, PublicTaskContext, QuerySeed
from easy_agentic_data.simulation import RuleBasedUserSimulator, user_callback
from easy_agentic_data.trace_exporters import analysis_record, trace_to_rl_episode, trace_to_sft
from easy_agentic_data.traces import TraceRecorder, load_trace, replay_trace


@dataclass(frozen=True)
class SynthesisTier:
    """Describe a supported synthesis path and the data quality it is meant to prove."""

    tier_id: str
    purpose: str
    runtime: str
    data_shape: str
    verifier_signal: str
    default_command: str
    artifacts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_synthesis_tiers() -> list[SynthesisTier]:
    return [
        SynthesisTier(
            tier_id="complex_synthetic",
            purpose="Exercise multi-turn agent trajectories before using external workspaces.",
            runtime=(
                "HeadlessAgent with MemorySandbox, user simulation, coding tools, and hidden "
                "checks."
            ),
            data_shape=(
                "Repository-like fixture with file reads, search, user clarification, patches, "
                "visible tests, diff inspection, hidden tests, and RL/SFT exports."
            ),
            verifier_signal="Hidden command, required-state, forbidden-state, and replay checks.",
            default_command="ead synthesis complex-demo --output runs/complex-synthetic-demo",
            artifacts=[
                "trace.jsonl",
                "report.json",
                "sft.json",
                "rl_episode.json",
                "analysis.json",
            ],
        ),
        SynthesisTier(
            tier_id="registry_backed",
            purpose="Production-style data from reusable query seeds and reproducible workspaces.",
            runtime="ScenarioRegistry materialization plus Docker-backed HeadlessAgent.",
            data_shape=(
                "Imported or curated query/workspace seeds with hidden evaluator references, "
                "sandboxed coding tools, repeated rollouts, and deterministic evaluation."
            ),
            verifier_signal=(
                "Executable workspace state, hidden tests, policy evidence, and trace replay."
            ),
            default_command=(
                "ead synthesis real-seed-demo --output runs/real-seed-demo "
                "--config examples/deepseek-v4-flash-thinking.json"
            ),
            artifacts=[
                "seed registry",
                "repository cache",
                "agent trace JSONL",
                "evaluation results",
                "SFT/preference/RL derived exports",
            ],
        ),
    ]


def run_complex_synthetic_demo(output: str | Path) -> dict[str, Any]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    trace_path = output_path / "trace.jsonl"
    if trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing trace: {trace_path}")

    sandbox = _complex_sandbox()
    sandbox.create()
    instance = _complex_instance(sandbox.state_hash())
    tools = CodingToolRuntime(
        sandbox,
        ToolPolicy(
            [
                "list_files",
                "read_file",
                "search_files",
                "apply_patch",
                "run_command",
                "git_diff",
                "ask_user",
            ]
        ),
    )
    user = RuleBasedUserSimulator(instance)
    client = _ComplexScriptedClient()
    with TraceRecorder(
        trace_path,
        session_id="session_complex_synthetic",
        scenario_instance=instance,
    ) as recorder:
        result = HeadlessAgent(
            client,
            tools,
            budgets=AgentBudgets(max_turns=12, max_tool_calls=16),
        ).run(instance, recorder, ask_user=user_callback(user, instance), finalize=False)
        partial_trace = load_trace(trace_path)
        turn_rewards = derive_turn_rewards(partial_trace, instance)
        report = EvaluationSuite(
            [
                HiddenCommandEvaluator(["python", "-m", "pytest", "tests/test_hidden_service.py"]),
                RequiredStateEvaluator(),
                ForbiddenStateEvaluator(),
            ]
        ).evaluate(
            sandbox,
            instance,
            diagnostics={
                "turns": float(result.turns),
                "tool_calls": float(result.tool_calls),
                "tokens": float(result.tokens),
                "user_turns": float(user.metrics.turns),
            },
            turn_rewards=turn_rewards,
        )
        report = apply_agent_termination(report, result.termination_reason)
        finalize_evaluation_trace(
            recorder,
            report,
            final_state_hash=sandbox.state_hash(),
            termination_reason=result.termination_reason,
        )

    trace = load_trace(trace_path)
    replay = replay_trace(trace)
    sft = trace_to_sft(trace, report)
    episode = trace_to_rl_episode(trace, report)
    analysis = analysis_record(trace, report)
    _write_json(output_path / "report.json", report.to_dict())
    _write_json(output_path / "sft.json", sft)
    _write_json(output_path / "rl_episode.json", episode)
    _write_json(output_path / "analysis.json", analysis)
    summary = {
        "tier": "complex_synthetic",
        "trace": str(trace_path),
        "trace_id": trace.trace_id,
        "event_count": len(trace.events),
        "success": report.success,
        "reward": report.reward,
        "termination_reason": replay.state.termination_reason,
        "turns": result.turns,
        "tool_calls": result.tool_calls,
        "user_turns": user.metrics.turns,
        "final_state_hash": replay.terminal_state_hash,
        "artifacts": {
            "report": str(output_path / "report.json"),
            "sft": str(output_path / "sft.json"),
            "rl_episode": str(output_path / "rl_episode.json"),
            "analysis": str(output_path / "analysis.json"),
        },
    }
    _write_json(output_path / "manifest.json", summary)
    return summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class _ComplexScriptedClient:
    model = "complex-scripted-agent"

    def __init__(self) -> None:
        self.index = 0
        self.script = [
            _tool("list_1", "list_files", {"path": "."}, "Inspect the workspace shape first."),
            _tool("read_1", "read_file", {"path": "service.py"}, "Read the parser before editing."),
            _tool("search_1", "search_files", {"query": "TODO"}, "Find the marked defect."),
            _tool(
                "ask_1",
                "ask_user",
                {"question": "What record format should the parser support?"},
                "The public task lacks the exact record format.",
            ),
            _tool(
                "patch_1",
                "apply_patch",
                {
                    "path": "service.py",
                    "old": _BUGGY_PARSE,
                    "new": _FIXED_PARSE,
                },
                "Patch the parser to implement the clarified format.",
            ),
            _tool(
                "patch_2",
                "apply_patch",
                {
                    "path": "README.md",
                    "old": "Parser supports comma-delimited records.",
                    "new": "Parser supports pipe-delimited records with quoted values.",
                },
                "Update public documentation so behavior and code agree.",
            ),
            _tool(
                "test_1",
                "run_command",
                {"command": ["python", "-m", "pytest", "tests/test_service.py"]},
                "Run the visible regression test.",
            ),
            _tool("diff_1", "git_diff", {}, "Inspect the final patch before answering."),
            Message(
                "assistant",
                (
                    "Implemented pipe-delimited parsing for quoted values, updated the README, "
                    "ran `python -m pytest tests/test_service.py`, and inspected the diff."
                ),
                reasoning_content=(
                    "All required code, docs, visible validation, and diff review are complete."
                ),
            ),
        ]

    def complete(self, messages, tools=None, **kwargs):
        del messages, tools, kwargs
        message = self.script[self.index]
        self.index += 1
        return LLMResponse(message, self.model, {"total_tokens": 24})


def _tool(call_id: str, name: str, arguments: dict[str, Any], reasoning: str) -> Message:
    return Message(
        "assistant",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
        reasoning_content=reasoning,
    )


def _complex_sandbox() -> MemorySandbox:
    return MemorySandbox(
        {
            "service.py": (
                "def parse_record(line):\n"
                "    # TODO: support the production record format.\n"
                "    name, value = line.split(',', 1)\n"
                "    return {'name': name.strip(), 'value': value.strip()}\n"
            ),
            "README.md": "Parser supports comma-delimited records.",
            "protected.txt": "DO NOT CHANGE\n",
            "tests/test_service.py": (
                "from service import parse_record\n\n"
                "def test_parse_simple_record():\n"
                "    assert parse_record('alpha|\"42\"') == {'name': 'alpha', 'value': '42'}\n"
            ),
        },
        {
            "python -m pytest tests/test_service.py": _visible_test,
            "python -m pytest tests/test_hidden_service.py": _hidden_test,
        },
    )


def _visible_test(sandbox: MemorySandbox) -> CommandResult:
    return _test_result(sandbox, "visible ok\n")


def _hidden_test(sandbox: MemorySandbox) -> CommandResult:
    service = sandbox.read("service.py")
    readme = sandbox.read("README.md")
    protected = sandbox.read("protected.txt")
    passed = (
        _parser_is_fixed(service)
        and "pipe-delimited records with quoted values" in readme
        and protected == "DO NOT CHANGE\n"
    )
    return CommandResult(
        0 if passed else 1,
        "hidden ok\n" if passed else "",
        "" if passed else "hidden failed\n",
        2.0,
    )


def _test_result(sandbox: MemorySandbox, success_stdout: str) -> CommandResult:
    passed = _parser_is_fixed(sandbox.read("service.py"))
    return CommandResult(
        0 if passed else 1,
        success_stdout if passed else "",
        "" if passed else "visible failed\n",
        1.0,
    )


def _parser_is_fixed(source: str) -> bool:
    return (
        "line.split('|', 1)" in source
        and ".strip().strip('\"')" in source
        and "return {'name': name.strip(), 'value': value}" in source
    )


def _complex_instance(state_hash: str) -> ScenarioInstance:
    scenario = Scenario(
        QuerySeed(
            PublicTaskContext(
                query=(
                    "Fix the parser so production records are handled correctly. Update the "
                    "documentation, run the focused test, and inspect the diff before finishing."
                ),
                context={"component": "record parser"},
                constraints=[
                    "Ask for the missing production record format if it is not present.",
                    "Keep protected.txt unchanged.",
                    "Run the focused parser test before the final answer.",
                    "Inspect the diff before the final answer.",
                ],
            ),
            hidden_user=HiddenUserContext(
                goal="Help the agent discover the production parser format.",
                goal_components={"record_format": "record format"},
                known_facts={"record_format": "pipe-delimited records with quoted values"},
                unavailable_facts=["deployment token"],
                patience_turns=4,
            ),
            category="software_engineering",
            difficulty=4,
            provenance="synthetic:complex-parser-fixture",
            license="Apache-2.0",
            metadata={"source_adapter": "complex_synthetic_fixture"},
        ),
        EnvironmentSpec(name="complex-memory-parser-fixture", version="1"),
        HiddenEvaluatorContext(
            hidden_tests=["python -m pytest tests/test_hidden_service.py"],
            required_state={
                "file_contains": {
                    "service.py": "line.split('|', 1)",
                    "README.md": "pipe-delimited records with quoted values",
                }
            },
            forbidden_state={"file_equals": {"protected.txt": "DO NOT CHANGE\n"}},
        ),
        metadata={"synthesis_tier": "complex_synthetic"},
    )
    return ScenarioInstance.materialize(
        scenario,
        random_seed=20260619,
        initial_state_hash=state_hash,
    )


_BUGGY_PARSE = (
    "def parse_record(line):\n"
    "    # TODO: support the production record format.\n"
    "    name, value = line.split(',', 1)\n"
    "    return {'name': name.strip(), 'value': value.strip()}\n"
)

_FIXED_PARSE = (
    "def parse_record(line):\n"
    "    name, value = line.split('|', 1)\n"
    "    value = value.strip().strip('\"')\n"
    "    return {'name': name.strip(), 'value': value}\n"
)
