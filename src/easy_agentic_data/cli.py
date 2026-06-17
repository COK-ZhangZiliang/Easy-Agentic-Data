from __future__ import annotations

import argparse
import json
import shlex
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from easy_agentic_data.agent import HeadlessAgent
from easy_agentic_data.batch import (
    PersistentScheduler,
    RolloutJob,
    RolloutOutcome,
    RunBudget,
)
from easy_agentic_data.coding_tools import SCHEMAS, CodingToolRuntime
from easy_agentic_data.config import PipelineConfig, load_config
from easy_agentic_data.evaluation import (
    EvaluationSuite,
    ForbiddenStateEvaluator,
    HiddenCommandEvaluator,
    RequiredStateEvaluator,
    derive_turn_rewards,
    finalize_evaluation_trace,
)
from easy_agentic_data.llm.mock import MockLLMClient
from easy_agentic_data.llm.observability import ObservedLLMClient
from easy_agentic_data.llm.openai_compatible import (
    LocalOpenAICompatibleClient,
    OpenAICompatibleClient,
)
from easy_agentic_data.pipeline import SynthesisPipeline
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.registry import ScenarioRegistry, materialize_environment_source
from easy_agentic_data.sandbox import DockerSandbox, SandboxLimits
from easy_agentic_data.simulation import RuleBasedUserSimulator, user_callback
from easy_agentic_data.tools import default_tool_registry
from easy_agentic_data.traces import TraceRecorder, load_trace, replay_trace
from easy_agentic_data.verification import (
    SemanticLLMVerifier,
    StructuralVerifier,
    ToolExecutionVerifier,
    VerificationSuite,
)


def build_pipeline(config: PipelineConfig) -> SynthesisPipeline:
    client = ObservedLLMClient(build_llm_client(config))
    verification = VerificationSuite(
        [
            StructuralVerifier(),
            ToolExecutionVerifier(),
            SemanticLLMVerifier(client),
        ]
    )
    return SynthesisPipeline(config, client, default_tool_registry(), verification)


def build_llm_client(config: PipelineConfig):
    if config.llm.provider == "mock":
        return MockLLMClient()
    elif config.llm.provider == "openai_compatible":
        return OpenAICompatibleClient(config.llm)
    elif config.llm.provider == "local_openai_compatible":
        return LocalOpenAICompatibleClient(config.llm)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ead")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run a synthesis pipeline")
    run_parser.add_argument("--config", required=True, help="Path to a JSON configuration file")
    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay an append-only trace without model or tool calls",
    )
    replay_parser.add_argument("--trace", required=True, help="Path to a trace JSONL file")
    replay_parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject an incomplete final JSONL record instead of ignoring it",
    )
    registry_parser = subparsers.add_parser("registry", help="Manage scenario registry entries")
    registry_subparsers = registry_parser.add_subparsers(dest="registry_command", required=True)
    for command in ("list", "validate"):
        item = registry_subparsers.add_parser(command)
        item.add_argument("--root", required=True)
    inspect_parser = registry_subparsers.add_parser("inspect")
    inspect_parser.add_argument("--root", required=True)
    inspect_parser.add_argument("--scenario-id", required=True)
    materialize_parser = registry_subparsers.add_parser("materialize")
    materialize_parser.add_argument("--root", required=True)
    materialize_parser.add_argument("--scenario-id", required=True)
    materialize_parser.add_argument("--random-seed", required=True, type=int)
    agent_parser = subparsers.add_parser("agent-run", help="Run one registry scenario in Docker")
    agent_parser.add_argument("--registry", required=True)
    agent_parser.add_argument("--scenario-id", required=True)
    agent_parser.add_argument("--config", required=True)
    agent_parser.add_argument("--trace", required=True)
    agent_parser.add_argument("--random-seed", type=int, default=42)
    batch_parser = subparsers.add_parser("batch", help="Manage recoverable synthesis jobs")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command", required=True)
    batch_enqueue = batch_subparsers.add_parser("enqueue")
    batch_enqueue.add_argument("--registry", required=True)
    batch_enqueue.add_argument("--database", required=True)
    batch_enqueue.add_argument("--model", required=True)
    batch_enqueue.add_argument("--config-hash", required=True)
    batch_enqueue.add_argument("--rollouts", type=int, default=1)
    batch_run = batch_subparsers.add_parser("run")
    batch_run.add_argument("--registry", required=True)
    batch_run.add_argument("--database", required=True)
    batch_run.add_argument("--config", required=True)
    batch_run.add_argument("--trace-directory", required=True)
    batch_run.add_argument("--max-workers", type=int, default=1)
    batch_status = batch_subparsers.add_parser("status")
    batch_status.add_argument("--database", required=True)
    args = parser.parse_args(argv)

    if args.command == "run":
        summary = build_pipeline(load_config(args.config)).run()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "replay":
        trace = load_trace(args.trace, tolerate_truncated=not args.strict)
        print(json.dumps(replay_trace(trace).to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "registry":
        registry = ScenarioRegistry(args.root)
        if args.registry_command == "list":
            print(json.dumps(registry.list_scenarios(), indent=2))
        elif args.registry_command == "validate":
            validation = registry.validate()
            print(
                json.dumps(
                    {
                        "valid": validation.valid,
                        "issues": [asdict(issue) for issue in validation.issues],
                    },
                    indent=2,
                )
            )
            return 0 if validation.valid else 2
        elif args.registry_command == "inspect":
            print(json.dumps(registry.get_scenario(args.scenario_id).to_dict(), indent=2))
        elif args.registry_command == "materialize":
            instance = registry.materialize(
                args.scenario_id,
                random_seed=args.random_seed,
            )
            print(json.dumps(instance.to_dict(), indent=2))
        return 0
    if args.command == "agent-run":
        outcome = _run_registry_scenario(
            ScenarioRegistry(args.registry),
            args.scenario_id,
            load_config(args.config),
            Path(args.trace),
            args.random_seed,
        )
        print(json.dumps(asdict(outcome), indent=2))
        return 0 if outcome.trace_id else 1
    if args.command == "batch":
        scheduler = PersistentScheduler(args.database)
        if args.batch_command == "enqueue":
            scenarios = ScenarioRegistry(args.registry).list_scenarios()
            scheduler.submit(
                RolloutJob(scenario["scenario_id"], rollout, args.model, args.config_hash)
                for scenario in scenarios
                for rollout in range(args.rollouts)
            )
            print(json.dumps(scheduler.status_counts(), indent=2))
        elif args.batch_command == "status":
            print(json.dumps(scheduler.status_counts(), indent=2))
        elif args.batch_command == "run":
            worker = _CLIRolloutWorker(
                ScenarioRegistry(args.registry),
                load_config(args.config),
                Path(args.trace_directory),
            )
            summary = scheduler.run(
                worker,
                max_workers=args.max_workers,
                budget=RunBudget(),
            )
            print(json.dumps(summary, indent=2))
        return 0
    return 1


class _CLIRolloutWorker:
    def __init__(self, registry: ScenarioRegistry, config: PipelineConfig, trace_directory: Path):
        self.registry = registry
        self.config = config
        self.trace_directory = trace_directory

    def run(self, job: RolloutJob) -> RolloutOutcome:
        try:
            return _run_registry_scenario(
                self.registry,
                job.scenario_id,
                self.config,
                self.trace_directory / f"{job.job_id}.jsonl",
                job.rollout_index,
            )
        except Exception as exc:
            return RolloutOutcome(
                infrastructure_failure=True,
                error=f"{type(exc).__name__}: {exc}",
            )


def _run_registry_scenario(
    registry: ScenarioRegistry,
    scenario_id: str,
    config: PipelineConfig,
    trace_path: Path,
    random_seed: int,
) -> RolloutOutcome:
    scenario = registry.get_scenario(scenario_id)
    with tempfile.TemporaryDirectory() as directory:
        source = materialize_environment_source(scenario.environment, directory)
        limits = SandboxLimits(**scenario.environment.resource_limits)
        sandbox = DockerSandbox(
            image_digest=scenario.environment.image_digest,
            source_directory=source,
            limits=limits,
            network_enabled=scenario.environment.network_policy != "disabled",
        )
        sandbox.create()
        try:
            instance = registry.materialize(
                scenario_id,
                random_seed=random_seed,
                initial_state_hash=sandbox.state_hash(),
            )
            policy = ToolPolicy(
                scenario.environment.capability_packs or SCHEMAS.keys(),
                network_enabled=scenario.environment.network_policy != "disabled",
            )
            tools = CodingToolRuntime(sandbox, policy)
            user = RuleBasedUserSimulator(instance)
            client = ObservedLLMClient(build_llm_client(config))
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with TraceRecorder(
                trace_path,
                session_id=f"session_{instance.instance_id}_{random_seed}",
                scenario_instance=instance,
            ) as recorder:
                run_result = HeadlessAgent(client, tools).run(
                    instance,
                    recorder,
                    ask_user=user_callback(user, instance),
                    finalize=False,
                )
                evaluators = [
                    HiddenCommandEvaluator(shlex.split(command))
                    for command in instance.hidden_evaluator.hidden_tests
                ]
                evaluators.extend([RequiredStateEvaluator(), ForbiddenStateEvaluator()])
                turn_rewards = derive_turn_rewards(load_trace(trace_path), instance)
                diagnostics = {
                    "turns": float(run_result.turns),
                    "tool_calls": float(run_result.tool_calls),
                    "tokens": float(run_result.tokens),
                    "user_turns": float(user.metrics.turns),
                }
                diagnostics.update(
                    {
                        key: float(value)
                        for key, value in user.metrics.to_dict().items()
                        if isinstance(value, (int, float))
                    }
                )
                report = EvaluationSuite(evaluators).evaluate(
                    sandbox,
                    instance,
                    diagnostics=diagnostics,
                    turn_rewards=turn_rewards,
                )
                finalize_evaluation_trace(recorder, report, final_state_hash=sandbox.state_hash())
            trace = load_trace(trace_path)
            return RolloutOutcome(
                trace_id=trace.trace_id,
                success=report.success,
                tokens=run_result.tokens,
                metrics=report.metrics,
            )
        finally:
            sandbox.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
