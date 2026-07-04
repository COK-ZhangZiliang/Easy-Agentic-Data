from __future__ import annotations

import argparse
import json
import shlex
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from easy_agentic_data.agent import AgentBudgets, HeadlessAgent
from easy_agentic_data.batch import (
    PersistentScheduler,
    RolloutJob,
    RolloutOutcome,
    RunBudget,
    audit_trace_logic,
    enqueue_human_review,
    estimate_scale_run,
    planned_batch_run,
    quality_report,
    scale_continuation_decision,
    scale_readiness_summary,
    select_scale_candidates,
    selected_job_status,
)
from easy_agentic_data.coding_tools import SCHEMAS, CodingToolRuntime
from easy_agentic_data.config import PipelineConfig, load_config
from easy_agentic_data.evaluation import (
    EvaluationSuite,
    ForbiddenStateEvaluator,
    HiddenCommandEvaluator,
    HiddenTestPatchEvaluator,
    RequiredStateEvaluator,
    TraceRequirementEvaluator,
    apply_agent_termination,
    derive_turn_rewards,
    evaluation_result_metrics,
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
from easy_agentic_data.real_seed_sources import (
    DEFAULT_DEMO_IMAGE_DIGEST,
    SWE_BENCH_LITE_DATASET,
    prepare_real_seed_registry,
)
from easy_agentic_data.registry import (
    ScenarioRegistry,
    materialize_environment_source,
)
from easy_agentic_data.registry_sources import (
    DEFAULT_TRAIN_LICENSE_ALLOWLIST,
    PUBLIC_CI_FORMATS,
    PUBLIC_ISSUE_PR_FORMATS,
    import_public_ci_records,
    import_public_issue_pr_records,
    import_swe_style_records,
    load_source_records,
)
from easy_agentic_data.repository_allowlist import (
    audit_repository_allowlist,
    load_repository_allowlist,
)
from easy_agentic_data.repository_synthetic import (
    DEFAULT_SYNTHETIC_TRAIN_LICENSE_ALLOWLIST,
    generate_repository_synthetic_scenarios,
    load_repository_synthesis_specs,
)
from easy_agentic_data.sandbox import DockerSandbox, SandboxLimits
from easy_agentic_data.scenario_decontamination import (
    audit_scenario_decontamination,
    scenarios_from_registry,
)
from easy_agentic_data.seed_library import (
    DEFAULT_BENCHMARK_SOURCE_ALIASES,
    SeedLibraryPolicy,
    audit_seed_library,
)
from easy_agentic_data.seed_corpus import (
    build_seed_backfill_plan,
    build_seed_corpus,
    build_seed_selection_plan,
    build_synthetic_backfill_spec_plan,
    rehearse_registry_import,
)
from easy_agentic_data.seed_review import build_seed_review_queue
from easy_agentic_data.source_collection import (
    audit_public_source_records,
    build_source_collection_shard_schedule,
    build_source_collection_retry_plan,
    build_source_collection_plan,
    export_public_source_records,
    filter_accepted_public_source_records,
    merge_source_export_summaries,
    run_source_collection_retry_plan,
    split_public_source_records,
    summarize_source_collection_preflight,
    summarize_source_collection_shard_status,
    summarize_source_collection_readiness,
)
from easy_agentic_data.simulation import RuleBasedUserSimulator, user_callback
from easy_agentic_data.synthesis_tiers import default_synthesis_tiers, run_complex_synthetic_demo
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


def _parse_train_eligible(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ead")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run a synthesis pipeline")
    run_parser.add_argument("--config", required=True, help="Path to a JSON configuration file")
    synthesis_parser = subparsers.add_parser("synthesis", help="Run or inspect synthesis tiers")
    synthesis_subparsers = synthesis_parser.add_subparsers(dest="synthesis_command", required=True)
    synthesis_subparsers.add_parser("tiers", help="List the supported synthesis tiers")
    complex_parser = synthesis_subparsers.add_parser(
        "complex-demo",
        help="Generate one deterministic complex synthetic agent trajectory",
    )
    complex_parser.add_argument(
        "--output",
        default="runs/complex-synthetic-demo",
        help="Directory for trace and derived export artifacts",
    )
    real_seed_parser = synthesis_subparsers.add_parser(
        "real-seed-demo",
        help="Prepare a real SWE-bench Lite seed, clone its repository, and optionally run it",
    )
    real_seed_parser.add_argument(
        "--output",
        default="runs/real-seed-demo",
        help="Root directory for the registry, workspace cache, and optional trace",
    )
    real_seed_parser.add_argument(
        "--source",
        default="",
        help="Optional local JSON/JSONL SWE-style seed file instead of Hugging Face rows",
    )
    real_seed_parser.add_argument("--dataset", default=SWE_BENCH_LITE_DATASET)
    real_seed_parser.add_argument("--split", default="dev")
    real_seed_parser.add_argument("--offset", type=int, default=0)
    real_seed_parser.add_argument("--limit", type=int, default=1)
    real_seed_parser.add_argument("--source-name", default="")
    real_seed_parser.add_argument("--license", default="")
    real_seed_parser.add_argument("--permitted-use", default="research")
    real_seed_parser.add_argument("--image-digest", default=DEFAULT_DEMO_IMAGE_DIGEST)
    real_seed_parser.add_argument(
        "--setup-command",
        action="append",
        default=[],
        help="Workspace setup command to run before the agent starts; may be repeated",
    )
    real_seed_parser.add_argument(
        "--network-policy",
        default="disabled",
        choices=["disabled", "enabled"],
        help="Runtime network policy for the prepared sandbox",
    )
    real_seed_parser.add_argument(
        "--test-command-template",
        default="python -m pytest {test}",
        help="Hidden evaluator command template; use an empty string to skip hidden tests",
    )
    real_seed_parser.add_argument(
        "--no-pull-repos",
        action="store_true",
        help="Import seed metadata without cloning repositories",
    )
    real_seed_parser.add_argument(
        "--config",
        default="",
        help="Optional LLM config. When set, run the first prepared scenario in Docker.",
    )
    real_seed_parser.add_argument(
        "--trace",
        default="",
        help="Optional trace path for --config runs. Defaults to output/trace.jsonl.",
    )
    real_seed_parser.add_argument("--random-seed", type=int, default=42)
    real_seed_parser.add_argument("--max-agent-turns", type=int, default=20)
    real_seed_parser.add_argument("--max-agent-tool-calls", type=int, default=50)
    real_seed_parser.add_argument(
        "--max-agent-tokens",
        type=int,
        default=200_000,
        help="Total agent-loop token budget. Thinking-mode runs often need more than smoke tests.",
    )
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
    audit_parser = registry_subparsers.add_parser("seed-audit")
    audit_parser.add_argument("--root", required=True)
    audit_parser.add_argument("--output", default="")
    audit_parser.add_argument(
        "--benchmark-source",
        action="append",
        default=[],
        help="Additional source alias treated as evaluation benchmark contamination",
    )
    audit_parser.add_argument(
        "--holdout-root",
        action="append",
        default=[],
        help="Additional registry root whose non-train or benchmark seeds are held out",
    )
    audit_parser.add_argument("--min-train-eligible", type=int, default=0)
    audit_parser.add_argument("--require-task-family", action="append", default=[])
    audit_parser.add_argument("--require-verifier-type", action="append", default=[])
    audit_parser.add_argument("--max-task-family-share", type=float, default=1.0)
    audit_parser.add_argument("--max-source-method-share", type=float, default=1.0)
    audit_parser.add_argument("--max-repository-share", type=float, default=1.0)
    audit_parser.add_argument("--max-language-share", type=float, default=1.0)
    scenario_audit_parser = registry_subparsers.add_parser("scenario-audit")
    scenario_audit_parser.add_argument("--root", required=True)
    scenario_audit_parser.add_argument("--output", default="")
    scenario_audit_parser.add_argument(
        "--benchmark-source",
        action="append",
        default=[],
        help="Additional source alias treated as evaluation benchmark contamination",
    )
    scenario_audit_parser.add_argument(
        "--holdout-root",
        action="append",
        default=[],
        help="Additional registry root whose non-train or benchmark scenarios are held out",
    )
    review_queue_parser = registry_subparsers.add_parser("review-queue")
    review_queue_parser.add_argument("--root", required=True)
    review_queue_parser.add_argument("--output", default="")
    review_queue_parser.add_argument("--sample-per-stratum", type=int, default=1)
    review_queue_parser.add_argument("--max-records", type=int)
    review_queue_parser.add_argument("--overwrite", action="store_true")
    build_corpus_parser = registry_subparsers.add_parser("build-corpus")
    build_corpus_parser.add_argument("--config", required=True)
    build_corpus_parser.add_argument("--manifest-output", default="")
    build_corpus_parser.add_argument(
        "--overwrite-outputs",
        action="store_true",
        help="Overwrite append-only review queue outputs declared by the corpus config",
    )
    seed_backfill_parser = registry_subparsers.add_parser("seed-backfill-plan")
    seed_backfill_parser.add_argument("--audit", required=True)
    seed_backfill_parser.add_argument("--policy", required=True)
    seed_backfill_parser.add_argument("--output", default="")
    seed_selection_parser = registry_subparsers.add_parser("seed-selection-plan")
    seed_selection_parser.add_argument("--root", required=True)
    seed_selection_parser.add_argument("--policy", required=True)
    seed_selection_parser.add_argument("--target-train-eligible", type=int)
    seed_selection_parser.add_argument("--output", default="")
    synthetic_backfill_parser = registry_subparsers.add_parser("seed-synthetic-backfill-spec")
    synthetic_backfill_parser.add_argument("--root", required=True)
    synthetic_backfill_parser.add_argument("--selection-plan", required=True)
    synthetic_backfill_parser.add_argument("--backfill-plan", required=True)
    synthetic_backfill_parser.add_argument("--max-repositories", type=int, default=10)
    synthetic_backfill_parser.add_argument("--output", default="")
    synthetic_backfill_parser.add_argument("--spec-output", default="")
    import_rehearsal_parser = registry_subparsers.add_parser("import-rehearsal")
    import_rehearsal_parser.add_argument("--root", required=True)
    import_rehearsal_parser.add_argument("--source", required=True)
    import_rehearsal_parser.add_argument(
        "--format",
        default="public-issue-pr",
        choices=[
            "auto",
            "swe-bench",
            "swe-smith",
            "multi-swe",
            "public-issue",
            "public-pr",
            "public-issue-pr",
            "public-ci",
        ],
    )
    import_rehearsal_parser.add_argument("--source-name", default="")
    import_rehearsal_parser.add_argument("--allowlist", default="")
    import_rehearsal_parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "evaluation", "dev", "eval_holdout", "quarantined"],
    )
    import_rehearsal_parser.add_argument("--license", default="")
    import_rehearsal_parser.add_argument("--permitted-use", default="research")
    import_rehearsal_parser.add_argument("--task-family", default="")
    import_rehearsal_parser.add_argument("--source-method", default="")
    import_rehearsal_parser.add_argument(
        "--train-eligible",
        default="auto",
        choices=["auto", "true", "false"],
    )
    import_rehearsal_parser.add_argument("--contamination-tag", action="append", default=[])
    import_rehearsal_parser.add_argument("--coverage-tag", action="append", default=[])
    import_rehearsal_parser.add_argument("--allow-train-license", action="append", default=[])
    import_rehearsal_parser.add_argument("--benchmark-source", action="append", default=[])
    import_rehearsal_parser.add_argument("--limit", type=int)
    import_rehearsal_parser.add_argument("--test-command-template", default="")
    import_rehearsal_parser.add_argument("--strict", action="store_true")
    import_rehearsal_parser.add_argument("--overwrite-registry", action="store_true")
    import_rehearsal_parser.add_argument("--min-imported", type=int, default=1)
    import_rehearsal_parser.add_argument("--max-quarantined", type=int, default=0)
    import_rehearsal_parser.add_argument("--min-train-eligible", type=int, default=0)
    import_rehearsal_parser.add_argument("--require-task-family", action="append", default=[])
    import_rehearsal_parser.add_argument("--require-verifier-type", action="append", default=[])
    import_rehearsal_parser.add_argument("--max-task-family-share", type=float, default=1.0)
    import_rehearsal_parser.add_argument("--max-source-method-share", type=float, default=1.0)
    import_rehearsal_parser.add_argument("--max-repository-share", type=float, default=1.0)
    import_rehearsal_parser.add_argument("--max-language-share", type=float, default=1.0)
    import_rehearsal_parser.add_argument("--materialize-sample-count", type=int, default=0)
    import_rehearsal_parser.add_argument("--materialize-root", default="")
    import_rehearsal_parser.add_argument("--run-hidden-commands", action="store_true")
    import_rehearsal_parser.add_argument("--output", default="")
    allowlist_audit_parser = registry_subparsers.add_parser("allowlist-audit")
    allowlist_audit_parser.add_argument("--source", required=True)
    allowlist_audit_parser.add_argument("--output", default="")
    allowlist_audit_parser.add_argument(
        "--allow-train-license",
        action="append",
        default=[],
        help="Additional license identifier allowed for train-eligible repositories",
    )
    allowlist_audit_parser.add_argument(
        "--benchmark-repository",
        action="append",
        default=[],
        help="Repository name that must be treated as benchmark-overlapping",
    )
    collection_plan_parser = registry_subparsers.add_parser("collection-plan")
    collection_plan_parser.add_argument("--allowlist", required=True)
    collection_plan_parser.add_argument("--output", default="")
    collection_plan_parser.add_argument("--output-root", default="runs/source-exports")
    collection_plan_parser.add_argument("--source-name", default="curated-public-sources")
    collection_shards_parser = registry_subparsers.add_parser("collection-shards")
    collection_shards_parser.add_argument("--plan", required=True)
    collection_shards_parser.add_argument("--source-output", required=True)
    collection_shards_parser.add_argument("--summary-output-dir", required=True)
    collection_shards_parser.add_argument("--preflight-output-dir", default="")
    collection_shards_parser.add_argument("--shard-size", type=int, default=4)
    collection_shards_parser.add_argument("--limit-per-task", type=int, default=5)
    collection_shards_parser.add_argument("--sleep-seconds", type=float, default=2.0)
    collection_shards_parser.add_argument("--resume", action="store_true")
    collection_shards_parser.add_argument("--allow-partial", action="store_true")
    collection_shards_parser.add_argument("--github-token-env", default="")
    collection_shards_parser.add_argument("--require-github-token", action="store_true")
    collection_shards_parser.add_argument("--output", default="")
    collection_shard_status_parser = registry_subparsers.add_parser("collection-shard-status")
    collection_shard_status_parser.add_argument("--schedule", required=True)
    collection_shard_status_parser.add_argument("--source", default="")
    collection_shard_status_parser.add_argument("--output", default="")
    collection_preflight_parser = registry_subparsers.add_parser("collection-preflight")
    collection_preflight_parser.add_argument("--plan", required=True)
    collection_preflight_parser.add_argument("--source", default="")
    collection_preflight_parser.add_argument("--summary", action="append", default=[])
    collection_preflight_parser.add_argument("--github-token-env", default="")
    collection_preflight_parser.add_argument("--require-github-token", action="store_true")
    collection_preflight_parser.add_argument("--task-offset", type=int, default=0)
    collection_preflight_parser.add_argument("--max-tasks", type=int)
    collection_preflight_parser.add_argument("--require-source", action="store_true")
    collection_preflight_parser.add_argument("--output", default="")
    collection_export_parser = registry_subparsers.add_parser("collection-export")
    collection_export_parser.add_argument("--plan", required=True)
    collection_export_parser.add_argument("--output", required=True)
    collection_export_parser.add_argument("--limit-per-task", type=int, default=10)
    collection_export_parser.add_argument("--task-offset", type=int, default=0)
    collection_export_parser.add_argument("--max-tasks", type=int)
    collection_export_parser.add_argument("--sleep-seconds", type=float, default=0.0)
    collection_export_parser.add_argument("--resume", action="store_true")
    collection_export_parser.add_argument("--allow-partial", action="store_true")
    collection_export_parser.add_argument("--summary-output", default="")
    collection_export_parser.add_argument("--fixture-root", default="")
    collection_export_parser.add_argument("--github-token-env", default="")
    collection_export_parser.add_argument("--require-github-token", action="store_true")
    collection_export_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    collection_retry_parser = registry_subparsers.add_parser("collection-retry-plan")
    collection_retry_parser.add_argument("--plan", required=True)
    collection_retry_parser.add_argument("--export-summary", required=True)
    collection_retry_parser.add_argument("--output", default="")
    collection_retry_parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Only plan retries for tasks inside the export summary's selected shard",
    )
    collection_retry_run_parser = registry_subparsers.add_parser("collection-retry-run")
    collection_retry_run_parser.add_argument("--plan", required=True)
    collection_retry_run_parser.add_argument("--retry-plan", required=True)
    collection_retry_run_parser.add_argument("--output", required=True)
    collection_retry_run_parser.add_argument("--summary-output", default="")
    collection_retry_run_parser.add_argument("--limit-per-task", type=int, default=10)
    collection_retry_run_parser.add_argument("--max-retry-tasks", type=int)
    collection_retry_run_parser.add_argument("--sleep-seconds", type=float, default=0.0)
    collection_retry_run_parser.add_argument("--fixture-root", default="")
    collection_retry_run_parser.add_argument("--github-token-env", default="")
    collection_retry_run_parser.add_argument("--require-github-token", action="store_true")
    collection_retry_run_parser.add_argument("--timeout-seconds", type=float, default=30.0)
    collection_retry_run_parser.add_argument("--allow-partial", action="store_true")
    collection_summary_parser = registry_subparsers.add_parser("collection-summary")
    collection_summary_parser.add_argument("--source", required=True)
    collection_summary_parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Collection export or retry-run summary to merge; may be repeated",
    )
    collection_summary_parser.add_argument("--plan", default="")
    collection_summary_parser.add_argument("--output", default="")
    collection_summary_parser.add_argument("--allow-partial", action="store_true")
    collection_split_parser = registry_subparsers.add_parser("collection-split")
    collection_split_parser.add_argument("--source", required=True)
    collection_split_parser.add_argument("--output", required=True)
    collection_split_parser.add_argument("--summary-output", default="")
    collection_split_parser.add_argument("--include-source-type", action="append", default=[])
    collection_split_parser.add_argument("--exclude-source-type", action="append", default=[])
    collection_audit_parser = registry_subparsers.add_parser("collection-audit")
    collection_audit_parser.add_argument("--source", required=True)
    collection_audit_parser.add_argument("--allowlist", required=True)
    collection_audit_parser.add_argument("--output", default="")
    collection_audit_parser.add_argument("--accepted-output", default="")
    collection_audit_parser.add_argument("--source-name", default="curated-public-sources")
    collection_readiness_parser = registry_subparsers.add_parser("collection-readiness")
    collection_readiness_parser.add_argument("--plan", required=True)
    collection_readiness_parser.add_argument("--export-summary", required=True)
    collection_readiness_parser.add_argument("--audit", required=True)
    collection_readiness_parser.add_argument("--output", default="")
    collection_readiness_parser.add_argument("--min-accepted", type=int, default=1)
    collection_readiness_parser.add_argument("--max-quarantined", type=int, default=0)
    collection_readiness_parser.add_argument("--require-source-type", action="append", default=[])
    collection_readiness_parser.add_argument("--require-clean-export", action="store_true")
    collection_readiness_parser.add_argument("--require-all-plan-tasks", action="store_true")
    inspect_parser = registry_subparsers.add_parser("inspect")
    inspect_parser.add_argument("--root", required=True)
    inspect_parser.add_argument("--scenario-id", required=True)
    materialize_parser = registry_subparsers.add_parser("materialize")
    materialize_parser.add_argument("--root", required=True)
    materialize_parser.add_argument("--scenario-id", required=True)
    materialize_parser.add_argument("--random-seed", required=True, type=int)
    import_parser = registry_subparsers.add_parser("import")
    import_parser.add_argument("--root", required=True)
    import_parser.add_argument("--source", required=True)
    import_parser.add_argument(
        "--format",
        default="auto",
        choices=[
            "auto",
            "swe-bench",
            "swe-smith",
            "multi-swe",
            "public-issue",
            "public-pr",
            "public-issue-pr",
            "public-ci",
        ],
        help="External source record shape",
    )
    import_parser.add_argument("--source-name", default="")
    import_parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "evaluation", "dev", "eval_holdout", "quarantined"],
    )
    import_parser.add_argument("--license", default="")
    import_parser.add_argument("--permitted-use", default="research")
    import_parser.add_argument(
        "--task-family",
        default="",
        help="Task family such as bug_repair, test_authoring, refactor, migration, or docs",
    )
    import_parser.add_argument(
        "--source-method",
        default="",
        help=(
            "Source construction method such as external_issue_workspace "
            "or synthetic_issue_workspace"
        ),
    )
    import_parser.add_argument(
        "--train-eligible",
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether imported seeds may be used for training; auto blocks known benchmarks",
    )
    import_parser.add_argument(
        "--contamination-tag",
        action="append",
        default=[],
        help="Additional contamination or holdout tag to attach to every imported seed",
    )
    import_parser.add_argument(
        "--coverage-tag",
        action="append",
        default=[],
        help="Additional coverage tag to attach to every imported seed",
    )
    import_parser.add_argument(
        "--allow-train-license",
        action="append",
        default=[],
        help="Additional license identifier allowed for train-eligible public issue/PR seeds",
    )
    import_parser.add_argument("--limit", type=int)
    import_parser.add_argument(
        "--test-command-template",
        default="",
        help="Optional command template such as 'python -m pytest {test}'",
    )
    import_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first malformed source record",
    )
    synthetic_parser = registry_subparsers.add_parser("generate-synthetic")
    synthetic_parser.add_argument("--root", required=True)
    synthetic_parser.add_argument(
        "--source",
        required=True,
        help="Local JSON or JSONL repository synthesis spec",
    )
    synthetic_parser.add_argument("--source-name", default="repository_synthetic")
    synthetic_parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation", "evaluation", "dev", "eval_holdout", "quarantined"],
    )
    synthetic_parser.add_argument(
        "--task-family",
        action="append",
        default=[],
        help="Task family to generate; repeat to select multiple families",
    )
    synthetic_parser.add_argument(
        "--train-eligible",
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether generated seeds may be used for training",
    )
    synthetic_parser.add_argument(
        "--allow-train-license",
        action="append",
        default=[],
        help="Additional license identifier allowed for train-eligible synthetic seeds",
    )
    synthetic_parser.add_argument("--limit", type=int)
    synthetic_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first repository synthesis issue",
    )
    agent_parser = subparsers.add_parser("agent-run", help="Run one registry scenario in Docker")
    agent_parser.add_argument("--registry", required=True)
    agent_parser.add_argument("--scenario-id", required=True)
    agent_parser.add_argument("--config", required=True)
    agent_parser.add_argument("--trace", required=True)
    agent_parser.add_argument("--random-seed", type=int, default=42)
    agent_parser.add_argument("--max-agent-turns", type=int, default=20)
    agent_parser.add_argument("--max-agent-tool-calls", type=int, default=50)
    agent_parser.add_argument("--max-agent-tokens", type=int, default=100_000)
    batch_parser = subparsers.add_parser("batch", help="Manage recoverable synthesis jobs")
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command", required=True)
    batch_enqueue = batch_subparsers.add_parser("enqueue")
    batch_enqueue.add_argument("--registry", required=True)
    batch_enqueue.add_argument("--database", required=True)
    batch_enqueue.add_argument("--model", required=True)
    batch_enqueue.add_argument("--config-hash", required=True)
    batch_enqueue.add_argument("--rollouts", type=int, default=1)
    batch_enqueue.add_argument("--selection-file", default="")
    batch_enqueue.add_argument("--scenario-id", action="append", default=[])
    batch_run = batch_subparsers.add_parser("run")
    batch_run.add_argument("--registry", required=True)
    batch_run.add_argument("--database", required=True)
    batch_run.add_argument("--config", required=True)
    batch_run.add_argument("--trace-directory", required=True)
    batch_run.add_argument("--max-workers", type=int, default=1)
    batch_run.add_argument("--max-retries", type=int, default=2)
    batch_run.add_argument("--max-jobs", type=int)
    batch_run.add_argument("--max-seconds", type=float, default=3600.0)
    batch_run.add_argument("--max-tokens", type=int, default=1_000_000)
    batch_run.add_argument("--max-cost", type=float, default=100.0)
    batch_run.add_argument("--max-agent-turns", type=int, default=20)
    batch_run.add_argument("--max-agent-tool-calls", type=int, default=50)
    batch_run.add_argument("--max-agent-tokens", type=int, default=100_000)
    batch_run.add_argument("--max-agent-seconds", type=float, default=600.0)
    batch_run.add_argument("--job-id", action="append", default=[])
    batch_run.add_argument("--job-id-file", default="")
    batch_run.add_argument("--shard-index", type=int)
    batch_run.add_argument("--dry-run", action="store_true")
    batch_status = batch_subparsers.add_parser("status")
    batch_status.add_argument("--database", required=True)
    batch_report = batch_subparsers.add_parser("report")
    batch_report.add_argument("--database", required=True)
    batch_report.add_argument("--output", default="")
    batch_report.add_argument("--trace-directory", default="")
    batch_report.add_argument("--review-sample", default="")
    batch_report.add_argument("--overwrite-review-sample", action="store_true")
    batch_report.add_argument("--sample-size", type=int, default=0)
    batch_report.add_argument("--job-id", action="append", default=[])
    batch_report.add_argument("--job-id-file", default="")
    batch_report.add_argument("--shard-index", type=int)
    batch_select = batch_subparsers.add_parser("select-scale-candidates")
    batch_select.add_argument("--database", required=True)
    batch_select.add_argument("--audit", default="")
    batch_select.add_argument("--output", default="")
    batch_select.add_argument("--min-rollouts", type=int, default=2)
    batch_select.add_argument("--min-success-rate", type=float, default=0.5)
    batch_select.add_argument("--min-hidden-command-pass-rate", type=float, default=0.5)
    batch_select.add_argument("--min-all-non-agent-pass-rate", type=float, default=0.5)
    batch_select.add_argument("--min-agent-stop-rate", type=float, default=0.0)
    batch_select.add_argument("--min-high-quality-rate", type=float, default=0.0)
    batch_select.add_argument("--min-closed-loop-rate", type=float, default=0.0)
    batch_select.add_argument("--min-multi-step-complex-rate", type=float, default=0.0)
    batch_select.add_argument("--max-infrastructure-failure-rate", type=float, default=0.0)
    batch_select.add_argument("--min-average-tool-calls", type=float, default=6.0)
    batch_estimate = batch_subparsers.add_parser("estimate-scale")
    batch_estimate.add_argument("--database", required=True)
    batch_estimate.add_argument("--pilot-database", required=True)
    batch_estimate.add_argument("--output", default="")
    batch_estimate.add_argument("--shard-size", type=int, default=20)
    batch_estimate.add_argument("--cost-per-million-tokens", type=float, default=0.0)
    batch_shard_status = batch_subparsers.add_parser("shard-status")
    batch_shard_status.add_argument("--database", required=True)
    batch_shard_status.add_argument("--job-id-file", required=True)
    batch_shard_status.add_argument("--shard-index", type=int, required=True)
    batch_shard_status.add_argument("--output", default="")
    batch_decision = batch_subparsers.add_parser("decide-continuation")
    batch_decision.add_argument("--report", required=True)
    batch_decision.add_argument("--status", required=True)
    batch_decision.add_argument("--audit", default="")
    batch_decision.add_argument("--output", default="")
    batch_decision.add_argument("--min-success-rate", type=float, default=0.3)
    batch_decision.add_argument("--min-unique-traces", type=int, default=1)
    batch_decision.add_argument("--min-hidden-command-pass-rate", type=float, default=0.4)
    batch_decision.add_argument("--min-high-quality-rate", type=float, default=0.0)
    batch_decision.add_argument("--min-closed-loop-rate", type=float, default=0.0)
    batch_decision.add_argument("--min-multi-step-complex-rate", type=float, default=0.0)
    batch_decision.add_argument("--max-infrastructure-failures", type=int, default=0)
    batch_audit = batch_subparsers.add_parser("audit-traces")
    batch_audit.add_argument("--database", required=True)
    batch_audit.add_argument("--trace-directory", required=True)
    batch_audit.add_argument("--output", default="")
    batch_audit.add_argument("--job-id", action="append", default=[])
    batch_audit.add_argument("--job-id-file", default="")
    batch_audit.add_argument("--shard-index", type=int)
    batch_audit.add_argument("--summary-only", action="store_true")
    batch_readiness = batch_subparsers.add_parser("scale-readiness")
    batch_readiness.add_argument("--selection", required=True)
    batch_readiness.add_argument("--estimate", required=True)
    batch_readiness.add_argument("--status", required=True)
    batch_readiness.add_argument("--audit", required=True)
    batch_readiness.add_argument("--decision", required=True)
    batch_readiness.add_argument("--output", default="")
    args = parser.parse_args(argv)

    if args.command == "run":
        summary = build_pipeline(load_config(args.config)).run()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "synthesis":
        if args.synthesis_command == "tiers":
            print(
                json.dumps(
                    [tier.to_dict() for tier in default_synthesis_tiers()],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.synthesis_command == "complex-demo":
            print(
                json.dumps(
                    run_complex_synthetic_demo(args.output),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.synthesis_command == "real-seed-demo":
            output = Path(args.output)
            registry_root = output / "registry"
            cache_root = output / "workspaces"
            summary = prepare_real_seed_registry(
                registry_root=registry_root,
                cache_root=cache_root,
                source=args.source or None,
                dataset=args.dataset,
                split=args.split,
                offset=args.offset,
                limit=args.limit,
                source_name=args.source_name,
                image_digest=args.image_digest,
                setup_commands=args.setup_command,
                network_policy=args.network_policy,
                pull_repositories=not args.no_pull_repos,
                test_command_template=args.test_command_template,
                license_name=args.license,
                permitted_use=args.permitted_use,
            ).to_dict()
            if args.config:
                scenario_ids = summary["import_summary"]["scenario_ids"]
                if not scenario_ids:
                    raise RuntimeError("No scenario was imported for the real seed run")
                trace_path = Path(args.trace) if args.trace else output / "trace.jsonl"
                outcome = _run_registry_scenario(
                    ScenarioRegistry(registry_root),
                    scenario_ids[0],
                    load_config(args.config),
                    trace_path,
                    args.random_seed,
                    _agent_budgets(args),
                )
                summary["agent_run"] = asdict(outcome)
                summary["trace"] = str(trace_path)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "replay":
        trace = load_trace(args.trace, tolerate_truncated=not args.strict)
        print(json.dumps(replay_trace(trace).to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "registry":
        if args.registry_command == "build-corpus":
            manifest = build_seed_corpus(
                args.config,
                manifest_output=args.manifest_output or None,
                overwrite_outputs=args.overwrite_outputs,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0 if manifest["valid"] else 2
        if args.registry_command == "allowlist-audit":
            audit = audit_repository_allowlist(
                load_repository_allowlist(args.source),
                license_allowlist=sorted(
                    set(DEFAULT_TRAIN_LICENSE_ALLOWLIST) | set(args.allow_train_license)
                ),
                benchmark_repositories=args.benchmark_repository,
            )
            payload = audit.to_dict()
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if audit.valid else 2
        if args.registry_command == "collection-plan":
            plan = build_source_collection_plan(
                load_repository_allowlist(args.allowlist),
                output_root=args.output_root,
                source_name=args.source_name,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(plan, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0 if plan["valid"] else 2
        if args.registry_command == "collection-shards":
            schedule = build_source_collection_shard_schedule(
                json.loads(Path(args.plan).read_text(encoding="utf-8")),
                plan_path=args.plan,
                source_output_path=args.source_output,
                summary_output_dir=args.summary_output_dir,
                preflight_output_dir=args.preflight_output_dir or None,
                shard_size=args.shard_size,
                limit_per_task=args.limit_per_task,
                sleep_seconds=args.sleep_seconds,
                resume=args.resume,
                allow_partial=args.allow_partial,
                github_token_env=args.github_token_env,
                require_github_token=args.require_github_token,
            )
            payload = schedule.to_dict()
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if schedule.valid else 2
        if args.registry_command == "collection-shard-status":
            status = summarize_source_collection_shard_status(
                json.loads(Path(args.schedule).read_text(encoding="utf-8")),
                source_path=args.source,
            )
            payload = status.to_dict()
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if status.ready_for_summary else 2
        if args.registry_command == "collection-preflight":
            preflight = summarize_source_collection_preflight(
                json.loads(Path(args.plan).read_text(encoding="utf-8")),
                source_path=args.source,
                summary_paths=args.summary,
                github_token_env=args.github_token_env,
                require_github_token=args.require_github_token,
                task_offset=args.task_offset,
                max_tasks=args.max_tasks,
                require_source=args.require_source,
            )
            payload = preflight.to_dict()
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if preflight.ready_for_collection else 2
        if args.registry_command == "collection-export":
            plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
            summary = export_public_source_records(
                plan,
                output_path=args.output,
                limit_per_task=args.limit_per_task,
                task_offset=args.task_offset,
                max_tasks=args.max_tasks,
                sleep_seconds=args.sleep_seconds,
                resume=args.resume,
                allow_partial=args.allow_partial,
                fixture_root=args.fixture_root or None,
                github_token_env=args.github_token_env,
                require_github_token=args.require_github_token,
                timeout_seconds=args.timeout_seconds,
            )
            payload = summary.to_dict()
            if args.summary_output:
                Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.summary_output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if summary.valid else 2
        if args.registry_command == "collection-retry-plan":
            retry_plan = build_source_collection_retry_plan(
                json.loads(Path(args.plan).read_text(encoding="utf-8")),
                json.loads(Path(args.export_summary).read_text(encoding="utf-8")),
                include_unselected=not args.selected_only,
            )
            payload = retry_plan.to_dict()
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if retry_plan.valid else 2
        if args.registry_command == "collection-retry-run":
            summary = run_source_collection_retry_plan(
                json.loads(Path(args.plan).read_text(encoding="utf-8")),
                json.loads(Path(args.retry_plan).read_text(encoding="utf-8")),
                output_path=args.output,
                limit_per_task=args.limit_per_task,
                max_retry_tasks=args.max_retry_tasks,
                sleep_seconds=args.sleep_seconds,
                fixture_root=args.fixture_root or None,
                github_token_env=args.github_token_env,
                require_github_token=args.require_github_token,
                timeout_seconds=args.timeout_seconds,
                allow_partial=args.allow_partial,
            )
            payload = summary.to_dict()
            if args.summary_output:
                Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.summary_output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if summary.valid else 2
        if args.registry_command == "collection-summary":
            plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
            summary = merge_source_export_summaries(
                load_source_records(args.source),
                [
                    json.loads(Path(summary_path).read_text(encoding="utf-8"))
                    for summary_path in args.summary
                ],
                collection_plan=plan,
                output_path=args.source,
                allow_partial=args.allow_partial,
            )
            payload = summary.to_dict()
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if summary.valid else 2
        if args.registry_command == "collection-split":
            summary = split_public_source_records(
                load_source_records(args.source),
                output_path=args.output,
                include_source_types=args.include_source_type,
                exclude_source_types=args.exclude_source_type,
            )
            payload = summary.to_dict()
            if args.summary_output:
                Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.summary_output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if summary.valid else 2
        if args.registry_command == "collection-audit":
            records = load_source_records(args.source)
            allowlist = load_repository_allowlist(args.allowlist)
            if args.accepted_output:
                accepted_records, audit = filter_accepted_public_source_records(
                    records,
                    allowlist,
                    source_name=args.source_name,
                )
                accepted_output = Path(args.accepted_output)
                accepted_output.parent.mkdir(parents=True, exist_ok=True)
                accepted_output.write_text(
                    "".join(
                        json.dumps(record, sort_keys=True) + "\n"
                        for record in accepted_records
                    ),
                    encoding="utf-8",
                )
            else:
                audit = audit_public_source_records(
                    records,
                    allowlist,
                    source_name=args.source_name,
                )
            payload = audit.to_dict()
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if audit.valid else 2
        if args.registry_command == "collection-readiness":
            readiness = summarize_source_collection_readiness(
                json.loads(Path(args.plan).read_text(encoding="utf-8")),
                json.loads(Path(args.export_summary).read_text(encoding="utf-8")),
                json.loads(Path(args.audit).read_text(encoding="utf-8")),
                min_accepted=args.min_accepted,
                max_quarantined=args.max_quarantined,
                require_clean_export=args.require_clean_export,
                require_all_plan_tasks=args.require_all_plan_tasks,
                required_source_types=args.require_source_type,
            )
            payload = readiness.to_dict()
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if readiness.ready_for_import else 2
        if args.registry_command == "import-rehearsal":
            benchmark_sources = sorted(
                set(DEFAULT_BENCHMARK_SOURCE_ALIASES) | set(args.benchmark_source)
            )
            payload = rehearse_registry_import(
                registry_root=args.root,
                source_path=args.source,
                source_format=args.format,
                source_name=args.source_name,
                allowlist_path=args.allowlist or None,
                split=args.split,
                license_name=args.license,
                permitted_use=args.permitted_use,
                test_command_template=args.test_command_template,
                task_family=args.task_family,
                source_method=args.source_method,
                train_eligible=_parse_train_eligible(args.train_eligible),
                contamination_tags=args.contamination_tag,
                coverage_tags=args.coverage_tag,
                allow_train_licenses=args.allow_train_license,
                limit=args.limit,
                strict=args.strict,
                overwrite_registry=args.overwrite_registry,
                min_imported=args.min_imported,
                max_quarantined=args.max_quarantined,
                seed_policy=SeedLibraryPolicy(
                    min_train_eligible=args.min_train_eligible,
                    required_task_families=args.require_task_family,
                    required_verifier_types=args.require_verifier_type,
                    max_task_family_share=args.max_task_family_share,
                    max_source_method_share=args.max_source_method_share,
                    max_repository_share=args.max_repository_share,
                    max_language_share=args.max_language_share,
                ),
                benchmark_sources=benchmark_sources,
                materialize_sample_count=args.materialize_sample_count,
                materialize_root=args.materialize_root or None,
                run_hidden_commands=args.run_hidden_commands,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["valid"] else 2
        if args.registry_command == "seed-backfill-plan":
            payload = build_seed_backfill_plan(
                json.loads(Path(args.audit).read_text(encoding="utf-8")),
                json.loads(Path(args.policy).read_text(encoding="utf-8")),
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["valid"] else 2
        if args.registry_command == "seed-selection-plan":
            payload = build_seed_selection_plan(
                ScenarioRegistry(args.root).list_seeds(),
                json.loads(Path(args.policy).read_text(encoding="utf-8")),
                target_train_eligible=args.target_train_eligible,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["valid"] else 2
        if args.registry_command == "seed-synthetic-backfill-spec":
            payload = build_synthetic_backfill_spec_plan(
                scenarios_from_registry(ScenarioRegistry(args.root)),
                json.loads(Path(args.selection_plan).read_text(encoding="utf-8")),
                json.loads(Path(args.backfill_plan).read_text(encoding="utf-8")),
                max_repositories=args.max_repositories,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if args.spec_output:
                Path(args.spec_output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.spec_output).write_text(
                    json.dumps(payload["generator_ready_specs"], indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload["valid"] else 2
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
        elif args.registry_command == "seed-audit":
            benchmark_sources = sorted(
                set(DEFAULT_BENCHMARK_SOURCE_ALIASES) | set(args.benchmark_source)
            )
            seeds = registry.list_seeds()
            holdout_seeds = list(seeds)
            for root in args.holdout_root:
                holdout_seeds.extend(ScenarioRegistry(root).list_seeds())
            policy = SeedLibraryPolicy(
                min_train_eligible=args.min_train_eligible,
                required_task_families=args.require_task_family,
                required_verifier_types=args.require_verifier_type,
                max_task_family_share=args.max_task_family_share,
                max_source_method_share=args.max_source_method_share,
                max_repository_share=args.max_repository_share,
                max_language_share=args.max_language_share,
            )
            audit = audit_seed_library(
                seeds,
                benchmark_sources=benchmark_sources,
                policy=policy,
                holdout_seeds=holdout_seeds,
            )
            payload = audit.to_dict()
            if args.output:
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if audit.valid else 2
        elif args.registry_command == "scenario-audit":
            benchmark_sources = sorted(
                set(DEFAULT_BENCHMARK_SOURCE_ALIASES) | set(args.benchmark_source)
            )
            scenarios = scenarios_from_registry(registry)
            holdout_scenarios = list(scenarios)
            for root in args.holdout_root:
                holdout_scenarios.extend(scenarios_from_registry(ScenarioRegistry(root)))
            audit = audit_scenario_decontamination(
                scenarios,
                benchmark_sources=benchmark_sources,
                holdout_scenarios=holdout_scenarios,
            )
            payload = audit.to_dict()
            if args.output:
                Path(args.output).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if audit.valid else 2
        elif args.registry_command == "review-queue":
            queue = build_seed_review_queue(
                scenarios_from_registry(registry),
                sample_per_stratum=args.sample_per_stratum,
                max_records=args.max_records,
            )
            payload = queue.to_dict()
            if args.output:
                output = Path(args.output)
                if args.overwrite:
                    output.unlink(missing_ok=True)
                for record in queue.records:
                    enqueue_human_review(output, record)
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.registry_command == "import":
            records = load_source_records(args.source)
            source_format = args.format.replace("-", "_")
            train_eligible = _parse_train_eligible(args.train_eligible)
            if source_format in PUBLIC_ISSUE_PR_FORMATS:
                summary = import_public_issue_pr_records(
                    registry,
                    records,
                    source_format=args.format,
                    source_name=args.source_name,
                    split=args.split,
                    license_name=args.license,
                    permitted_use=args.permitted_use,
                    limit=args.limit,
                    test_command_template=args.test_command_template,
                    task_family=args.task_family,
                    source_method=args.source_method,
                    train_eligible=train_eligible,
                    contamination_tags=args.contamination_tag,
                    coverage_tags=args.coverage_tag,
                    train_license_allowlist=sorted(
                        set(DEFAULT_TRAIN_LICENSE_ALLOWLIST) | set(args.allow_train_license)
                    ),
                    strict=args.strict,
                )
            elif source_format in PUBLIC_CI_FORMATS:
                summary = import_public_ci_records(
                    registry,
                    records,
                    source_format=args.format,
                    source_name=args.source_name,
                    split=args.split,
                    license_name=args.license,
                    permitted_use=args.permitted_use,
                    limit=args.limit,
                    task_family=args.task_family,
                    source_method=args.source_method,
                    train_eligible=train_eligible,
                    contamination_tags=args.contamination_tag,
                    coverage_tags=args.coverage_tag,
                    train_license_allowlist=sorted(
                        set(DEFAULT_TRAIN_LICENSE_ALLOWLIST) | set(args.allow_train_license)
                    ),
                    strict=args.strict,
                )
            else:
                summary = import_swe_style_records(
                    registry,
                    records,
                    source_format=args.format,
                    source_name=args.source_name,
                    split=args.split,
                    license_name=args.license,
                    permitted_use=args.permitted_use,
                    limit=args.limit,
                    test_command_template=args.test_command_template,
                    task_family=args.task_family,
                    source_method=args.source_method,
                    train_eligible=train_eligible,
                    contamination_tags=args.contamination_tag,
                    coverage_tags=args.coverage_tag,
                    strict=args.strict,
                )
            print(json.dumps(summary.to_dict(), indent=2))
        elif args.registry_command == "generate-synthetic":
            summary = generate_repository_synthetic_scenarios(
                registry,
                load_repository_synthesis_specs(args.source),
                source_name=args.source_name,
                split=args.split,
                task_families=args.task_family,
                train_eligible=_parse_train_eligible(args.train_eligible),
                train_license_allowlist=sorted(
                    set(DEFAULT_SYNTHETIC_TRAIN_LICENSE_ALLOWLIST)
                    | set(args.allow_train_license)
                ),
                limit=args.limit,
                strict=args.strict,
            )
            print(json.dumps(summary.to_dict(), indent=2))
        return 0
    if args.command == "agent-run":
        outcome = _run_registry_scenario(
            ScenarioRegistry(args.registry),
            args.scenario_id,
            load_config(args.config),
            Path(args.trace),
            args.random_seed,
            _agent_budgets(args),
        )
        print(json.dumps(asdict(outcome), indent=2))
        return 0 if outcome.trace_id else 1
    if args.command == "batch":
        scheduler = PersistentScheduler(args.database) if hasattr(args, "database") else None
        if args.batch_command == "enqueue":
            assert scheduler is not None
            scenarios = _filter_scenarios_for_enqueue(
                ScenarioRegistry(args.registry).list_scenarios(),
                selection_file=Path(args.selection_file) if args.selection_file else None,
                scenario_ids=args.scenario_id,
            )
            scheduler.submit(
                RolloutJob(scenario["scenario_id"], rollout, args.model, args.config_hash)
                for scenario in scenarios
                for rollout in range(args.rollouts)
            )
            print(json.dumps(scheduler.status_counts(), indent=2))
        elif args.batch_command == "status":
            assert scheduler is not None
            print(json.dumps(scheduler.status_counts(), indent=2))
        elif args.batch_command == "report":
            assert scheduler is not None
            selected_job_ids = _selected_job_ids_for_run(
                explicit_job_ids=args.job_id,
                job_id_file=Path(args.job_id_file) if args.job_id_file else None,
                shard_index=args.shard_index,
            )
            rows = _reportable_rows(scheduler.rows(), selected_job_ids)
            report = quality_report(rows)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            if args.review_sample and args.sample_size > 0:
                if args.overwrite_review_sample:
                    Path(args.review_sample).unlink(missing_ok=True)
                for record in _review_sample_rows(
                    rows,
                    sample_size=args.sample_size,
                    trace_directory=Path(args.trace_directory) if args.trace_directory else None,
                ):
                    enqueue_human_review(args.review_sample, record)
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.batch_command == "select-scale-candidates":
            assert scheduler is not None
            rows = [
                row for row in scheduler.rows() if row.get("status") not in {"pending", "running"}
            ]
            selection = select_scale_candidates(
                rows,
                audit=json.loads(Path(args.audit).read_text(encoding="utf-8"))
                if args.audit
                else None,
                min_rollouts=args.min_rollouts,
                min_success_rate=args.min_success_rate,
                min_hidden_command_pass_rate=args.min_hidden_command_pass_rate,
                min_all_non_agent_pass_rate=args.min_all_non_agent_pass_rate,
                min_agent_stop_rate=args.min_agent_stop_rate,
                min_high_quality_rate=args.min_high_quality_rate,
                min_closed_loop_rate=args.min_closed_loop_rate,
                min_multi_step_complex_rate=args.min_multi_step_complex_rate,
                max_infrastructure_failure_rate=args.max_infrastructure_failure_rate,
                min_average_tool_calls=args.min_average_tool_calls,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(selection, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
            )
            print(json.dumps(selection, indent=2, sort_keys=True))
        elif args.batch_command == "estimate-scale":
            assert scheduler is not None
            pilot_scheduler = PersistentScheduler(args.pilot_database)
            estimate = estimate_scale_run(
                scheduler.rows(),
                pilot_scheduler.rows(),
                shard_size=args.shard_size,
                cost_per_million_tokens=args.cost_per_million_tokens,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(estimate, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
            )
            print(json.dumps(estimate, indent=2, sort_keys=True))
        elif args.batch_command == "shard-status":
            assert scheduler is not None
            payload = json.loads(Path(args.job_id_file).read_text(encoding="utf-8"))
            job_ids = _selected_job_ids_for_run(
                explicit_job_ids=[],
                job_id_file=Path(args.job_id_file),
                shard_index=args.shard_index,
            )
            status = selected_job_status(scheduler.rows(), job_ids or [])
            if isinstance(payload, dict) and isinstance(payload.get("shards"), list):
                status["estimate"] = payload["shards"][args.shard_index]
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(status, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(status, indent=2, sort_keys=True))
        elif args.batch_command == "decide-continuation":
            decision = scale_continuation_decision(
                json.loads(Path(args.report).read_text(encoding="utf-8")),
                json.loads(Path(args.status).read_text(encoding="utf-8")),
                audit=json.loads(Path(args.audit).read_text(encoding="utf-8"))
                if args.audit
                else None,
                min_success_rate=args.min_success_rate,
                min_unique_traces=args.min_unique_traces,
                min_hidden_command_pass_rate=args.min_hidden_command_pass_rate,
                min_high_quality_rate=args.min_high_quality_rate,
                min_closed_loop_rate=args.min_closed_loop_rate,
                min_multi_step_complex_rate=args.min_multi_step_complex_rate,
                max_infrastructure_failures=args.max_infrastructure_failures,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(decision, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
            )
            print(json.dumps(decision, indent=2, sort_keys=True))
        elif args.batch_command == "audit-traces":
            assert scheduler is not None
            selected_job_ids = _selected_job_ids_for_run(
                explicit_job_ids=args.job_id,
                job_id_file=Path(args.job_id_file) if args.job_id_file else None,
                shard_index=args.shard_index,
            )
            audit = audit_trace_logic(
                scheduler.rows(),
                args.trace_directory,
                job_ids=selected_job_ids,
                include_items=not args.summary_only,
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(audit, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(audit, indent=2, sort_keys=True))
        elif args.batch_command == "scale-readiness":
            readiness = scale_readiness_summary(
                selection=json.loads(Path(args.selection).read_text(encoding="utf-8")),
                estimate=json.loads(Path(args.estimate).read_text(encoding="utf-8")),
                status=json.loads(Path(args.status).read_text(encoding="utf-8")),
                audit=json.loads(Path(args.audit).read_text(encoding="utf-8")),
                decision=json.loads(Path(args.decision).read_text(encoding="utf-8")),
            )
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(
                    json.dumps(readiness, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps(readiness, indent=2, sort_keys=True))
        elif args.batch_command == "run":
            assert scheduler is not None
            selected_job_ids = _selected_job_ids_for_run(
                explicit_job_ids=args.job_id,
                job_id_file=Path(args.job_id_file) if args.job_id_file else None,
                shard_index=args.shard_index,
            )
            if args.dry_run:
                plan = planned_batch_run(
                    scheduler.rows(),
                    job_ids=selected_job_ids,
                    max_jobs=args.max_jobs,
                )
                print(
                    json.dumps(
                        {
                            "dry_run": True,
                            "database": args.database,
                            "registry": args.registry,
                            "config": args.config,
                            "trace_directory": args.trace_directory,
                            "max_workers": args.max_workers,
                            "max_retries": args.max_retries,
                            "max_jobs": args.max_jobs,
                            "budgets": {
                                "max_seconds": args.max_seconds,
                                "max_tokens": args.max_tokens,
                                "max_cost": args.max_cost,
                                "max_agent_turns": args.max_agent_turns,
                                "max_agent_tool_calls": args.max_agent_tool_calls,
                                "max_agent_tokens": args.max_agent_tokens,
                                "max_agent_seconds": args.max_agent_seconds,
                            },
                            **plan,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            worker = _CLIRolloutWorker(
                ScenarioRegistry(args.registry),
                load_config(args.config),
                Path(args.trace_directory),
                _agent_budgets(args),
            )
            summary = scheduler.run(
                worker,
                max_workers=args.max_workers,
                max_retries=args.max_retries,
                budget=RunBudget(
                    max_seconds=args.max_seconds,
                    max_tokens=args.max_tokens,
                    max_cost=args.max_cost,
                ),
                max_jobs=args.max_jobs,
                job_ids=selected_job_ids,
            )
            print(json.dumps(summary, indent=2))
        return 0
    return 1


def _review_sample_rows(
    rows: Sequence[dict[str, object]],
    *,
    sample_size: int,
    trace_directory: Path | None,
) -> list[dict[str, object]]:
    candidates = sorted(
        rows,
        key=lambda row: (
            str(row.get("status") or "") == "completed",
            str(row.get("status") or ""),
            str(row.get("job_id") or ""),
        ),
    )
    records: list[dict[str, object]] = []
    for row in candidates[:sample_size]:
        record: dict[str, object] = {
            "job_id": row.get("job_id", ""),
            "scenario_id": row.get("scenario_id", ""),
            "rollout_index": row.get("rollout_index", 0),
            "model": row.get("model", ""),
            "status": row.get("status", ""),
            "success": bool(row.get("success", 0)),
            "trace_id": row.get("trace_id", ""),
            "reason": "batch quality sample",
        }
        if trace_directory is not None and row.get("job_id"):
            record["trace_path"] = str(trace_directory / f"{row['job_id']}.jsonl")
        records.append(record)
    return records


def _reportable_rows(
    rows: Sequence[dict[str, object]],
    selected_job_ids: Sequence[str] | None,
) -> list[dict[str, object]]:
    selected = set(selected_job_ids or [])
    return [
        row
        for row in rows
        if row.get("status") not in {"pending", "running"}
        and (not selected or str(row.get("job_id") or "") in selected)
    ]


def _filter_scenarios_for_enqueue(
    scenarios: Sequence[dict[str, object]],
    *,
    selection_file: Path | None,
    scenario_ids: Sequence[str],
) -> list[dict[str, object]]:
    filter_requested = bool(scenario_ids) or selection_file is not None
    selected = set(str(value) for value in scenario_ids)
    if selection_file is not None:
        payload = json.loads(selection_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            file_ids = payload.get("candidates", [])
        else:
            file_ids = payload
        if not isinstance(file_ids, list) or not all(isinstance(item, str) for item in file_ids):
            raise ValueError("Selection file must contain a candidates list of scenario IDs")
        selected = set(file_ids) if not selected else selected & set(file_ids)
    if not filter_requested:
        return list(scenarios)
    return [scenario for scenario in scenarios if str(scenario.get("scenario_id")) in selected]


def _selected_job_ids_for_run(
    *,
    explicit_job_ids: Sequence[str],
    job_id_file: Path | None,
    shard_index: int | None,
) -> list[str] | None:
    selected = list(explicit_job_ids)
    if job_id_file is not None:
        payload = json.loads(job_id_file.read_text(encoding="utf-8"))
        if shard_index is not None:
            if not isinstance(payload, dict) or not isinstance(payload.get("shards"), list):
                raise ValueError("Shard selection requires an estimate file with a shards list")
            shards = payload["shards"]
            if shard_index < 0 or shard_index >= len(shards):
                raise ValueError(f"Shard index {shard_index} is outside the estimate shard range")
            file_ids = shards[shard_index].get("job_ids", [])
        elif isinstance(payload, dict):
            file_ids = payload.get("job_ids", [])
        else:
            file_ids = payload
        if not isinstance(file_ids, list) or not all(isinstance(item, str) for item in file_ids):
            raise ValueError("Job selection file must contain a list of job IDs")
        selected.extend(file_ids)
    elif shard_index is not None:
        raise ValueError("--shard-index requires --job-id-file")
    if not selected:
        return None
    return sorted(set(selected))


class _CLIRolloutWorker:
    def __init__(
        self,
        registry: ScenarioRegistry,
        config: PipelineConfig,
        trace_directory: Path,
        budgets: AgentBudgets | None = None,
    ):
        self.registry = registry
        self.config = config
        self.trace_directory = trace_directory
        self.budgets = budgets

    def run(self, job: RolloutJob) -> RolloutOutcome:
        trace_path = self.trace_directory / f"{job.job_id}.jsonl"
        if trace_path.exists():
            return _rollout_outcome_from_existing_trace(trace_path)
        try:
            return _run_registry_scenario(
                self.registry,
                job.scenario_id,
                self.config,
                trace_path,
                job.rollout_index,
                self.budgets,
            )
        except Exception as exc:
            return RolloutOutcome(
                infrastructure_failure=True,
                error=f"{type(exc).__name__}: {exc}",
            )


def _rollout_outcome_from_existing_trace(trace_path: Path) -> RolloutOutcome:
    trace = load_trace(trace_path)
    if not trace.events or trace.events[-1].event_type.value != "session_finished":
        return RolloutOutcome(
            infrastructure_failure=True,
            error=f"Incomplete existing trace: {trace_path}",
        )
    success = False
    tokens = 0
    tool_calls = 0
    metrics: dict[str, float] = {}
    for event in trace.events:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.event_type.value == "model_response":
            usage = payload.get("usage", {})
            if isinstance(usage, dict):
                tokens += int(usage.get("total_tokens", 0) or 0)
        elif event.event_type.value == "tool_requested":
            tool_calls += 1
        elif event.event_type.value == "verification_result":
            verifier = str(payload.get("verifier") or "").replace("-", "_")
            if verifier:
                metrics[f"verifier_{verifier}_passed"] = (
                    1.0 if bool(payload.get("passed", False)) else 0.0
                )
        elif event.event_type.value == "session_finished":
            success = bool(payload.get("success", False))
    non_agent_verifiers = [
        value
        for key, value in metrics.items()
        if key.startswith("verifier_") and key != "verifier_agent_termination_passed"
    ]
    metrics["verifier_all_non_agent_passed"] = (
        1.0 if non_agent_verifiers and all(value == 1.0 for value in non_agent_verifiers) else 0.0
    )
    metrics["tool_calls"] = float(tool_calls)
    metrics["tokens"] = float(tokens)
    return RolloutOutcome(
        trace_id=trace.trace_id,
        success=success,
        tokens=tokens,
        metrics=metrics,
    )


def _run_registry_scenario(
    registry: ScenarioRegistry,
    scenario_id: str,
    config: PipelineConfig,
    trace_path: Path,
    random_seed: int,
    budgets: AgentBudgets | None = None,
) -> RolloutOutcome:
    scenario = registry.get_scenario(scenario_id)
    with tempfile.TemporaryDirectory() as directory:
        source = materialize_environment_source(
            scenario.environment,
            directory,
            run_health_checks=False,
        )
        limits = SandboxLimits(**scenario.environment.resource_limits)
        sandbox = DockerSandbox(
            image_digest=scenario.environment.image_digest,
            source_directory=source,
            limits=limits,
            network_enabled=scenario.environment.network_policy != "disabled",
        )
        sandbox.create()
        try:
            _run_setup_commands(sandbox, scenario.environment.setup_commands)
            _run_health_check_commands(sandbox, scenario.environment.health_check)
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
                run_result = HeadlessAgent(client, tools, budgets=budgets).run(
                    instance,
                    recorder,
                    ask_user=user_callback(user, instance),
                    finalize=False,
                )
                trace = load_trace(trace_path)
                evaluators = _deterministic_evaluators(instance, trace)
                turn_rewards = derive_turn_rewards(trace, instance)
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
                report = apply_agent_termination(report, run_result.termination_reason)
                finalize_evaluation_trace(
                    recorder,
                    report,
                    final_state_hash=sandbox.state_hash(),
                    termination_reason=run_result.termination_reason,
                )
            trace = load_trace(trace_path)
            return RolloutOutcome(
                trace_id=trace.trace_id,
                success=report.success,
                tokens=run_result.tokens,
                metrics={**report.metrics, **evaluation_result_metrics(report)},
            )
        finally:
            sandbox.destroy()


def _deterministic_evaluators(instance, trace=None):
    evaluators = []
    if instance.hidden_evaluator.metadata.get("test_patch"):
        evaluators.append(HiddenTestPatchEvaluator())
    evaluators.extend(
        HiddenCommandEvaluator(shlex.split(command))
        for command in instance.hidden_evaluator.hidden_tests
    )
    if instance.hidden_evaluator.required_state:
        evaluators.append(RequiredStateEvaluator())
    if instance.hidden_evaluator.forbidden_state:
        evaluators.append(ForbiddenStateEvaluator())
    retrieval_requirements = instance.hidden_evaluator.metadata.get("retrieval_requirements", [])
    trace_quality_rubric = instance.hidden_evaluator.metadata.get("trace_quality_rubric", [])
    if retrieval_requirements or trace_quality_rubric:
        evaluators.append(
            TraceRequirementEvaluator(
                trace,
                retrieval_requirements=retrieval_requirements,
                trace_quality_rubric=trace_quality_rubric,
            )
        )
    return evaluators


def _run_setup_commands(sandbox: DockerSandbox, commands: Sequence[str]) -> None:
    for command in commands:
        result = sandbox.execute_as_root(shlex.split(command))
        if result.exit_code != 0:
            raise RuntimeError(
                "Environment setup command failed "
                f"({command!r}, exit={result.exit_code}): "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )


def _run_health_check_commands(sandbox: DockerSandbox, commands: Sequence[str]) -> None:
    for command in commands:
        result = sandbox.execute(shlex.split(command))
        if result.exit_code != 0:
            raise RuntimeError(
                "Environment health check failed "
                f"({command!r}, exit={result.exit_code}): "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )


def _agent_budgets(args) -> AgentBudgets:
    return AgentBudgets(
        max_turns=args.max_agent_turns,
        max_tool_calls=args.max_agent_tool_calls,
        max_tokens=args.max_agent_tokens,
        max_seconds=getattr(args, "max_agent_seconds", 600.0),
    )


if __name__ == "__main__":
    raise SystemExit(main())
