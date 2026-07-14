from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import easy_agentic_data.agent.runtime as agent_runtime_module
import easy_agentic_data.batch as batch_module
import easy_agentic_data.coding_tools as coding_tools_module
import easy_agentic_data.config as config_module
import easy_agentic_data.environments.models as environment_models_module
import easy_agentic_data.evaluation as evaluation_module
import easy_agentic_data.llm.base as llm_base_module
import easy_agentic_data.llm.observability as llm_observability_module
import easy_agentic_data.llm.openai_compatible as llm_adapter_module
import easy_agentic_data.models as models_module
import easy_agentic_data.pilot_artifacts as pilot_artifacts_module
import easy_agentic_data.pilot_contract as pilot_contract_module
import easy_agentic_data.pilot_usage_ledger as pilot_usage_ledger_module
import easy_agentic_data.policy as policy_module
import easy_agentic_data.registry as registry_module
import easy_agentic_data.registry_rollouts as registry_rollouts_module
import easy_agentic_data.sandbox.base as sandbox_base_module
import easy_agentic_data.scenarios as scenarios_module
import easy_agentic_data.seeds.models as seed_models_module
import easy_agentic_data.simulation as simulation_module
import easy_agentic_data.trace_exporters as trace_exporters_module
import easy_agentic_data.traces.artifacts as trace_artifacts_module
import easy_agentic_data.traces.events as trace_events_module
import easy_agentic_data.traces.recorder as trace_recorder_module
import easy_agentic_data.traces.replay as trace_replay_module
import easy_agentic_data.trajectory_review as trajectory_review_module
from easy_agentic_data.agent import DEFAULT_SYSTEM_PROMPT, AgentBudgets
from easy_agentic_data.batch import JobStatus, PersistentScheduler, RolloutJob, RolloutOutcome
from easy_agentic_data.coding_tools import SCHEMAS, CodingToolRuntime
from easy_agentic_data.config import LLMConfig
from easy_agentic_data.evaluation import (
    EvaluationReport,
    contamination_findings,
    evaluation_result_metrics,
    public_evaluation_result,
)
from easy_agentic_data.models import stable_id
from easy_agentic_data.pilot_contract import (
    Gold20Binding,
    PilotBudgets,
    PilotRolloutAssignment,
    PilotRunContract,
    PilotVersionHashes,
    PricingSpec,
    ProviderConfigBinding,
    canonical_sha256,
)
from easy_agentic_data.pilot_usage_ledger import (
    PilotUsageAttempt,
    UnknownProviderUsageError,
    audit_pilot_usage_ledger,
    load_pilot_job_usage,
    recover_running_pilot_usage_attempt,
)
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_rollouts import (
    RolloutArtifactPaths,
    deterministic_evaluators,
    publish_registry_rollout,
    run_registry_rollout,
    safe_error_message,
)
from easy_agentic_data.sandbox import MemorySandbox
from easy_agentic_data.sandbox import docker as docker_module
from easy_agentic_data.traces import EventType, load_trace, replay_trace

M2_GOLD20_CORPUS_ID = "gold20_8757dfd30b43612a"
M2_GOLD20_MANIFEST_SHA256 = "4bf92be3d308891e47ad1b3ca71e123dfca6c404dbf7dee5fe3cd354e43325fd"


def build_pilot_run_contract(
    manifest: Mapping[str, Any] | str | Path,
    registry: ScenarioRegistry,
    config: LLMConfig,
    *,
    budgets: PilotBudgets,
    pricing: PricingSpec,
    rollout_seeds: tuple[int, int] = (0, 1),
) -> PilotRunContract:
    """Bind the exact Gold-20 registry, provider settings, budgets, and code versions."""

    corpus = Gold20Binding.from_manifest(manifest)
    if (
        corpus.corpus_id != M2_GOLD20_CORPUS_ID
        or corpus.manifest_sha256 != M2_GOLD20_MANIFEST_SHA256
    ):
        raise ValueError("M2 requires the repository's exact frozen Gold-20 manifest")
    _validate_gold20_registry(corpus, registry)
    versions = current_pilot_versions(corpus, registry)
    return PilotRunContract(
        corpus=corpus,
        provider=ProviderConfigBinding.from_config(config),
        budgets=budgets,
        versions=versions,
        pricing=pricing,
        rollout_seeds=rollout_seeds,
    )


def current_pilot_versions(
    corpus: Gold20Binding,
    registry: ScenarioRegistry,
) -> PilotVersionHashes:
    tool_schemas: dict[str, list[dict[str, Any]]] = {}
    schema_sandbox = MemorySandbox()
    for scenario_id in corpus.scenario_ids:
        scenario = registry.get_scenario(scenario_id)
        policy = ToolPolicy(
            scenario.environment.capability_packs or SCHEMAS.keys(),
            network_enabled=scenario.environment.network_policy != "disabled",
        )
        tool_schemas[scenario_id] = CodingToolRuntime(schema_sandbox, policy).schemas()
    return PilotVersionHashes(
        prompt_sha256=_sha256_text(DEFAULT_SYSTEM_PROMPT),
        tool_schema_sha256=canonical_sha256(
            {
                "schemas": tool_schemas,
                "coding_tools": _module_sha256(coding_tools_module),
                "policy": _module_sha256(policy_module),
                "agent_runtime": _module_sha256(agent_runtime_module),
                "models": _module_sha256(models_module),
            }
        ),
        evaluator_sha256=canonical_sha256(
            {
                "corpus_evaluators": corpus.evaluator_bundle_sha256,
                "implementation": _module_sha256(evaluation_module),
            }
        ),
        environment_sha256=canonical_sha256(
            {
                "corpus_environments": corpus.environment_bundle_sha256,
                "docker_sandbox": _module_sha256(docker_module),
                "sandbox_base": _module_sha256(sandbox_base_module),
                "batch_scheduler": _module_sha256(batch_module),
                "config": _module_sha256(config_module),
                "environment_models": _module_sha256(environment_models_module),
                "seed_models": _module_sha256(seed_models_module),
                "scenarios": _module_sha256(scenarios_module),
                "registry": _module_sha256(registry_module),
                "registry_rollout": _module_sha256(registry_rollouts_module),
                "simulation": _module_sha256(simulation_module),
                "pilot_cli": _sha256_bytes(Path(__file__).with_name("cli.py").read_bytes()),
                "pilot_workflow": _sha256_bytes(Path(__file__).read_bytes()),
                "pilot_contract": _module_sha256(pilot_contract_module),
                "pilot_usage_ledger": _module_sha256(pilot_usage_ledger_module),
                "llm_base": _module_sha256(llm_base_module),
                "llm_adapter": _module_sha256(llm_adapter_module),
                "llm_observability": _module_sha256(llm_observability_module),
                "trace_artifacts": _module_sha256(trace_artifacts_module),
                "trace_events": _module_sha256(trace_events_module),
                "trace_recorder": _module_sha256(trace_recorder_module),
                "trace_replay": _module_sha256(trace_replay_module),
            }
        ),
        exporter_sha256=canonical_sha256(
            {
                "trace_exporters": _module_sha256(trace_exporters_module),
                "pilot_artifacts": _module_sha256(pilot_artifacts_module),
                "trajectory_review": _module_sha256(trajectory_review_module),
            }
        ),
    )


def load_pilot_run_contract(path: str | Path) -> PilotRunContract:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Pilot contract must be a JSON object")
    return PilotRunContract.from_dict(value)


def write_pilot_run_contract(path: str | Path, contract: PilotRunContract) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def validate_pilot_runtime(
    contract: PilotRunContract,
    registry: ScenarioRegistry,
    config: LLMConfig,
) -> None:
    actual_provider = ProviderConfigBinding.from_config(config)
    if actual_provider != contract.provider:
        raise ValueError("Runtime provider configuration does not match the pilot contract")
    validate_pilot_versions(contract, registry)


def validate_pilot_versions(
    contract: PilotRunContract,
    registry: ScenarioRegistry,
) -> None:
    """Require the frozen Gold-20 registry and implementation bound by a contract."""

    _validate_gold20_registry(contract.corpus, registry)
    if current_pilot_versions(contract.corpus, registry) != contract.versions:
        raise ValueError("Runtime implementation versions do not match the pilot contract")


def validate_provider_availability(config: LLMConfig) -> None:
    if config.provider == "openai_compatible":
        variable = config.api_key_env or ""
        if not variable or not os.environ.get(variable):
            variable_name = variable or "<unset>"
            raise RuntimeError(
                f"Pilot provider credential is missing from environment variable {variable_name}"
            )


def submit_pilot_jobs(
    scheduler: PersistentScheduler,
    contract: PilotRunContract,
) -> list[str]:
    existing = scheduler.rows()
    expected = {assignment.job_id: assignment for assignment in contract.rollouts}
    expected_ids = set(expected)
    unexpected = sorted(str(row["job_id"]) for row in existing if row["job_id"] not in expected_ids)
    if unexpected:
        raise ValueError(f"Pilot database contains jobs outside the contract: {unexpected}")
    for row in existing:
        job_id = str(row.get("job_id") or "")
        _validate_pilot_queue_row(contract, expected[job_id], row)
    scheduler.submit(
        RolloutJob(
            assignment.scenario_id,
            assignment.rollout_index,
            contract.provider.model,
            contract.contract_id,
            assignment.job_id,
        )
        for assignment in contract.rollouts
    )
    validate_pilot_jobs(scheduler, contract)
    return list(expected)


def validate_pilot_jobs(
    scheduler: PersistentScheduler,
    contract: PilotRunContract,
) -> list[dict[str, Any]]:
    rows = scheduler.rows()
    rows_by_id = {str(row.get("job_id") or ""): row for row in rows}
    expected = {assignment.job_id: assignment for assignment in contract.rollouts}
    if set(rows_by_id) != set(expected):
        missing = sorted(set(expected) - set(rows_by_id))
        unexpected = sorted(set(rows_by_id) - set(expected))
        raise ValueError(f"Pilot queue mismatch: missing={missing} unexpected={unexpected}")
    for job_id, assignment in expected.items():
        _validate_pilot_queue_row(contract, assignment, rows_by_id[job_id])
    return rows


def _validate_pilot_queue_row(
    contract: PilotRunContract,
    assignment: PilotRolloutAssignment,
    row: Mapping[str, Any],
) -> None:
    fields = {
        "job_id": assignment.job_id,
        "scenario_id": assignment.scenario_id,
        "rollout_index": assignment.rollout_index,
        "model": contract.provider.model,
        "config_hash": contract.contract_id,
    }
    invalid = [key for key, value in fields.items() if row.get(key) != value]
    if invalid:
        raise ValueError(f"Pilot queue row mismatch for {assignment.job_id}: {invalid}")


class PilotRolloutWorker:
    """Contract-validating rollout worker with immutable retry attempts and real cost."""

    def __init__(
        self,
        registry: ScenarioRegistry,
        config: LLMConfig,
        trace_directory: str | Path,
        contract: PilotRunContract,
    ) -> None:
        validate_pilot_runtime(contract, registry, config)
        self.registry = registry
        self.config = config
        self.trace_directory = Path(trace_directory)
        self.contract = contract
        self.assignments = {assignment.job_id: assignment for assignment in contract.rollouts}
        self.budgets = AgentBudgets(
            max_turns=contract.budgets.max_agent_turns,
            max_tool_calls=contract.budgets.max_agent_tool_calls,
            max_tokens=contract.budgets.max_agent_tokens,
            max_seconds=contract.budgets.max_agent_seconds,
            malformed_tool_retries=contract.budgets.malformed_tool_retries,
        )

    def run(self, job: RolloutJob) -> RolloutOutcome:
        started = time.perf_counter()
        assignment = self.assignments.get(job.job_id)
        if assignment is None:
            return RolloutOutcome(error="Job is not part of the pilot contract")
        expected = (
            assignment.scenario_id,
            assignment.rollout_index,
            self.contract.provider.model,
            self.contract.contract_id,
        )
        actual = (job.scenario_id, job.rollout_index, job.model, job.config_hash)
        if actual != expected:
            return RolloutOutcome(error="Job fields do not match the pilot contract")
        trace_path = self.trace_directory / f"{job.job_id}.jsonl"
        if trace_path.exists():
            try:
                outcome = self._existing_outcome(trace_path, assignment)
                state = load_pilot_job_usage(
                    self.contract,
                    assignment.job_id,
                    self.trace_directory,
                )
                return replace(outcome, absolute_consumed_usage=state.totals)
            except Exception as exc:
                return RolloutOutcome(
                    error=safe_error_message(
                        PilotIntegrityError(f"Existing canonical rollout is invalid: {exc}")
                    ),
                    metrics={"elapsed_ms": (time.perf_counter() - started) * 1000},
                )
        usage_attempt = PilotUsageAttempt(
            self.trace_directory,
            contract_id=self.contract.contract_id,
            job_id=assignment.job_id,
        )
        attempt_finalized = False
        try:
            config_snapshot = copy.deepcopy(self.config)
            if ProviderConfigBinding.from_config(config_snapshot) != self.contract.provider:
                raise ValueError("Runtime provider configuration does not match the pilot contract")
            run_config = _config_for_assignment(config_snapshot, assignment.random_seed)
            result = run_registry_rollout(
                self.registry,
                assignment.scenario_id,
                run_config,
                trace_path,
                assignment.random_seed,
                self.budgets,
                cost_calculator=lambda usage: float(
                    self.contract.pricing.calculate_cost(usage).cost_usd
                ),
                run_contract_id=self.contract.contract_id,
                provider_binding_sha256=self.contract.provider.config_sha256,
                provider_binding=self.contract.provider.to_dict(),
                version_hashes=self.contract.versions.to_dict(),
                usage_attempt=usage_attempt,
                publish=False,
            )
            usage_cost = self.contract.pricing.calculate_cost(result.usage)
            staged_row = {
                "job_id": assignment.job_id,
                "scenario_id": assignment.scenario_id,
                "rollout_index": assignment.rollout_index,
                "model": self.contract.provider.model,
                "config_hash": self.contract.contract_id,
                "status": "completed",
                "trace_id": result.trace.trace_id,
                "success": int(result.report.success),
                "tokens": usage_cost.total_tokens,
                "cost": float(usage_cost.cost_usd),
                "metrics": result.metrics,
            }
            validated_artifact = pilot_artifacts_module.validate_pilot_rollout_artifact(
                self.contract,
                self.registry,
                assignment,
                staged_row,
                result.artifacts.trace,
                artifact_paths=result.artifacts,
            )
            outcome = RolloutOutcome(
                trace_id=result.trace.trace_id,
                success=result.report.success,
                infrastructure_failure=result.report.infrastructure_failure,
                tokens=usage_cost.total_tokens,
                cost=float(usage_cost.cost_usd),
                metrics=result.metrics,
            )
            usage_attempt.finalize(
                outcome,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
            attempt_finalized = True
            result = publish_registry_rollout(
                result,
                trace_path,
                validation_receipt=validated_artifact.validation_receipt,
            )
            state = load_pilot_job_usage(
                self.contract,
                assignment.job_id,
                self.trace_directory,
            )
            return replace(outcome, absolute_consumed_usage=state.totals)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if attempt_finalized:
                state = load_pilot_job_usage(
                    self.contract,
                    assignment.job_id,
                    self.trace_directory,
                )
                if state.latest_outcome is None:
                    raise RuntimeError("Finalized usage attempt has no terminal outcome") from exc
                return replace(
                    state.latest_outcome,
                    error=safe_error_message(exc),
                    absolute_consumed_usage=state.totals,
                )
            usage_error = None
            try:
                usage = usage_attempt.usage()
            except (UnknownProviderUsageError, ValueError) as accounting_exc:
                usage = {}
                usage_error = accounting_exc
            usage_cost = None
            if usage and usage_error is None:
                try:
                    usage_cost = self.contract.pricing.calculate_cost(usage)
                except ValueError as accounting_exc:
                    usage_error = accounting_exc
            retryable = _is_retryable_rollout_failure(exc) and usage_error is None
            error = safe_error_message(exc)
            if usage_error is not None:
                error = f"{error}; {safe_error_message(usage_error)}"
            outcome = RolloutOutcome(
                infrastructure_failure=retryable,
                tokens=usage_cost.total_tokens if usage_cost is not None else 0,
                cost=float(usage_cost.cost_usd) if usage_cost is not None else 0.0,
                metrics={
                    "elapsed_ms": elapsed_ms,
                    "failed_attempt_usage_known": 1.0 if usage_cost is not None else 0.0,
                    "failed_attempt_usage_invalid": 1.0 if usage_error is not None else 0.0,
                },
                error=error,
            )
            if usage_error is not None:
                return outcome
            usage_attempt.finalize(outcome, elapsed_ms=elapsed_ms)
            state = load_pilot_job_usage(
                self.contract,
                assignment.job_id,
                self.trace_directory,
            )
            return replace(outcome, absolute_consumed_usage=state.totals)

    def _existing_outcome(
        self,
        trace_path: Path,
        assignment: PilotRolloutAssignment,
    ) -> RolloutOutcome:
        trace = load_trace(trace_path, tolerate_truncated=False)
        replay = replay_trace(trace)
        if trace.truncated or not trace.events:
            raise PilotIntegrityError("Existing trace is incomplete")
        if trace.events[-1].event_type is not EventType.SESSION_FINISHED:
            raise PilotIntegrityError("Existing trace has no terminal event")
        paths = RolloutArtifactPaths.for_trace(trace_path)
        for path in (
            paths.candidate_patch,
            paths.private_evaluation,
            paths.run_evidence,
        ):
            if not path.is_file():
                raise PilotIntegrityError(f"Existing rollout sidecar is missing: {path.name}")
        candidate_patch = paths.candidate_patch.read_text(encoding="utf-8")
        private = _read_json(paths.private_evaluation)
        evidence = _read_json(paths.run_evidence)
        _validate_content_id(private, "private_evaluation_id", "private_evaluation")
        _validate_content_id(evidence, "evidence_id", "run_evidence")
        if private.get("schema") != "easy_agentic_data.private_evaluation.v1":
            raise PilotIntegrityError("Existing private evaluation schema is invalid")
        if evidence.get("schema") != "easy_agentic_data.registry_rollout_evidence.v1":
            raise PilotIntegrityError("Existing run evidence schema is invalid")
        report_value = private.get("report")
        if not isinstance(report_value, dict):
            raise PilotIntegrityError("Existing private evaluation report is invalid")
        report = EvaluationReport.from_dict(report_value)
        if report.to_dict() != report_value:
            raise PilotIntegrityError("Existing evaluation report has non-canonical values")
        if report.infrastructure_failure:
            raise PilotIntegrityError("Canonical rollout cannot contain infrastructure failure")
        if not any(item.evaluator != "agent_termination" for item in report.results):
            raise PilotIntegrityError("Existing report has no independent hard verifier")
        start = trace.events[0]
        if start.event_type is not EventType.SESSION_STARTED:
            raise PilotIntegrityError("Existing trace has no session start")
        if start.payload.get("scenario_id") != assignment.scenario_id:
            raise PilotIntegrityError("Existing trace scenario does not match assignment")
        if start.payload.get("random_seed") != assignment.random_seed:
            raise PilotIntegrityError("Existing trace seed does not match assignment")
        if start.payload.get("scenario_instance_id") != report.scenario_instance_id:
            raise PilotIntegrityError("Existing report does not match trace instance")
        instance = self.registry.materialize(
            assignment.scenario_id,
            random_seed=assignment.random_seed,
            initial_state_hash=str(start.payload.get("initial_state_hash") or ""),
        )
        if instance.instance_id != report.scenario_instance_id:
            raise PilotIntegrityError("Existing scenario materialization differs")
        start_expected = {
            "environment_id": instance.environment_id,
            "public_task": instance.public_task.to_dict(),
            "parameters": instance.parameters,
        }
        if any(start.payload.get(key) != value for key, value in start_expected.items()):
            raise PilotIntegrityError("Existing trace start payload differs")
        if contamination_findings(trace_path, instance):
            raise PilotIntegrityError("Existing public trace contains hidden context")
        expected_evaluator_names = [
            evaluator.name for evaluator in deterministic_evaluators(instance, trace)
        ] + ["agent_termination"]
        actual_evaluator_names = [item.evaluator for item in report.results]
        if actual_evaluator_names != expected_evaluator_names:
            raise PilotIntegrityError("Existing evaluator set does not match the scenario")
        expected_infrastructure_failure = any(
            item.infrastructure_failure for item in report.results
        )
        expected_success = (
            bool(report.results)
            and all(item.passed for item in report.results)
            and not expected_infrastructure_failure
        )
        if (
            report.infrastructure_failure is not expected_infrastructure_failure
            or report.success is not expected_success
            or report.reward != (1 if expected_success else 0)
        ):
            raise PilotIntegrityError("Existing report aggregate fields are inconsistent")
        expected_public_results = []
        for result in report.results:
            public = public_evaluation_result(result)
            expected_public_results.append({"verifier": public.pop("evaluator"), **public})
        actual_public_results = [
            event.payload
            for event in trace.events
            if event.event_type is EventType.VERIFICATION_RESULT
        ]
        if actual_public_results != expected_public_results:
            raise PilotIntegrityError("Existing public and private evaluations differ")
        if replay.state.success is not report.success:
            raise PilotIntegrityError("Existing trace success does not match evaluation")
        if replay.terminal_state_hash != evidence.get("candidate_state_hash"):
            raise PilotIntegrityError("Existing candidate state hash does not match trace")
        patch_sha256 = _sha256_text(candidate_patch)
        termination_results = [
            item for item in report.results if item.evaluator == "agent_termination"
        ]
        if len(termination_results) != 1:
            raise PilotIntegrityError("Existing report has invalid agent termination evidence")
        agent_termination_reason = termination_results[0].evidence.get("termination_reason")
        if not isinstance(agent_termination_reason, str) or not agent_termination_reason:
            raise PilotIntegrityError("Existing agent termination reason is invalid")
        scenario = self.registry.get_scenario(assignment.scenario_id)
        expected = {
            "run_contract_id": self.contract.contract_id,
            "provider_binding_sha256": self.contract.provider.config_sha256,
            "trace_id": trace.trace_id,
            "scenario_id": assignment.scenario_id,
            "scenario_instance_id": report.scenario_instance_id,
            "environment_id": scenario.environment.environment_id,
            "image_digest": scenario.environment.image_digest,
            "random_seed": assignment.random_seed,
            "candidate_patch_sha256": patch_sha256,
            "initial_state_hash": start.payload.get("initial_state_hash"),
            "success": report.success,
            "infrastructure_failure": report.infrastructure_failure,
            "reward": report.reward,
            "termination_reason": agent_termination_reason,
        }
        invalid = [key for key, value in expected.items() if evidence.get(key) != value]
        if invalid:
            raise PilotIntegrityError(
                f"Existing trace evidence does not match the pilot contract: {invalid}"
            )
        private_expected = {
            "trace_id": trace.trace_id,
            "candidate_patch_sha256": patch_sha256,
            "clean_reset": True,
        }
        private_invalid = [
            key for key, value in private_expected.items() if private.get(key) != value
        ]
        if private_invalid:
            raise PilotIntegrityError(
                f"Existing private evaluation contract mismatch: {private_invalid}"
            )
        if evidence.get("clean_reset_verification") is not True:
            raise PilotIntegrityError("Existing evidence lacks clean-reset verification")
        if evidence.get("contract_versions") != self.contract.versions.to_dict():
            raise PilotIntegrityError("Existing evidence version contract mismatch")
        expected_provider_config = self.contract.provider.to_dict()
        if evidence.get("provider_config") != expected_provider_config:
            raise PilotIntegrityError("Existing provider runtime configuration differs")
        if evidence.get("provider_runtime_sha256") != canonical_sha256(expected_provider_config):
            raise PilotIntegrityError("Existing provider runtime hash differs")
        if evidence.get("budgets") != asdict(self.budgets):
            raise PilotIntegrityError("Existing rollout budgets differ")
        if evidence.get("prompt_sha256") != self.contract.versions.prompt_sha256:
            raise PilotIntegrityError("Existing prompt version differs")
        policy = ToolPolicy(
            scenario.environment.capability_packs or SCHEMAS.keys(),
            network_enabled=scenario.environment.network_policy != "disabled",
        )
        expected_tool_schemas = CodingToolRuntime(MemorySandbox(), policy).schemas()
        expected_tool_hash = canonical_sha256(expected_tool_schemas)
        if evidence.get("tool_schema_sha256") != expected_tool_hash:
            raise PilotIntegrityError("Existing tool schema hash differs")
        evaluator_names = actual_evaluator_names
        if evidence.get("evaluator_names") != evaluator_names:
            raise PilotIntegrityError("Existing evaluator names differ")
        if evidence.get("evaluator_set_sha256") != canonical_sha256(evaluator_names):
            raise PilotIntegrityError("Existing evaluator hash differs")
        observed_calls = evidence.get("observed_calls")
        if not isinstance(observed_calls, list) or not observed_calls:
            raise PilotIntegrityError("Existing observed model calls are missing")
        if any(
            not isinstance(item, Mapping) or item.get("status") != "completed"
            for item in observed_calls
        ):
            raise PilotIntegrityError("Existing observed model calls are incomplete")
        if [item.get("call_index") for item in observed_calls] != list(range(len(observed_calls))):
            raise PilotIntegrityError("Existing observed model call indexes are invalid")
        llm_observability_module.validate_observed_prompt_lineage(
            observed_calls,
            trace,
            DEFAULT_SYSTEM_PROMPT,
            expected_tool_schemas,
        )
        model_events = [
            event for event in trace.events if event.event_type is EventType.MODEL_RESPONSE
        ]
        if len(model_events) != len(observed_calls):
            raise PilotIntegrityError("Existing model calls do not match public responses")
        for observed, event in zip(observed_calls, model_events, strict=True):
            if observed.get("response_model") != event.payload.get("model") or observed.get(
                "usage"
            ) != event.payload.get("usage"):
                raise PilotIntegrityError(
                    "Existing observed model call differs from public response"
                )
        retry_count = sum(int(item.get("retry_count", 0)) for item in observed_calls)
        if evidence.get("retry_count") != retry_count:
            raise PilotIntegrityError("Existing model retry count differs")
        observed_usage = _aggregate_usage_values(
            item.get("usage") for item in observed_calls if isinstance(item, Mapping)
        )
        usage = evidence.get("usage")
        if not isinstance(usage, Mapping):
            raise PilotIntegrityError("Existing trace usage is invalid")
        if dict(usage) != observed_usage:
            raise PilotIntegrityError("Existing aggregate usage differs from model calls")
        usage_cost = self.contract.pricing.calculate_cost(usage)
        metric_expected = {
            "tool_calls": evidence.get("tool_calls"),
            "tokens": usage_cost.total_tokens,
        }
        invalid_metrics = [
            key for key, value in metric_expected.items() if report.metrics.get(key) != value
        ]
        if invalid_metrics:
            raise PilotIntegrityError(
                f"Existing evaluation metrics differ from run evidence: {invalid_metrics}"
            )
        evidence_cost = evidence.get("cost")
        if (
            isinstance(evidence_cost, bool)
            or not isinstance(evidence_cost, (int, float))
            or not math.isfinite(float(evidence_cost))
            or abs(float(evidence_cost) - float(usage_cost.cost_usd)) > 1e-12
        ):
            raise PilotIntegrityError("Existing rollout cost differs from token usage")
        elapsed_ms = evidence.get("elapsed_ms")
        if (
            isinstance(elapsed_ms, bool)
            or not isinstance(elapsed_ms, (int, float))
            or not math.isfinite(float(elapsed_ms))
            or elapsed_ms < 0
        ):
            raise PilotIntegrityError("Existing rollout elapsed time is invalid")
        outcome_metrics = {
            **report.metrics,
            **evaluation_result_metrics(report),
            "elapsed_ms": float(elapsed_ms),
        }
        validated = pilot_artifacts_module.validate_pilot_rollout_artifact(
            self.contract,
            self.registry,
            assignment,
            {
                "job_id": assignment.job_id,
                "scenario_id": assignment.scenario_id,
                "rollout_index": assignment.rollout_index,
                "model": self.contract.provider.model,
                "config_hash": self.contract.contract_id,
                "status": "completed",
                "trace_id": trace.trace_id,
                "success": int(report.success),
                "tokens": usage_cost.total_tokens,
                "cost": float(usage_cost.cost_usd),
                "metrics": outcome_metrics,
            },
            trace_path,
            artifact_paths=paths,
        )
        return RolloutOutcome(
            trace_id=validated.trace_id,
            success=validated.report.success,
            infrastructure_failure=validated.report.infrastructure_failure,
            tokens=usage_cost.total_tokens,
            cost=float(usage_cost.cost_usd),
            metrics=outcome_metrics,
        )


class PilotIntegrityError(RuntimeError):
    """A non-retryable mismatch in contract-bound rollout state."""


def reconcile_pilot_usage_ledger(
    scheduler: PersistentScheduler,
    contract: PilotRunContract,
    worker: PilotRolloutWorker,
) -> dict[str, Any]:
    """Recover durable terminal attempts and reconcile absolute usage before scheduling."""

    rows = validate_pilot_jobs(scheduler, contract)
    for row in rows:
        recover_running_pilot_usage_attempt(
            contract,
            row,
            worker.trace_directory,
        )
    audit = audit_pilot_usage_ledger(
        contract,
        rows,
        worker.trace_directory,
    )
    scheduler.reconcile_consumed_usage(audit.consumed_totals)
    rows_by_id = {str(row["job_id"]): row for row in scheduler.rows()}
    assignments = {assignment.job_id: assignment for assignment in contract.rollouts}
    for job_id, row in sorted(rows_by_id.items()):
        if row.get("status") != JobStatus.RUNNING.value:
            continue
        state = audit.jobs[job_id]
        outcome = state.latest_outcome
        if outcome is None:
            raise UnknownProviderUsageError(
                f"Running pilot job has no durable terminal outcome: {job_id}"
            )
        trace_path = worker.trace_directory / f"{job_id}.jsonl"
        if trace_path.is_file():
            canonical = worker._existing_outcome(trace_path, assignments[job_id])
            outcome = replace(
                canonical,
                absolute_consumed_usage=state.totals,
            )
        scheduler.reconcile_interrupted_outcome(
            RolloutJob(
                assignments[job_id].scenario_id,
                assignments[job_id].rollout_index,
                contract.provider.model,
                contract.contract_id,
                job_id,
            ),
            outcome,
            max_retries=contract.budgets.max_infrastructure_retries,
        )
    final_rows = validate_pilot_jobs(scheduler, contract)
    final_audit = audit_pilot_usage_ledger(
        contract,
        final_rows,
        worker.trace_directory,
        require_database_match=True,
    )
    return final_audit.to_evidence()


def _config_for_assignment(config: LLMConfig, random_seed: int) -> LLMConfig:
    request_body = dict(config.request_body)
    if config.seed_request_field is not None:
        request_body[config.seed_request_field] = random_seed
    return replace(
        config,
        request_body=request_body,
        seed_request_field=None,
    )


def _validate_gold20_registry(
    corpus: Gold20Binding,
    registry: ScenarioRegistry,
) -> None:
    scenario_hashes: dict[str, str] = {}
    bindings = {item.scenario_id: item for item in corpus.scenarios}
    for row in registry.list_scenarios():
        scenario_id = row["scenario_id"]
        scenario = registry.get_scenario(scenario_id)
        scenario_hashes[scenario_id] = canonical_sha256(scenario.to_dict())
        binding = bindings.get(scenario_id)
        if binding is None:
            continue
        if (
            row.get("seed_id") != binding.seed_id
            or row.get("environment_id") != binding.environment_id
            or scenario.query_seed.seed_id != binding.seed_id
            or scenario.environment.environment_id != binding.environment_id
            or canonical_sha256(scenario.environment.to_dict()) != binding.environment_sha256
            or canonical_sha256(scenario.hidden_evaluator.to_dict()) != binding.evaluator_sha256
        ):
            raise ValueError(f"Registry component hash mismatch: {scenario_id}")
    corpus.assert_exact_scenarios(scenario_hashes)
    if _registry_snapshot_sha256(registry.root) != corpus.registry_snapshot_sha256:
        raise ValueError("Registry snapshot does not match the frozen Gold-20 manifest")


def _registry_snapshot_sha256(root: Path) -> str:
    entries: list[dict[str, str]] = []
    for directory in ("seeds", "environments", "scenarios"):
        for path in sorted((root / directory).glob("*.json")):
            entries.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return canonical_sha256(entries)


def _validate_content_id(value: Mapping[str, Any], key: str, prefix: str) -> None:
    supplied = value.get(key)
    payload = {item_key: item for item_key, item in value.items() if item_key != key}
    if supplied != stable_id(prefix, payload):
        raise PilotIntegrityError(f"Existing {prefix} content ID is invalid")


def _usage_from_new_attempts(
    attempt_root: Path,
    prior_attempts: set[Path],
) -> dict[str, Any]:
    new_attempts = [
        path for path in attempt_root.glob("*.jsonl.partial") if path not in prior_attempts
    ]
    usages = []
    for path in sorted(new_attempts):
        try:
            trace = load_trace(path, tolerate_truncated=True)
        except (OSError, ValueError):
            continue
        usages.extend(
            event.payload.get("usage")
            for event in trace.events
            if event.event_type is EventType.MODEL_RESPONSE
        )
    return _aggregate_usage_values(usages)


def _aggregate_usage_values(values: Any) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("Observed model usage must be an object")
        if not value:
            raise ValueError("Observed model usage must not be empty")
        _merge_usage_value(aggregate, value)
    return dict(sorted(aggregate.items()))


def _merge_usage_value(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    for raw_key, raw_amount in value.items():
        key = str(raw_key)
        if isinstance(raw_amount, bool):
            raise ValueError(f"Token usage value must be an integer: {key}")
        if isinstance(raw_amount, int):
            if raw_amount < 0:
                raise ValueError(f"Token usage value must be non-negative: {key}")
            prior = target.get(key, 0)
            if not isinstance(prior, int):
                raise ValueError(f"Conflicting token usage shape: {key}")
            target[key] = prior + raw_amount
        elif isinstance(raw_amount, Mapping):
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Conflicting token usage shape: {key}")
            _merge_usage_value(nested, raw_amount)
        else:
            raise ValueError(f"Unsupported token usage value: {key}")


_INTEGRITY_RUNTIME_FRAGMENTS = (
    "capturing the candidate patch changed workspace state",
    "clean reset initial state does not match",
    "clean reset candidate state does not match",
    "public trace contains",
    "canonical trace already exists",
)


def _is_retryable_rollout_failure(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            PilotIntegrityError,
            AssertionError,
            AttributeError,
            FileExistsError,
            KeyError,
            TypeError,
            ValueError,
        ),
    ):
        return False
    message = str(exc).lower()
    if any(fragment in message for fragment in _INTEGRITY_RUNTIME_FRAGMENTS):
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _module_sha256(module: Any) -> str:
    path = Path(module.__file__ or "")
    if not path.is_file():
        raise RuntimeError(f"Cannot hash module source: {module.__name__}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
