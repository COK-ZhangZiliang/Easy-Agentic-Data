from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

from easy_agentic_data.agent import AgentBudgets, AgentRunResult, HeadlessAgent
from easy_agentic_data.coding_tools import SCHEMAS, CodingToolRuntime
from easy_agentic_data.config import LLMConfig
from easy_agentic_data.evaluation import (
    EvaluationReport,
    EvaluationSuite,
    ForbiddenStateEvaluator,
    HiddenCommandEvaluator,
    HiddenTestPatchEvaluator,
    RequiredStateEvaluator,
    TraceRequirementEvaluator,
    apply_agent_termination,
    contamination_findings,
    derive_turn_rewards,
    evaluation_result_metrics,
    finalize_evaluation_trace,
)
from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.llm.observability import (
    ObservedLLMClient,
    validate_observed_prompt_lineage,
)
from easy_agentic_data.llm.openai_compatible import (
    LocalOpenAICompatibleClient,
    OpenAICompatibleClient,
)
from easy_agentic_data.models import stable_id, utc_now
from easy_agentic_data.pilot_contract import ProviderConfigBinding
from easy_agentic_data.pilot_usage_ledger import PilotUsageAttempt
from easy_agentic_data.policy import ToolPolicy
from easy_agentic_data.registry import ScenarioRegistry, materialize_environment_source
from easy_agentic_data.sandbox import DockerSandbox, SandboxLimits
from easy_agentic_data.scenarios import Scenario, ScenarioInstance
from easy_agentic_data.simulation import RuleBasedUserSimulator, user_callback
from easy_agentic_data.traces import Trace, TraceRecorder, load_trace, replay_trace

_ROLLOUT_ARTIFACT_HASH_KEYS = frozenset(
    {"trace", "candidate_patch", "private_evaluation", "run_evidence"}
)
_VALIDATION_RECEIPT_AUTHORITY = object()


class CandidateSandbox(Protocol):
    def create(self) -> None: ...
    def destroy(self) -> None: ...
    def execute(self, command: list[str], *, timeout_seconds: float | None = None): ...
    def execute_as_root(self, command: list[str], *, timeout_seconds: float | None = None): ...
    def state_hash(self) -> str: ...
    def prepare_git_baseline(self) -> str: ...
    def candidate_patch(self) -> str: ...
    def apply_candidate_patch(self, patch: str) -> str: ...


@dataclass(frozen=True)
class RolloutArtifactPaths:
    trace: Path
    candidate_patch: Path
    private_evaluation: Path
    run_evidence: Path

    @classmethod
    def for_trace(cls, trace_path: str | Path) -> RolloutArtifactPaths:
        trace = Path(trace_path)
        return cls(
            trace=trace,
            candidate_patch=trace.parent / "candidate-patches" / f"{trace.stem}.patch",
            private_evaluation=(trace.parent / "private-evaluations" / f"{trace.stem}.json"),
            run_evidence=trace.parent / "run-evidence" / f"{trace.stem}.json",
        )


@dataclass(frozen=True)
class RolloutValidationReceipt:
    """Content binding issued only after strict validation of a staged rollout."""

    SCHEMA: ClassVar[str] = "easy_agentic_data.rollout_validation_receipt.v1"

    contract_id: str
    job_id: str
    trace_id: str
    artifact_sha256: Mapping[str, str]
    receipt_id: str = ""
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority is not _VALIDATION_RECEIPT_AUTHORITY:
            raise TypeError("Rollout validation receipts must be issued by the strict validator")
        for name in ("contract_id", "job_id", "trace_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Validation receipt {name} must be non-empty")
        hashes = dict(self.artifact_sha256)
        if set(hashes) != _ROLLOUT_ARTIFACT_HASH_KEYS:
            raise ValueError("Validation receipt artifact hashes are incomplete")
        for name, digest in hashes.items():
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"Validation receipt hash is invalid: {name}")
        object.__setattr__(
            self,
            "artifact_sha256",
            MappingProxyType(dict(sorted(hashes.items()))),
        )
        expected = stable_id("rollout_validation_receipt", self._identity_payload())
        if self.receipt_id and self.receipt_id != expected:
            raise ValueError("Validation receipt ID does not match its content")
        object.__setattr__(self, "receipt_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_id": self.contract_id,
            "job_id": self.job_id,
            "trace_id": self.trace_id,
            "artifact_sha256": dict(self.artifact_sha256),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "receipt_id": self.receipt_id}


@dataclass(frozen=True)
class RegistryRolloutResult:
    trace: Trace
    report: EvaluationReport
    run_result: AgentRunResult
    candidate_patch_sha256: str
    usage: dict[str, Any]
    observed_calls: list[dict[str, Any]]
    cost: float
    elapsed_ms: float
    artifacts: RolloutArtifactPaths

    @property
    def metrics(self) -> dict[str, float]:
        return {
            **self.report.metrics,
            **evaluation_result_metrics(self.report),
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class CleanVerificationResult:
    report: EvaluationReport
    initial_state_hash: str
    candidate_state_hash: str
    candidate_patch_sha256: str


SandboxFactory = Callable[[Scenario, Path], CandidateSandbox]
ClientBuilder = Callable[[LLMConfig], LLMClient]
SourceMaterializer = Callable[..., Path]
CostCalculator = Callable[[Mapping[str, Any]], float]


def run_registry_rollout(
    registry: ScenarioRegistry,
    scenario_id: str,
    config: LLMConfig,
    trace_path: str | Path,
    random_seed: int,
    budgets: AgentBudgets | None = None,
    *,
    sandbox_factory: SandboxFactory | None = None,
    client_builder: ClientBuilder | None = None,
    source_materializer: SourceMaterializer | None = None,
    cost_calculator: CostCalculator | None = None,
    run_contract_id: str = "",
    provider_binding_sha256: str = "",
    version_hashes: Mapping[str, str] | None = None,
    provider_binding: Mapping[str, Any] | None = None,
    usage_attempt: PilotUsageAttempt | None = None,
    publish: bool = True,
) -> RegistryRolloutResult:
    """Run an agent and verify its exact patch in an independently reset sandbox."""

    runtime_provider_binding: ProviderConfigBinding | None = None
    if run_contract_id:
        if sandbox_factory is not None:
            raise ValueError("Pilot contract rollouts do not allow a custom sandbox_factory")
        if client_builder is not None:
            raise ValueError("Pilot contract rollouts do not allow a custom client_builder")
        if source_materializer is not None:
            raise ValueError("Pilot contract rollouts do not allow a custom source_materializer")
        if publish:
            raise ValueError("Pilot contract rollouts must be staged with publish=False")
        if usage_attempt is None:
            raise ValueError("Pilot contract rollouts require an immutable usage ledger")
        if (
            usage_attempt.contract_id != run_contract_id
            or usage_attempt.job_id != Path(trace_path).stem
        ):
            raise ValueError("Pilot usage ledger lineage does not match the rollout")
        if not isinstance(provider_binding, Mapping):
            raise ValueError("Pilot contract rollout requires its frozen provider binding")
        declared_provider_binding = ProviderConfigBinding.from_dict(provider_binding)
        runtime_provider_binding = _provider_binding_from_execution_config(
            config,
            declared_provider_binding,
            random_seed,
        )
        if runtime_provider_binding != declared_provider_binding:
            raise ValueError("Runtime provider configuration does not match the pilot contract")
        if provider_binding_sha256 != runtime_provider_binding.config_sha256:
            raise ValueError("Runtime provider binding hash does not match the pilot contract")
    elif usage_attempt is not None:
        raise ValueError("Usage ledger attempts are reserved for pilot contract rollouts")

    canonical_path = Path(trace_path)
    if canonical_path.exists():
        raise FileExistsError(f"Canonical trace already exists: {canonical_path}")
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path = _attempt_trace_path(canonical_path)
    scenario = registry.get_scenario(scenario_id)
    actual_budgets = budgets or AgentBudgets()
    make_sandbox = sandbox_factory or _docker_sandbox
    build_client = client_builder or _build_llm_client
    materialize_source = source_materializer or materialize_environment_source
    staged_artifacts = RolloutArtifactPaths.for_trace(attempt_path)
    agent_sandbox: CandidateSandbox | None = None
    started_at = utc_now()
    started = time.perf_counter()

    try:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            source = materialize_source(
                scenario.environment,
                directory,
                run_health_checks=False,
            )
            agent_sandbox = make_sandbox(scenario, source)
            agent_sandbox.create()
            _prepare_sandbox(agent_sandbox, scenario)
            initial_state_hash = agent_sandbox.prepare_git_baseline()
            instance = registry.materialize(
                scenario_id,
                random_seed=random_seed,
                initial_state_hash=initial_state_hash,
            )
            policy = ToolPolicy(
                scenario.environment.capability_packs or SCHEMAS.keys(),
                network_enabled=scenario.environment.network_policy != "disabled",
            )
            tools = CodingToolRuntime(agent_sandbox, policy)
            user = RuleBasedUserSimulator(instance)
            observed = ObservedLLMClient(
                build_client(config),
                call_journal=usage_attempt,
            )
            agent = HeadlessAgent(observed, tools, budgets=actual_budgets)
            with TraceRecorder(
                attempt_path,
                session_id=f"session_{instance.instance_id}_{random_seed}",
                scenario_instance=instance,
            ) as recorder:
                run_result = agent.run(
                    instance,
                    recorder,
                    ask_user=user_callback(user, instance),
                    finalize=False,
                )
                candidate_state_hash = run_result.final_state_hash
                candidate_patch = agent_sandbox.candidate_patch()
                if agent_sandbox.state_hash() != candidate_state_hash:
                    raise RuntimeError("Capturing the candidate patch changed workspace state")
                partial_trace = load_trace(attempt_path)
                turn_rewards = derive_turn_rewards(partial_trace, instance)
                diagnostics = _rollout_diagnostics(run_result, user)
                clean = verify_candidate_from_clean_reset(
                    scenario,
                    instance,
                    partial_trace,
                    candidate_patch,
                    expected_initial_state_hash=initial_state_hash,
                    expected_candidate_state_hash=candidate_state_hash,
                    diagnostics=diagnostics,
                    turn_rewards=turn_rewards,
                    sandbox_factory=make_sandbox,
                    source_materializer=materialize_source,
                )
                report = apply_agent_termination(clean.report, run_result.termination_reason)
                if report.infrastructure_failure:
                    raise RuntimeError("Clean verifier reported an infrastructure failure")
                finalize_evaluation_trace(
                    recorder,
                    report,
                    final_state_hash=candidate_state_hash,
                    termination_reason=run_result.termination_reason,
                )

            completed_trace = load_trace(attempt_path, tolerate_truncated=False)
            replay_trace(completed_trace)
            leaks = contamination_findings(attempt_path, instance)
            if leaks:
                raise ValueError(f"Public trace contains {len(leaks)} hidden-context value(s)")
            usage = _aggregate_usage(observed.records)
            cost = float(cost_calculator(usage)) if cost_calculator is not None else 0.0
            patch_sha256 = _sha256_text(candidate_patch)
            tool_schemas = tools.schemas()
            validate_observed_prompt_lineage(
                observed.records,
                completed_trace,
                agent.system_prompt,
                tool_schemas,
            )
            safe_calls = [_safe_observed_call(record) for record in observed.records]
            elapsed_ms = (time.perf_counter() - started) * 1000
            evidence = _run_evidence(
                scenario=scenario,
                instance=instance,
                config=config,
                budgets=actual_budgets,
                trace=completed_trace,
                report=report,
                run_result=run_result,
                candidate_patch_sha256=patch_sha256,
                candidate_state_hash=candidate_state_hash,
                observed_calls=safe_calls,
                usage=usage,
                cost=cost,
                run_contract_id=run_contract_id,
                provider_binding_sha256=provider_binding_sha256,
                system_prompt=agent.system_prompt,
                tool_schemas=tool_schemas,
                started_at=started_at,
                elapsed_ms=elapsed_ms,
                version_hashes=version_hashes,
                provider_binding=(
                    runtime_provider_binding.to_dict()
                    if runtime_provider_binding is not None
                    else provider_binding
                ),
                usage_attempt_id=(usage_attempt.attempt_id if usage_attempt is not None else ""),
            )
            private_report = {
                "schema": "easy_agentic_data.private_evaluation.v1",
                "trace_id": completed_trace.trace_id,
                "candidate_patch_sha256": patch_sha256,
                "clean_reset": True,
                "report": report.to_dict(),
            }
            private_report["private_evaluation_id"] = stable_id(
                "private_evaluation",
                private_report,
            )
            _write_text_atomic(staged_artifacts.candidate_patch, candidate_patch)
            _write_json_atomic(staged_artifacts.private_evaluation, private_report)
            _write_json_atomic(staged_artifacts.run_evidence, evidence)
            result = RegistryRolloutResult(
                trace=completed_trace,
                report=report,
                run_result=run_result,
                candidate_patch_sha256=patch_sha256,
                usage=usage,
                observed_calls=safe_calls,
                cost=cost,
                elapsed_ms=elapsed_ms,
                artifacts=staged_artifacts,
            )
            return publish_registry_rollout(result, canonical_path) if publish else result
    except Exception as exc:
        _write_json_atomic(
            attempt_path.with_suffix(".error.json"),
            {
                "schema": "easy_agentic_data.rollout_attempt_error.v1",
                "error": safe_error_message(exc),
                "failed_at": utc_now(),
            },
        )
        raise
    finally:
        if agent_sandbox is not None:
            agent_sandbox.destroy()


def publish_registry_rollout(
    result: RegistryRolloutResult,
    canonical_trace_path: str | Path,
    *,
    validation_receipt: RolloutValidationReceipt | None = None,
) -> RegistryRolloutResult:
    """Publish staged artifacts without overwriting an existing canonical set.

    Sidecars are linked first and the trace is linked last as the commit marker.
    Pilot evidence and the reserved ``rollout_`` namespace require a receipt
    issued after strict artifact validation.
    """

    canonical_path = Path(canonical_trace_path)
    if result.trace.path != result.artifacts.trace:
        raise ValueError("Staged trace and rollout artifact paths do not match")
    staged_paths = _rollout_artifact_path_map(result.artifacts)
    invalid = [path for path in staged_paths.values() if path.is_symlink() or not path.is_file()]
    if invalid:
        raise ValueError(f"Staged rollout artifact is missing or unsafe: {invalid[0].name}")

    staged_trace = load_trace(result.artifacts.trace, tolerate_truncated=False)
    if staged_trace.trace_id != result.trace.trace_id:
        raise ValueError("Staged trace content does not match rollout result")
    evidence = _read_run_evidence(result.artifacts.run_evidence)
    evidence_contract_id = evidence.get("run_contract_id", "")
    if not isinstance(evidence_contract_id, str):
        raise ValueError("Run evidence contract ID must be a string")
    if evidence.get("trace_id") != staged_trace.trace_id:
        raise ValueError("Run evidence trace does not match staged trace")

    artifact_hashes = _rollout_artifact_hashes(result.artifacts)
    receipt_required = bool(evidence_contract_id) or canonical_path.stem.startswith("rollout_")
    if receipt_required:
        if validation_receipt is None:
            raise ValueError("Pilot rollout publication requires a validation receipt")
        _validate_publication_receipt(
            validation_receipt,
            contract_id=evidence_contract_id,
            job_id=canonical_path.stem,
            trace_id=staged_trace.trace_id,
            artifact_hashes=artifact_hashes,
        )
    elif validation_receipt is not None:
        raise ValueError("A validation receipt cannot be used for non-Pilot evidence")

    canonical_artifacts = RolloutArtifactPaths.for_trace(canonical_path)
    publication_items = (
        (
            "candidate_patch",
            result.artifacts.candidate_patch,
            canonical_artifacts.candidate_patch,
        ),
        (
            "private_evaluation",
            result.artifacts.private_evaluation,
            canonical_artifacts.private_evaluation,
        ),
        (
            "run_evidence",
            result.artifacts.run_evidence,
            canonical_artifacts.run_evidence,
        ),
        ("trace", result.artifacts.trace, canonical_artifacts.trace),
    )
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_root = canonical_path.parent.resolve(strict=True)
    for _, _, destination in publication_items:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require_canonical_destination_root(destination, canonical_root)
    existing = {
        name: _canonical_destination_is_exact(destination, artifact_hashes[name])
        for name, _, destination in publication_items
    }
    if existing["trace"] and not all(
        existing[name] for name in ("candidate_patch", "private_evaluation", "run_evidence")
    ):
        raise FileExistsError(
            "Canonical trace already exists before all bound sidecars are present"
        )

    created: list[tuple[Path, int, int]] = []
    try:
        for name, staged, destination in publication_items:
            destination_exists = _canonical_destination_is_exact(
                destination,
                artifact_hashes[name],
            )
            if name != "trace" and not destination_exists:
                trace_exists = _canonical_destination_is_exact(
                    canonical_artifacts.trace,
                    artifact_hashes["trace"],
                )
                if trace_exists:
                    raise FileExistsError(
                        "Canonical trace already exists before all bound sidecars are present"
                    )
            if name == "trace":
                for sidecar_name, _, sidecar_destination in publication_items[:-1]:
                    if not _canonical_destination_is_exact(
                        sidecar_destination,
                        artifact_hashes[sidecar_name],
                    ):
                        raise RuntimeError("Canonical sidecar disappeared before trace publication")
            if destination_exists:
                _fsync_directory(destination.parent)
                continue
            source_stat = staged.stat(follow_symlinks=False)
            try:
                os.link(staged, destination, follow_symlinks=False)
            except FileExistsError as exc:
                if _canonical_destination_is_exact(
                    destination,
                    artifact_hashes[name],
                ):
                    _fsync_directory(destination.parent)
                    continue
                raise FileExistsError(
                    f"Canonical rollout artifact appeared during publication: {destination}"
                ) from exc
            created.append((destination, source_stat.st_dev, source_stat.st_ino))
            if not _canonical_destination_is_exact(
                destination,
                artifact_hashes[name],
            ):
                raise RuntimeError(
                    f"Published rollout artifact changed during promotion: {destination}"
                )
            _fsync_directory(destination.parent)
    except Exception:
        _rollback_publication_links(created)
        raise

    for staged in staged_paths.values():
        try:
            staged.unlink()
        except OSError:
            pass
    published_trace = replace(staged_trace, path=canonical_path)
    return replace(result, trace=published_trace, artifacts=canonical_artifacts)


def verify_candidate_from_clean_reset(
    scenario: Scenario,
    instance: ScenarioInstance,
    trace: Trace,
    candidate_patch: str,
    *,
    expected_initial_state_hash: str,
    expected_candidate_state_hash: str,
    diagnostics: dict[str, float] | None = None,
    turn_rewards: Sequence[Any] = (),
    sandbox_factory: SandboxFactory | None = None,
    source_materializer: SourceMaterializer = materialize_environment_source,
) -> CleanVerificationResult:
    """Materialize a fresh workspace, apply only the candidate patch, and run hard checks."""

    import tempfile

    make_sandbox = sandbox_factory or _docker_sandbox
    verifier: CandidateSandbox | None = None
    try:
        with tempfile.TemporaryDirectory() as directory:
            source = source_materializer(
                scenario.environment,
                directory,
                run_health_checks=False,
            )
            verifier = make_sandbox(scenario, source)
            verifier.create()
            _prepare_sandbox(verifier, scenario)
            initial_hash = verifier.prepare_git_baseline()
            if initial_hash != expected_initial_state_hash:
                raise RuntimeError(
                    "Clean reset initial state does not match the agent workspace baseline"
                )
            candidate_hash = verifier.apply_candidate_patch(candidate_patch)
            if candidate_hash != expected_candidate_state_hash:
                raise RuntimeError(
                    "Clean reset candidate state does not match the agent workspace state"
                )
            report = EvaluationSuite(deterministic_evaluators(instance, trace)).evaluate(
                verifier,
                instance,
                diagnostics=diagnostics,
                turn_rewards=turn_rewards,
            )
            return CleanVerificationResult(
                report=report,
                initial_state_hash=initial_hash,
                candidate_state_hash=candidate_hash,
                candidate_patch_sha256=_sha256_text(candidate_patch),
            )
    finally:
        if verifier is not None:
            verifier.destroy()


def deterministic_evaluators(instance: ScenarioInstance, trace: Trace | None = None) -> list[Any]:
    evaluators: list[Any] = []
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
    retrieval = instance.hidden_evaluator.metadata.get("retrieval_requirements", [])
    rubric = instance.hidden_evaluator.metadata.get("trace_quality_rubric", [])
    if retrieval or rubric:
        evaluators.append(
            TraceRequirementEvaluator(
                trace,
                retrieval_requirements=retrieval,
                trace_quality_rubric=rubric,
            )
        )
    return evaluators


def safe_error_message(exc: BaseException) -> str:
    """Return bounded diagnostics without persisting endpoints or authorization material."""

    message = str(exc)
    message = re.sub(r"https?://[^\s'\"]+", "[redacted endpoint]", message)
    message = re.sub(
        r"(?i)(authorization|api[_-]?key|token|password)\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        message,
    )
    return f"{type(exc).__name__}: {message[:500]}"


def _docker_sandbox(scenario: Scenario, source: Path) -> DockerSandbox:
    return DockerSandbox(
        image_digest=scenario.environment.image_digest,
        source_directory=source,
        limits=SandboxLimits(**scenario.environment.resource_limits),
        network_enabled=scenario.environment.network_policy != "disabled",
    )


def _build_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == "openai_compatible":
        return OpenAICompatibleClient(config)
    if config.provider == "local_openai_compatible":
        return LocalOpenAICompatibleClient(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


def _provider_binding_from_execution_config(
    config: LLMConfig,
    declared: ProviderConfigBinding,
    random_seed: int,
) -> ProviderConfigBinding:
    normalized = config
    seed_field = declared.seed_request_field
    if seed_field is not None:
        if config.seed_request_field is not None:
            raise ValueError("Pilot execution config must consume seed_request_field")
        if config.request_body.get(seed_field) != random_seed:
            raise ValueError("Pilot execution config does not contain the assignment seed")
        request_body = dict(config.request_body)
        request_body.pop(seed_field)
        normalized = replace(
            config,
            request_body=request_body,
            seed_request_field=seed_field,
        )
    return ProviderConfigBinding.from_config(normalized)


def _prepare_sandbox(sandbox: CandidateSandbox, scenario: Scenario) -> None:
    for command in scenario.environment.setup_commands:
        result = sandbox.execute_as_root(shlex.split(command))
        if result.exit_code != 0:
            raise RuntimeError(
                f"Environment setup command failed ({command!r}, exit={result.exit_code})"
            )
    for command in scenario.environment.health_check:
        result = sandbox.execute(shlex.split(command))
        if result.exit_code != 0:
            raise RuntimeError(
                f"Environment health check failed ({command!r}, exit={result.exit_code})"
            )


def _rollout_diagnostics(
    run_result: AgentRunResult,
    user: RuleBasedUserSimulator,
) -> dict[str, float]:
    diagnostics = {
        "turns": float(run_result.turns),
        "tool_calls": float(run_result.tool_calls),
        "tokens": float(run_result.tokens),
        "agent_elapsed_ms": run_result.elapsed_ms,
        "user_turns": float(user.metrics.turns),
    }
    diagnostics.update(
        {
            key: float(value)
            for key, value in user.metrics.to_dict().items()
            if key != "turns" and isinstance(value, (int, float))
        }
    )
    return diagnostics


def _aggregate_usage(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for record in records:
        value = record.get("usage")
        if not isinstance(value, Mapping):
            continue
        _merge_usage(usage, value)
    return dict(sorted(usage.items()))


def _merge_usage(target: dict[str, Any], value: Mapping[str, Any]) -> None:
    for raw_key, amount in value.items():
        key = str(raw_key)
        if isinstance(amount, int) and not isinstance(amount, bool):
            target[key] = int(target.get(key, 0)) + amount
        elif isinstance(amount, Mapping):
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ValueError(f"Conflicting token usage shape for {key}")
            _merge_usage(nested, amount)


def _safe_observed_call(record: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "call_index",
        "started_at",
        "model",
        "message_count",
        "tool_count",
        "temperature",
        "max_tokens",
        "retry_count",
        "prompt_hash",
        "prompt_token_upper_bound",
        "response_format",
        "status",
        "response_model",
        "provider_response_identity",
        "provider_response_identity_sha256",
        "provider_response_sha256",
        "usage",
        "latency_ms",
    }
    safe = {key: record[key] for key in sorted(allowed) if key in record}
    if record.get("status") == "failed":
        safe["error_type"] = str(record.get("error") or "Error").split(":", 1)[0]
    return safe


def _run_evidence(
    *,
    scenario: Scenario,
    instance: ScenarioInstance,
    config: LLMConfig,
    budgets: AgentBudgets,
    trace: Trace,
    report: EvaluationReport,
    run_result: AgentRunResult,
    candidate_patch_sha256: str,
    candidate_state_hash: str,
    observed_calls: list[dict[str, Any]],
    usage: dict[str, Any],
    cost: float,
    run_contract_id: str,
    provider_binding_sha256: str,
    system_prompt: str,
    tool_schemas: list[dict[str, Any]],
    started_at: str,
    elapsed_ms: float,
    version_hashes: Mapping[str, str] | None,
    provider_binding: Mapping[str, Any] | None,
    usage_attempt_id: str,
) -> dict[str, Any]:
    config_payload = (
        dict(provider_binding)
        if provider_binding is not None
        else {
            "provider": config.provider,
            "model": config.model,
            "endpoint_sha256": hashlib.sha256(config.base_url.encode("utf-8")).hexdigest(),
            "api_key_env": config.api_key_env,
            "ca_bundle_env": config.ca_bundle_env,
            "chat_completions_path_sha256": _sha256_text(config.chat_completions_path),
            "timeout_seconds": config.timeout_seconds,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "max_retries": config.max_retries,
            "retry_backoff_seconds": config.retry_backoff_seconds,
            "request_body_sha256": _sha256_json(config.request_body),
        }
    )
    evaluator_names = [item.evaluator for item in report.results]
    payload: dict[str, Any] = {
        "schema": "easy_agentic_data.registry_rollout_evidence.v1",
        "run_contract_id": run_contract_id,
        "provider_binding_sha256": provider_binding_sha256,
        "usage_attempt_id": usage_attempt_id,
        "trace_id": trace.trace_id,
        "scenario_id": scenario.scenario_id,
        "scenario_instance_id": instance.instance_id,
        "environment_id": scenario.environment.environment_id,
        "image_digest": scenario.environment.image_digest,
        "random_seed": instance.random_seed,
        "provider_config": config_payload,
        "provider_runtime_sha256": _sha256_json(config_payload),
        "contract_versions": dict(version_hashes or {}),
        "budgets": asdict(budgets),
        "prompt_sha256": _sha256_text(system_prompt),
        "tool_schema_sha256": _sha256_json(tool_schemas),
        "evaluator_names": evaluator_names,
        "evaluator_set_sha256": _sha256_json(evaluator_names),
        "candidate_patch_sha256": candidate_patch_sha256,
        "initial_state_hash": instance.initial_state_hash,
        "candidate_state_hash": candidate_state_hash,
        "clean_reset_verification": True,
        "success": report.success,
        "infrastructure_failure": report.infrastructure_failure,
        "reward": report.reward,
        "termination_reason": run_result.termination_reason.value,
        "turns": run_result.turns,
        "tool_calls": run_result.tool_calls,
        "usage": usage,
        "cost": cost,
        "retry_count": sum(int(item.get("retry_count", 0) or 0) for item in observed_calls),
        "observed_calls": observed_calls,
        "started_at": started_at,
        "elapsed_ms": elapsed_ms,
    }
    payload["evidence_id"] = stable_id("run_evidence", payload)
    return payload


def _attempt_trace_path(canonical_path: Path) -> Path:
    root = canonical_path.parent / ".attempts" / canonical_path.stem
    root.mkdir(parents=True, exist_ok=True)
    return root / f"attempt-{time.time_ns()}.jsonl.partial"


def _rollout_artifact_path_map(
    artifacts: RolloutArtifactPaths,
) -> dict[str, Path]:
    return {
        "trace": artifacts.trace,
        "candidate_patch": artifacts.candidate_patch,
        "private_evaluation": artifacts.private_evaluation,
        "run_evidence": artifacts.run_evidence,
    }


def _rollout_artifact_hashes(
    artifacts: RolloutArtifactPaths,
) -> dict[str, str]:
    return {
        name: _sha256_file(path)
        for name, path in sorted(_rollout_artifact_path_map(artifacts).items())
    }


def _read_run_evidence(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Run evidence is not readable canonical JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Run evidence must be a JSON object")
    return value


def _validate_publication_receipt(
    receipt: RolloutValidationReceipt,
    *,
    contract_id: str,
    job_id: str,
    trace_id: str,
    artifact_hashes: Mapping[str, str],
) -> None:
    if not isinstance(receipt, RolloutValidationReceipt):
        raise ValueError("Pilot rollout validation receipt has an invalid type")
    if receipt._authority is not _VALIDATION_RECEIPT_AUTHORITY:
        raise ValueError("Pilot rollout validation receipt is not trusted")
    expected_id = stable_id("rollout_validation_receipt", receipt._identity_payload())
    if receipt.receipt_id != expected_id:
        raise ValueError("Pilot rollout validation receipt ID is invalid")
    expected = {
        "contract_id": contract_id,
        "job_id": job_id,
        "trace_id": trace_id,
        "artifact_sha256": dict(artifact_hashes),
    }
    actual = {
        "contract_id": receipt.contract_id,
        "job_id": receipt.job_id,
        "trace_id": receipt.trace_id,
        "artifact_sha256": dict(receipt.artifact_sha256),
    }
    mismatches = sorted(key for key in expected if expected[key] != actual[key])
    if mismatches:
        raise ValueError("Pilot rollout validation receipt mismatch: " + ", ".join(mismatches))


def _canonical_destination_is_exact(path: Path, expected_sha256: str) -> bool:
    try:
        before = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(before.st_mode):
        raise FileExistsError(f"Canonical rollout artifact already exists as a symlink: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise FileExistsError(
            f"Canonical rollout artifact already exists and is not a regular file: {path}"
        )
    try:
        actual_sha256 = _sha256_file(path)
        after = path.lstat()
    except FileNotFoundError:
        return False
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise FileExistsError(f"Canonical rollout artifact changed while being validated: {path}")
    if actual_sha256 != expected_sha256:
        raise FileExistsError(
            f"Canonical rollout artifact already exists with different content: {path}"
        )
    return True


def _require_canonical_destination_root(destination: Path, canonical_root: Path) -> None:
    try:
        destination.parent.resolve(strict=True).relative_to(canonical_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Canonical rollout artifact resolves outside its root: {destination}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rollback_publication_links(created: Sequence[tuple[Path, int, int]]) -> None:
    changed_directories: set[Path] = set()
    for destination, source_device, source_inode in reversed(created):
        try:
            current = destination.stat(follow_symlinks=False)
        except OSError:
            continue
        if (current.st_dev, current.st_ino) != (source_device, source_inode):
            continue
        try:
            destination.unlink()
            changed_directories.add(destination.parent)
        except OSError:
            continue
    for directory in changed_directories:
        try:
            _fsync_directory(directory)
        except OSError:
            continue


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_text(encoded)
