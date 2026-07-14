from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from easy_agentic_data.environments import EnvironmentSpec, is_immutable_image_reference
from easy_agentic_data.models import stable_id, utc_now
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_sources import DEFAULT_TRAIN_LICENSE_ALLOWLIST
from easy_agentic_data.sandbox import SandboxLimits
from easy_agentic_data.scenario_decontamination import audit_scenario_decontamination
from easy_agentic_data.scenarios import Scenario, ScenarioInstance
from easy_agentic_data.seed_corpus import REGISTRY_IMPORT_REHEARSAL_SCHEMA_VERSION
from easy_agentic_data.seed_library import (
    DEFAULT_BENCHMARK_SOURCE_ALIASES,
    SeedLibraryPolicy,
    audit_seed_library,
    is_benchmark_seed,
)
from easy_agentic_data.seeds import QuerySeed

GOLD20_FREEZE_CONFIG_SCHEMA_VERSION = "easy_agentic_data.gold20_freeze_config.v1"
GOLD20_MANIFEST_SCHEMA_VERSION = "easy_agentic_data.gold20_manifest.v1"
GOLD20_MATERIALIZATION_SCHEMA_VERSION = "easy_agentic_data.gold20_materialization.v1"
HIDDEN_TEST_PATCH_VALIDATION_SCHEMA_VERSION = "easy_agentic_data.hidden_test_patch_validation.v1"
GOLD_REPAIR_VALIDATION_SCHEMA_VERSION = "easy_agentic_data.gold_repair_validation.v1"
GOLD20_REFERENCE_REPAIRS_SCHEMA_VERSION = "easy_agentic_data.gold20_reference_repairs.v1"
GOLD20_CONTAINER_REPLAY_SCHEMA_VERSION = "easy_agentic_data.gold20_container_replay.v1"
GOLD20_RUNTIME_PLATFORM = "linux/arm64"
GOLD20_RUNTIME_RANDOM_SEED = 0
GOLD20_BUILD_VERIFICATION_MODE = "local_image_id_plus_declared_spec"
EXPECTED_SEED_COUNT = 20
MIN_REPOSITORIES = 8
MIN_LANGUAGES = 2
ALLOWED_PERMITTED_USES = {"research", "research_and_training", "training"}

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_IMAGE_PATTERN = re.compile(r"(?:.+@)?sha256:[0-9a-fA-F]{64}")


class _Gate:
    def __init__(self) -> None:
        self.issues: list[dict[str, str]] = []

    def require(self, condition: bool, code: str, entry_id: str = "") -> bool:
        if not condition:
            issue = {"code": code}
            if entry_id:
                issue["entry_id"] = entry_id
            self.issues.append(issue)
        return condition


def freeze_gold20(
    config_path: str | Path,
    *,
    manifest_output: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and freeze the exact Gold-20 corpus as a metadata-only manifest."""

    config_file = Path(config_path).expanduser().resolve()
    config = _read_object(config_file)
    if config.get("schema_version") != GOLD20_FREEZE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Gold-20 config must use schema version {GOLD20_FREEZE_CONFIG_SCHEMA_VERSION}"
        )
    config_dir = config_file.parent
    registry_root = _required_path(config, "registry_root", config_dir)
    source_path = _required_path(config, "source_snapshot", config_dir)
    rehearsal_path = _required_path(config, "hidden_patch_rehearsal", config_dir)
    materialization_path = _required_path(config, "materialization_reset_evidence", config_dir)
    validation_paths = _required_path_list(config, "repair_validation_evidence", config_dir)
    reference_repair_path = _required_path(config, "reference_repair_evidence", config_dir)
    container_replay_path = _required_path(config, "container_replay_evidence", config_dir)
    holdout_roots = _required_path_list(config, "holdout_registry_roots", config_dir)

    gate = _Gate()
    runtime_builds = _load_container_runtime_builds(config, config_dir, gate)
    declared_resolutions = _load_contamination_resolutions(config, gate)
    scenarios = _load_registry_scenarios(registry_root)
    scenario_ids = {scenario.scenario_id for scenario in scenarios}
    seed_ids = {scenario.query_seed.seed_id for scenario in scenarios}
    gate.require(
        len(scenarios) == EXPECTED_SEED_COUNT,
        "gold20_scenario_count_not_exact",
    )
    gate.require(
        len(seed_ids) == EXPECTED_SEED_COUNT,
        "gold20_seed_count_not_exact",
    )
    gate.require(
        len(scenario_ids) == len(scenarios),
        "gold20_duplicate_scenario_id",
    )

    registry = ScenarioRegistry(registry_root)
    registry_validation = registry.validate()
    for issue in registry_validation.issues:
        gate.require(False, f"registry_{issue.code}", issue.entry_id)
    registry_seeds = [
        QuerySeed.from_dict(_read_object(path))
        for path in sorted((registry_root / "seeds").glob("*.json"))
    ]
    registry_seed_by_id = {seed.seed_id: seed for seed in registry_seeds}
    registry_seed_ids = set(registry_seed_by_id)
    gate.require(
        len(registry_seeds) == len(registry_seed_by_id),
        "registry_duplicate_seed_id",
    )
    gate.require(
        registry_seed_ids == seed_ids,
        "registry_seed_set_mismatch",
    )
    registry_environments = [
        EnvironmentSpec.from_dict(_read_object(path))
        for path in sorted((registry_root / "environments").glob("*.json"))
    ]
    registry_environment_by_id = {
        environment.environment_id: environment for environment in registry_environments
    }
    gate.require(
        len(registry_environments) == len(registry_environment_by_id),
        "registry_duplicate_environment_id",
    )
    scenario_environment_ids = {scenario.environment.environment_id for scenario in scenarios}
    gate.require(
        set(registry_environment_by_id) == scenario_environment_ids,
        "registry_environment_set_mismatch",
    )
    for scenario in scenarios:
        seed = registry_seed_by_id.get(scenario.query_seed.seed_id)
        environment = registry_environment_by_id.get(scenario.environment.environment_id)
        gate.require(
            seed is not None and seed.to_dict() == scenario.query_seed.to_dict(),
            "registry_seed_payload_mismatch",
            scenario.scenario_id,
        )
        gate.require(
            environment is not None and environment.to_dict() == scenario.environment.to_dict(),
            "registry_environment_payload_mismatch",
            scenario.scenario_id,
        )

    source_records = _load_records(source_path)
    source_by_id = _index_records(source_records, _source_instance_id, gate, "source")
    gate.require(
        len(source_records) == EXPECTED_SEED_COUNT,
        "source_record_count_not_exact",
    )

    scenario_by_source: dict[str, Scenario] = {}
    for scenario in scenarios:
        source_id = _scenario_source_instance_id(scenario)
        gate.require(bool(source_id), "scenario_missing_source_instance", scenario.scenario_id)
        if source_id in scenario_by_source:
            gate.require(False, "duplicate_scenario_source_instance", source_id)
        elif source_id:
            scenario_by_source[source_id] = scenario
    source_ids = set(scenario_by_source)
    gate.require(
        len(source_ids) == EXPECTED_SEED_COUNT,
        "scenario_source_instance_count_not_exact",
    )
    gate.require(set(source_by_id) == source_ids, "source_record_set_mismatch")

    rehearsal = _read_object(rehearsal_path)
    rehearsal_by_scenario = _validate_rehearsal(
        rehearsal,
        source_path=source_path,
        expected_scenario_ids=scenario_ids,
        gate=gate,
    )
    materialization = _read_object(materialization_path)
    materialization_by_scenario = _validate_materialization_artifact(
        materialization,
        expected_scenario_ids=scenario_ids,
        gate=gate,
    )
    validation_by_source = _load_validation_evidence(validation_paths, gate)
    gate.require(
        len(validation_by_source) == EXPECTED_SEED_COUNT,
        "repair_validation_record_count_not_exact",
    )
    gate.require(
        set(validation_by_source) == source_ids,
        "repair_validation_record_set_mismatch",
    )
    reference_repair_by_source = _load_reference_repairs(reference_repair_path, gate)
    gate.require(
        len(reference_repair_by_source) == EXPECTED_SEED_COUNT,
        "reference_repair_record_count_not_exact",
    )
    gate.require(
        set(reference_repair_by_source) == source_ids,
        "reference_repair_record_set_mismatch",
    )
    container_replay = _read_object(container_replay_path)
    container_replay_by_source = _validate_container_replay_artifact(
        container_replay,
        expected_source_ids=source_ids,
        expected_scenario_ids=scenario_ids,
        runtime_builds=runtime_builds,
        gate=gate,
    )

    records: list[dict[str, Any]] = []
    repositories: set[str] = set()
    languages: set[str] = set()
    corpus_private_canaries = {
        canary for scenario in scenarios for canary in _privacy_canaries(scenario)
    }
    for scenario in sorted(scenarios, key=lambda item: item.scenario_id):
        source_id = _scenario_source_instance_id(scenario)
        source_record = source_by_id.get(source_id, {})
        rehearsal_record = rehearsal_by_scenario.get(scenario.scenario_id, {})
        materialization_record = materialization_by_scenario.get(scenario.scenario_id, {})
        validation_record = validation_by_source.get(source_id, {})
        reference_repair_record = reference_repair_by_source.get(source_id, {})
        container_replay_record = container_replay_by_source.get(source_id, {})
        record = _validate_scenario_record(
            scenario,
            source_id=source_id,
            source_record=source_record,
            rehearsal_record=rehearsal_record,
            materialization_record=materialization_record,
            validation_record=validation_record,
            reference_repair_record=reference_repair_record,
            container_replay_record=container_replay_record,
            runtime_builds=runtime_builds,
            corpus_private_canaries=corpus_private_canaries,
            gate=gate,
        )
        if record["repository"]:
            repositories.add(record["repository"])
        if record["language"]:
            languages.add(record["language"])
        records.append(record)

    _check_corpus_hidden_isolation(scenarios, gate)
    gate.require(len(repositories) >= MIN_REPOSITORIES, "repository_coverage_not_met")
    gate.require(len(languages) >= MIN_LANGUAGES, "language_coverage_not_met")

    benchmark_sources = sorted(
        set(DEFAULT_BENCHMARK_SOURCE_ALIASES) | set(_string_list(config.get("benchmark_sources")))
    )
    holdout_scenarios: list[Scenario] = []
    benchmark_holdouts: list[Scenario] = []
    holdout_snapshot_hashes = []
    for root in holdout_roots:
        loaded = _load_registry_scenarios(root)
        holdout_validation = ScenarioRegistry(root).validate()
        for issue in holdout_validation.issues:
            gate.require(False, f"holdout_registry_{issue.code}", issue.entry_id)
        for scenario in loaded:
            gate.require(
                _seed_id_is_stable(scenario.query_seed),
                "holdout_seed_content_id_mismatch",
                scenario.query_seed.seed_id,
            )
            gate.require(
                _environment_id_is_stable(scenario.environment),
                "holdout_environment_content_id_mismatch",
                scenario.environment.environment_id,
            )
            gate.require(
                _scenario_id_is_stable(scenario),
                "holdout_scenario_content_id_mismatch",
                scenario.scenario_id,
            )
            test_patch = str(scenario.hidden_evaluator.metadata.get("test_patch") or "")
            declared_test_patch_sha256 = str(
                scenario.hidden_evaluator.metadata.get("test_patch_sha256") or ""
            ).lower()
            if test_patch or declared_test_patch_sha256:
                gate.require(
                    bool(test_patch)
                    and declared_test_patch_sha256 == _text_sha256(test_patch),
                    "holdout_hidden_test_patch_hash_mismatch",
                    scenario.scenario_id,
                )
            if is_benchmark_seed(
                scenario.query_seed,
                benchmark_sources=benchmark_sources,
            ):
                benchmark_holdouts.append(scenario)
                gate.require(
                    bool(test_patch),
                    "benchmark_holdout_hidden_test_patch_missing",
                    scenario.scenario_id,
                )
        holdout_scenarios.extend(loaded)
        holdout_snapshot_hashes.append(_registry_snapshot_sha256(root))
    usable_holdouts = [
        scenario
        for scenario in holdout_scenarios
        if not scenario.query_seed.train_eligible
        or is_benchmark_seed(
            scenario.query_seed,
            benchmark_sources=benchmark_sources,
        )
    ]
    gate.require(bool(holdout_scenarios), "holdout_registry_empty")
    gate.require(bool(usable_holdouts), "holdout_registry_has_no_holdout_scenarios")
    gate.require(bool(benchmark_holdouts), "holdout_registry_has_no_benchmark_scenarios")
    seed_audit = audit_seed_library(
        [scenario.query_seed for scenario in scenarios],
        benchmark_sources=benchmark_sources,
        policy=SeedLibraryPolicy(
            min_train_eligible=EXPECTED_SEED_COUNT,
            required_verifier_types=["hidden_test_patch"],
        ),
        holdout_seeds=[scenario.query_seed for scenario in holdout_scenarios],
    )
    scenario_audit = audit_scenario_decontamination(
        scenarios,
        holdout_scenarios=holdout_scenarios,
        benchmark_sources=benchmark_sources,
    )
    resolved_findings, unresolved_findings = _apply_contamination_resolutions(
        seed_issues=seed_audit.issues,
        scenario_issues=scenario_audit.issues,
        declared=declared_resolutions,
        gate=gate,
    )

    for record in records:
        record["valid"] = not gate.issues
        record["record_sha256"] = _stable_json_sha256(record)

    source_snapshot_sha256 = _file_sha256(source_path)
    registry_snapshot_sha256 = _registry_snapshot_sha256(registry_root)
    rehearsal_sha256 = _file_sha256(rehearsal_path)
    materialization_sha256 = _file_sha256(materialization_path)
    repair_validation_sha256 = sorted(_file_sha256(path) for path in validation_paths)
    reference_repair_evidence_sha256 = _file_sha256(reference_repair_path)
    container_replay_sha256 = _file_sha256(container_replay_path)
    runtime_build_spec_sha256s = sorted(
        build["build_spec_sha256"] for build in runtime_builds.values()
    )
    holdout_snapshot_hashes = sorted(holdout_snapshot_hashes)
    seed_audit_sha256 = _stable_json_sha256(seed_audit.to_dict())
    scenario_audit_sha256 = _stable_json_sha256(scenario_audit.to_dict())
    validation_policy_sha256 = _stable_json_sha256(
        {
            "benchmark_sources": benchmark_sources,
            "expected_seed_count": EXPECTED_SEED_COUNT,
            "min_repositories": MIN_REPOSITORIES,
            "min_languages": MIN_LANGUAGES,
        }
    )
    corpus_id = stable_id(
        "gold20",
        {
            "records": [record["record_sha256"] for record in records],
            "source_snapshot_sha256": source_snapshot_sha256,
            "registry_snapshot_sha256": registry_snapshot_sha256,
            "hidden_patch_rehearsal_sha256": rehearsal_sha256,
            "materialization_reset_sha256": materialization_sha256,
            "repair_validation_artifact_sha256": repair_validation_sha256,
            "reference_repair_evidence_sha256": reference_repair_evidence_sha256,
            "container_replay_sha256": container_replay_sha256,
            "runtime_build_spec_sha256": runtime_build_spec_sha256s,
            "holdout_registry_snapshot_sha256": holdout_snapshot_hashes,
            "seed_library_audit_sha256": seed_audit_sha256,
            "scenario_decontamination_audit_sha256": scenario_audit_sha256,
            "resolved_contamination_findings": resolved_findings,
            "validation_policy_sha256": validation_policy_sha256,
        },
    )
    manifest = {
        "schema_version": GOLD20_MANIFEST_SCHEMA_VERSION,
        "corpus_id": corpus_id,
        "created_at": utc_now(),
        "expected_seed_count": EXPECTED_SEED_COUNT,
        "records": records,
        "coverage": {
            "repositories": len(repositories),
            "languages": len(languages),
        },
        "evidence": {
            "source_snapshot_sha256": source_snapshot_sha256,
            "registry_snapshot_sha256": registry_snapshot_sha256,
            "hidden_patch_rehearsal_sha256": rehearsal_sha256,
            "materialization_reset_sha256": materialization_sha256,
            "repair_validation_artifact_sha256": repair_validation_sha256,
            "reference_repair_evidence_sha256": reference_repair_evidence_sha256,
            "container_replay_sha256": container_replay_sha256,
            "runtime_build_spec_sha256": runtime_build_spec_sha256s,
            "holdout_registry_snapshot_sha256": holdout_snapshot_hashes,
            "validation_policy_sha256": validation_policy_sha256,
        },
        "audits": {
            "seed_library_sha256": seed_audit_sha256,
            "seed_library_issue_count": len(seed_audit.issues),
            "scenario_decontamination_sha256": scenario_audit_sha256,
            "scenario_decontamination_issue_count": len(scenario_audit.issues),
            "resolved_finding_count": len(resolved_findings),
            "unresolved_finding_count": len(unresolved_findings),
            "resolved_findings_sha256": _stable_json_sha256(resolved_findings),
        },
        "validation": {
            "exact_count": len(scenarios) == EXPECTED_SEED_COUNT
            and len(seed_ids) == EXPECTED_SEED_COUNT,
            "exact_source_set": set(source_by_id) == source_ids,
            "registry_valid": registry_validation.valid,
            "hidden_patch_rehearsal_valid": not any(
                issue["code"].startswith("hidden_patch_rehearsal_") for issue in gate.issues
            ),
            "materialization_reset_valid": not any(
                issue["code"].startswith("materialization_") for issue in gate.issues
            ),
            "repair_validation_valid": not any(
                issue["code"].startswith("repair_validation_") for issue in gate.issues
            ),
            "container_replay_valid": not any(
                issue["code"].startswith("container_replay_") for issue in gate.issues
            ),
            "decontamination_valid": not unresolved_findings
            and not any(
                issue["code"].startswith("contamination_resolution_")
                for issue in gate.issues
            ),
        },
        "issues": sorted(
            (_manifest_issue(issue) for issue in gate.issues),
            key=lambda item: (item["code"], item.get("entry_id_sha256", "")),
        ),
        "valid": not gate.issues,
    }
    output_value = manifest_output or config.get("manifest_output")
    if output_value and manifest["valid"]:
        output_path = _resolve_path(config_dir, output_value)
        _write_json_atomically(output_path, manifest)
    return manifest


def _load_contamination_resolutions(
    config: dict[str, Any],
    gate: _Gate,
) -> dict[tuple[str, str, str], dict[str, str]]:
    raw = config.get("resolved_contamination_findings", [])
    records = _strict_object_list(
        raw,
        gate,
        "contamination_resolution_records_invalid",
    )
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    for record in records:
        audit = _normalize_label(record.get("audit"))
        code = str(record.get("code") or "").strip()
        entry_id = str(record.get("entry_id") or "").strip()
        rationale = str(record.get("rationale") or "").strip()
        valid = (
            audit in {"seed_library", "scenario_decontamination"}
            and bool(code)
            and bool(rationale)
        )
        gate.require(valid, "contamination_resolution_record_invalid", entry_id)
        if not valid:
            continue
        key = (audit, code, entry_id)
        gate.require(
            key not in indexed,
            "contamination_resolution_duplicate",
            entry_id,
        )
        if key not in indexed:
            indexed[key] = {
                "audit": audit,
                "code": code,
                "entry_id": entry_id,
                "rationale": rationale,
            }
    return indexed


def _apply_contamination_resolutions(
    *,
    seed_issues: Iterable[Any],
    scenario_issues: Iterable[Any],
    declared: dict[tuple[str, str, str], dict[str, str]],
    gate: _Gate,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    findings = [
        ("seed_library", "seed_audit", issue, str(issue.seed_id or ""))
        for issue in seed_issues
    ] + [
        (
            "scenario_decontamination",
            "scenario_audit",
            issue,
            str(issue.scenario_id or ""),
        )
        for issue in scenario_issues
    ]
    matched: set[tuple[str, str, str]] = set()
    resolved: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    for audit, issue_prefix, issue, entry_id in findings:
        code = str(issue.code)
        severity = _normalize_label(getattr(issue, "severity", "error"))
        key = (audit, code, entry_id)
        resolution = declared.get(key)
        if resolution is not None:
            matched.add(key)
        if severity == "warning" and resolution is not None:
            resolved.append(
                {
                    "audit": audit,
                    "code": code,
                    "entry_id_sha256": _text_sha256(entry_id),
                    "rationale_sha256": _text_sha256(resolution["rationale"]),
                }
            )
            continue
        if resolution is not None:
            gate.require(
                False,
                "contamination_resolution_cannot_waive_error",
                entry_id,
            )
        gate.require(False, f"{issue_prefix}_{code}", entry_id)
        unresolved.append(
            {
                "audit": audit,
                "code": code,
                "entry_id_sha256": _text_sha256(entry_id),
                "severity": severity or "error",
            }
        )
    for key, resolution in declared.items():
        if key not in matched:
            gate.require(
                False,
                "contamination_resolution_unmatched",
                resolution["entry_id"],
            )
    return (
        sorted(
            resolved,
            key=lambda item: (item["audit"], item["code"], item["entry_id_sha256"]),
        ),
        sorted(
            unresolved,
            key=lambda item: (item["audit"], item["code"], item["entry_id_sha256"]),
        ),
    )


def _validate_scenario_record(
    scenario: Scenario,
    *,
    source_id: str,
    source_record: dict[str, Any],
    rehearsal_record: dict[str, Any],
    materialization_record: dict[str, Any],
    validation_record: dict[str, Any],
    reference_repair_record: dict[str, Any],
    container_replay_record: dict[str, Any],
    runtime_builds: dict[str, dict[str, Any]],
    corpus_private_canaries: set[str],
    gate: _Gate,
) -> dict[str, Any]:
    seed = scenario.query_seed
    environment = scenario.environment
    entry_id = scenario.scenario_id
    repository = _scenario_repository(scenario)
    language = _scenario_language(scenario)
    source_name = _scenario_source_name(scenario)
    source_url = str(source_record.get("source_url") or "").strip()
    source_query = _source_query_text(source_record)
    source_identity = _public_source_identity(source_url)
    source_type = _normalize_public_source_type(
        scenario.query_seed.public.context.get("source_type")
        or scenario.query_seed.metadata.get("source_type")
        or scenario.metadata.get("source_type")
    )
    permitted_use = _normalize_label(
        seed.metadata.get("permitted_use") or environment.metadata.get("permitted_use") or ""
    )
    license_name = _normalize_label(seed.license)
    test_patch = str(scenario.hidden_evaluator.metadata.get("test_patch") or "")
    patch_sha256 = _text_sha256(test_patch)
    command_hashes = sorted(_text_sha256(item) for item in scenario.hidden_evaluator.hidden_tests)
    setup_sha256 = _stable_json_sha256(environment.setup_commands)
    health_sha256 = _stable_json_sha256(environment.health_check)

    gate.require(seed.train_eligible, "scenario_not_train_eligible", entry_id)
    gate.require(seed.split == "train", "scenario_not_train_split", entry_id)
    provenance_source, separator, provenance_instance = seed.provenance.rpartition(":")
    gate.require(
        bool(separator)
        and _normalize_source_name(provenance_source) == source_name
        and provenance_instance == source_id,
        "scenario_provenance_invalid",
        entry_id,
    )
    gate.require(bool(source_name), "scenario_source_name_missing", entry_id)
    gate.require(
        license_name in DEFAULT_TRAIN_LICENSE_ALLOWLIST,
        "scenario_license_not_allowed",
        entry_id,
    )
    gate.require(
        permitted_use in ALLOWED_PERMITTED_USES,
        "scenario_permitted_use_not_allowed",
        entry_id,
    )
    gate.require(bool(repository), "scenario_missing_repository", entry_id)
    gate.require(bool(language), "scenario_missing_language", entry_id)
    gate.require(
        bool(_REVISION_PATTERN.fullmatch(environment.source_revision.lower())),
        "scenario_revision_not_fixed",
        entry_id,
    )
    gate.require(
        bool(environment.image_digest)
        and is_immutable_image_reference(environment.image_digest)
        and bool(_IMAGE_PATTERN.fullmatch(environment.image_digest)),
        "scenario_image_not_content_addressed",
        entry_id,
    )
    gate.require(
        bool(environment.health_check)
        and all(str(command).strip() for command in environment.health_check),
        "scenario_health_check_empty",
        entry_id,
    )
    gate.require(
        environment.network_policy == "disabled",
        "scenario_network_policy_not_disabled",
        entry_id,
    )
    gate.require(
        not scenario.hidden_evaluator.required_state
        and not scenario.hidden_evaluator.forbidden_state
        and not scenario.hidden_evaluator.metadata.get("retrieval_requirements")
        and not scenario.hidden_evaluator.metadata.get("trace_quality_rubric"),
        "scenario_unsupported_evaluator_surface",
        entry_id,
    )
    gate.require(bool(test_patch.strip()), "scenario_hidden_test_patch_empty", entry_id)
    gate.require(
        str(scenario.hidden_evaluator.metadata.get("test_patch_sha256") or "").lower()
        == patch_sha256,
        "scenario_hidden_test_patch_hash_mismatch",
        entry_id,
    )
    gate.require(bool(command_hashes), "scenario_hidden_test_command_empty", entry_id)
    gate.require(
        "hidden_test_patch" in seed.verifier_types,
        "scenario_hidden_test_patch_verifier_missing",
        entry_id,
    )
    gate.require(_seed_id_is_stable(seed), "seed_content_id_mismatch", seed.seed_id)
    gate.require(
        _environment_id_is_stable(environment),
        "environment_content_id_mismatch",
        environment.environment_id,
    )
    gate.require(
        _scenario_id_is_stable(scenario),
        "scenario_content_id_mismatch",
        scenario.scenario_id,
    )

    source_patch_sha = _record_patch_sha256(source_record)
    gate.require(source_patch_sha == patch_sha256, "source_patch_hash_mismatch", entry_id)
    gate.require(
        _normalize_repository(source_record.get("repo") or source_record.get("repository"))
        == repository,
        "source_repository_mismatch",
        entry_id,
    )
    gate.require(
        str(source_record.get("base_commit") or source_record.get("source_revision") or "")
        == environment.source_revision,
        "source_revision_mismatch",
        entry_id,
    )
    gate.require(
        _normalize_label(source_record.get("license")) == license_name,
        "source_license_mismatch",
        entry_id,
    )
    gate.require(
        _normalize_label(source_record.get("permitted_use")) == permitted_use,
        "source_permitted_use_mismatch",
        entry_id,
    )
    gate.require(
        _normalize_label(source_record.get("language")) == language,
        "source_language_mismatch",
        entry_id,
    )
    gate.require(
        _normalize_source_name(source_record.get("source_name")) == source_name,
        "source_name_mismatch",
        entry_id,
    )
    gate.require(
        _public_source_repository(source_url) == repository,
        "source_url_missing_or_untrusted",
        entry_id,
    )
    registry_source_urls = (
        str(seed.public.context.get("source_url") or "").strip(),
        str(seed.metadata.get("source_url") or "").strip(),
        str(environment.metadata.get("source_url") or "").strip(),
    )
    gate.require(
        bool(source_url) and all(item == source_url for item in registry_source_urls),
        "source_url_registry_mismatch",
        entry_id,
    )
    gate.require(
        bool(source_query) and source_query == seed.public.query,
        "source_query_registry_mismatch",
        entry_id,
    )
    gate.require(
        source_identity[1] == source_type
        and _normalize_public_source_type(
            source_record.get("source_type") or source_record.get("type")
        )
        == source_type,
        "source_type_mismatch",
        entry_id,
    )
    gate.require(
        source_id.casefold() == _expected_public_source_id(source_identity).casefold(),
        "source_instance_url_identity_mismatch",
        entry_id,
    )
    gate.require(
        source_record.get("setup_commands") == environment.setup_commands,
        "source_setup_commands_mismatch",
        entry_id,
    )
    gate.require(
        source_record.get("health_check") == environment.health_check,
        "source_health_check_mismatch",
        entry_id,
    )
    gate.require(
        str(source_record.get("image_digest") or "") == environment.image_digest,
        "source_image_digest_mismatch",
        entry_id,
    )
    gate.require(
        str(source_record.get("source_uri") or "") == environment.source_uri,
        "source_workspace_uri_mismatch",
        entry_id,
    )
    gate.require(
        _public_clone_repository(source_record.get("workspace_original_source_uri"))
        == repository,
        "source_original_workspace_uri_mismatch",
        entry_id,
    )

    _check_rehearsal_record(
        rehearsal_record,
        scenario=scenario,
        patch_sha256=patch_sha256,
        command_hashes=command_hashes,
        gate=gate,
    )
    _check_materialization_record(
        materialization_record,
        scenario=scenario,
        setup_sha256=setup_sha256,
        health_sha256=health_sha256,
        gate=gate,
    )
    normalized_validation = _check_validation_record(
        validation_record,
        scenario=scenario,
        source_id=source_id,
        patch_sha256=patch_sha256,
        command_hashes=command_hashes,
        repository=repository,
        materialization_record=materialization_record,
        setup_sha256=setup_sha256,
        health_sha256=health_sha256,
        reference_repair_record=reference_repair_record,
        gate=gate,
    )
    _check_container_replay_record(
        container_replay_record,
        scenario=scenario,
        source_id=source_id,
        patch_sha256=patch_sha256,
        command_hashes=command_hashes,
        materialization_record=materialization_record,
        setup_sha256=setup_sha256,
        health_sha256=health_sha256,
        validated_repair_sha256=normalized_validation.get("validated_repair_sha256", ""),
        runtime_builds=runtime_builds,
        gate=gate,
    )
    _check_hidden_isolation(scenario, gate)

    manifest_metadata = {
        "source_instance_id": source_id,
        "source_name": source_name,
        "repository": repository,
        "language": language,
        "license": license_name,
        "permitted_use": permitted_use,
    }
    safe_manifest_metadata = {
        key: _redact_private_metadata(value, corpus_private_canaries)
        for key, value in manifest_metadata.items()
    }
    gate.require(
        safe_manifest_metadata == manifest_metadata,
        "manifest_metadata_contains_private_canary",
        entry_id,
    )

    return {
        "scenario_id": scenario.scenario_id,
        "seed_id": seed.seed_id,
        "environment_id": environment.environment_id,
        "source_instance_id": safe_manifest_metadata["source_instance_id"],
        "source_name": safe_manifest_metadata["source_name"],
        "source_url_sha256": _text_sha256(source_url),
        "repository": safe_manifest_metadata["repository"],
        "language": safe_manifest_metadata["language"],
        "source_revision": environment.source_revision,
        "license": safe_manifest_metadata["license"],
        "permitted_use": safe_manifest_metadata["permitted_use"],
        "hashes": {
            "seed_sha256": _stable_json_sha256(seed.to_dict()),
            "environment_sha256": _stable_json_sha256(environment.to_dict()),
            "evaluator_sha256": _stable_json_sha256(scenario.hidden_evaluator.to_dict()),
            "scenario_sha256": _stable_json_sha256(scenario.to_dict()),
            "source_record_sha256": _stable_json_sha256(source_record),
            "hidden_test_patch_sha256": patch_sha256,
            "hidden_test_commands_sha256": _stable_json_sha256(command_hashes),
            "setup_commands_sha256": setup_sha256,
            "health_check_sha256": health_sha256,
            "hidden_patch_rehearsal_record_sha256": _stable_json_sha256(
                _safe_rehearsal_record(rehearsal_record)
            ),
            "materialization_reset_record_sha256": _stable_json_sha256(
                _safe_materialization_record(materialization_record)
            ),
            "workspace_tree_sha256": _materialization_tree_sha256(materialization_record),
            "workspace_origin_sha256": _text_sha256(
                str(source_record.get("workspace_original_source_uri") or "")
            ),
            "repair_validation_artifact_sha256": validation_record.get("_artifact_sha256", ""),
            "reference_repair_record_sha256": _stable_json_sha256(
                _safe_reference_repair_record(reference_repair_record)
            ),
            "validated_repair_sha256": normalized_validation.get("validated_repair_sha256", ""),
            "container_replay_record_sha256": _stable_json_sha256(
                _safe_container_replay_record(container_replay_record)
            ),
        },
    }


def _validate_rehearsal(
    value: dict[str, Any],
    *,
    source_path: Path,
    expected_scenario_ids: set[str],
    gate: _Gate,
) -> dict[str, dict[str, Any]]:
    gate.require(
        value.get("schema_version") == REGISTRY_IMPORT_REHEARSAL_SCHEMA_VERSION,
        "hidden_patch_rehearsal_schema_invalid",
    )
    gate.require(value.get("valid") is True, "hidden_patch_rehearsal_invalid")
    source = _object(value.get("source"))
    gate.require(
        source.get("sha256") == _file_sha256(source_path),
        "hidden_patch_rehearsal_source_hash_mismatch",
    )
    imported = _object(value.get("import"))
    imported_ids = _string_list(imported.get("scenario_ids"))
    gate.require(
        _first_int(imported.get("imported")) == EXPECTED_SEED_COUNT,
        "hidden_patch_rehearsal_import_count_not_exact",
    )
    gate.require(
        set(imported_ids) == expected_scenario_ids and len(imported_ids) == EXPECTED_SEED_COUNT,
        "hidden_patch_rehearsal_import_set_mismatch",
    )
    rehearsal = _object(value.get("hidden_test_patch_rehearsal"))
    gate.require(rehearsal.get("enabled") is True, "hidden_patch_rehearsal_not_enabled")
    gate.require(rehearsal.get("valid") is True, "hidden_patch_rehearsal_invalid")
    gate.require(
        rehearsal.get("expected_outcome") == "fail",
        "hidden_patch_rehearsal_outcome_not_fail",
    )
    gate.require(
        _first_int(rehearsal.get("requested")) == EXPECTED_SEED_COUNT
        and _first_int(rehearsal.get("sampled")) == EXPECTED_SEED_COUNT,
        "hidden_patch_rehearsal_sample_count_not_exact",
    )
    results = _strict_object_list(
        rehearsal.get("results"),
        gate,
        "hidden_patch_rehearsal_results_invalid",
    )
    indexed = _index_records(
        results,
        lambda item: str(item.get("scenario_id") or ""),
        gate,
        "hidden_patch_rehearsal",
    )
    gate.require(
        set(indexed) == expected_scenario_ids and len(results) == EXPECTED_SEED_COUNT,
        "hidden_patch_rehearsal_result_set_mismatch",
    )
    return indexed


def _check_rehearsal_record(
    value: dict[str, Any],
    *,
    scenario: Scenario,
    patch_sha256: str,
    command_hashes: list[str],
    gate: _Gate,
) -> None:
    entry_id = scenario.scenario_id
    gate.require(bool(value), "hidden_patch_rehearsal_record_missing", entry_id)
    if not value:
        return
    gate.require(
        value.get("valid") is True,
        "hidden_patch_rehearsal_record_invalid",
        entry_id,
    )
    gate.require(
        value.get("environment_id") == scenario.environment.environment_id,
        "hidden_patch_rehearsal_environment_mismatch",
        entry_id,
    )
    gate.require(
        value.get("source_revision") == scenario.environment.source_revision,
        "hidden_patch_rehearsal_revision_mismatch",
        entry_id,
    )
    gate.require(
        (value.get("test_patch_sha256") or value.get("hidden_test_patch_sha256")) == patch_sha256,
        "hidden_patch_rehearsal_patch_hash_mismatch",
        entry_id,
    )
    gate.require(
        _exit_code(value, "patch_check_exit_code", ("patch_check", "exit_code")) == 0,
        "hidden_patch_rehearsal_patch_check_failed",
        entry_id,
    )
    gate.require(
        _exit_code(value, "patch_apply_exit_code", ("patch_apply", "exit_code")) == 0,
        "hidden_patch_rehearsal_patch_apply_failed",
        entry_id,
    )
    command_results = _strict_object_list(
        value.get("command_results"),
        gate,
        "hidden_patch_rehearsal_command_results_invalid",
        entry_id,
    )
    result_hashes = sorted(str(item.get("command_sha256") or "") for item in command_results)
    exit_codes = [_first_int(item.get("exit_code")) for item in command_results]
    gate.require(
        value.get("hidden_commands_ran") is True
        and _first_int(value.get("commands_run")) == len(command_hashes)
        and _first_int(value.get("hidden_command_count")) == len(command_hashes),
        "hidden_patch_rehearsal_commands_not_run",
        entry_id,
    )
    gate.require(
        result_hashes == command_hashes,
        "hidden_patch_rehearsal_command_set_mismatch",
        entry_id,
    )
    gate.require(
        len(exit_codes) == len(command_hashes)
        and all(exit_code is not None for exit_code in exit_codes)
        and any(exit_code != 0 for exit_code in exit_codes),
        "hidden_patch_rehearsal_original_did_not_fail",
        entry_id,
    )
    derived_outcome = (
        "fail"
        if exit_codes
        and all(exit_code is not None for exit_code in exit_codes)
        and any(exit_code != 0 for exit_code in exit_codes)
        else "pass"
    )
    gate.require(
        value.get("expected_outcome") == "fail" and value.get("command_outcome") == derived_outcome,
        "hidden_patch_rehearsal_outcome_label_mismatch",
        entry_id,
    )


def _validate_materialization_artifact(
    value: dict[str, Any],
    *,
    expected_scenario_ids: set[str],
    gate: _Gate,
) -> dict[str, dict[str, Any]]:
    gate.require(
        value.get("schema_version") == GOLD20_MATERIALIZATION_SCHEMA_VERSION,
        "materialization_schema_invalid",
    )
    gate.require(value.get("valid") is True, "materialization_artifact_invalid")
    records = _strict_object_list(
        value.get("records"),
        gate,
        "materialization_records_invalid",
    )
    indexed = _index_records(
        records,
        lambda item: str(item.get("scenario_id") or ""),
        gate,
        "materialization",
    )
    gate.require(
        len(records) == EXPECTED_SEED_COUNT and set(indexed) == expected_scenario_ids,
        "materialization_record_set_mismatch",
    )
    return indexed


def _check_materialization_record(
    value: dict[str, Any],
    *,
    scenario: Scenario,
    setup_sha256: str,
    health_sha256: str,
    gate: _Gate,
) -> None:
    entry_id = scenario.scenario_id
    gate.require(bool(value), "materialization_record_missing", entry_id)
    if not value:
        return
    environment = scenario.environment
    hashes = _string_list(value.get("workspace_tree_hashes"))
    gate.require(value.get("valid") is True, "materialization_record_invalid", entry_id)
    gate.require(
        value.get("environment_id") == environment.environment_id,
        "materialization_environment_mismatch",
        entry_id,
    )
    gate.require(
        value.get("source_revision") == environment.source_revision,
        "materialization_revision_mismatch",
        entry_id,
    )
    gate.require(
        value.get("image_digest") == environment.image_digest,
        "materialization_image_mismatch",
        entry_id,
    )
    gate.require(
        value.get("setup_commands_sha256") == setup_sha256,
        "materialization_setup_hash_mismatch",
        entry_id,
    )
    gate.require(
        value.get("health_check_sha256") == health_sha256,
        "materialization_health_hash_mismatch",
        entry_id,
    )
    attempts = _first_int(value.get("attempts"))
    gate.require(
        attempts is not None and attempts >= 2 and attempts == len(hashes),
        "materialization_attempts_insufficient",
        entry_id,
    )
    gate.require(
        bool(hashes) and len(set(hashes)) == 1 and all(_is_sha256(item) for item in hashes),
        "materialization_tree_hash_mismatch",
        entry_id,
    )
    attempt_results = _strict_object_list(
        value.get("attempt_results"),
        gate,
        "materialization_attempt_results_invalid",
        entry_id,
    )
    gate.require(
        attempts is not None and len(attempt_results) == attempts,
        "materialization_attempt_result_count_mismatch",
        entry_id,
    )
    attempts_valid = attempts is not None and len(attempt_results) == attempts
    for index, result in enumerate(attempt_results):
        setup_exits = _strict_exit_code_list(result.get("setup_exit_codes"))
        health_exits = _strict_exit_code_list(result.get("health_check_exit_codes"))
        result_valid = (
            _first_int(result.get("attempt")) == index
            and index < len(hashes)
            and result.get("workspace_tree_sha256") == hashes[index]
            and setup_exits is not None
            and len(setup_exits) == len(environment.setup_commands)
            and all(exit_code == 0 for exit_code in setup_exits)
            and health_exits is not None
            and len(health_exits) == len(environment.health_check)
            and all(exit_code == 0 for exit_code in health_exits)
            and result.get("valid") is True
        )
        gate.require(result_valid, "materialization_attempt_result_invalid", entry_id)
        attempts_valid = attempts_valid and result_valid
    gate.require(
        value.get("health_checks_passed") is True and attempts_valid,
        "materialization_health_check_failed",
        entry_id,
    )


def _load_container_runtime_builds(
    config: dict[str, Any],
    config_dir: Path,
    gate: _Gate,
) -> dict[str, dict[str, Any]]:
    records = _strict_object_list(
        config.get("container_runtime_builds"),
        gate,
        "container_replay_runtime_build_config_invalid",
    )
    indexed = _index_records(
        records,
        lambda item: str(item.get("image_digest") or ""),
        gate,
        "container_replay_runtime_build_config",
    )
    gate.require(bool(indexed), "container_replay_runtime_build_config_empty")
    normalized: dict[str, dict[str, Any]] = {}
    for image_digest, record in indexed.items():
        build_spec_value = str(record.get("build_spec") or "").strip()
        build_spec = _resolve_path(config_dir, build_spec_value) if build_spec_value else None
        build_spec_exists = build_spec is not None and build_spec.is_file()
        gate.require(
            bool(_IMAGE_PATTERN.fullmatch(image_digest))
            and is_immutable_image_reference(image_digest),
            "container_replay_runtime_image_not_content_addressed",
            image_digest,
        )
        gate.require(
            record.get("platform") == GOLD20_RUNTIME_PLATFORM,
            "container_replay_runtime_platform_invalid",
            image_digest,
        )
        gate.require(
            build_spec_exists,
            "container_replay_runtime_build_spec_missing",
            image_digest,
        )
        normalized[image_digest] = {
            "image_digest": image_digest,
            "platform": str(record.get("platform") or ""),
            "build_spec_sha256": _file_sha256(build_spec) if build_spec_exists else "",
        }
    return normalized


def _validate_container_replay_artifact(
    value: dict[str, Any],
    *,
    expected_source_ids: set[str],
    expected_scenario_ids: set[str],
    runtime_builds: dict[str, dict[str, Any]],
    gate: _Gate,
) -> dict[str, dict[str, Any]]:
    gate.require(
        value.get("schema_version") == GOLD20_CONTAINER_REPLAY_SCHEMA_VERSION,
        "container_replay_schema_invalid",
    )
    producer = _object(value.get("producer"))
    runtime_module = Path(__file__).with_name("gold20_runtime.py")
    sandbox_backend = Path(__file__).with_name("sandbox") / "docker.py"
    gate.require(
        producer.get("module_sha256") == _file_sha256(runtime_module),
        "container_replay_producer_hash_mismatch",
    )
    gate.require(
        producer.get("sandbox_backend_sha256") == _file_sha256(sandbox_backend),
        "container_replay_sandbox_hash_mismatch",
    )
    gate.require(
        _object(producer.get("component_sha256s"))
        == _runtime_producer_component_sha256s(),
        "container_replay_producer_component_hash_mismatch",
    )
    execution = _object(value.get("execution"))
    gate.require(
        execution.get("backend") == "DockerSandbox",
        "container_replay_backend_invalid",
    )
    gate.require(
        execution.get("platform") == GOLD20_RUNTIME_PLATFORM,
        "container_replay_platform_invalid",
    )
    gate.require(
        _first_int(execution.get("random_seed")) == GOLD20_RUNTIME_RANDOM_SEED,
        "container_replay_random_seed_invalid",
    )
    gate.require(
        bool(str(execution.get("docker_server_version") or "").strip()),
        "container_replay_docker_version_missing",
    )

    runtime_records = _strict_object_list(
        value.get("runtime_builds"),
        gate,
        "container_replay_runtime_builds_invalid",
    )
    runtime_by_image = _index_records(
        runtime_records,
        lambda item: str(item.get("image_digest") or ""),
        gate,
        "container_replay_runtime_build",
    )
    gate.require(
        set(runtime_by_image) == set(runtime_builds),
        "container_replay_runtime_build_set_mismatch",
    )
    for image_digest, expected in runtime_builds.items():
        record = runtime_by_image.get(image_digest, {})
        gate.require(
            record.get("platform") == expected["platform"],
            "container_replay_runtime_build_platform_mismatch",
            image_digest,
        )
        gate.require(
            record.get("build_spec_sha256") == expected["build_spec_sha256"],
            "container_replay_runtime_build_spec_hash_mismatch",
            image_digest,
        )
        gate.require(
            record.get("image_id") == _image_content_digest(image_digest),
            "container_replay_runtime_image_id_mismatch",
            image_digest,
        )
        gate.require(
            record.get("build_verification_mode") == GOLD20_BUILD_VERIFICATION_MODE,
            "container_replay_runtime_build_verification_mode_invalid",
            image_digest,
        )
        gate.require(
            (_first_int(record.get("image_size_bytes")) or 0) > 0,
            "container_replay_runtime_image_size_invalid",
            image_digest,
        )

    records = _strict_object_list(
        value.get("records"),
        gate,
        "container_replay_records_invalid",
    )
    indexed = _index_records(
        records,
        _source_instance_id,
        gate,
        "container_replay",
    )
    scenario_ids = {str(record.get("scenario_id") or "") for record in records}
    gate.require(
        len(records) == EXPECTED_SEED_COUNT and set(indexed) == expected_source_ids,
        "container_replay_record_set_mismatch",
    )
    gate.require(
        scenario_ids == expected_scenario_ids and len(scenario_ids) == EXPECTED_SEED_COUNT,
        "container_replay_scenario_set_mismatch",
    )
    valid_count = sum(record.get("valid") is True for record in records)
    counts = _object(value.get("counts"))
    gate.require(
        _first_int(counts.get("records")) == len(records)
        and _first_int(counts.get("valid")) == valid_count
        and _first_int(counts.get("invalid")) == len(records) - valid_count,
        "container_replay_counts_mismatch",
    )
    gate.require(
        value.get("valid") is True and valid_count == EXPECTED_SEED_COUNT,
        "container_replay_artifact_invalid",
    )
    return indexed


def _check_container_replay_record(
    value: dict[str, Any],
    *,
    scenario: Scenario,
    source_id: str,
    patch_sha256: str,
    command_hashes: list[str],
    materialization_record: dict[str, Any],
    setup_sha256: str,
    health_sha256: str,
    validated_repair_sha256: str,
    runtime_builds: dict[str, dict[str, Any]],
    gate: _Gate,
) -> None:
    entry_id = scenario.scenario_id
    gate.require(bool(value), "container_replay_record_missing", entry_id)
    if not value:
        return
    environment = scenario.environment
    runtime_build = runtime_builds.get(environment.image_digest, {})
    expected_tree = _materialization_tree_sha256(materialization_record)
    base_tree = str(value.get("base_workspace_tree_sha256") or "").lower()
    repaired_tree = str(value.get("repaired_workspace_tree_sha256") or "").lower()
    base_state = str(value.get("base_initial_state_sha256") or "").lower()
    repaired_state = str(value.get("repaired_initial_state_sha256") or "").lower()
    setup_count = len(environment.setup_commands)
    health_count = len(environment.health_check)
    hidden_count = len(scenario.hidden_evaluator.hidden_tests)
    expected_limits = asdict(SandboxLimits(**environment.resource_limits))
    expected_policy = _expected_container_sandbox_policy(
        SandboxLimits(**environment.resource_limits),
        environment.image_digest,
    )

    gate.require(value.get("valid") is True, "container_replay_record_invalid", entry_id)
    gate.require(
        _source_instance_id(value) == source_id,
        "container_replay_source_instance_mismatch",
        entry_id,
    )
    gate.require(
        value.get("scenario_id") == scenario.scenario_id,
        "container_replay_scenario_id_mismatch",
        entry_id,
    )
    gate.require(
        value.get("seed_id") == scenario.query_seed.seed_id,
        "container_replay_seed_id_mismatch",
        entry_id,
    )
    gate.require(
        value.get("environment_id") == environment.environment_id,
        "container_replay_environment_id_mismatch",
        entry_id,
    )
    gate.require(
        value.get("source_revision") == environment.source_revision,
        "container_replay_revision_mismatch",
        entry_id,
    )
    gate.require(
        value.get("image_digest") == environment.image_digest,
        "container_replay_image_mismatch",
        entry_id,
    )
    gate.require(
        value.get("runtime_build_spec_sha256") == runtime_build.get("build_spec_sha256"),
        "container_replay_runtime_build_spec_mismatch",
        entry_id,
    )
    gate.require(
        value.get("limits") == expected_limits,
        "container_replay_limits_mismatch",
        entry_id,
    )
    gate.require(
        value.get("sandbox_policy") == expected_policy,
        "container_replay_sandbox_policy_mismatch",
        entry_id,
    )
    gate.require(
        _is_sha256(expected_tree) and base_tree == repaired_tree == expected_tree,
        "container_replay_materialization_tree_mismatch",
        entry_id,
    )
    gate.require(
        _is_sha256(base_state) and base_state == repaired_state,
        "container_replay_initial_state_mismatch",
        entry_id,
    )
    expected_instance_id = (
        ScenarioInstance.materialize(
            scenario,
            random_seed=GOLD20_RUNTIME_RANDOM_SEED,
            initial_state_hash=base_state,
        ).instance_id
        if _is_sha256(base_state)
        else ""
    )
    gate.require(
        value.get("scenario_instance_id") == expected_instance_id,
        "container_replay_instance_id_mismatch",
        entry_id,
    )
    gate.require(
        value.get("hidden_test_patch_sha256") == patch_sha256,
        "container_replay_patch_hash_mismatch",
        entry_id,
    )
    gate.require(
        _string_list(value.get("hidden_test_command_sha256s")) == command_hashes,
        "container_replay_command_hash_mismatch",
        entry_id,
    )
    gate.require(
        value.get("validated_repair_sha256") == validated_repair_sha256
        and _is_sha256(validated_repair_sha256),
        "container_replay_repair_hash_mismatch",
        entry_id,
    )
    gate.require(
        value.get("setup_commands_sha256") == setup_sha256,
        "container_replay_setup_hash_mismatch",
        entry_id,
    )
    gate.require(
        value.get("health_check_sha256") == health_sha256,
        "container_replay_health_hash_mismatch",
        entry_id,
    )

    for key, expected_count in (
        ("base_setup_exit_codes", setup_count),
        ("repaired_setup_exit_codes", setup_count),
        ("base_health_exit_codes", health_count),
        ("repaired_health_exit_codes", health_count),
        ("base_post_health_exit_codes", health_count),
        ("repaired_post_health_exit_codes", health_count),
    ):
        exits = _strict_exit_code_list(value.get(key))
        gate.require(
            exits is not None
            and len(exits) == expected_count
            and all(exit_code == 0 for exit_code in exits),
            "container_replay_setup_or_health_failed",
            entry_id,
        )
    base_health_hashes = _string_list(value.get("base_health_result_sha256s"))
    repaired_health_hashes = _string_list(value.get("repaired_health_result_sha256s"))
    gate.require(
        len(base_health_hashes) == health_count
        and base_health_hashes == repaired_health_hashes
        and all(_is_sha256(item) for item in base_health_hashes),
        "container_replay_health_result_hash_mismatch",
        entry_id,
    )
    gate.require(
        _first_int(value.get("base_hidden_patch_exit")) == 0
        and _first_int(value.get("repaired_hidden_patch_exit")) == 0
        and value.get("base_hidden_patch_infrastructure_failure") is False
        and value.get("repaired_hidden_patch_infrastructure_failure") is False,
        "container_replay_hidden_patch_failed",
        entry_id,
    )
    gate.require(
        _first_int(value.get("repair_check_exit")) == 0
        and _first_int(value.get("repair_apply_exit")) == 0,
        "container_replay_repair_apply_failed",
        entry_id,
    )
    base_hidden_exits = _strict_exit_code_list(value.get("base_hidden_test_exit_codes"))
    repaired_hidden_exits = _strict_exit_code_list(
        value.get("repaired_hidden_test_exit_codes")
    )
    gate.require(
        base_hidden_exits is not None
        and len(base_hidden_exits) == hidden_count
        and any(exit_code != 0 for exit_code in base_hidden_exits),
        "container_replay_original_did_not_fail",
        entry_id,
    )
    gate.require(
        repaired_hidden_exits is not None
        and len(repaired_hidden_exits) == hidden_count
        and all(exit_code == 0 for exit_code in repaired_hidden_exits),
        "container_replay_repair_did_not_pass",
        entry_id,
    )
    base_infrastructure = _strict_bool_list(
        value.get("base_hidden_test_infrastructure_failures")
    )
    repaired_infrastructure = _strict_bool_list(
        value.get("repaired_hidden_test_infrastructure_failures")
    )
    gate.require(
        base_infrastructure is not None
        and repaired_infrastructure is not None
        and len(base_infrastructure) == len(repaired_infrastructure) == hidden_count
        and not any(base_infrastructure)
        and not any(repaired_infrastructure),
        "container_replay_hidden_test_infrastructure_failure",
        entry_id,
    )
    expected_base_result_hashes = (
        [
            _hidden_test_result_sha256(exit_code, infrastructure_failure)
            for exit_code, infrastructure_failure in zip(
                base_hidden_exits,
                base_infrastructure,
                strict=True,
            )
        ]
        if base_hidden_exits is not None
        and base_infrastructure is not None
        and len(base_hidden_exits) == len(base_infrastructure) == hidden_count
        else []
    )
    expected_repaired_result_hashes = (
        [
            _hidden_test_result_sha256(exit_code, infrastructure_failure)
            for exit_code, infrastructure_failure in zip(
                repaired_hidden_exits,
                repaired_infrastructure,
                strict=True,
            )
        ]
        if repaired_hidden_exits is not None
        and repaired_infrastructure is not None
        and len(repaired_hidden_exits) == len(repaired_infrastructure) == hidden_count
        else []
    )
    gate.require(
        _string_list(value.get("base_hidden_test_result_sha256s"))
        == expected_base_result_hashes
        and _string_list(value.get("repaired_hidden_test_result_sha256s"))
        == expected_repaired_result_hashes,
        "container_replay_hidden_test_result_hash_invalid",
        entry_id,
    )


def _load_reference_repairs(path: Path, gate: _Gate) -> dict[str, dict[str, Any]]:
    container = _read_object(path)
    gate.require(
        container.get("schema_version") == GOLD20_REFERENCE_REPAIRS_SCHEMA_VERSION,
        "reference_repair_schema_invalid",
    )
    records = _strict_object_list(
        container.get("records"),
        gate,
        "reference_repair_records_invalid",
    )
    return _index_records(
        records,
        _source_instance_id,
        gate,
        "reference_repair",
    )


def _load_validation_evidence(paths: list[Path], gate: _Gate) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        container = _read_object(path)
        schema = container.get("schema_version")
        raw_records = container.get("records")
        aggregate = isinstance(raw_records, list)
        if aggregate:
            records = _strict_object_list(
                raw_records,
                gate,
                "repair_validation_aggregate_records_invalid",
            )
            gate.require(
                schema == GOLD_REPAIR_VALIDATION_SCHEMA_VERSION,
                "repair_validation_schema_invalid",
            )
            gate.require(
                container.get("valid") is True,
                "repair_validation_aggregate_invalid",
            )
            counts = _object(container.get("counts"))
            valid_records = sum(record.get("valid") is True for record in records)
            gate.require(
                _first_int(counts.get("records")) == len(records)
                and _first_int(counts.get("valid")) == valid_records
                and _first_int(counts.get("invalid")) == len(records) - valid_records,
                "repair_validation_aggregate_counts_mismatch",
            )
        else:
            records = [container]
            gate.require(
                schema == HIDDEN_TEST_PATCH_VALIDATION_SCHEMA_VERSION,
                "repair_validation_schema_invalid",
            )
        for raw_record in records:
            record_schema = raw_record.get("schema_version")
            gate.require(
                record_schema
                in {
                    None,
                    "",
                    HIDDEN_TEST_PATCH_VALIDATION_SCHEMA_VERSION,
                    GOLD_REPAIR_VALIDATION_SCHEMA_VERSION,
                },
                "repair_validation_schema_invalid",
            )
            source_id = _source_instance_id(raw_record)
            gate.require(bool(source_id), "repair_validation_source_instance_missing")
            if not source_id:
                continue
            if source_id in indexed:
                gate.require(False, "repair_validation_duplicate_source_instance", source_id)
                continue
            value = dict(raw_record)
            value["_artifact_sha256"] = _file_sha256(path)
            indexed[source_id] = value
    return indexed


def _check_validation_record(
    value: dict[str, Any],
    *,
    scenario: Scenario,
    source_id: str,
    patch_sha256: str,
    command_hashes: list[str],
    repository: str,
    materialization_record: dict[str, Any],
    setup_sha256: str,
    health_sha256: str,
    reference_repair_record: dict[str, Any],
    gate: _Gate,
) -> dict[str, Any]:
    entry_id = scenario.scenario_id
    gate.require(bool(value), "repair_validation_record_missing", entry_id)
    if not value:
        return {}
    normalized = _normalize_validation_record(value)
    gate.require(bool(normalized["valid"]), "repair_validation_record_invalid", entry_id)
    gate.require(
        normalized["source_instance_id"] == source_id,
        "repair_validation_source_instance_mismatch",
        entry_id,
    )
    gate.require(
        normalized["scenario_id"] == scenario.scenario_id,
        "repair_validation_scenario_id_mismatch",
        entry_id,
    )
    gate.require(
        normalized["seed_id"] == scenario.query_seed.seed_id,
        "repair_validation_seed_id_mismatch",
        entry_id,
    )
    gate.require(
        normalized["environment_id"] == scenario.environment.environment_id,
        "repair_validation_environment_id_mismatch",
        entry_id,
    )
    gate.require(
        normalized["repository"] == repository,
        "repair_validation_repository_mismatch",
        entry_id,
    )
    gate.require(
        normalized["source_revision"] == scenario.environment.source_revision,
        "repair_validation_revision_mismatch",
        entry_id,
    )
    gate.require(
        normalized["hidden_test_patch_sha256"] == patch_sha256,
        "repair_validation_patch_hash_mismatch",
        entry_id,
    )
    gate.require(
        normalized["hidden_test_command_sha256s"] == command_hashes,
        "repair_validation_command_hash_mismatch",
        entry_id,
    )
    gate.require(
        normalized["patch_check_exit"] == 0,
        "repair_validation_patch_check_failed",
        entry_id,
    )
    gate.require(
        normalized["base_patch_apply_exit"] == 0,
        "repair_validation_patch_apply_failed",
        entry_id,
    )
    gate.require(
        normalized["original_hidden_test_exit"] is not None
        and normalized["original_hidden_test_exit"] != 0,
        "repair_validation_original_did_not_fail",
        entry_id,
    )
    gate.require(
        normalized["fixed_hidden_test_exit"] == 0,
        "repair_validation_repair_did_not_pass",
        entry_id,
    )
    gate.require(
        not normalized["repair_check_present"] or normalized["repair_check_exit"] == 0,
        "repair_validation_repair_check_failed",
        entry_id,
    )
    gate.require(
        normalized["repair_apply_exit"] == 0,
        "repair_validation_repair_apply_failed",
        entry_id,
    )
    gate.require(
        not normalized["repaired_patch_check_present"]
        or normalized["repaired_patch_check_exit"] == 0,
        "repair_validation_repaired_patch_check_failed",
        entry_id,
    )
    gate.require(
        normalized["repaired_patch_apply_exit"] == 0,
        "repair_validation_repaired_patch_apply_failed",
        entry_id,
    )
    gate.require(
        normalized["workspace_revision_matches"] is True,
        "repair_validation_workspace_revision_mismatch",
        entry_id,
    )
    materialization_hashes = _string_list(materialization_record.get("workspace_tree_hashes"))
    expected_tree_sha256 = (
        materialization_hashes[0]
        if materialization_hashes and len(set(materialization_hashes)) == 1
        else ""
    )
    gate.require(
        normalized["materialization_tree_sha256"] == expected_tree_sha256
        and _is_sha256(expected_tree_sha256),
        "repair_validation_materialization_tree_mismatch",
        entry_id,
    )
    gate.require(
        normalized["image_digest"] == scenario.environment.image_digest,
        "repair_validation_image_mismatch",
        entry_id,
    )
    gate.require(
        normalized["setup_commands_sha256"] == setup_sha256,
        "repair_validation_setup_hash_mismatch",
        entry_id,
    )
    gate.require(
        normalized["health_check_sha256"] == health_sha256,
        "repair_validation_health_hash_mismatch",
        entry_id,
    )
    gate.require(
        _is_sha256(normalized["validated_repair_sha256"]),
        "repair_validation_repair_hash_missing",
        entry_id,
    )
    actual_repair_sha256 = _check_reference_repair_record(
        reference_repair_record,
        scenario=scenario,
        source_id=source_id,
        gate=gate,
    )
    gate.require(
        bool(actual_repair_sha256)
        and normalized["validated_repair_sha256"] == actual_repair_sha256,
        "repair_validation_reference_repair_hash_mismatch",
        entry_id,
    )
    return normalized


def _normalize_validation_record(value: dict[str, Any]) -> dict[str, Any]:
    command = str(
        value.get("hidden_test_command")
        or value.get("command")
        or _nested(value, "checks", "fails_on_original_source", "hidden_test_run", "command")
        or ""
    )
    return {
        "source_instance_id": _source_instance_id(value),
        "scenario_id": str(value.get("scenario_id") or ""),
        "seed_id": str(value.get("seed_id") or ""),
        "environment_id": str(value.get("environment_id") or ""),
        "repository": _normalize_repository(value.get("repository") or value.get("repo")),
        "source_revision": str(value.get("source_revision") or value.get("base_commit") or ""),
        "hidden_test_patch_sha256": str(
            value.get("hidden_test_patch_sha256") or value.get("test_patch_sha256") or ""
        ).lower(),
        "hidden_test_command_sha256s": _validation_command_hashes(value, command),
        "patch_check_exit": _first_int(
            value.get("patch_check_exit"),
            _nested(value, "checks", "patch_apply_check", "exit_code"),
            _nested(value, "patch_check", "exit_code"),
            _nested(value, "base", "hidden_patch_check", "exit_code"),
        ),
        "base_patch_apply_exit": _first_int(
            value.get("patch_apply_exit"),
            _nested(value, "base", "hidden_patch_apply", "exit_code"),
        ),
        "original_hidden_test_exit": _first_int(
            value.get("original_hidden_test_exit"),
            _nested(
                value,
                "checks",
                "fails_on_original_source",
                "hidden_test_run",
                "exit_code",
            ),
            _nested(value, "original_command", "exit_code"),
            _nested(value, "base", "hidden_test_run", "exit_code"),
        ),
        "fixed_hidden_test_exit": _first_int(
            value.get("fixed_hidden_test_exit"),
            _nested(
                value,
                "checks",
                "passes_after_minimal_fix",
                "hidden_test_run",
                "exit_code",
            ),
            _nested(value, "fixed_command", "exit_code"),
            _nested(value, "repaired", "hidden_test_run", "exit_code"),
        ),
        "repair_check_exit": _first_int(
            value.get("repair_check_exit"),
            _nested(value, "repaired", "repair_patch_check", "exit_code"),
        ),
        "repair_check_present": _any_path_present(
            value,
            ("repair_check_exit",),
            ("repaired", "repair_patch_check", "exit_code"),
        ),
        "repair_apply_exit": _first_int(
            value.get("repair_apply_exit"),
            _nested(value, "repaired", "repair_patch_apply", "exit_code"),
        ),
        "repaired_patch_check_exit": _first_int(
            value.get("repaired_patch_check_exit"),
            _nested(value, "repaired", "hidden_patch_check", "exit_code"),
        ),
        "repaired_patch_check_present": _any_path_present(
            value,
            ("repaired_patch_check_exit",),
            ("repaired", "hidden_patch_check", "exit_code"),
        ),
        "repaired_patch_apply_exit": _first_int(
            value.get("repaired_patch_apply_exit"),
            _nested(value, "repaired", "hidden_patch_apply", "exit_code"),
        ),
        "validated_repair_sha256": str(
            value.get("validated_repair_sha256")
            or value.get("fix_patch_sha256")
            or value.get("repair_patch_sha256")
            or ""
        ).lower(),
        "materialization_tree_sha256": str(
            value.get("materialization_tree_sha256") or value.get("workspace_tree_sha256") or ""
        ).lower(),
        "image_digest": str(value.get("image_digest") or ""),
        "setup_commands_sha256": str(value.get("setup_commands_sha256") or "").lower(),
        "health_check_sha256": str(value.get("health_check_sha256") or "").lower(),
        "workspace_revision_matches": value.get("workspace_revision_matches") is True,
        "valid": value.get("valid") is True,
    }


def _check_reference_repair_record(
    value: dict[str, Any],
    *,
    scenario: Scenario,
    source_id: str,
    gate: _Gate,
) -> str:
    entry_id = scenario.scenario_id
    gate.require(bool(value), "reference_repair_record_missing", entry_id)
    if not value:
        return ""
    patch_value = value.get("repair_patch")
    patch = patch_value if isinstance(patch_value, str) else ""
    declared_sha256 = str(value.get("repair_patch_sha256") or "").lower()
    actual_sha256 = _text_sha256(patch) if patch else ""
    gate.require(
        bool(patch) and patch.startswith("diff --git "),
        "reference_repair_patch_invalid",
        entry_id,
    )
    gate.require(
        _is_sha256(declared_sha256) and declared_sha256 == actual_sha256,
        "reference_repair_patch_hash_mismatch",
        entry_id,
    )
    gate.require(
        _source_instance_id(value) == source_id,
        "reference_repair_source_instance_mismatch",
        entry_id,
    )
    gate.require(
        str(value.get("scenario_id") or "") == scenario.scenario_id,
        "reference_repair_scenario_id_mismatch",
        entry_id,
    )
    gate.require(
        str(value.get("seed_id") or "") == scenario.query_seed.seed_id,
        "reference_repair_seed_id_mismatch",
        entry_id,
    )
    gate.require(
        str(value.get("environment_id") or "") == scenario.environment.environment_id,
        "reference_repair_environment_id_mismatch",
        entry_id,
    )
    gate.require(
        str(value.get("source_revision") or "") == scenario.environment.source_revision,
        "reference_repair_revision_mismatch",
        entry_id,
    )
    return actual_sha256


def _validation_command_hashes(value: dict[str, Any], command: str) -> list[str]:
    declared = _string_list(value.get("hidden_test_command_sha256s"))
    if not declared:
        singular = str(value.get("hidden_test_command_sha256") or "")
        if singular:
            declared = [singular]
        elif command:
            declared = [_text_sha256(command)]
    return sorted(item.lower() for item in declared)


def _check_hidden_isolation(scenario: Scenario, gate: _Gate) -> None:
    gate.require(
        not _hidden_context_leaks_into_public(scenario, scenario),
        "hidden_evaluator_leaked_to_public_task",
        scenario.scenario_id,
    )


def _check_corpus_hidden_isolation(scenarios: Iterable[Scenario], gate: _Gate) -> None:
    scenario_list = list(scenarios)
    for hidden_scenario in scenario_list:
        for public_scenario in scenario_list:
            if hidden_scenario.scenario_id == public_scenario.scenario_id:
                continue
            gate.require(
                not _hidden_context_leaks_into_public(hidden_scenario, public_scenario),
                "hidden_evaluator_leaked_across_public_tasks",
                f"{hidden_scenario.scenario_id}:{public_scenario.scenario_id}",
            )


def _hidden_context_leaks_into_public(
    hidden_scenario: Scenario,
    public_scenario: Scenario,
) -> bool:
    public_values: list[str] = []
    _collect_strings(public_scenario.query_seed.public.to_dict(), public_values)
    canaries = _privacy_canaries(hidden_scenario)
    hidden_values: list[str] = []
    _collect_strings(hidden_scenario.query_seed.hidden_user.to_dict(), hidden_values)
    evaluator = hidden_scenario.hidden_evaluator
    _collect_strings(evaluator.reference_answer, hidden_values)
    _collect_strings(evaluator.reference_artifacts, hidden_values)
    _collect_strings(evaluator.hidden_tests, hidden_values)
    _collect_strings(evaluator.required_state, hidden_values)
    _collect_strings(evaluator.forbidden_state, hidden_values)
    public_lineage_fields = {
        "source_adapter",
        "source_format",
        "source_name",
        "source_instance_id",
        "source_type",
    }
    _collect_strings(
        {
            key: value
            for key, value in evaluator.metadata.items()
            if key not in public_lineage_fields
        },
        hidden_values,
    )
    _collect_strings(hidden_scenario.environment.evaluator_refs, hidden_values)
    exact_private_values = canaries | {value for value in hidden_values if len(value) >= 8}
    sensitive = {
        fragment for value in hidden_values for fragment in _sensitive_fragments(value) if fragment
    }
    public_lineage_values = [
        hidden_scenario.metadata.get(field_name)
        or hidden_scenario.query_seed.metadata.get(field_name)
        or hidden_scenario.query_seed.public.context.get(field_name)
        or hidden_scenario.environment.metadata.get(field_name)
        for field_name in (
            "source_adapter",
            "source_format",
            "source_name",
            "source_instance_id",
            "source_type",
            "repository",
            "source_url",
        )
    ]
    public_lineage_fragments = {
        fragment
        for value in public_lineage_values
        for fragment in _sensitive_fragments(str(value or ""))
        if fragment
    }
    sensitive.difference_update(public_lineage_fragments)
    return any(
        hidden in public for hidden in exact_private_values for public in public_values
    ) or any(
        hidden in public for hidden in sensitive if len(hidden) >= 8 for public in public_values
    )


def _privacy_canaries(scenario: Scenario) -> set[str]:
    values: list[str] = []
    evaluator = scenario.hidden_evaluator
    _collect_strings(evaluator.reference_answer, values)
    _collect_strings(evaluator.reference_artifacts, values)
    _collect_strings(evaluator.hidden_tests, values)
    _collect_strings(evaluator.metadata.get("test_patch"), values)
    _collect_strings(scenario.environment.evaluator_refs, values)
    hidden_user = scenario.query_seed.hidden_user
    _collect_strings(hidden_user.goal, values)
    _collect_strings(hidden_user.known_facts, values)
    _collect_strings(hidden_user.unavailable_facts, values)
    _collect_strings(hidden_user.business_knowledge_refs, values)
    return {value for value in values if value}


def _redact_private_metadata(value: str, canaries: Iterable[str]) -> str:
    for canary in canaries:
        for candidate in {canary, _normalize_label(canary)}:
            if candidate and (candidate in value if len(candidate) >= 4 else candidate == value):
                return ""
    return value


def _manifest_issue(issue: dict[str, str]) -> dict[str, str]:
    value = {"code": issue["code"]}
    if issue.get("entry_id"):
        value["entry_id_sha256"] = _text_sha256(issue["entry_id"])
    return value


def _sensitive_fragments(value: str) -> set[str]:
    fragments: set[str] = set()
    paths = re.findall(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+", value)
    fragments.update(paths)
    fragments.update(path.rsplit("/", 1)[-1] for path in paths)
    fragments.update(re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])", value))
    fragments.update(
        token
        for token in re.findall(r"[A-Za-z0-9_.-]{8,}", value)
        if any(marker in token.lower() for marker in ("hidden", "private", "secret", "evaluator"))
    )
    return fragments


def _safe_rehearsal_record(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": value.get("scenario_id", ""),
        "environment_id": value.get("environment_id", ""),
        "source_revision": value.get("source_revision", ""),
        "test_patch_sha256": value.get("test_patch_sha256", ""),
        "patch_check_exit_code": value.get("patch_check_exit_code"),
        "patch_apply_exit_code": value.get("patch_apply_exit_code"),
        "hidden_command_count": value.get("hidden_command_count"),
        "hidden_commands_ran": value.get("hidden_commands_ran"),
        "commands_run": value.get("commands_run"),
        "command_outcome": value.get("command_outcome", ""),
        "command_results": [
            {
                "command_sha256": item.get("command_sha256", ""),
                "exit_code": item.get("exit_code"),
                "stdout_sha256": item.get("stdout_sha256", ""),
                "stderr_sha256": item.get("stderr_sha256", ""),
            }
            for item in _object_list(value.get("command_results"))
        ],
        "valid": value.get("valid"),
    }


def _safe_materialization_record(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "scenario_id",
            "environment_id",
            "source_revision",
            "image_digest",
            "setup_commands_sha256",
            "health_check_sha256",
            "attempts",
            "workspace_tree_hashes",
            "attempt_results",
            "health_checks_passed",
            "valid",
        )
    }


def _materialization_tree_sha256(value: dict[str, Any]) -> str:
    hashes = _string_list(value.get("workspace_tree_hashes"))
    if hashes and len(set(hashes)) == 1 and _is_sha256(hashes[0]):
        return hashes[0].lower()
    return ""


def _safe_reference_repair_record(value: dict[str, Any]) -> dict[str, Any]:
    patch = value.get("repair_patch")
    actual_sha256 = _text_sha256(patch) if isinstance(patch, str) and patch else ""
    return {
        "source_instance_id": _source_instance_id(value),
        "scenario_id": str(value.get("scenario_id") or ""),
        "seed_id": str(value.get("seed_id") or ""),
        "environment_id": str(value.get("environment_id") or ""),
        "source_revision": str(value.get("source_revision") or ""),
        "repair_patch_sha256": actual_sha256,
    }


def _safe_container_replay_record(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "source_instance_id",
            "scenario_id",
            "seed_id",
            "environment_id",
            "scenario_instance_id",
            "source_revision",
            "image_digest",
            "runtime_build_spec_sha256",
            "sandbox_policy",
            "limits",
            "base_workspace_tree_sha256",
            "repaired_workspace_tree_sha256",
            "base_initial_state_sha256",
            "repaired_initial_state_sha256",
            "hidden_test_patch_sha256",
            "hidden_test_command_sha256s",
            "validated_repair_sha256",
            "setup_commands_sha256",
            "health_check_sha256",
            "base_setup_exit_codes",
            "repaired_setup_exit_codes",
            "base_health_exit_codes",
            "repaired_health_exit_codes",
            "base_post_health_exit_codes",
            "repaired_post_health_exit_codes",
            "base_health_result_sha256s",
            "repaired_health_result_sha256s",
            "base_hidden_patch_exit",
            "repaired_hidden_patch_exit",
            "base_hidden_patch_infrastructure_failure",
            "repaired_hidden_patch_infrastructure_failure",
            "repair_check_exit",
            "repair_apply_exit",
            "base_hidden_test_exit_codes",
            "repaired_hidden_test_exit_codes",
            "base_hidden_test_infrastructure_failures",
            "repaired_hidden_test_infrastructure_failures",
            "base_hidden_test_result_sha256s",
            "repaired_hidden_test_result_sha256s",
            "valid",
        )
    }


def _load_registry_scenarios(root: Path) -> list[Scenario]:
    scenario_dir = root / "scenarios"
    if not scenario_dir.is_dir():
        raise ValueError(f"Registry has no scenario directory: {root}")
    return [Scenario.from_dict(_read_object(path)) for path in sorted(scenario_dir.glob("*.json"))]


def _registry_snapshot_sha256(root: Path) -> str:
    entries = []
    for directory in ("seeds", "environments", "scenarios"):
        for path in sorted((root / directory).glob("*.json")):
            entries.append(
                {"path": path.relative_to(root).as_posix(), "sha256": _file_sha256(path)}
            )
    return _stable_json_sha256(entries)


def _seed_id_is_stable(seed: QuerySeed) -> bool:
    value = seed.to_dict()
    value.pop("seed_id", None)
    return QuerySeed.from_dict(value).seed_id == seed.seed_id


def _environment_id_is_stable(environment: EnvironmentSpec) -> bool:
    value = environment.to_dict()
    value.pop("environment_id", None)
    return EnvironmentSpec.from_dict(value).environment_id == environment.environment_id


def _scenario_id_is_stable(scenario: Scenario) -> bool:
    value = scenario.to_dict()
    value.pop("scenario_id", None)
    return Scenario.from_dict(value).scenario_id == scenario.scenario_id


def _scenario_source_instance_id(scenario: Scenario) -> str:
    return str(
        scenario.metadata.get("source_instance_id")
        or scenario.query_seed.metadata.get("source_instance_id")
        or scenario.query_seed.public.context.get("source_instance_id")
        or ""
    ).strip()


def _scenario_source_name(scenario: Scenario) -> str:
    return _normalize_source_name(
        scenario.metadata.get("source_name")
        or scenario.query_seed.metadata.get("source_name")
        or scenario.query_seed.provenance.rpartition(":")[0]
    )


def _source_instance_id(value: dict[str, Any]) -> str:
    return str(
        value.get("source_instance_id")
        or value.get("task_id")
        or value.get("instance_id")
        or value.get("id")
        or ""
    ).strip()


def _scenario_repository(scenario: Scenario) -> str:
    return _normalize_repository(
        scenario.query_seed.public.context.get("repository")
        or scenario.query_seed.metadata.get("repository")
        or scenario.environment.metadata.get("repository")
    )


def _scenario_language(scenario: Scenario) -> str:
    for tag in scenario.query_seed.coverage_tags:
        if tag.startswith("language:"):
            return _normalize_label(tag.split(":", 1)[1])
    return _normalize_label(
        scenario.query_seed.metadata.get("language")
        or scenario.environment.metadata.get("language")
    )


def _record_patch_sha256(value: dict[str, Any]) -> str:
    declared = str(
        value.get("hidden_test_patch_sha256") or value.get("test_patch_sha256") or ""
    ).lower()
    patch = str(value.get("test_patch") or value.get("hidden_test_patch") or "")
    if patch:
        actual = _text_sha256(patch)
        return actual if not declared or declared == actual else ""
    return declared


def _index_records(
    records: Iterable[dict[str, Any]],
    key_fn: Any,
    gate: _Gate,
    prefix: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(key_fn(record) or "").strip()
        gate.require(bool(key), f"{prefix}_record_id_missing")
        if not key:
            continue
        if key in indexed:
            gate.require(False, f"{prefix}_duplicate_record_id", key)
            continue
        indexed[key] = record
    return indexed


def _exit_code(value: dict[str, Any], top: str, nested: tuple[str, ...]) -> int | None:
    return _first_int(value.get(top), _nested(value, *nested))


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value.strip()):
            return int(value)
    return None


def _strict_exit_code_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    exit_codes = [_first_int(item) for item in value]
    if any(exit_code is None for exit_code in exit_codes):
        return None
    return [int(exit_code) for exit_code in exit_codes]


def _strict_bool_list(value: Any) -> list[bool] | None:
    if not isinstance(value, list) or any(not isinstance(item, bool) for item in value):
        return None
    return list(value)


def _expected_container_sandbox_policy(
    limits: SandboxLimits,
    image_digest: str,
) -> dict[str, Any]:
    return {
        "image_id": _image_content_digest(image_digest),
        "user": "65532:65532",
        "network_mode": "none",
        "rootfs_read_only": True,
        "privileged": False,
        "workspace_mount_type": "volume",
        "workspace_mount_read_write": True,
        "docker_socket_mounted": False,
        "tmpfs": "rw,noexec,nosuid,size=64m",
        "memory_bytes": _docker_memory_bytes(limits.memory),
        "nano_cpus": int(limits.cpus * 1_000_000_000),
        "pids_limit": limits.pids,
    }


def _image_content_digest(value: str) -> str:
    return value.rsplit("@", 1)[-1]


def _docker_memory_bytes(value: str) -> int:
    normalized = value.strip().lower()
    multipliers = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}
    if not normalized or normalized[-1] not in multipliers:
        return -1
    try:
        return int(float(normalized[:-1]) * multipliers[normalized[-1]])
    except ValueError:
        return -1


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _any_path_present(value: dict[str, Any], *paths: tuple[str, ...]) -> bool:
    for path in paths:
        current: Any = value
        for key in path[:-1]:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            if isinstance(current, dict) and path[-1] in current:
                return True
    return False


def _collect_strings(value: Any, output: list[str]) -> None:
    if isinstance(value, str):
        if value:
            output.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, output)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_strings(item, output)


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("records", payload.get("evidence_records"))
        else:
            records = None
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Expected a JSON or JSONL record list: {path}")
    return list(records)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(serialized)
        handle.flush()
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _required_path(config: dict[str, Any], key: str, base: Path) -> Path:
    value = config.get(key)
    if not value:
        raise ValueError(f"Gold-20 config requires {key}")
    path = _resolve_path(base, value)
    if not path.exists():
        raise ValueError(f"Gold-20 input does not exist for {key}: {path}")
    return path


def _required_path_list(config: dict[str, Any], key: str, base: Path) -> list[Path]:
    values = config.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Gold-20 config requires a non-empty {key} list")
    paths = [_resolve_path(base, value) for value in values]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"Gold-20 input does not exist for {key}: {missing[0]}")
    return paths


def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _strict_object_list(
    value: Any,
    gate: _Gate,
    code: str,
    entry_id: str = "",
) -> list[dict[str, Any]]:
    records = _object_list(value)
    gate.require(
        isinstance(value, list) and len(records) == len(value),
        code,
        entry_id,
    )
    return records


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_repository(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if "/" not in normalized and "__" in normalized:
        normalized = normalized.replace("__", "/", 1)
    return normalized


def _normalize_source_name(value: Any) -> str:
    return _normalize_label(value).replace("__", "/")


def _source_query_text(record: dict[str, Any]) -> str:
    for key in ("problem_statement", "issue_text", "query", "prompt"):
        if key in record and record[key] is not None:
            return str(record[key]).strip()
    title = str(record.get("title") or "").strip()
    body = str(record.get("body") or "").strip()
    return f"{title}\n\n{body}".strip()


def _normalize_public_source_type(value: Any) -> str:
    normalized = _normalize_label(value)
    if normalized in {"pr", "pull_request", "public_pr"}:
        return "public_pr"
    if normalized in {"issue", "public_issue"}:
        return "public_issue"
    return ""


def _public_source_identity(value: str) -> tuple[str, str, str]:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return ("", "", "")
    path_parts = [part for part in parsed.path.split("/") if part]
    valid = (
        parsed.scheme == "https"
        and parsed.hostname == "github.com"
        and port is None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and len(path_parts) == 4
        and path_parts[2] in {"issues", "pull"}
        and path_parts[3].isdigit()
    )
    if not valid:
        return ("", "", "")
    source_type = "public_pr" if path_parts[2] == "pull" else "public_issue"
    return (_normalize_repository("/".join(path_parts[:2])), source_type, path_parts[3])


def _expected_public_source_id(identity: tuple[str, str, str]) -> str:
    repository, source_type, number = identity
    if not repository or not source_type or not number:
        return ""
    short_type = "pr" if source_type == "public_pr" else "issue"
    return f"{repository.replace('/', '__')}-{short_type}-{number}"


def _public_source_repository(value: str) -> str:
    return _public_source_identity(value)[0]


def _public_clone_repository(value: Any) -> str:
    try:
        parsed = urlsplit(str(value or ""))
        port = parsed.port
    except ValueError:
        return ""
    path_parts = [part for part in parsed.path.split("/") if part]
    valid = (
        parsed.scheme == "https"
        and parsed.hostname == "github.com"
        and port is None
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and len(path_parts) == 2
        and path_parts[1].endswith(".git")
    )
    if not valid:
        return ""
    return _normalize_repository(f"{path_parts[0]}/{path_parts[1][:-4]}")


def _runtime_producer_component_sha256s() -> dict[str, str]:
    package_root = Path(__file__).parent
    return {
        name: _file_sha256(package_root / name)
        for name in ("evaluation.py", "registry.py", "scenarios.py")
    }


def _hidden_test_result_sha256(
    exit_code: int,
    infrastructure_failure: bool,
) -> str:
    return _stable_json_sha256(
        {
            "exit_code": exit_code,
            "passed": exit_code == 0 and not infrastructure_failure,
            "infrastructure_failure": infrastructure_failure,
        }
    )


def _is_sha256(value: Any) -> bool:
    return bool(_SHA256_PATTERN.fullmatch(str(value or "").lower()))


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
