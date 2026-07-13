from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from easy_agentic_data.batch import enqueue_human_review
from easy_agentic_data.registry import ScenarioRegistry, materialize_environment_source
from easy_agentic_data.registry_sources import (
    DEFAULT_TRAIN_LICENSE_ALLOWLIST,
    PUBLIC_CI_FORMATS,
    PUBLIC_ISSUE_PR_FORMATS,
    RegistryImportSummary,
    import_public_ci_records,
    import_public_issue_pr_records,
    import_swe_style_records,
    load_source_records,
)
from easy_agentic_data.repository_allowlist import (
    AllowlistFilterSummary,
    audit_repository_allowlist,
    filter_records_by_allowlist,
    load_repository_allowlist,
)
from easy_agentic_data.repository_synthetic import (
    DEFAULT_SYNTHETIC_TRAIN_LICENSE_ALLOWLIST,
    RepositorySyntheticSummary,
    generate_repository_synthetic_scenarios,
    load_repository_synthesis_specs,
)
from easy_agentic_data.scenario_decontamination import (
    audit_scenario_decontamination,
    scenarios_from_registry,
)
from easy_agentic_data.scenarios import Scenario
from easy_agentic_data.seed_library import (
    DEFAULT_BENCHMARK_SOURCE_ALIASES,
    TASK_FAMILY_VERIFIER_TEMPLATES,
    SeedLibraryPolicy,
    audit_seed_library,
)
from easy_agentic_data.seed_review import build_seed_review_queue
from easy_agentic_data.seeds import QuerySeed

SEED_CORPUS_SCHEMA_VERSION = "easy_agentic_data.seed_corpus.v1"
REGISTRY_IMPORT_REHEARSAL_SCHEMA_VERSION = "easy_agentic_data.registry_import_rehearsal.v1"
SEED_BACKFILL_PLAN_SCHEMA_VERSION = "easy_agentic_data.seed_backfill_plan.v1"
SEED_SELECTION_PLAN_SCHEMA_VERSION = "easy_agentic_data.seed_selection_plan.v1"
SEED_REMEDIATION_PLAN_SCHEMA_VERSION = "easy_agentic_data.seed_remediation_plan.v1"
HIDDEN_TEST_PATCH_CURATION_PLAN_SCHEMA_VERSION = (
    "easy_agentic_data.hidden_test_patch_curation_plan.v1"
)
HIDDEN_TEST_PATCH_CURATION_RECORD_TEMPLATE_SCHEMA_VERSION = (
    "easy_agentic_data.hidden_test_patch_curation_record_template.v1"
)
HIDDEN_TEST_PATCH_CURATION_APPLY_SCHEMA_VERSION = (
    "easy_agentic_data.hidden_test_patch_curation_apply.v1"
)
HIDDEN_COMMAND_CURATION_PLAN_SCHEMA_VERSION = (
    "easy_agentic_data.hidden_command_curation_plan.v1"
)
HIDDEN_COMMAND_CURATION_RECORD_TEMPLATE_SCHEMA_VERSION = (
    "easy_agentic_data.hidden_command_curation_record_template.v1"
)
HIDDEN_COMMAND_CURATION_APPLY_SCHEMA_VERSION = (
    "easy_agentic_data.hidden_command_curation_apply.v1"
)
SOURCE_WORKSPACE_MATERIALIZATION_PLAN_SCHEMA_VERSION = (
    "easy_agentic_data.source_workspace_materialization_plan.v1"
)
SOURCE_WORKSPACE_MATERIALIZATION_RUN_SCHEMA_VERSION = (
    "easy_agentic_data.source_workspace_materialization_run.v1"
)
SYNTHETIC_BACKFILL_SPEC_SCHEMA_VERSION = (
    "easy_agentic_data.synthetic_backfill_spec_plan.v1"
)
SEED_CANDIDATE_ASSEMBLY_SCHEMA_VERSION = "easy_agentic_data.seed_candidate_assembly.v1"
SYNTHETIC_EVIDENCE_BACKFILL_PLAN_SCHEMA_VERSION = (
    "easy_agentic_data.synthetic_evidence_backfill_plan.v1"
)
SYNTHETIC_EVIDENCE_APPLY_SCHEMA_VERSION = (
    "easy_agentic_data.synthetic_evidence_apply.v1"
)
SYNTHETIC_EVIDENCE_SHARD_SCHEDULE_SCHEMA_VERSION = (
    "easy_agentic_data.synthetic_evidence_shard_schedule.v1"
)
SYNTHETIC_EVIDENCE_SHARD_STATUS_SCHEMA_VERSION = (
    "easy_agentic_data.synthetic_evidence_shard_status.v1"
)
SYNTHETIC_EVIDENCE_RECORD_TEMPLATE_SCHEMA_VERSION = (
    "easy_agentic_data.synthetic_evidence_record_template.v1"
)
SYNTHETIC_READY_SPEC_COMBINE_SCHEMA_VERSION = (
    "easy_agentic_data.synthetic_ready_spec_combine.v1"
)


def build_seed_corpus(
    config_path: str | Path,
    *,
    manifest_output: str | Path | None = None,
    overwrite_outputs: bool = False,
) -> dict[str, Any]:
    """Build train and holdout registries from a seed-corpus config and freeze a manifest."""

    config_file = Path(config_path)
    config = _read_json(config_file)
    config_dir = config_file.parent
    overwrite = overwrite_outputs or bool(config.get("overwrite_outputs", False))
    overwrite_registries = bool(config.get("overwrite_registries", False))
    train_root = _required_path(config_dir, config, "train_registry_root")
    holdout_root = _optional_path(config_dir, config.get("holdout_registry_root"))

    _prepare_registry_root(train_root, overwrite=overwrite_registries)
    if holdout_root is not None:
        _prepare_registry_root(holdout_root, overwrite=overwrite_registries)
    allowlist_path = _optional_path(config_dir, config.get("repository_allowlist"))
    allowlist_records = load_repository_allowlist(allowlist_path) if allowlist_path else []
    allowlist_audit = (
        audit_repository_allowlist(
            allowlist_records,
            license_allowlist=sorted(
                set(DEFAULT_TRAIN_LICENSE_ALLOWLIST)
                | set(_string_list(config.get("allow_train_licenses")))
            ),
            benchmark_repositories=_string_list(config.get("benchmark_repositories")),
        )
        if allowlist_path is not None
        else None
    )
    train_registry = ScenarioRegistry(train_root)
    train_registry.initialize()
    holdout_registry = ScenarioRegistry(holdout_root) if holdout_root is not None else None
    if holdout_registry is not None:
        holdout_registry.initialize()

    import_summaries = []
    allowlist_filters: list[AllowlistFilterSummary] = []
    import_summaries.extend(
        _import_record_sources(
            train_registry,
            _dict_list(config.get("public_issue_sources")),
            config_dir=config_dir,
            default_format="public_issue_pr",
            default_split="train",
            default_train_eligible=None,
            allowlist_records=allowlist_records,
            allowlist_filters=allowlist_filters,
        )
    )
    import_summaries.extend(
        _import_record_sources(
            train_registry,
            _dict_list(config.get("public_ci_sources")),
            config_dir=config_dir,
            default_format="public_ci",
            default_split="train",
            default_train_eligible=None,
            allowlist_records=allowlist_records,
            allowlist_filters=allowlist_filters,
        )
    )
    import_summaries.extend(
        _import_record_sources(
            train_registry,
            _dict_list(config.get("swe_style_sources")),
            config_dir=config_dir,
            default_format="auto",
            default_split="train",
            default_train_eligible=None,
            allowlist_records=allowlist_records,
            allowlist_filters=allowlist_filters,
        )
    )
    synthetic_summaries = _generate_synthetic_sources(
        train_registry,
        _dict_list(config.get("repository_synthetic_sources")),
        config_dir=config_dir,
        default_split="train",
        default_train_eligible=None,
        allowlist_records=allowlist_records,
        allowlist_filters=allowlist_filters,
    )
    holdout_summaries: list[dict[str, Any]] = []
    if holdout_registry is not None:
        holdout_summaries.extend(
            summary.to_dict()
            for summary in _import_record_sources(
                holdout_registry,
                _dict_list(config.get("holdout_sources")),
                config_dir=config_dir,
                default_format="auto",
                default_split="evaluation",
                default_train_eligible=False,
            )
        )

    benchmark_sources = sorted(
        set(DEFAULT_BENCHMARK_SOURCE_ALIASES)
        | set(_string_list(config.get("benchmark_sources")))
    )
    train_validation = train_registry.validate()
    holdout_validation = holdout_registry.validate() if holdout_registry is not None else None
    train_seeds = train_registry.list_seeds()
    holdout_seeds = list(train_seeds)
    if holdout_registry is not None:
        holdout_seeds.extend(holdout_registry.list_seeds())
    seed_policy = _seed_policy(config.get("seed_policy", {}))
    seed_audit = audit_seed_library(
        train_seeds,
        benchmark_sources=benchmark_sources,
        policy=seed_policy,
        holdout_seeds=holdout_seeds,
    )
    seed_audit_payload = seed_audit.to_dict()
    seed_audit_payload["train_verifier_type_counts"] = _train_verifier_type_counts(train_seeds)
    train_scenarios = scenarios_from_registry(train_registry)
    holdout_scenarios = list(train_scenarios)
    if holdout_registry is not None:
        holdout_scenarios.extend(scenarios_from_registry(holdout_registry))
    scenario_audit = audit_scenario_decontamination(
        train_scenarios,
        benchmark_sources=benchmark_sources,
        holdout_scenarios=holdout_scenarios,
    )
    quarantine = _quarantine_summary(
        import_summaries,
        synthetic_summaries,
        holdout_summaries,
        allowlist_filters,
    )
    coverage_budget = _coverage_budget_report(
        seed_audit_payload,
        config.get("coverage_budgets", {}),
        quarantine_count=int(quarantine["records"]),
    )
    review_config = _dict(config.get("review", {}))
    review_queue = build_seed_review_queue(
        train_scenarios,
        sample_per_stratum=_int(review_config.get("sample_per_stratum"), default=1),
        max_records=_optional_int(review_config.get("max_records")),
    )
    review_output = _optional_path(config_dir, review_config.get("output"))
    if review_output is None:
        review_output = _optional_path(config_dir, config.get("review_queue_output"))
    if review_output is not None:
        if overwrite:
            review_output.unlink(missing_ok=True)
        for record in review_queue.records:
            enqueue_human_review(review_output, record)

    seed_audit_output = _optional_path(config_dir, config.get("seed_audit_output"))
    scenario_audit_output = _optional_path(config_dir, config.get("scenario_audit_output"))
    _write_optional_json(seed_audit_output, seed_audit_payload)
    _write_optional_json(scenario_audit_output, scenario_audit.to_dict())

    scale_decision = _dict(config.get("scale_decision", {}))
    validation = {
        "allowlist_valid": allowlist_audit.valid if allowlist_audit is not None else True,
        "registry_valid": train_validation.valid
        and (holdout_validation.valid if holdout_validation is not None else True),
        "seed_audit_valid": seed_audit.valid,
        "scenario_audit_valid": scenario_audit.valid,
        "coverage_budget_valid": coverage_budget["valid"],
        "review_queue_valid": review_queue.selected > 0
        or not bool(review_config.get("required", True)),
    }
    valid = all(validation.values())
    manifest_path = (
        Path(manifest_output)
        if manifest_output is not None
        else _optional_path(config_dir, config.get("manifest_output"))
    )
    if manifest_path is None:
        manifest_path = train_root.parent / "seed-corpus-manifest.json"
    manifest = {
        "schema_version": SEED_CORPUS_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": str(config_file),
            "sha256": _file_sha256(config_file),
        },
        "train_registry_root": str(train_root),
        "holdout_registry_root": str(holdout_root) if holdout_root is not None else "",
        "benchmark_sources": benchmark_sources,
        "repository_allowlist": {
            "path": str(allowlist_path) if allowlist_path else "",
            "sha256": _file_sha256(allowlist_path) if allowlist_path else "",
            "audit": allowlist_audit.to_dict() if allowlist_audit is not None else None,
            "filters": [summary.to_dict() for summary in allowlist_filters],
        },
        "source_snapshots": _source_snapshots(config, config_dir),
        "imports": [summary.to_dict() for summary in import_summaries],
        "synthetic_generation": [summary.to_dict() for summary in synthetic_summaries],
        "holdout_imports": holdout_summaries,
        "quarantine": quarantine,
        "registry_validation": {
            "train": _registry_validation_payload(train_validation),
            "holdout": _registry_validation_payload(holdout_validation)
            if holdout_validation is not None
            else None,
        },
        "seed_policy": asdict(seed_policy),
        "coverage_budget": coverage_budget,
        "seed_audit": seed_audit_payload,
        "seed_audit_output": str(seed_audit_output) if seed_audit_output else "",
        "scenario_audit": scenario_audit.to_dict(),
        "scenario_audit_output": str(scenario_audit_output) if scenario_audit_output else "",
        "review_queue": {
            "selected": review_queue.selected,
            "total_scenarios": review_queue.total_scenarios,
            "stratum_counts": review_queue.stratum_counts,
            "output": str(review_output) if review_output else "",
        },
        "validation": validation,
        "valid": valid,
        "scale_decision": {
            "approved": bool(scale_decision.get("approved", False)),
            "reason": str(scale_decision.get("reason", "")),
        },
        "approved_for_scale": valid and bool(scale_decision.get("approved", False)),
    }
    manifest["manifest_output"] = str(manifest_path)
    _write_json(manifest_path, manifest)
    return manifest


def rehearse_registry_import(
    *,
    registry_root: str | Path,
    source_path: str | Path,
    source_format: str = "public_issue_pr",
    source_name: str = "",
    allowlist_path: str | Path | None = None,
    split: str = "train",
    license_name: str = "",
    permitted_use: str = "research",
    test_command_template: str = "",
    task_family: str = "",
    source_method: str = "",
    train_eligible: bool | None = None,
    contamination_tags: Iterable[str] = (),
    coverage_tags: Iterable[str] = (),
    allow_train_licenses: Iterable[str] = (),
    limit: int | None = None,
    strict: bool = False,
    overwrite_registry: bool = False,
    min_imported: int = 1,
    max_quarantined: int = 0,
    seed_policy: SeedLibraryPolicy | None = None,
    benchmark_sources: Iterable[str] = DEFAULT_BENCHMARK_SOURCE_ALIASES,
    materialize_sample_count: int = 0,
    materialize_root: str | Path | None = None,
    run_hidden_commands: bool = False,
    hidden_test_patch_sample_count: int = 0,
    hidden_test_patch_root: str | Path | None = None,
    hidden_test_patch_expected_outcome: str = "fail",
) -> dict[str, Any]:
    """Import source records into a temporary registry and run pre-materialization gates."""

    root = Path(registry_root)
    source = Path(source_path)
    _prepare_registry_root(root, overwrite=overwrite_registry)
    registry = ScenarioRegistry(root)
    registry.initialize()
    records = load_source_records(source)
    normalized_format = source_format.replace("-", "_")
    allowlist_filter = None
    if allowlist_path is not None:
        records, allowlist_filter = filter_records_by_allowlist(
            records,
            load_repository_allowlist(allowlist_path),
            source_name=source_name or normalized_format,
        )
    if normalized_format in PUBLIC_ISSUE_PR_FORMATS:
        import_summary = import_public_issue_pr_records(
            registry,
            records,
            source_format=normalized_format,
            source_name=source_name,
            split=split,
            license_name=license_name,
            permitted_use=permitted_use,
            limit=limit,
            test_command_template=test_command_template,
            task_family=task_family,
            source_method=source_method,
            train_eligible=train_eligible,
            contamination_tags=contamination_tags,
            coverage_tags=coverage_tags,
            train_license_allowlist=sorted(
                set(DEFAULT_TRAIN_LICENSE_ALLOWLIST) | set(allow_train_licenses)
            ),
            strict=strict,
        )
    elif normalized_format in PUBLIC_CI_FORMATS:
        import_summary = import_public_ci_records(
            registry,
            records,
            source_format=normalized_format,
            source_name=source_name,
            split=split,
            license_name=license_name,
            permitted_use=permitted_use,
            limit=limit,
            task_family=task_family,
            source_method=source_method,
            train_eligible=train_eligible,
            contamination_tags=contamination_tags,
            coverage_tags=coverage_tags,
            train_license_allowlist=sorted(
                set(DEFAULT_TRAIN_LICENSE_ALLOWLIST) | set(allow_train_licenses)
            ),
            strict=strict,
        )
    else:
        import_summary = import_swe_style_records(
            registry,
            records,
            source_format=normalized_format,
            source_name=source_name,
            split=split,
            license_name=license_name,
            permitted_use=permitted_use,
            limit=limit,
            test_command_template=test_command_template,
            task_family=task_family,
            source_method=source_method,
            train_eligible=train_eligible,
            contamination_tags=contamination_tags,
            coverage_tags=coverage_tags,
            strict=strict,
        )

    registry_validation = registry.validate()
    seeds = registry.list_seeds()
    policy = seed_policy or SeedLibraryPolicy()
    seed_audit = audit_seed_library(
        seeds,
        benchmark_sources=benchmark_sources,
        policy=policy,
        holdout_seeds=list(seeds),
    )
    seed_audit_payload = seed_audit.to_dict()
    seed_audit_payload["train_verifier_type_counts"] = _train_verifier_type_counts(seeds)
    materialization = _rehearse_materialization(
        registry,
        import_summary.scenario_ids,
        sample_count=materialize_sample_count,
        materialize_root=materialize_root or (root / "materialization-rehearsal"),
        run_hidden_commands=run_hidden_commands,
    )
    hidden_test_patch_rehearsal = _rehearse_hidden_test_patches(
        registry,
        import_summary.scenario_ids,
        sample_count=hidden_test_patch_sample_count,
        materialize_root=hidden_test_patch_root or (root / "hidden-test-patch-rehearsal"),
        expected_outcome=hidden_test_patch_expected_outcome,
    )
    quarantined = import_summary.skipped
    if allowlist_filter is not None:
        quarantined += allowlist_filter.quarantined
    gate_issues = _import_rehearsal_gate_issues(
        imported=import_summary.imported,
        min_imported=min_imported,
        quarantined=quarantined,
        max_quarantined=max_quarantined,
    )
    validation = {
        "imported_minimum_valid": import_summary.imported >= max(0, min_imported),
        "quarantine_budget_valid": quarantined <= max(0, max_quarantined),
        "registry_valid": registry_validation.valid,
        "seed_audit_valid": seed_audit.valid,
        "materialization_valid": materialization["valid"],
        "hidden_test_patch_rehearsal_valid": hidden_test_patch_rehearsal["valid"],
    }
    valid = all(validation.values())
    return {
        "schema_version": REGISTRY_IMPORT_REHEARSAL_SCHEMA_VERSION,
        "registry_root": str(root),
        "source": {
            "path": str(source),
            "sha256": _file_sha256(source),
            "format": normalized_format,
            "source_name": source_name or normalized_format,
        },
        "allowlist_filter": allowlist_filter.to_dict() if allowlist_filter is not None else None,
        "import": import_summary.to_dict(),
        "quarantine": {
            "records": quarantined,
            "issues": list(import_summary.issues)
            + (allowlist_filter.issues if allowlist_filter is not None else []),
        },
        "registry_validation": _registry_validation_payload(registry_validation),
        "seed_policy": asdict(policy),
        "seed_audit": seed_audit_payload,
        "materialization": materialization,
        "hidden_test_patch_rehearsal": hidden_test_patch_rehearsal,
        "gate_issues": gate_issues,
        "validation": validation,
        "valid": valid,
    }


def build_seed_backfill_plan(
    seed_audit: dict[str, Any],
    policy_config: dict[str, Any],
) -> dict[str, Any]:
    """Convert seed-audit coverage failures into deterministic backfill actions."""

    audit = _dict(seed_audit)
    config = _dict(policy_config)
    policy = _seed_policy(config.get("seed_policy", config))
    coverage_budgets = _dict(config.get("coverage_budgets"))
    train_eligible = _int(audit.get("train_eligible"), default=0)
    target_train_eligible = max(
        _int(config.get("target_train_eligible"), default=0),
        policy.min_train_eligible,
    )
    family_counts = _normalized_int_counts(
        audit.get("train_task_family_counts") or audit.get("task_family_counts")
    )
    verifier_counts = _normalized_int_counts(
        audit.get("train_verifier_type_counts") or audit.get("verifier_type_counts")
    )
    source_method_counts = _normalized_int_counts(
        audit.get("train_source_method_counts") or audit.get("source_method_counts")
    )
    language_counts = _normalized_int_counts(
        audit.get("train_language_counts") or audit.get("language_counts")
    )
    repository_counts = _normalized_int_counts(audit.get("train_repository_counts"))

    task_family_gaps = _count_gaps(
        family_counts,
        minimum_counts=_int_dict(coverage_budgets.get("min_task_family_counts")),
        required_present=policy.required_task_families,
        dimension="task_family",
    )
    verifier_type_gaps = _count_gaps(
        verifier_counts,
        minimum_counts=_int_dict(coverage_budgets.get("min_verifier_type_counts")),
        required_present=policy.required_verifier_types,
        dimension="verifier_type",
    )
    source_method_gaps = _count_gaps(
        source_method_counts,
        minimum_counts=_int_dict(coverage_budgets.get("min_source_method_counts")),
        required_present=(),
        dimension="source_method",
    )
    language_count_gaps = _count_gaps(
        language_counts,
        minimum_counts=_int_dict(coverage_budgets.get("min_language_counts")),
        required_present=(),
        dimension="language",
    )
    dominance = {
        "task_family": _dominance_gaps(
            family_counts,
            total=train_eligible,
            max_share=policy.max_task_family_share,
            dimension="task_family",
        ),
        "source_method": _dominance_gaps(
            source_method_counts,
            total=train_eligible,
            max_share=policy.max_source_method_share,
            dimension="source_method",
        ),
        "repository": _dominance_gaps(
            repository_counts,
            total=train_eligible,
            max_share=policy.max_repository_share,
            dimension="repository",
        ),
        "language": _dominance_gaps(
            language_counts,
            total=train_eligible,
            max_share=policy.max_language_share,
            dimension="language",
        ),
    }
    train_eligible_gap = max(0, target_train_eligible - train_eligible)
    actions = _backfill_actions(
        train_eligible_gap=train_eligible_gap,
        task_family_gaps=task_family_gaps,
        verifier_type_gaps=verifier_type_gaps,
        source_method_gaps=source_method_gaps,
        language_count_gaps=language_count_gaps,
        dominance=dominance,
    )
    requires_backfill = bool(train_eligible_gap or actions)
    issue_count = len(audit.get("issues", [])) if isinstance(audit.get("issues"), list) else 0
    return {
        "schema_version": SEED_BACKFILL_PLAN_SCHEMA_VERSION,
        "audit": {
            "valid": bool(audit.get("valid", False)),
            "total": _int(audit.get("total"), default=train_eligible),
            "train_eligible": train_eligible,
            "issue_count": issue_count,
        },
        "policy": {
            "target_train_eligible": target_train_eligible,
            "seed_policy": asdict(policy),
            "coverage_budgets": coverage_budgets,
        },
        "counts": {
            "task_family": family_counts,
            "verifier_type": verifier_counts,
            "source_method": source_method_counts,
            "language": language_counts,
            "repository": repository_counts,
        },
        "gaps": {
            "train_eligible": {
                "current": train_eligible,
                "minimum": target_train_eligible,
                "shortfall": train_eligible_gap,
            },
            "task_family": task_family_gaps,
            "verifier_type": verifier_type_gaps,
            "source_method": source_method_gaps,
            "language_count": language_count_gaps,
            "dominance": dominance,
        },
        "recommended_actions": actions,
        "requires_backfill": requires_backfill,
        "valid": True,
    }


def build_seed_selection_plan(
    seeds: Iterable[QuerySeed],
    policy_config: dict[str, Any],
    *,
    target_train_eligible: int | None = None,
) -> dict[str, Any]:
    """Plan a balanced train slice while preserving room for required backfill."""

    seed_list = [seed for seed in seeds if seed.train_eligible]
    config = _dict(policy_config)
    policy = _seed_policy(config.get("seed_policy", config))
    target = _selection_target(
        config,
        policy,
        explicit_target=target_train_eligible,
        candidate_count=len(seed_list),
    )
    seed_audit = audit_seed_library(seed_list, policy=policy, holdout_seeds=list(seed_list))
    seed_audit_payload = seed_audit.to_dict()
    seed_audit_payload["train_verifier_type_counts"] = _train_verifier_type_counts(seed_list)
    backfill_plan = build_seed_backfill_plan(seed_audit_payload, config)
    reserved = _reserved_backfill_slots(
        backfill_plan["gaps"],
        target=target,
        seeds=seed_list,
        policy=policy,
    )
    existing_target = max(0, min(len(seed_list), target - reserved["minimum_reserved_slots"]))
    selected = _select_existing_seed_slice(
        seed_list,
        target=existing_target,
        final_target=target,
        policy=policy,
    )
    selected_audit = audit_seed_library(
        selected,
        policy=policy,
        holdout_seeds=list(selected),
    )
    selected_audit_payload = selected_audit.to_dict()
    selected_audit_payload["train_verifier_type_counts"] = _train_verifier_type_counts(selected)
    issues = _selection_plan_issues(
        target=target,
        existing_target=existing_target,
        selected_count=len(selected),
        selected_audit=selected_audit_payload,
        reserved=reserved,
    )
    selected_ids = [seed.seed_id for seed in selected]
    ready_for_rollout = (
        len(selected) == target
        and not reserved["slots"]
        and bool(selected_audit_payload["valid"])
    )
    return {
        "schema_version": SEED_SELECTION_PLAN_SCHEMA_VERSION,
        "target_train_eligible": target,
        "candidate_train_eligible": len(seed_list),
        "existing_selection_target": existing_target,
        "selected_existing_count": len(selected),
        "selected_seed_ids": selected_ids,
        "selected_seed_ids_sha256": hashlib.sha256(
            json.dumps(selected_ids, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "reserved_backfill": reserved,
        "selected_counts": _selection_counts(selected),
        "selected_shares_against_target": _selection_shares_against_target(
            selected,
            target=target,
        ),
        "candidate_seed_audit": seed_audit_payload,
        "selected_seed_audit": selected_audit_payload,
        "backfill_plan_summary": {
            "requires_backfill": bool(backfill_plan["requires_backfill"]),
            "recommended_action_count": len(backfill_plan["recommended_actions"]),
        },
        "issues": issues,
        "requires_backfill": bool(reserved["slots"]),
        "ready_for_rollout": ready_for_rollout,
        "valid": True,
    }


def build_seed_remediation_plan(
    selection_plan: dict[str, Any],
    policy_config: dict[str, Any],
    *,
    allowlist_records: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Convert reserved backfill slots into next-collection remediation requirements."""

    selection = _dict(selection_plan)
    config = _dict(policy_config)
    policy = _seed_policy(config.get("seed_policy", config))
    allowlist = list(allowlist_records)
    allowlist_summary = _remediation_allowlist_summary(allowlist)
    slots = _dict_list(_dict(selection.get("reserved_backfill")).get("slots"))
    requirements: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for slot in slots:
        requirement = _remediation_requirement_for_slot(
            slot,
            allowlist_summary=allowlist_summary,
        )
        if requirement:
            requirements.append(requirement)
    if slots and allowlist and not allowlist_summary["non_python_repositories"]:
        issues.append(
            {
                "code": "missing_cross_language_allowlist_candidate",
                "message": (
                    "Reserved non-Python slots cannot be collected from the current "
                    "repository allowlist."
                ),
                "severity": "error",
            }
        )
    missing_repository_candidates = any(
        requirement.get("action") == "collect_non_dominant_repository_sources"
        and not requirement.get("candidate_allowlist_repositories")
        for requirement in requirements
    )
    if slots and allowlist and missing_repository_candidates:
        issues.append(
            {
                "code": "missing_non_dominant_repository_candidate",
                "message": (
                    "Reserved repository-diversity slots have no non-dominant "
                    "allowlist candidates."
                ),
                "severity": "error",
            }
        )
    if slots and not allowlist:
        issues.append(
            {
                "code": "allowlist_not_provided",
                "message": (
                    "No repository allowlist was provided, so source expansion "
                    "readiness could not be checked."
                ),
                "severity": "warning",
            }
        )
    if not slots:
        issues.append(
            {
                "code": "no_reserved_backfill_slots",
                "message": "The selection plan does not require future backfill slots.",
                "severity": "warning",
            }
        )
    future_slots = _int(
        _dict(selection.get("reserved_backfill")).get("minimum_reserved_slots"),
        default=0,
    )
    return {
        "schema_version": SEED_REMEDIATION_PLAN_SCHEMA_VERSION,
        "inputs": {
            "selection_plan_hash": _stable_json_sha256(selection),
            "policy_hash": _stable_json_sha256(config),
            "allowlist_repositories": len(allowlist),
        },
        "target": {
            "target_train_eligible": _int(
                selection.get("target_train_eligible"),
                default=max(policy.min_train_eligible, 0),
            ),
            "selected_existing_count": _int(
                selection.get("selected_existing_count"),
                default=0,
            ),
            "minimum_future_slots": future_slots,
        },
        "overlap_policy": (
            "A future seed may satisfy multiple requirements only when its provenance, "
            "task family, language, repository, and verifier evidence independently "
            "meet each listed constraint."
        ),
        "requirements": requirements,
        "allowlist": allowlist_summary,
        "issues": issues,
        "ready_for_collection": bool(requirements)
        and not any(issue["severity"] == "error" for issue in issues),
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def build_hidden_test_patch_curation_plan(
    source_records: Iterable[dict[str, Any]],
    remediation_plan: dict[str, Any],
    *,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Create leakage-safe curation tasks for withheld hidden test patches."""

    records = list(source_records)
    remediation = _dict(remediation_plan)
    requirement = _hidden_test_patch_requirement(remediation)
    has_requirement = bool(requirement)
    required_count = _int(requirement.get("minimum_count"), default=0)
    accepted_source_types = set(
        _string_list(requirement.get("accepted_source_types"))
        or ["public_issue", "public_pr"]
    )
    limit = (
        max(0, max_records)
        if max_records is not None and has_requirement
        else required_count
    )
    if not has_requirement:
        limit = 0
    elif limit <= 0:
        limit = len(records)
    eligible_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        normalized_record = _dict(record)
        rejection = _hidden_test_patch_record_rejection(
            normalized_record,
            accepted_source_types=accepted_source_types,
        )
        if rejection:
            rejected.append(_curation_rejection(index, normalized_record, rejection))
            continue
        eligible_records.append(normalized_record)
    selected_records = _select_hidden_test_patch_curation_records(
        eligible_records,
        limit=limit,
    )
    selected = [
        _hidden_test_patch_curation_task(
            record,
            task_index=task_index,
            leakage_constraints=_string_list(requirement.get("leakage_constraints")),
        )
        for task_index, record in enumerate(selected_records)
    ]

    shortfall = max(0, required_count - len(selected))
    issues: list[dict[str, Any]] = []
    if not has_requirement:
        issues.append(
            {
                "code": "hidden_test_patch_requirement_missing",
                "message": (
                    "The remediation plan does not request hidden test patch "
                    "evidence."
                ),
                "severity": "warning",
            }
        )
    if required_count and shortfall:
        issues.append(
            {
                "code": "hidden_test_patch_curation_shortfall",
                "message": (
                    f"Selected {len(selected)} curation tasks for "
                    f"{required_count} required hidden test patch records."
                ),
                "severity": "warning",
            }
        )
    if required_count and not selected:
        issues.append(
            {
                "code": "no_hidden_test_patch_curation_candidates",
                "message": (
                    "No public issue or PR records were eligible for hidden test "
                    "patch curation."
                ),
                "severity": "error",
            }
        )

    return {
        "schema_version": HIDDEN_TEST_PATCH_CURATION_PLAN_SCHEMA_VERSION,
        "inputs": {
            "source_records": len(records),
            "source_records_hash": _stable_json_sha256({"records": records}),
            "remediation_plan_hash": _stable_json_sha256(remediation),
            "max_records": max_records,
        },
        "target": {
            "required_hidden_test_patch_records": required_count,
            "accepted_source_types": sorted(accepted_source_types),
        },
        "counts": {
            "source_records": len(records),
            "eligible_records": len(eligible_records),
            "selected_curation_tasks": len(selected),
            "rejected_records": len(rejected),
            "shortfall": shortfall,
        },
        "curation_tasks": selected,
        "rejected_records": rejected,
        "issues": issues,
        "ready_for_curation": has_requirement
        and bool(selected)
        and not any(issue["severity"] == "error" for issue in issues),
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def build_hidden_test_patch_curation_record_template(
    curation_plan: dict[str, Any],
) -> dict[str, Any]:
    """Create fillable hidden-test-patch curation records from a plan."""

    tasks = _dict_list(curation_plan.get("curation_tasks"))
    records = [
        {
            "curation_task_id": str(task.get("curation_task_id", "")),
            "source_instance_id": str(task.get("source_instance_id", "")),
            "repository": str(task.get("repository", "")),
            "source_type": str(task.get("source_type", "")),
            "source_revision": str(task.get("source_revision", "")),
            "title": str(task.get("title", "")),
            "candidate_verifier_commands": _optional_string_list(
                task.get("candidate_verifier_commands")
            ),
            "public_behavior_summary": "",
            "hidden_test_patch": "",
            "hidden_test_commands": [],
            "withheld_evaluator_notes": "",
        }
        for task in tasks
    ]
    issues = []
    if not curation_plan.get("ready_for_curation", False):
        issues.append(
            {
                "code": "hidden_test_patch_curation_plan_not_ready",
                "message": "Curation plan is not ready for hidden-test-patch curation.",
                "severity": "warning",
            }
        )
    if not tasks:
        issues.append(
            {
                "code": "hidden_test_patch_curation_tasks_missing",
                "message": "Curation plan does not contain curation tasks.",
                "severity": "error",
            }
        )
    return {
        "schema_version": HIDDEN_TEST_PATCH_CURATION_RECORD_TEMPLATE_SCHEMA_VERSION,
        "inputs": {
            "curation_plan_hash": _stable_json_sha256(curation_plan),
            "curation_tasks": len(tasks),
        },
        "collector_instructions": [
            "Fill hidden_test_patch only with a unified diff that adds withheld tests.",
            "Fill hidden_test_commands only with stable commands validated on the fixed workspace.",
            "Keep curation_task_id and source_instance_id unchanged for the apply gate.",
            "Do not copy benchmark oracle patches, FAIL_TO_PASS, PASS_TO_PASS, or answers.",
            "Write completed records to a separate records file, not back into this template.",
        ],
        "counts": {
            "template_records": len(records),
        },
        "records": records,
        "issues": issues,
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def apply_hidden_test_patch_curation_records(
    curation_plan: dict[str, Any],
    source_records: Iterable[dict[str, Any]],
    curation_records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply completed hidden-test-patch records to a derived source-record view."""

    source_list = [_dict(record) for record in source_records]
    task_by_id = {
        str(task.get("curation_task_id", "")): task
        for task in _dict_list(curation_plan.get("curation_tasks"))
        if str(task.get("curation_task_id", ""))
    }
    source_by_instance = {
        str(record.get("source_instance_id", "")): record
        for record in source_list
        if str(record.get("source_instance_id", ""))
    }
    records_by_task_id: dict[str, dict[str, Any]] = {}
    invalid_records: list[dict[str, Any]] = []
    unused_records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(curation_records):
        record = _dict(raw_record)
        task_id = str(record.get("curation_task_id", "")).strip()
        if not task_id:
            invalid_records.append(
                {
                    "record_index": index,
                    "curation_task_id": "",
                    "reason": "Curation record is missing curation_task_id.",
                }
            )
            continue
        task = task_by_id.get(task_id)
        if task is None:
            unused_records.append(
                {
                    "record_index": index,
                    "curation_task_id": task_id,
                    "reason": "Curation task ID is not present in the curation plan.",
                }
            )
            continue
        if task_id in records_by_task_id:
            invalid_records.append(
                {
                    "record_index": index,
                    "curation_task_id": task_id,
                    "reason": "Duplicate curation record for the same task.",
                }
            )
            continue
        validation_errors = _hidden_test_patch_curation_record_errors(record, task)
        if validation_errors:
            invalid_records.append(
                {
                    "record_index": index,
                    "curation_task_id": task_id,
                    "source_instance_id": str(record.get("source_instance_id", "")),
                    "reason": "Curation record is malformed or incomplete.",
                    "errors": validation_errors,
                }
            )
            continue
        records_by_task_id[task_id] = record

    task_id_by_source_instance = {
        str(task.get("source_instance_id", "")): task_id
        for task_id, task in task_by_id.items()
        if str(task.get("source_instance_id", ""))
    }
    applied_source_instance_ids: list[str] = []
    rewritten_records: list[dict[str, Any]] = []
    for source_record in source_list:
        rewritten = dict(source_record)
        source_instance_id = str(source_record.get("source_instance_id", ""))
        task_id = task_id_by_source_instance.get(source_instance_id, "")
        curation_record = records_by_task_id.get(task_id)
        if curation_record is not None:
            rewritten = _apply_hidden_test_patch_curation_to_source_record(
                rewritten,
                curation_record,
                task_id=task_id,
            )
            applied_source_instance_ids.append(source_instance_id)
        rewritten_records.append(rewritten)

    missing_source_instance_ids = sorted(
        source_instance_id
        for source_instance_id, task_id in task_id_by_source_instance.items()
        if task_id in records_by_task_id and source_instance_id not in source_by_instance
    )
    remaining_tasks = [
        {
            "curation_task_id": task_id,
            "source_instance_id": str(task.get("source_instance_id", "")),
            "repository": str(task.get("repository", "")),
            "source_type": str(task.get("source_type", "")),
        }
        for task_id, task in sorted(task_by_id.items())
        if task_id not in records_by_task_id
    ]
    issues = []
    if invalid_records:
        issues.append(
            {
                "code": "invalid_hidden_test_patch_curation_records",
                "message": "Some hidden-test-patch curation records were malformed.",
                "severity": "error",
            }
        )
    if missing_source_instance_ids:
        issues.append(
            {
                "code": "curation_source_records_missing",
                "message": "Some curated source instances are absent from source records.",
                "severity": "error",
                "source_instance_ids": missing_source_instance_ids,
            }
        )
    if unused_records:
        issues.append(
            {
                "code": "unused_hidden_test_patch_curation_records",
                "message": "Some curation records did not match the curation plan.",
                "severity": "warning",
            }
        )
    if remaining_tasks:
        issues.append(
            {
                "code": "hidden_test_patch_curation_remaining",
                "message": f"{len(remaining_tasks)} curation tasks still need test patches.",
                "severity": "warning",
            }
        )
    valid = not any(issue["severity"] == "error" for issue in issues)
    return (
        {
            "schema_version": HIDDEN_TEST_PATCH_CURATION_APPLY_SCHEMA_VERSION,
            "inputs": {
                "curation_plan_hash": _stable_json_sha256(curation_plan),
                "source_records_hash": _stable_json_sha256({"records": source_list}),
                "curation_records_hash": _stable_json_sha256(
                    {"records": list(records_by_task_id.values())}
                ),
                "curation_plan_tasks": len(task_by_id),
                "source_records": len(source_list),
            },
            "counts": {
                "curation_records": len(records_by_task_id)
                + len(invalid_records)
                + len(unused_records),
                "applied_curation_records": len(applied_source_instance_ids),
                "remaining_curation_tasks": len(remaining_tasks),
                "invalid_curation_records": len(invalid_records),
                "unused_curation_records": len(unused_records),
                "missing_source_records": len(missing_source_instance_ids),
                "rewritten_source_records": len(rewritten_records),
            },
            "applied_source_instance_ids": sorted(applied_source_instance_ids),
            "remaining_curation_tasks": remaining_tasks,
            "invalid_curation_records": invalid_records,
            "unused_curation_records": unused_records,
            "missing_source_instance_ids": missing_source_instance_ids,
            "issues": issues,
            "ready_for_import_rehearsal": bool(applied_source_instance_ids) and valid,
            "valid": valid,
        },
        rewritten_records,
    )


def build_hidden_command_curation_plan(
    source_records: Iterable[dict[str, Any]],
    *,
    rehearsal_summaries: Iterable[dict[str, Any]] = (),
    max_records: int | None = None,
) -> dict[str, Any]:
    """Create curation tasks for stable hidden commands and setup evidence."""

    records = list(source_records)
    summaries = list(rehearsal_summaries)
    failure_evidence = _hidden_command_failure_evidence_by_source_instance(summaries)
    eligible_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        normalized_record = _dict(record)
        rejection = _hidden_command_record_rejection(normalized_record)
        if rejection:
            rejected.append(_curation_rejection(index, normalized_record, rejection))
            continue
        eligible_records.append(normalized_record)
    failed_records = [
        record
        for record in eligible_records
        if str(record.get("source_instance_id", "")) in failure_evidence
    ]
    remaining_records = [
        record
        for record in eligible_records
        if str(record.get("source_instance_id", "")) not in failure_evidence
    ]
    limit = max(0, max_records) if max_records is not None else len(eligible_records)
    selected_records = _select_hidden_command_curation_records(
        failed_records,
        remaining_records,
        limit=limit,
    )
    selected = [
        _hidden_command_curation_task(
            record,
            task_index=task_index,
            failure_evidence=failure_evidence.get(
                str(record.get("source_instance_id", "")),
                {},
            ),
        )
        for task_index, record in enumerate(selected_records)
    ]
    issues: list[dict[str, Any]] = []
    if records and not selected:
        issues.append(
            {
                "code": "no_hidden_command_curation_candidates",
                "message": (
                    "No public source records had stable command evidence eligible "
                    "for hidden command curation."
                ),
                "severity": "error",
            }
        )
    if summaries and not failure_evidence:
        issues.append(
            {
                "code": "no_rehearsal_failure_mapping",
                "message": (
                    "Rehearsal summaries were provided, but no hidden-command "
                    "failures could be mapped to source instance IDs."
                ),
                "severity": "warning",
            }
        )
    return {
        "schema_version": HIDDEN_COMMAND_CURATION_PLAN_SCHEMA_VERSION,
        "inputs": {
            "source_records": len(records),
            "source_records_hash": _stable_json_sha256({"records": records}),
            "rehearsal_summary_count": len(summaries),
            "rehearsal_summaries_hash": _stable_json_sha256({"summaries": summaries}),
            "max_records": max_records,
        },
        "counts": {
            "source_records": len(records),
            "eligible_records": len(eligible_records),
            "selected_curation_tasks": len(selected),
            "records_with_observed_failures": len(failed_records),
            "rejected_records": len(rejected),
            "shortfall": max(0, len(eligible_records) - len(selected)),
        },
        "selected_source_type_counts": _source_type_counts(selected_records),
        "selected_repository_counts": _repository_counts(selected_records),
        "curation_tasks": selected,
        "rejected_records": rejected,
        "issues": issues,
        "ready_for_curation": bool(selected)
        and not any(issue["severity"] == "error" for issue in issues),
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def build_hidden_command_curation_record_template(
    curation_plan: dict[str, Any],
) -> dict[str, Any]:
    """Create fillable hidden-command curation records from a curation plan."""

    tasks = _dict_list(curation_plan.get("curation_tasks"))
    records = [
        {
            "curation_task_id": str(task.get("curation_task_id", "")),
            "source_instance_id": str(task.get("source_instance_id", "")),
            "repository": str(task.get("repository", "")),
            "source_type": str(task.get("source_type", "")),
            "source_revision": str(task.get("source_revision", "")),
            "current_setup_commands": _optional_string_list(
                task.get("current_setup_commands")
            ),
            "current_candidate_verifier_commands": _optional_string_list(
                task.get("current_candidate_verifier_commands")
            ),
            "observed_failure": _dict(task.get("observed_failure")),
            "curated_setup_commands": [],
            "curated_hidden_commands": [],
            "command_runtime": "",
            "expected_runtime_seconds": None,
            "withheld_curation_notes": "",
        }
        for task in tasks
    ]
    issues = []
    if not curation_plan.get("ready_for_curation", False):
        issues.append(
            {
                "code": "hidden_command_curation_plan_not_ready",
                "message": "Curation plan is not ready for command curation.",
                "severity": "warning",
            }
        )
    if not tasks:
        issues.append(
            {
                "code": "hidden_command_curation_tasks_missing",
                "message": "Curation plan does not contain curation tasks.",
                "severity": "error",
            }
        )
    return {
        "schema_version": HIDDEN_COMMAND_CURATION_RECORD_TEMPLATE_SCHEMA_VERSION,
        "inputs": {
            "curation_plan_hash": _stable_json_sha256(curation_plan),
            "curation_tasks": len(tasks),
        },
        "collector_instructions": [
            "Fill curated_hidden_commands only with stable commands validated on the fixed "
            "workspace revision.",
            "Fill curated_setup_commands only when setup is required before hidden commands.",
            "Keep curation_task_id and source_instance_id unchanged for the apply gate.",
            "Do not copy benchmark oracle patches, hidden test patches, or evaluator answers.",
            "Write completed records to a separate records file, not back into this template.",
        ],
        "counts": {
            "template_records": len(records),
            "records_with_observed_failures": sum(
                1 for record in records if record["observed_failure"]
            ),
        },
        "records": records,
        "issues": issues,
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def apply_hidden_command_curation_records(
    curation_plan: dict[str, Any],
    source_records: Iterable[dict[str, Any]],
    curation_records: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply completed hidden-command curation records to a derived source-record view."""

    source_list = [_dict(record) for record in source_records]
    task_by_id = {
        str(task.get("curation_task_id", "")): task
        for task in _dict_list(curation_plan.get("curation_tasks"))
        if str(task.get("curation_task_id", ""))
    }
    source_by_instance = {
        str(record.get("source_instance_id", "")): record
        for record in source_list
        if str(record.get("source_instance_id", ""))
    }
    records_by_task_id: dict[str, dict[str, Any]] = {}
    invalid_records: list[dict[str, Any]] = []
    unused_records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(curation_records):
        record = _dict(raw_record)
        task_id = str(record.get("curation_task_id", "")).strip()
        if not task_id:
            invalid_records.append(
                {
                    "record_index": index,
                    "curation_task_id": "",
                    "reason": "Curation record is missing curation_task_id.",
                }
            )
            continue
        task = task_by_id.get(task_id)
        if task is None:
            unused_records.append(
                {
                    "record_index": index,
                    "curation_task_id": task_id,
                    "reason": "Curation task ID is not present in the curation plan.",
                }
            )
            continue
        if task_id in records_by_task_id:
            invalid_records.append(
                {
                    "record_index": index,
                    "curation_task_id": task_id,
                    "reason": "Duplicate curation record for the same task.",
                }
            )
            continue
        validation_errors = _hidden_command_curation_record_errors(record, task)
        if validation_errors:
            invalid_records.append(
                {
                    "record_index": index,
                    "curation_task_id": task_id,
                    "source_instance_id": str(record.get("source_instance_id", "")),
                    "reason": "Curation record is malformed or incomplete.",
                    "errors": validation_errors,
                }
            )
            continue
        records_by_task_id[task_id] = record

    task_id_by_source_instance = {
        str(task.get("source_instance_id", "")): task_id
        for task_id, task in task_by_id.items()
        if str(task.get("source_instance_id", ""))
    }
    applied_source_instance_ids: list[str] = []
    rewritten_records: list[dict[str, Any]] = []
    for source_record in source_list:
        rewritten = dict(source_record)
        source_instance_id = str(source_record.get("source_instance_id", ""))
        task_id = task_id_by_source_instance.get(source_instance_id, "")
        curation_record = records_by_task_id.get(task_id)
        if curation_record is not None:
            rewritten = _apply_hidden_command_curation_to_source_record(
                rewritten,
                curation_record,
                task_id=task_id,
            )
            applied_source_instance_ids.append(source_instance_id)
        rewritten_records.append(rewritten)

    missing_source_instance_ids = sorted(
        source_instance_id
        for source_instance_id, task_id in task_id_by_source_instance.items()
        if task_id in records_by_task_id and source_instance_id not in source_by_instance
    )
    remaining_tasks = [
        {
            "curation_task_id": task_id,
            "source_instance_id": str(task.get("source_instance_id", "")),
            "repository": str(task.get("repository", "")),
            "source_type": str(task.get("source_type", "")),
        }
        for task_id, task in sorted(task_by_id.items())
        if task_id not in records_by_task_id
    ]
    issues = []
    if invalid_records:
        issues.append(
            {
                "code": "invalid_hidden_command_curation_records",
                "message": "Some hidden-command curation records were malformed.",
                "severity": "error",
            }
        )
    if missing_source_instance_ids:
        issues.append(
            {
                "code": "curation_source_records_missing",
                "message": "Some curated source instances are absent from source records.",
                "severity": "error",
                "source_instance_ids": missing_source_instance_ids,
            }
        )
    if unused_records:
        issues.append(
            {
                "code": "unused_hidden_command_curation_records",
                "message": "Some curation records did not match the curation plan.",
                "severity": "warning",
            }
        )
    if remaining_tasks:
        issues.append(
            {
                "code": "hidden_command_curation_remaining",
                "message": f"{len(remaining_tasks)} curation tasks still need commands.",
                "severity": "warning",
            }
        )
    valid = not any(issue["severity"] == "error" for issue in issues)
    return (
        {
            "schema_version": HIDDEN_COMMAND_CURATION_APPLY_SCHEMA_VERSION,
            "inputs": {
                "curation_plan_hash": _stable_json_sha256(curation_plan),
                "source_records_hash": _stable_json_sha256({"records": source_list}),
                "curation_records_hash": _stable_json_sha256(
                    {"records": list(records_by_task_id.values())}
                ),
                "curation_plan_tasks": len(task_by_id),
                "source_records": len(source_list),
            },
            "counts": {
                "curation_records": len(records_by_task_id)
                + len(invalid_records)
                + len(unused_records),
                "applied_curation_records": len(applied_source_instance_ids),
                "remaining_curation_tasks": len(remaining_tasks),
                "invalid_curation_records": len(invalid_records),
                "unused_curation_records": len(unused_records),
                "missing_source_records": len(missing_source_instance_ids),
                "rewritten_source_records": len(rewritten_records),
            },
            "applied_source_instance_ids": sorted(applied_source_instance_ids),
            "remaining_curation_tasks": remaining_tasks,
            "invalid_curation_records": invalid_records,
            "unused_curation_records": unused_records,
            "missing_source_instance_ids": missing_source_instance_ids,
            "issues": issues,
            "ready_for_import_rehearsal": bool(applied_source_instance_ids) and valid,
            "valid": valid,
        },
        rewritten_records,
    )


def build_source_workspace_materialization_plan(
    source_records: Iterable[dict[str, Any]],
    *,
    workspace_root: str | Path,
    max_records: int | None = None,
    shard_size: int = 20,
) -> dict[str, Any]:
    """Plan reproducible workspace caches for audited public source records."""

    records = list(source_records)
    root = Path(workspace_root)
    limit = max(0, max_records) if max_records is not None else len(records)
    eligible_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        normalized_record = _dict(record)
        rejection = _source_workspace_record_rejection(normalized_record)
        if rejection:
            rejected.append(
                _workspace_materialization_rejection(
                    index,
                    normalized_record,
                    rejection,
                )
            )
            continue
        eligible_records.append(normalized_record)
    selected_records = _select_balanced_source_records(eligible_records, limit=limit)
    tasks = _source_workspace_materialization_tasks(selected_records, root)
    normalized_shard_size = max(0, int(shard_size))
    if normalized_shard_size <= 0:
        issues.append(
            {
                "code": "invalid_workspace_materialization_shard_size",
                "message": "Workspace materialization shard size must be positive.",
                "severity": "error",
            }
        )
        normalized_shard_size = 1
    if records and not tasks:
        issues.append(
            {
                "code": "no_workspace_materialization_candidates",
                "message": "No source records were eligible for workspace materialization.",
                "severity": "error",
            }
        )
    shards = _source_workspace_materialization_shards(tasks, normalized_shard_size)
    return {
        "schema_version": SOURCE_WORKSPACE_MATERIALIZATION_PLAN_SCHEMA_VERSION,
        "inputs": {
            "source_records": len(records),
            "source_records_hash": _stable_json_sha256({"records": records}),
            "workspace_root": str(root),
            "max_records": max_records,
            "shard_size": shard_size,
        },
        "workspace_root": str(root),
        "counts": {
            "source_records": len(records),
            "eligible_records": len(eligible_records),
            "selected_records": len(selected_records),
            "rejected_records": len(rejected),
            "materialization_tasks": len(tasks),
            "shard_count": len(shards),
        },
        "selected_source_type_counts": _source_type_counts(selected_records),
        "selected_repository_counts": _repository_counts(selected_records),
        "materialization_tasks": tasks,
        "shard_size": normalized_shard_size,
        "shard_count": len(shards),
        "shards": shards,
        "rejected_records": rejected,
        "issues": issues,
        "ready_for_materialization": bool(tasks)
        and not any(issue["severity"] == "error" for issue in issues),
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def materialize_source_workspaces(
    materialization_plan: dict[str, Any],
    source_records: Iterable[dict[str, Any]],
    *,
    shard_id: str = "",
    task_offset: int = 0,
    max_tasks: int | None = None,
    timeout_seconds: float = 300.0,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Execute workspace materialization tasks and return rewritten source records."""

    plan = _dict(materialization_plan)
    records = list(source_records)
    selection, selection_issues = _select_source_workspace_materialization_tasks(
        plan,
        shard_id=shard_id,
        task_offset=task_offset,
        max_tasks=max_tasks,
    )
    workspace_root = Path(str(plan.get("workspace_root", "")))
    timeout = max(1.0, float(timeout_seconds))
    issues = list(selection_issues)
    task_results: list[dict[str, Any]] = []
    rewrites_by_source_instance_id: dict[str, dict[str, Any]] = {}
    for task in selection:
        result = _materialize_source_workspace_task(
            task,
            workspace_root=workspace_root,
            timeout_seconds=timeout,
            dry_run=dry_run,
        )
        task_results.append(result)
        if not result["valid"]:
            issues.append(
                {
                    "code": "source_workspace_materialization_failed",
                    "message": result["error"],
                    "materialization_task_id": result["materialization_task_id"],
                    "severity": "error",
                }
            )
            continue
        if dry_run:
            continue
        for source_instance_id in result["source_instance_ids"]:
            rewrites_by_source_instance_id[source_instance_id] = {
                "source_uri": result["planned_file_source_uri"],
                "source_revision": result["source_revision"],
                "workspace_cache_path": result["cache_path"],
                "workspace_materialization_task_id": result["materialization_task_id"],
            }

    rewritten_records: list[dict[str, Any]] = []
    for record in records:
        source_instance_id = str(record.get("source_instance_id", ""))
        update = rewrites_by_source_instance_id.get(source_instance_id)
        if not update:
            continue
        rewritten = dict(record)
        rewritten.update(
            {
                "workspace_original_source_uri": str(record.get("source_uri", "")),
                "source_uri": update["source_uri"],
                "source_revision": update["source_revision"],
                "workspace_materialized": True,
                "workspace_cache_path": update["workspace_cache_path"],
                "workspace_materialization_task_id": update[
                    "workspace_materialization_task_id"
                ],
            }
        )
        rewritten_records.append(rewritten)

    expected_rewrites = sum(
        len(_string_list(result.get("source_instance_ids")))
        for result in task_results
        if result["valid"] and not dry_run
    )
    if expected_rewrites and len(rewritten_records) < expected_rewrites:
        issues.append(
            {
                "code": "missing_source_records_for_materialized_tasks",
                "message": (
                    f"Materialized tasks referenced {expected_rewrites} source records "
                    f"but only {len(rewritten_records)} were present in the input file."
                ),
                "severity": "error",
            }
        )
    if not selection:
        issues.append(
            {
                "code": "no_source_workspace_materialization_tasks_selected",
                "message": "No source workspace materialization tasks were selected.",
                "severity": "error",
            }
        )

    failed_tasks = sum(1 for result in task_results if not result["valid"])
    validation = {
        "tasks_selected": bool(selection),
        "tasks_succeeded": bool(selection) and failed_tasks == 0,
        "record_rewrites_complete": expected_rewrites == len(rewritten_records),
        "not_dry_run": not dry_run,
    }
    valid = not any(issue["severity"] == "error" for issue in issues)
    return (
        {
            "schema_version": SOURCE_WORKSPACE_MATERIALIZATION_RUN_SCHEMA_VERSION,
            "inputs": {
                "plan_hash": _stable_json_sha256(plan),
                "source_records_hash": _stable_json_sha256({"records": records}),
                "source_records": len(records),
                "shard_id": shard_id,
                "task_offset": task_offset,
                "max_tasks": max_tasks,
                "timeout_seconds": timeout,
                "dry_run": bool(dry_run),
            },
            "workspace_root": str(workspace_root),
            "counts": {
                "source_records": len(records),
                "selected_tasks": len(selection),
                "succeeded_tasks": len(selection) - failed_tasks,
                "failed_tasks": failed_tasks,
                "rewritten_records": len(rewritten_records),
            },
            "validation": validation,
            "tasks": task_results,
            "issues": issues,
            "ready_for_import_rehearsal": valid
            and all(validation.values())
            and bool(rewritten_records),
            "valid": valid,
        },
        rewritten_records,
    )


def build_synthetic_backfill_spec_plan(
    scenarios: Iterable[Scenario],
    selection_plan: dict[str, Any],
    backfill_plan: dict[str, Any],
    *,
    max_repositories: int = 10,
) -> dict[str, Any]:
    """Draft repository-grounded synthetic specs from reserved backfill slots."""

    scenario_list = list(scenarios)
    slots = _repository_synthetic_family_slots(selection_plan, backfill_plan)
    snapshots = _scenario_repository_snapshots(scenario_list)
    selected_snapshots = snapshots[: max(0, int(max_repositories))]
    ready_specs, draft_specs, report = _build_synthetic_backfill_specs(
        slots,
        selected_snapshots,
    )
    ready_count = _planned_synthetic_count(ready_specs)
    draft_count = _planned_synthetic_count(draft_specs)
    issues = []
    if not slots:
        issues.append(
            {
                "code": "no_repository_synthetic_family_slots",
                "message": "No repository-grounded synthetic task-family slots were found.",
                "severity": "warning",
            }
        )
    if slots and not selected_snapshots:
        issues.append(
            {
                "code": "no_fixed_repository_snapshots",
                "message": "No fixed-revision repository snapshots were available.",
                "severity": "error",
            }
        )
    if draft_count:
        issues.append(
            {
                "code": "synthetic_evidence_incomplete",
                "message": (
                    f"{draft_count} planned synthetic records still require verifier "
                    "evidence before generation."
                ),
                "severity": "warning",
            }
        )
    return {
        "schema_version": SYNTHETIC_BACKFILL_SPEC_SCHEMA_VERSION,
        "inputs": {
            "candidate_scenarios": len(scenario_list),
            "repository_snapshots": len(snapshots),
            "selected_repository_snapshots": len(selected_snapshots),
            "max_repositories": max(0, int(max_repositories)),
            "selection_plan_hash": _stable_json_sha256(selection_plan),
            "backfill_plan_hash": _stable_json_sha256(backfill_plan),
        },
        "family_slots": slots,
        "repository_snapshots": selected_snapshots,
        "generator_ready_specs": {"repositories": ready_specs},
        "draft_specs": {"repositories": draft_specs},
        "evidence_report": report,
        "planned_ready_records": ready_count,
        "planned_draft_records": draft_count,
        "ready_to_generate": bool(ready_count and not draft_count),
        "issues": issues,
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def assemble_seed_candidate_registry(
    *,
    source_root: str | Path,
    output_root: str | Path,
    selection_plan: dict[str, Any],
    synthetic_specs: Iterable[dict[str, Any]],
    source_name: str = "repository_synthetic_backfill",
    overwrite_output: bool = False,
    policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble selected existing seeds and generated backfill into a new registry."""

    source_registry = ScenarioRegistry(source_root)
    output_path = Path(output_root)
    _prepare_registry_root(output_path, overwrite=overwrite_output)
    output_registry = ScenarioRegistry(output_path)
    output_registry.initialize()

    selected_seed_ids = _string_list(selection_plan.get("selected_seed_ids"))
    source_rows = source_registry.list_scenarios()
    scenario_ids_by_seed: dict[str, list[str]] = {}
    for row in source_rows:
        scenario_ids_by_seed.setdefault(row["seed_id"], []).append(row["scenario_id"])

    missing_seed_ids: list[str] = []
    duplicate_seed_ids: list[str] = []
    copied_seed_ids: list[str] = []
    copied_scenario_ids: list[str] = []
    for seed_id in selected_seed_ids:
        scenario_ids = scenario_ids_by_seed.get(seed_id, [])
        if not scenario_ids:
            missing_seed_ids.append(seed_id)
            continue
        if len(scenario_ids) > 1:
            duplicate_seed_ids.append(seed_id)
            continue
        scenario_id = scenario_ids[0]
        output_registry.add_scenario(source_registry.get_scenario(scenario_id))
        copied_seed_ids.append(seed_id)
        copied_scenario_ids.append(scenario_id)

    spec_list = list(synthetic_specs)
    synthetic_summary = generate_repository_synthetic_scenarios(
        output_registry,
        spec_list,
        source_name=source_name,
        strict=False,
    )
    output_validation = _registry_validation_payload(output_registry.validate())
    output_seeds = output_registry.list_seeds()
    policy = _seed_policy(_dict(policy_config or {}).get("seed_policy", policy_config or {}))
    seed_audit = audit_seed_library(output_seeds, policy=policy)
    seed_audit_payload = seed_audit.to_dict()
    seed_audit_payload["train_verifier_type_counts"] = _train_verifier_type_counts(
        output_seeds
    )

    issues: list[dict[str, Any]] = []
    if not selected_seed_ids:
        issues.append(
            {
                "code": "missing_selected_seed_ids",
                "message": "Selection plan did not contain selected_seed_ids.",
                "severity": "error",
            }
        )
    if duplicate_seed_ids:
        issues.append(
            {
                "code": "duplicate_source_seed_ids",
                "message": "Selected seed IDs map to multiple source scenarios.",
                "seed_ids": sorted(set(duplicate_seed_ids)),
                "severity": "error",
            }
        )
    if missing_seed_ids:
        issues.append(
            {
                "code": "selected_seed_ids_missing_from_source",
                "message": "Some selected seed IDs were not present in the source registry.",
                "seed_ids": missing_seed_ids,
                "severity": "error",
            }
        )
    if synthetic_summary.skipped:
        issues.append(
            {
                "code": "synthetic_generation_skipped_records",
                "message": "Some repository-grounded synthetic records were skipped.",
                "skipped": synthetic_summary.skipped,
                "severity": "error",
            }
        )
    if not output_validation["valid"]:
        issues.append(
            {
                "code": "assembled_registry_invalid",
                "message": "The assembled candidate registry failed registry validation.",
                "severity": "error",
            }
        )

    assembly_valid = not any(issue["severity"] == "error" for issue in issues)
    return {
        "schema_version": SEED_CANDIDATE_ASSEMBLY_SCHEMA_VERSION,
        "inputs": {
            "source_root": str(source_root),
            "output_root": str(output_path),
            "source_registry_scenarios": len(source_rows),
            "selection_plan_hash": _stable_json_sha256(selection_plan),
            "synthetic_specs_hash": _stable_json_sha256({"repositories": spec_list}),
            "synthetic_specs": len(spec_list),
        },
        "selection": {
            "requested_existing_count": len(selected_seed_ids),
            "copied_existing_count": len(copied_seed_ids),
            "copied_scenario_count": len(copied_scenario_ids),
            "selected_seed_ids_sha256": hashlib.sha256(
                json.dumps(copied_seed_ids, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "missing_seed_ids": missing_seed_ids,
            "duplicate_source_seed_ids": sorted(set(duplicate_seed_ids)),
        },
        "synthetic_generation": synthetic_summary.to_dict(),
        "output_registry": {
            "root": str(output_path),
            "scenario_count": len(output_registry.list_scenarios()),
            "seed_count": len(output_seeds),
            "validation": output_validation,
        },
        "seed_audit": seed_audit_payload,
        "issues": issues,
        "ready_for_rollout": assembly_valid and bool(seed_audit_payload["valid"]),
        "valid": assembly_valid,
    }


def build_synthetic_evidence_backfill_plan(
    synthetic_backfill_plan: dict[str, Any],
    backfill_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a runbook for draft synthetic specs that still need verifier evidence."""

    draft_specs = _synthetic_draft_specs(synthetic_backfill_plan)
    evidence_tasks: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    for spec in draft_specs:
        families = _string_list(spec.get("task_families")) or ["unknown"]
        targets = (
            _dict_list(spec.get("targets"))
            if spec.get("targets") is not None
            else [dict(spec)]
        )
        for family_value in families:
            family = _normalize_label(family_value)
            for target in targets:
                requirements = _synthetic_target_evidence_requirements(
                    spec,
                    target,
                    family,
                )
                if not requirements["missing_evidence"]:
                    continue
                family_counts[family] += 1
                missing_counts.update(requirements["missing_evidence"])
                evidence_tasks.append(
                    {
                        "evidence_task_id": _evidence_task_id(spec, target, family),
                        "task_family": family,
                        "repository": str(spec.get("repository", "")),
                        "source_uri": str(spec.get("source_uri", "")),
                        "source_revision": str(spec.get("source_revision", "")),
                        "language": _normalize_label(spec.get("language")),
                        "license": str(spec.get("license", "")),
                        "target_name": str(target.get("name", "")),
                        "paths": _string_list(target.get("paths")),
                        "source_instances": _string_list(target.get("source_instances")),
                        "source_urls": _string_list(target.get("source_urls")),
                        "accepted_verifier_types": requirements["accepted_verifier_types"],
                        "missing_evidence": requirements["missing_evidence"],
                        "required_fields": requirements["required_fields"],
                        "suggested_field_examples": requirements["suggested_field_examples"],
                    }
                )

    backfill_summary = _evidence_backfill_gap_summary(backfill_plan or {})
    issues = []
    if evidence_tasks:
        issues.append(
            {
                "code": "synthetic_evidence_tasks_required",
                "message": (
                    f"{len(evidence_tasks)} draft synthetic targets need verifier "
                    "evidence before generation."
                ),
                "severity": "warning",
            }
        )
    return {
        "schema_version": SYNTHETIC_EVIDENCE_BACKFILL_PLAN_SCHEMA_VERSION,
        "inputs": {
            "synthetic_backfill_plan_hash": _stable_json_sha256(synthetic_backfill_plan),
            "backfill_plan_hash": _stable_json_sha256(backfill_plan or {}),
            "draft_specs": len(draft_specs),
        },
        "counts": {
            "evidence_tasks": len(evidence_tasks),
            "task_family": dict(sorted(family_counts.items())),
            "missing_evidence": dict(sorted(missing_counts.items())),
        },
        "backfill_gap_summary": backfill_summary,
        "evidence_tasks": evidence_tasks,
        "recommended_next_actions": _synthetic_evidence_next_actions(
            evidence_tasks,
            backfill_summary,
        ),
        "issues": issues,
        "ready_for_generation": not evidence_tasks,
        "valid": True,
    }


def apply_synthetic_evidence_records(
    synthetic_backfill_plan: dict[str, Any],
    evidence_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Merge collected verifier evidence into draft synthetic specs."""

    evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_backfill_plan)
    task_by_id = {
        str(task["evidence_task_id"]): task for task in evidence_plan["evidence_tasks"]
    }
    records_by_id: dict[str, dict[str, Any]] = {}
    invalid_records: list[dict[str, Any]] = []
    unused_records: list[dict[str, Any]] = []
    for record in evidence_records:
        task_id = str(record.get("evidence_task_id", ""))
        if not task_id:
            invalid_records.append(
                {
                    "evidence_task_id": "",
                    "reason": "Evidence record is missing evidence_task_id.",
                }
            )
            continue
        if task_id not in task_by_id:
            unused_records.append(
                {
                    "evidence_task_id": task_id,
                    "reason": "Evidence task ID is not present in the synthetic plan.",
                }
            )
            continue
        if task_id in records_by_id:
            invalid_records.append(
                {
                    "evidence_task_id": task_id,
                    "reason": "Duplicate evidence record for the same task.",
                }
            )
            continue
        records_by_id[task_id] = dict(record)

    ready_specs: list[dict[str, Any]] = []
    remaining_tasks: list[dict[str, Any]] = []
    applied_task_ids: list[str] = []
    for spec in _synthetic_draft_specs(synthetic_backfill_plan):
        families = _string_list(spec.get("task_families")) or ["unknown"]
        targets = (
            _dict_list(spec.get("targets"))
            if spec.get("targets") is not None
            else [dict(spec)]
        )
        for family_value in families:
            family = _normalize_label(family_value)
            ready_targets: list[dict[str, Any]] = []
            for target in targets:
                task_id = _evidence_task_id(spec, target, family)
                requirements = _synthetic_target_evidence_requirements(spec, target, family)
                if not requirements["missing_evidence"]:
                    ready_targets.append(dict(target))
                    continue
                record = records_by_id.get(task_id)
                if record is None:
                    remaining_tasks.append(task_by_id[task_id])
                    continue
                merged_target = _merge_synthetic_evidence_record(target, record, family)
                merged_requirements = _synthetic_target_evidence_requirements(
                    spec,
                    merged_target,
                    family,
                )
                if merged_requirements["missing_evidence"]:
                    invalid_records.append(
                        {
                            "evidence_task_id": task_id,
                            "reason": "Evidence record does not satisfy required verifier fields.",
                            "missing_evidence": merged_requirements["missing_evidence"],
                        }
                    )
                    remaining_tasks.append(task_by_id[task_id])
                    continue
                ready_targets.append(merged_target)
                applied_task_ids.append(task_id)
            if ready_targets:
                ready_spec = dict(spec)
                ready_spec["task_families"] = [family]
                ready_spec["targets"] = ready_targets
                ready_spec["generator_ready"] = True
                ready_specs.append(ready_spec)

    issues = []
    if invalid_records:
        issues.append(
            {
                "code": "invalid_synthetic_evidence_records",
                "message": "Some evidence records were malformed or incomplete.",
                "severity": "error",
            }
        )
    if unused_records:
        issues.append(
            {
                "code": "unused_synthetic_evidence_records",
                "message": "Some evidence records did not match draft synthetic tasks.",
                "severity": "warning",
            }
        )
    if remaining_tasks:
        issues.append(
            {
                "code": "synthetic_evidence_remaining",
                "message": (
                    f"{len(remaining_tasks)} draft synthetic targets still need evidence."
                ),
                "severity": "warning",
            }
        )
    return {
        "schema_version": SYNTHETIC_EVIDENCE_APPLY_SCHEMA_VERSION,
        "inputs": {
            "synthetic_backfill_plan_hash": _stable_json_sha256(synthetic_backfill_plan),
            "evidence_records": len(list(records_by_id.values())) + len(unused_records),
        },
        "counts": {
            "applied_evidence_records": len(applied_task_ids),
            "ready_records": _planned_synthetic_count(ready_specs),
            "remaining_evidence_tasks": len(remaining_tasks),
            "invalid_evidence_records": len(invalid_records),
            "unused_evidence_records": len(unused_records),
        },
        "applied_evidence_task_ids": sorted(applied_task_ids),
        "remaining_evidence_tasks": remaining_tasks,
        "invalid_evidence_records": invalid_records,
        "unused_evidence_records": unused_records,
        "generator_ready_specs": {"repositories": ready_specs},
        "issues": issues,
        "ready_for_generation": not remaining_tasks and not invalid_records,
        "valid": not invalid_records,
    }


def build_synthetic_evidence_shard_schedule(
    evidence_plan: dict[str, Any],
    *,
    synthetic_backfill_plan_path: str | Path,
    output_dir: str | Path,
    shard_size: int = 20,
) -> dict[str, Any]:
    """Create a deterministic shard runbook for synthetic evidence collection."""

    tasks = _dict_list(evidence_plan.get("evidence_tasks"))
    normalized_size = max(0, int(shard_size))
    issues = []
    if normalized_size <= 0:
        issues.append(
            {
                "code": "invalid_evidence_shard_size",
                "message": "Evidence shard size must be positive.",
                "severity": "error",
            }
        )
        normalized_size = 1
    output_root = Path(output_dir)
    synthetic_plan_text = str(synthetic_backfill_plan_path)
    shards = []
    for shard_index, task_offset in enumerate(range(0, len(tasks), normalized_size)):
        selected = tasks[task_offset : task_offset + normalized_size]
        shard_id = f"synthetic-evidence-shard-{shard_index:04d}"
        records_output = str(output_root / f"{shard_id}-evidence-records.json")
        record_template_output = str(
            output_root / f"{shard_id}-evidence-record-template.json"
        )
        apply_output = str(output_root / f"{shard_id}-evidence-apply.json")
        spec_output = str(output_root / f"{shard_id}-generator-ready.json")
        shards.append(
            {
                "shard_id": shard_id,
                "task_offset": task_offset,
                "max_tasks": normalized_size,
                "selected_tasks": len(selected),
                "evidence_task_ids": [
                    str(task.get("evidence_task_id", "")) for task in selected
                ],
                "task_family_counts": _task_count(selected, "task_family"),
                "missing_evidence_counts": _multi_value_task_count(
                    selected,
                    "missing_evidence",
                ),
                "repository_counts": _task_count(selected, "repository"),
                "records_output": records_output,
                "record_template_output": record_template_output,
                "apply_output": apply_output,
                "spec_output": spec_output,
                "apply_args": [
                    "registry",
                    "seed-synthetic-evidence-apply",
                    "--synthetic-backfill-plan",
                    synthetic_plan_text,
                    "--evidence-records",
                    records_output,
                    "--output",
                    apply_output,
                    "--spec-output",
                    spec_output,
                ],
            }
        )
    if not tasks:
        issues.append(
            {
                "code": "no_synthetic_evidence_tasks",
                "message": "Evidence plan contains no draft synthetic evidence tasks.",
                "severity": "warning",
            }
        )
    return {
        "schema_version": SYNTHETIC_EVIDENCE_SHARD_SCHEDULE_SCHEMA_VERSION,
        "inputs": {
            "evidence_plan_hash": _stable_json_sha256(evidence_plan),
            "synthetic_backfill_plan_path": synthetic_plan_text,
            "output_dir": str(output_root),
        },
        "evidence_tasks": len(tasks),
        "shard_size": normalized_size,
        "shard_count": len(shards),
        "shards": shards,
        "issues": issues,
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def build_synthetic_evidence_record_templates(
    evidence_plan: dict[str, Any],
    shard_schedule: dict[str, Any],
) -> dict[str, Any]:
    """Create fillable per-shard evidence record templates from a shard schedule."""

    issues: list[dict[str, Any]] = []
    tasks_by_id = {
        str(task.get("evidence_task_id", "")): task
        for task in _dict_list(evidence_plan.get("evidence_tasks"))
    }
    tasks_by_id.pop("", None)
    raw_shards = shard_schedule.get("shards")
    if not isinstance(raw_shards, list):
        return {
            "schema_version": SYNTHETIC_EVIDENCE_RECORD_TEMPLATE_SCHEMA_VERSION,
            "inputs": {
                "evidence_plan_hash": _stable_json_sha256(evidence_plan),
                "shard_schedule_hash": _stable_json_sha256(shard_schedule),
            },
            "counts": {
                "template_shards": 0,
                "template_records": 0,
                "missing_evidence_tasks": 0,
            },
            "shard_templates": [],
            "issues": [
                {
                    "code": "synthetic_evidence_shard_schedule_invalid",
                    "message": "Shard schedule must contain a shards list.",
                    "severity": "error",
                }
            ],
            "valid": False,
        }

    shard_templates = []
    total_records = 0
    missing_task_ids: list[str] = []
    for shard_index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, dict):
            issues.append(
                {
                    "code": "synthetic_evidence_shard_invalid",
                    "message": f"Shard entry {shard_index} must contain a JSON object.",
                    "severity": "error",
                }
            )
            continue
        shard_id = str(
            raw_shard.get("shard_id")
            or f"synthetic-evidence-shard-{shard_index:04d}"
        )
        records_output = str(raw_shard.get("records_output", ""))
        template_output = str(
            raw_shard.get("record_template_output")
            or _record_template_output(records_output, shard_id)
        )
        task_ids = _string_list(raw_shard.get("evidence_task_ids"))
        template_records = []
        for task_id in task_ids:
            task = tasks_by_id.get(task_id)
            if task is None:
                missing_task_ids.append(task_id)
                continue
            template_records.append(_synthetic_evidence_record_template(task))
        template_payload = {
            "schema_version": SYNTHETIC_EVIDENCE_RECORD_TEMPLATE_SCHEMA_VERSION,
            "shard_id": shard_id,
            "records_output": records_output,
            "collector_instructions": [
                "Fill verifier fields only with commands or thresholds validated against the "
                "listed source revision.",
                "Keep evidence_task_id values unchanged so the apply gate can merge records.",
                "Write completed records to records_output, not to this template file.",
            ],
            "records": template_records,
        }
        total_records += len(template_records)
        shard_templates.append(
            {
                "shard_id": shard_id,
                "selected_tasks": _int(raw_shard.get("selected_tasks"), default=0),
                "record_template_output": template_output,
                "records_output": records_output,
                "template_records": len(template_records),
                "missing_task_ids": [
                    task_id for task_id in task_ids if task_id not in tasks_by_id
                ],
                "template_payload": template_payload,
            }
        )
    if missing_task_ids:
        issues.append(
            {
                "code": "synthetic_evidence_tasks_missing_from_plan",
                "message": "Some scheduled evidence task IDs are absent from the evidence plan.",
                "severity": "error",
                "task_ids": sorted(set(missing_task_ids)),
            }
        )
    return {
        "schema_version": SYNTHETIC_EVIDENCE_RECORD_TEMPLATE_SCHEMA_VERSION,
        "inputs": {
            "evidence_plan_hash": _stable_json_sha256(evidence_plan),
            "shard_schedule_hash": _stable_json_sha256(shard_schedule),
        },
        "counts": {
            "template_shards": len(shard_templates),
            "template_records": total_records,
            "missing_evidence_tasks": len(set(missing_task_ids)),
        },
        "shard_templates": shard_templates,
        "issues": issues,
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def combine_synthetic_generator_ready_specs(
    synthetic_specs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Combine generator-ready synthetic spec files with duplicate-target checks."""

    issues: list[dict[str, Any]] = []
    combined_specs: list[dict[str, Any]] = []
    spec_summaries = []
    seen_targets: dict[str, str] = {}
    spec_list = list(synthetic_specs)
    for spec_index, spec in enumerate(spec_list):
        repositories_value = spec.get("repositories")
        if not isinstance(repositories_value, list) or not all(
            isinstance(item, dict) for item in repositories_value
        ):
            issues.append(
                {
                    "code": "invalid_generator_ready_spec",
                    "message": "Generator-ready spec must contain a repositories list.",
                    "spec_index": spec_index,
                    "severity": "error",
                }
            )
            continue
        repositories = list(repositories_value)
        spec_summaries.append(
            {
                "spec_index": spec_index,
                "spec_hash": _stable_json_sha256({"repositories": repositories}),
                "repositories": len(repositories),
                "planned_records": _planned_synthetic_count(repositories),
                "task_family_counts": _synthetic_spec_task_family_counts(repositories),
            }
        )
        for repository_index, repository_spec in enumerate(repositories):
            if not repository_spec.get("generator_ready", False):
                issues.append(
                    {
                        "code": "non_ready_synthetic_spec",
                        "message": "Combined synthetic specs must be generator-ready.",
                        "spec_index": spec_index,
                        "repository_index": repository_index,
                        "severity": "error",
                    }
                )
            for target in _dict_list(repository_spec.get("targets")):
                for family in _string_list(repository_spec.get("task_families")):
                    target_key = _synthetic_target_key(repository_spec, target, family)
                    previous = seen_targets.get(target_key)
                    if previous is not None:
                        issues.append(
                            {
                                "code": "duplicate_generator_ready_target",
                                "message": "Duplicate generator-ready synthetic target.",
                                "target": str(target.get("name", "")),
                                "task_family": family,
                                "previous": previous,
                                "current": f"{spec_index}:{repository_index}",
                                "severity": "error",
                            }
                        )
                    else:
                        seen_targets[target_key] = f"{spec_index}:{repository_index}"
            combined_specs.append(dict(repository_spec))
    if not spec_list:
        issues.append(
            {
                "code": "missing_generator_ready_specs",
                "message": "At least one generator-ready spec is required.",
                "severity": "error",
            }
        )
    return {
        "schema_version": SYNTHETIC_READY_SPEC_COMBINE_SCHEMA_VERSION,
        "inputs": {
            "specs": spec_summaries,
        },
        "counts": {
            "input_specs": len(spec_list),
            "input_repositories": sum(item["repositories"] for item in spec_summaries),
            "combined_repositories": len(combined_specs),
            "planned_ready_records": _planned_synthetic_count(combined_specs),
            "task_family": _synthetic_spec_task_family_counts(combined_specs),
        },
        "combined_generator_ready_specs": {"repositories": combined_specs},
        "issues": issues,
        "ready_for_generation": not issues,
        "valid": not any(issue["severity"] == "error" for issue in issues),
    }


def summarize_synthetic_evidence_shard_status(
    shard_schedule: dict[str, Any],
) -> dict[str, Any]:
    """Summarize per-shard next actions from a synthetic evidence runbook."""

    issues: list[dict[str, Any]] = []
    if not isinstance(shard_schedule, dict):
        return {
            "schema_version": SYNTHETIC_EVIDENCE_SHARD_STATUS_SCHEMA_VERSION,
            "schedule": {"shard_count": 0, "evidence_tasks": 0},
            "counts": _synthetic_evidence_shard_status_counts([]),
            "shards": [],
            "issues": [
                {
                    "code": "synthetic_evidence_shard_schedule_invalid",
                    "message": "Shard schedule must contain a JSON object.",
                    "severity": "error",
                }
            ],
            "valid": False,
        }

    raw_shards = shard_schedule.get("shards")
    if not isinstance(raw_shards, list):
        return {
            "schema_version": SYNTHETIC_EVIDENCE_SHARD_STATUS_SCHEMA_VERSION,
            "inputs": {"schedule_hash": _stable_json_sha256(shard_schedule)},
            "schedule": {
                "shard_count": _int(shard_schedule.get("shard_count"), default=0),
                "evidence_tasks": _int(shard_schedule.get("evidence_tasks"), default=0),
            },
            "counts": _synthetic_evidence_shard_status_counts([]),
            "shards": [],
            "issues": [
                {
                    "code": "synthetic_evidence_shard_schedule_invalid",
                    "message": "Shard schedule must contain a shards list.",
                    "severity": "error",
                }
            ],
            "valid": False,
        }

    if not shard_schedule.get("valid", False):
        issues.append(
            {
                "code": "synthetic_evidence_shard_schedule_invalid",
                "message": "Shard schedule is not valid.",
                "severity": "warning",
            }
        )

    shard_statuses = []
    for shard_index, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, dict):
            shard_statuses.append(
                {
                    "shard_id": f"synthetic-evidence-shard-{shard_index:04d}",
                    "status": "blocked",
                    "next_action": "inspect_schedule",
                    "issues": ["Shard entry must contain a JSON object."],
                }
            )
            continue
        shard_statuses.append(_synthetic_evidence_shard_status(raw_shard, shard_index))

    counts = _synthetic_evidence_shard_status_counts(shard_statuses)
    return {
        "schema_version": SYNTHETIC_EVIDENCE_SHARD_STATUS_SCHEMA_VERSION,
        "inputs": {"schedule_hash": _stable_json_sha256(shard_schedule)},
        "schedule": {
            "shard_count": len(raw_shards),
            "declared_shard_count": _int(shard_schedule.get("shard_count"), default=0),
            "evidence_tasks": _int(shard_schedule.get("evidence_tasks"), default=0),
            "shard_size": _int(shard_schedule.get("shard_size"), default=0),
        },
        "counts": counts,
        "shards": shard_statuses,
        "issues": issues,
        "valid": True,
    }


def _import_rehearsal_gate_issues(
    *,
    imported: int,
    min_imported: int,
    quarantined: int,
    max_quarantined: int,
) -> list[dict[str, Any]]:
    issues = []
    required = max(0, min_imported)
    allowed_quarantine = max(0, max_quarantined)
    if imported < required:
        issues.append(
            {
                "code": "min_imported_not_met",
                "message": (
                    f"Imported source record count {imported} is below "
                    f"required minimum {required}"
                ),
                "severity": "error",
            }
        )
    if quarantined > allowed_quarantine:
        issues.append(
            {
                "code": "quarantine_budget_exceeded",
                "message": (
                    f"Quarantined source records {quarantined} exceeds budget "
                    f"{allowed_quarantine}"
                ),
                "severity": "error",
            }
        )
    return issues


def _rehearse_materialization(
    registry: ScenarioRegistry,
    scenario_ids: Iterable[str],
    *,
    sample_count: int,
    materialize_root: str | Path,
    run_hidden_commands: bool,
) -> dict[str, Any]:
    requested = max(0, int(sample_count))
    if requested == 0:
        return {
            "enabled": False,
            "requested": 0,
            "sampled": 0,
            "root": "",
            "run_hidden_commands": bool(run_hidden_commands),
            "issues": [],
            "results": [],
            "valid": True,
        }
    root = Path(materialize_root)
    root.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    selected = list(scenario_ids)[:requested]
    for index, scenario_id in enumerate(selected):
        scenario = registry.get_scenario(scenario_id)
        destination = root / f"{index:04d}-{scenario_id}"
        if destination.exists():
            _ensure_safe_generated_root(destination)
            shutil.rmtree(destination)
        result = {
            "scenario_id": scenario_id,
            "environment_id": scenario.environment.environment_id,
            "workspace": str(destination),
            "source_uri": scenario.environment.source_uri,
            "source_revision": scenario.environment.source_revision,
            "hidden_command_count": len(scenario.hidden_evaluator.hidden_tests),
            "hidden_commands_ran": False,
            "commands_run": 0,
            "valid": True,
        }
        try:
            materialize_environment_source(scenario.environment, destination)
            if run_hidden_commands:
                _run_hidden_commands_for_rehearsal(
                    scenario.hidden_evaluator.hidden_tests,
                    destination,
                    result,
                )
        except Exception as exc:  # noqa: BLE001 - convert any rehearsal failure into gate data.
            result["valid"] = False
            issue = {
                "code": "materialization_rehearsal_failed",
                "message": str(exc),
                "scenario_id": scenario_id,
                "severity": "error",
            }
            issues.append(issue)
        results.append(result)
    if len(selected) < requested:
        issues.append(
            {
                "code": "materialization_sample_shortfall",
                "message": (
                    f"Requested {requested} materialization samples but only "
                    f"{len(selected)} imported scenarios were available"
                ),
                "scenario_id": "",
                "severity": "error",
            }
        )
    return {
        "enabled": True,
        "requested": requested,
        "sampled": len(selected),
        "root": str(root),
        "run_hidden_commands": bool(run_hidden_commands),
        "issues": issues,
        "results": results,
        "valid": not issues,
    }


def _rehearse_hidden_test_patches(
    registry: ScenarioRegistry,
    scenario_ids: Iterable[str],
    *,
    sample_count: int,
    materialize_root: str | Path,
    expected_outcome: str,
) -> dict[str, Any]:
    normalized_expected_outcome = str(expected_outcome or "fail").strip().lower()
    if normalized_expected_outcome not in {"fail", "pass", "any"}:
        raise ValueError("hidden_test_patch_expected_outcome must be fail, pass, or any")
    requested = max(0, int(sample_count))
    if requested == 0:
        return {
            "enabled": False,
            "requested": 0,
            "sampled": 0,
            "root": "",
            "expected_outcome": normalized_expected_outcome,
            "issues": [],
            "results": [],
            "valid": True,
        }
    root = Path(materialize_root)
    root.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    selected = []
    for scenario_id in scenario_ids:
        scenario = registry.get_scenario(scenario_id)
        if scenario.hidden_evaluator.metadata.get("test_patch"):
            selected.append((scenario_id, scenario))
        if len(selected) >= requested:
            break
    for index, (scenario_id, scenario) in enumerate(selected):
        destination = root / f"{index:04d}-{scenario_id}"
        if destination.exists():
            _ensure_safe_generated_root(destination)
            shutil.rmtree(destination)
        test_patch = str(scenario.hidden_evaluator.metadata.get("test_patch") or "")
        result = {
            "scenario_id": scenario_id,
            "environment_id": scenario.environment.environment_id,
            "workspace": str(destination),
            "source_uri": scenario.environment.source_uri,
            "source_revision": scenario.environment.source_revision,
            "test_patch_sha256": hashlib.sha256(test_patch.encode("utf-8")).hexdigest(),
            "hidden_command_count": len(scenario.hidden_evaluator.hidden_tests),
            "patch_check_exit_code": None,
            "patch_apply_exit_code": None,
            "hidden_commands_ran": False,
            "commands_run": 0,
            "expected_outcome": normalized_expected_outcome,
            "valid": True,
        }
        try:
            if not test_patch.strip():
                raise RuntimeError("Hidden-test-patch rehearsal requested a scenario with no patch")
            materialize_environment_source(scenario.environment, destination)
            patch_check = _run_git_apply_for_rehearsal(
                test_patch,
                destination,
                check=True,
            )
            result["patch_check_exit_code"] = patch_check["exit_code"]
            result["patch_check"] = patch_check
            if patch_check["exit_code"] != 0:
                raise RuntimeError(
                    "Hidden test patch failed git apply --check "
                    f"(sha256={result['test_patch_sha256']}, exit={patch_check['exit_code']})"
                )
            patch_apply = _run_git_apply_for_rehearsal(
                test_patch,
                destination,
                check=False,
            )
            result["patch_apply_exit_code"] = patch_apply["exit_code"]
            result["patch_apply"] = patch_apply
            if patch_apply["exit_code"] != 0:
                raise RuntimeError(
                    "Hidden test patch failed git apply "
                    f"(sha256={result['test_patch_sha256']}, exit={patch_apply['exit_code']})"
                )
            _run_hidden_commands_for_hidden_test_patch_rehearsal(
                scenario.hidden_evaluator.hidden_tests,
                destination,
                result,
                expected_outcome=normalized_expected_outcome,
            )
        except Exception as exc:  # noqa: BLE001 - convert any rehearsal failure into gate data.
            result["valid"] = False
            issues.append(
                {
                    "code": "hidden_test_patch_rehearsal_failed",
                    "message": str(exc),
                    "scenario_id": scenario_id,
                    "severity": "error",
                }
            )
        results.append(result)
    if len(selected) < requested:
        issues.append(
            {
                "code": "hidden_test_patch_sample_shortfall",
                "message": (
                    f"Requested {requested} hidden-test-patch samples but only "
                    f"{len(selected)} imported scenarios had hidden test patches"
                ),
                "scenario_id": "",
                "severity": "error",
            }
        )
    return {
        "enabled": True,
        "requested": requested,
        "sampled": len(selected),
        "root": str(root),
        "expected_outcome": normalized_expected_outcome,
        "issues": issues,
        "results": results,
        "valid": not issues,
    }


def _run_git_apply_for_rehearsal(
    patch_text: str,
    workspace: Path,
    *,
    check: bool,
) -> dict[str, Any]:
    arguments = ["git", "apply"]
    if check:
        arguments.append("--check")
    env = os.environ.copy()
    ceiling = str(workspace.resolve().parent)
    existing_ceiling = env.get("GIT_CEILING_DIRECTORIES")
    env["GIT_CEILING_DIRECTORIES"] = (
        f"{ceiling}{os.pathsep}{existing_ceiling}" if existing_ceiling else ceiling
    )
    completed = subprocess.run(
        arguments,
        cwd=workspace,
        env=env,
        input=patch_text,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return {
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }


def _run_hidden_commands_for_hidden_test_patch_rehearsal(
    commands: Iterable[str],
    workspace: Path,
    result: dict[str, Any],
    *,
    expected_outcome: str,
) -> None:
    command_list = list(commands)
    if not command_list:
        raise RuntimeError("Hidden-test-patch rehearsal requested commands, but none exist")
    command_results = []
    for command in command_list:
        arguments = _hidden_rehearsal_command_arguments(command)
        if not arguments:
            raise ValueError("Hidden verifier command cannot be empty")
        completed = subprocess.run(
            arguments,
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=30,
        )
        command_results.append(
            {
                "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                "exit_code": completed.returncode,
                "stdout_sha256": hashlib.sha256(
                    completed.stdout.encode("utf-8")
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    completed.stderr.encode("utf-8")
                ).hexdigest(),
            }
        )
    result["hidden_commands_ran"] = True
    result["commands_run"] = len(command_results)
    result["command_results"] = command_results
    exit_codes = [int(item["exit_code"]) for item in command_results]
    all_passed = all(exit_code == 0 for exit_code in exit_codes)
    any_failed = any(exit_code != 0 for exit_code in exit_codes)
    result["command_outcome"] = "pass" if all_passed else "fail"
    if expected_outcome == "pass" and any_failed:
        raise RuntimeError(
            "Hidden verifier command failed during hidden-test-patch rehearsal "
            f"(sha256={command_results[-1]['command_sha256']}, "
            f"exit={command_results[-1]['exit_code']})"
        )
    if expected_outcome == "fail" and all_passed:
        raise RuntimeError(
            "Hidden-test-patch commands unexpectedly passed on the original workspace"
        )


def _run_hidden_commands_for_rehearsal(
    commands: Iterable[str],
    workspace: Path,
    result: dict[str, Any],
) -> None:
    command_list = list(commands)
    if not command_list:
        raise RuntimeError("Materialization rehearsal requested hidden commands, but none exist")
    command_results = []
    for command in command_list:
        arguments = _hidden_rehearsal_command_arguments(command)
        if not arguments:
            raise ValueError("Hidden verifier command cannot be empty")
        completed = subprocess.run(
            arguments,
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=30,
        )
        command_results.append(
            {
                "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                "exit_code": completed.returncode,
                "stdout_sha256": hashlib.sha256(
                    completed.stdout.encode("utf-8")
                ).hexdigest(),
                "stderr_sha256": hashlib.sha256(
                    completed.stderr.encode("utf-8")
                ).hexdigest(),
            }
        )
        if completed.returncode != 0:
            result["commands_run"] = len(command_results)
            result["command_results"] = command_results
            raise RuntimeError(
                "Hidden verifier command failed during materialization rehearsal "
                f"(sha256={command_results[-1]['command_sha256']}, "
                f"exit={completed.returncode})"
            )
    result["hidden_commands_ran"] = True
    result["commands_run"] = len(command_results)
    result["command_results"] = command_results


def _hidden_rehearsal_command_arguments(command: str) -> list[str]:
    arguments = shlex.split(command)
    if arguments and arguments[0] == "python" and shutil.which("python") is None:
        python3 = shutil.which("python3")
        if python3:
            arguments[0] = python3
    return arguments


def _import_record_sources(
    registry: ScenarioRegistry,
    sources: Iterable[dict[str, Any]],
    *,
    config_dir: Path,
    default_format: str,
    default_split: str,
    default_train_eligible: bool | None,
    allowlist_records: Iterable[dict[str, Any]] = (),
    allowlist_filters: list[AllowlistFilterSummary] | None = None,
) -> list[RegistryImportSummary]:
    summaries = []
    for source in sources:
        source_path = _required_source_path(config_dir, source)
        source_format = _source_format(source, default_format)
        records = load_source_records(source_path)
        if allowlist_records and str(source.get("split", default_split)) == "train":
            records, filter_summary = filter_records_by_allowlist(
                records,
                allowlist_records,
                source_name=str(source.get("source_name", "")) or source_format,
            )
            if allowlist_filters is not None:
                allowlist_filters.append(filter_summary)
        train_eligible = _train_eligible(
            source.get("train_eligible", default_train_eligible)
        )
        if source_format in PUBLIC_ISSUE_PR_FORMATS:
            summaries.append(
                import_public_issue_pr_records(
                    registry,
                    records,
                    source_format=source_format,
                    source_name=str(source.get("source_name", "")),
                    split=str(source.get("split", default_split)),
                    license_name=str(source.get("license", "")),
                    permitted_use=str(source.get("permitted_use", "research")),
                    limit=_optional_int(source.get("limit")),
                    test_command_template=str(source.get("test_command_template", "")),
                    task_family=str(source.get("task_family", "")),
                    source_method=str(source.get("source_method", "")),
                    train_eligible=train_eligible,
                    contamination_tags=_string_list(source.get("contamination_tags")),
                    coverage_tags=_string_list(source.get("coverage_tags")),
                    train_license_allowlist=sorted(
                        set(DEFAULT_TRAIN_LICENSE_ALLOWLIST)
                        | set(_string_list(source.get("allow_train_licenses")))
                    ),
                    strict=bool(source.get("strict", False)),
                )
            )
        elif source_format in PUBLIC_CI_FORMATS:
            summaries.append(
                import_public_ci_records(
                    registry,
                    records,
                    source_format=source_format,
                    source_name=str(source.get("source_name", "")),
                    split=str(source.get("split", default_split)),
                    license_name=str(source.get("license", "")),
                    permitted_use=str(source.get("permitted_use", "research")),
                    limit=_optional_int(source.get("limit")),
                    task_family=str(source.get("task_family", "")),
                    source_method=str(source.get("source_method", "")),
                    train_eligible=train_eligible,
                    contamination_tags=_string_list(source.get("contamination_tags")),
                    coverage_tags=_string_list(source.get("coverage_tags")),
                    train_license_allowlist=sorted(
                        set(DEFAULT_TRAIN_LICENSE_ALLOWLIST)
                        | set(_string_list(source.get("allow_train_licenses")))
                    ),
                    strict=bool(source.get("strict", False)),
                )
            )
        else:
            summaries.append(
                import_swe_style_records(
                    registry,
                    records,
                    source_format=source_format,
                    source_name=str(source.get("source_name", "")),
                    split=str(source.get("split", default_split)),
                    license_name=str(source.get("license", "")),
                    permitted_use=str(source.get("permitted_use", "research")),
                    limit=_optional_int(source.get("limit")),
                    test_command_template=str(source.get("test_command_template", "")),
                    task_family=str(source.get("task_family", "")),
                    source_method=str(source.get("source_method", "")),
                    train_eligible=train_eligible,
                    contamination_tags=_string_list(source.get("contamination_tags")),
                    coverage_tags=_string_list(source.get("coverage_tags")),
                    strict=bool(source.get("strict", False)),
                )
            )
    return summaries


def _generate_synthetic_sources(
    registry: ScenarioRegistry,
    sources: Iterable[dict[str, Any]],
    *,
    config_dir: Path,
    default_split: str,
    default_train_eligible: bool | None,
    allowlist_records: Iterable[dict[str, Any]] = (),
    allowlist_filters: list[AllowlistFilterSummary] | None = None,
) -> list[RepositorySyntheticSummary]:
    summaries = []
    for source in sources:
        source_path = _required_source_path(config_dir, source)
        specs = load_repository_synthesis_specs(source_path)
        if allowlist_records and str(source.get("split", default_split)) == "train":
            specs, filter_summary = filter_records_by_allowlist(
                specs,
                allowlist_records,
                source_name=str(source.get("source_name", "repository_synthetic")),
            )
            if allowlist_filters is not None:
                allowlist_filters.append(filter_summary)
        summaries.append(
            generate_repository_synthetic_scenarios(
                registry,
                specs,
                source_name=str(source.get("source_name", "repository_synthetic")),
                split=str(source.get("split", default_split)),
                task_families=_string_list(source.get("task_families")),
                train_eligible=_train_eligible(
                    source.get("train_eligible", default_train_eligible)
                ),
                train_license_allowlist=sorted(
                    set(DEFAULT_SYNTHETIC_TRAIN_LICENSE_ALLOWLIST)
                    | set(_string_list(source.get("allow_train_licenses")))
                ),
                limit=_optional_int(source.get("limit")),
                strict=bool(source.get("strict", False)),
            )
        )
    return summaries


def _coverage_budget_report(
    seed_audit: dict[str, Any],
    budget_value: Any,
    *,
    quarantine_count: int,
) -> dict[str, Any]:
    budget = _dict(budget_value)
    issues = []
    _add_min_count_issues(
        issues,
        "task_family",
        _dict(seed_audit.get("train_task_family_counts")),
        _int_dict(budget.get("min_task_family_counts")),
    )
    _add_min_count_issues(
        issues,
        "language",
        _dict(seed_audit.get("train_language_counts")),
        _int_dict(budget.get("min_language_counts")),
    )
    _add_min_count_issues(
        issues,
        "source_method",
        _dict(seed_audit.get("train_source_method_counts")),
        _int_dict(budget.get("min_source_method_counts")),
    )
    _add_min_count_issues(
        issues,
        "verifier_type",
        _dict(seed_audit.get("train_verifier_type_counts")),
        _int_dict(budget.get("min_verifier_type_counts")),
    )
    max_quarantined = _optional_int(budget.get("max_quarantined_records"))
    if max_quarantined is not None and quarantine_count > max_quarantined:
        issues.append(
            {
                "code": "quarantine_budget_exceeded",
                "message": (
                    f"Quarantined record count {quarantine_count} exceeds "
                    f"budget {max_quarantined}"
                ),
                "severity": "error",
            }
        )
    return {
        "valid": not any(issue["severity"] == "error" for issue in issues),
        "issues": issues,
        "configured": budget,
    }


def _add_min_count_issues(
    issues: list[dict[str, str]],
    field_name: str,
    actual: dict[str, Any],
    required: dict[str, int],
) -> None:
    normalized_actual = {_normalize_label(key): int(value) for key, value in actual.items()}
    for key, minimum in sorted(required.items()):
        actual_count = normalized_actual.get(_normalize_label(key), 0)
        if actual_count < minimum:
            issues.append(
                {
                    "code": f"min_{field_name}_count_not_met",
                    "message": (
                        f"{field_name} {key} count {actual_count} is below "
                        f"required minimum {minimum}"
                    ),
                    "severity": "error",
                }
            )


def _normalized_int_counts(value: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for key, count in _dict(value).items():
        label = _normalize_label(key)
        if label:
            counts[label] += int(count)
    return dict(sorted(counts.items()))


def _count_gaps(
    counts: dict[str, int],
    *,
    minimum_counts: dict[str, int],
    required_present: Iterable[str],
    dimension: str,
) -> list[dict[str, Any]]:
    normalized_minimums = {
        _normalize_label(key): max(0, int(value)) for key, value in minimum_counts.items()
    }
    required_labels = {_normalize_label(label) for label in required_present}
    required_labels.discard("")
    targets = sorted(set(normalized_minimums) | required_labels)
    gaps = []
    for target in targets:
        minimum = max(normalized_minimums.get(target, 0), 1 if target in required_labels else 0)
        current = int(counts.get(target, 0))
        if current >= minimum:
            continue
        gap = {
            "dimension": dimension,
            "target": target,
            "current": current,
            "minimum": minimum,
            "shortfall": minimum - current,
            "required_by_presence_policy": target in required_labels,
            "required_by_minimum_budget": target in normalized_minimums,
        }
        if dimension == "task_family":
            template = TASK_FAMILY_VERIFIER_TEMPLATES.get(target)
            if template is not None:
                gap["accepted_verifier_types"] = list(template.accepted_verifier_types)
                gap["minimum_evidence"] = template.minimum_evidence
        gaps.append(gap)
    return gaps


def _dominance_gaps(
    counts: dict[str, int],
    *,
    total: int,
    max_share: float,
    dimension: str,
) -> list[dict[str, Any]]:
    if total <= 0 or max_share >= 1.0:
        return []
    gaps = []
    for target, count in counts.items():
        if count <= 0:
            continue
        share = count / total
        if share <= max_share:
            continue
        add_only_total = math.ceil(count / max_share)
        gaps.append(
            {
                "dimension": dimension,
                "target": target,
                "count": count,
                "total": total,
                "current_share": round(share, 6),
                "max_share": max_share,
                "additional_non_target_if_no_downsampling": max(
                    0,
                    add_only_total - total,
                ),
            }
        )
    return sorted(gaps, key=lambda item: (-float(item["current_share"]), item["target"]))


def _backfill_actions(
    *,
    train_eligible_gap: int,
    task_family_gaps: list[dict[str, Any]],
    verifier_type_gaps: list[dict[str, Any]],
    source_method_gaps: list[dict[str, Any]],
    language_count_gaps: list[dict[str, Any]],
    dominance: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if train_eligible_gap > 0:
        actions.append(
            {
                "action": "increase_trainable_seed_pool",
                "target": "train_eligible",
                "minimum_count": train_eligible_gap,
                "reason": "Trainable seed count is below the configured minimum.",
            }
        )
    for gap in task_family_gaps:
        target = str(gap["target"])
        action = (
            "collect_public_source_family"
            if target == "bug_repair"
            else "generate_repository_grounded_synthetic_family"
        )
        actions.append(
            {
                "action": action,
                "target": target,
                "minimum_count": gap["shortfall"],
                "current_count": gap["current"],
                "accepted_verifier_types": gap.get("accepted_verifier_types", []),
                "reason": (
                    "Task-family coverage is below the required presence or minimum-count gate."
                ),
            }
        )
    for gap in verifier_type_gaps:
        actions.append(
            {
                "action": "add_verifier_evidence_backfill",
                "target": gap["target"],
                "minimum_count": gap["shortfall"],
                "current_count": gap["current"],
                "reason": "Verifier evidence coverage is below the configured gate.",
            }
        )
    for gap in source_method_gaps:
        target = str(gap["target"])
        action = (
            "generate_repository_grounded_synthetic_records"
            if target == "repository_grounded_synthetic"
            else "collect_source_method_records"
        )
        actions.append(
            {
                "action": action,
                "target": target,
                "minimum_count": gap["shortfall"],
                "current_count": gap["current"],
                "reason": "Source-method coverage is below the configured minimum.",
            }
        )
    for gap in language_count_gaps:
        actions.append(
            {
                "action": "collect_language_sources",
                "target": gap["target"],
                "minimum_count": gap["shortfall"],
                "current_count": gap["current"],
                "reason": "Language coverage is below the configured minimum.",
            }
        )
    _append_dominance_actions(actions, "task_family", dominance["task_family"])
    _append_dominance_actions(actions, "source_method", dominance["source_method"])
    _append_dominance_actions(actions, "repository", dominance["repository"])
    _append_dominance_actions(actions, "language", dominance["language"])
    if actions:
        actions.append(
            {
                "action": "refresh_holdout_and_decontamination",
                "target": "holdout_registry",
                "minimum_count": 0,
                "reason": (
                    "Any backfill or balancing change must be followed by seed and "
                    "scenario decontamination before provider rollout."
                ),
            }
        )
    return actions


def _append_dominance_actions(
    actions: list[dict[str, Any]],
    dimension: str,
    gaps: list[dict[str, Any]],
) -> None:
    action_by_dimension = {
        "task_family": "balance_or_sample_task_family",
        "source_method": "balance_or_sample_source_method",
        "repository": "balance_or_sample_repository",
        "language": "add_cross_language_sources_or_downsample",
    }
    for gap in gaps:
        actions.append(
            {
                "action": action_by_dimension[dimension],
                "target": gap["target"],
                "minimum_count": gap["additional_non_target_if_no_downsampling"],
                "current_count": gap["count"],
                "current_share": gap["current_share"],
                "max_share": gap["max_share"],
                "reason": (
                    "Current share exceeds the configured cap; use targeted backfill, "
                    "balanced sampling, or both."
                ),
            }
        )


def _selection_target(
    config: dict[str, Any],
    policy: SeedLibraryPolicy,
    *,
    explicit_target: int | None,
    candidate_count: int,
) -> int:
    if explicit_target is not None:
        return max(0, int(explicit_target))
    configured = _optional_int(config.get("target_train_eligible"))
    if configured is not None:
        return max(0, configured)
    if policy.min_train_eligible > 0:
        return policy.min_train_eligible
    return max(0, candidate_count)


def _reserved_backfill_slots(
    gaps: dict[str, Any],
    *,
    target: int,
    seeds: list[QuerySeed],
    policy: SeedLibraryPolicy,
) -> dict[str, Any]:
    slots: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    train_gap = _dict(gaps.get("train_eligible"))
    train_shortfall = max(0, int(train_gap.get("shortfall", 0) or 0))
    if train_shortfall:
        _add_reserved_component(
            components,
            slots,
            component="train_eligible_shortfall",
            amount=train_shortfall,
            slot={
                "type": "train_eligible_minimum",
                "target": "train_eligible",
                "minimum_count": train_shortfall,
            },
        )
    _add_gap_slots(
        components,
        slots,
        component="task_family_minimums",
        slot_type="task_family_minimum",
        gaps=_dict_list(gaps.get("task_family")),
    )
    _add_gap_slots(
        components,
        slots,
        component="source_method_minimums",
        slot_type="source_method_minimum",
        gaps=_dict_list(gaps.get("source_method")),
    )
    _add_gap_slots(
        components,
        slots,
        component="language_minimums",
        slot_type="language_minimum",
        gaps=_dict_list(gaps.get("language_count")),
    )
    verifier_gaps = _dict_list(gaps.get("verifier_type"))
    if verifier_gaps:
        verifier_lower_bound = max(int(gap.get("shortfall", 0) or 0) for gap in verifier_gaps)
        if verifier_lower_bound > 0:
            components.append(
                {
                    "component": "verifier_type_minimums",
                    "minimum_reserved_slots": verifier_lower_bound,
                    "reason": "Verifier evidence gaps may overlap on the same backfill seeds.",
                }
            )
        for gap in verifier_gaps:
            shortfall = int(gap.get("shortfall", 0) or 0)
            if shortfall > 0:
                slots.append(
                    {
                        "type": "verifier_type_minimum",
                        "target": gap.get("target", ""),
                        "minimum_count": shortfall,
                    }
                )
    diversity_slots = _diversity_reserved_slots(seeds, target=target, policy=policy)
    if diversity_slots:
        diversity_lower_bound = max(slot["minimum_count"] for slot in diversity_slots)
        components.append(
            {
                "component": "share_cap_diversity",
                "minimum_reserved_slots": diversity_lower_bound,
                "reason": "Dominant labels need future non-target records to satisfy caps.",
            }
        )
        slots.extend(diversity_slots)
    minimum_reserved = max(
        (int(component["minimum_reserved_slots"]) for component in components),
        default=0,
    )
    return {
        "minimum_reserved_slots": min(target, minimum_reserved),
        "components": components,
        "slots": sorted(slots, key=lambda item: (str(item["type"]), str(item["target"]))),
    }


def _add_gap_slots(
    components: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    *,
    component: str,
    slot_type: str,
    gaps: list[dict[str, Any]],
) -> None:
    amount = sum(max(0, int(gap.get("shortfall", 0) or 0)) for gap in gaps)
    if amount <= 0:
        return
    components.append(
        {
            "component": component,
            "minimum_reserved_slots": amount,
            "reason": "Each seed contributes to one label in this dimension.",
        }
    )
    for gap in gaps:
        shortfall = max(0, int(gap.get("shortfall", 0) or 0))
        if shortfall <= 0:
            continue
        slot = {
            "type": slot_type,
            "target": gap.get("target", ""),
            "minimum_count": shortfall,
        }
        if "accepted_verifier_types" in gap:
            slot["accepted_verifier_types"] = gap["accepted_verifier_types"]
        slots.append(slot)


def _add_reserved_component(
    components: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    *,
    component: str,
    amount: int,
    slot: dict[str, Any],
) -> None:
    components.append(
        {
            "component": component,
            "minimum_reserved_slots": amount,
            "reason": "The current candidate pool is below the configured target.",
        }
    )
    slots.append(slot)


def _diversity_reserved_slots(
    seeds: list[QuerySeed],
    *,
    target: int,
    policy: SeedLibraryPolicy,
) -> list[dict[str, Any]]:
    if target <= 0:
        return []
    dimensions = {
        "task_family": (policy.max_task_family_share, _seed_task_family),
        "source_method": (policy.max_source_method_share, _seed_source_method),
        "repository": (policy.max_repository_share, _seed_repository_value),
        "language": (policy.max_language_share, _seed_language_value),
    }
    slots = []
    for dimension, (max_share, value_fn) in dimensions.items():
        if max_share >= 1.0:
            continue
        counts = _dimension_counts(seeds, value_fn)
        for label, count in counts.items():
            required_non_target = max(0, target - math.floor(target * max_share))
            available_non_target = len(seeds) - count
            shortfall = max(0, required_non_target - available_non_target)
            if shortfall <= 0:
                continue
            slots.append(
                {
                    "type": "share_cap_diversity",
                    "dimension": dimension,
                    "target": f"non_{label}",
                    "dominant_label": label,
                    "minimum_count": shortfall,
                    "max_share": max_share,
                }
            )
    return sorted(slots, key=lambda item: (item["dimension"], item["target"]))


def _select_existing_seed_slice(
    seeds: list[QuerySeed],
    *,
    target: int,
    final_target: int,
    policy: SeedLibraryPolicy,
) -> list[QuerySeed]:
    if target <= 0:
        return []
    remaining = sorted(seeds, key=lambda seed: seed.seed_id)
    selected: list[QuerySeed] = []
    counts = _empty_selection_counters()
    while len(selected) < target:
        eligible = [
            seed
            for seed in remaining
            if _seed_fits_final_caps(seed, counts, final_target=final_target, policy=policy)
        ]
        if not eligible:
            break
        chosen = min(
            eligible,
            key=lambda seed: _selection_priority(
                seed,
                counts,
                final_target=final_target,
                policy=policy,
            ),
        )
        selected.append(chosen)
        _update_selection_counters(counts, chosen)
        remaining.remove(chosen)
    return selected


def _empty_selection_counters() -> dict[str, Counter[str]]:
    return {
        "task_family": Counter(),
        "source_method": Counter(),
        "repository": Counter(),
        "language": Counter(),
    }


def _seed_fits_final_caps(
    seed: QuerySeed,
    counts: dict[str, Counter[str]],
    *,
    final_target: int,
    policy: SeedLibraryPolicy,
) -> bool:
    if final_target <= 0:
        return False
    for dimension, max_share, value in _seed_dimension_values(seed, policy):
        if max_share >= 1.0:
            continue
        cap = math.floor(final_target * max_share)
        if cap <= 0 or counts[dimension][value] + 1 > cap:
            return False
    return True


def _selection_priority(
    seed: QuerySeed,
    counts: dict[str, Counter[str]],
    *,
    final_target: int,
    policy: SeedLibraryPolicy,
) -> tuple[float, float, float, float, str]:
    pressure = []
    for dimension, max_share, value in _seed_dimension_values(seed, policy):
        if max_share >= 1.0:
            pressure.append(0.0)
            continue
        cap = max(1, math.floor(final_target * max_share))
        pressure.append(counts[dimension][value] / cap)
    return (
        pressure[0],
        pressure[1],
        pressure[2],
        pressure[3],
        seed.seed_id,
    )


def _seed_dimension_values(
    seed: QuerySeed,
    policy: SeedLibraryPolicy,
) -> tuple[tuple[str, float, str], ...]:
    return (
        ("task_family", policy.max_task_family_share, _seed_task_family(seed)),
        ("source_method", policy.max_source_method_share, _seed_source_method(seed)),
        ("repository", policy.max_repository_share, _seed_repository_value(seed)),
        ("language", policy.max_language_share, _seed_language_value(seed)),
    )


def _update_selection_counters(
    counts: dict[str, Counter[str]],
    seed: QuerySeed,
) -> None:
    counts["task_family"][_seed_task_family(seed)] += 1
    counts["source_method"][_seed_source_method(seed)] += 1
    counts["repository"][_seed_repository_value(seed)] += 1
    counts["language"][_seed_language_value(seed)] += 1


def _selection_counts(seeds: list[QuerySeed]) -> dict[str, dict[str, int]]:
    counts = _empty_selection_counters()
    verifier_counts: Counter[str] = Counter()
    for seed in seeds:
        _update_selection_counters(counts, seed)
        verifier_counts.update(seed.verifier_types)
    return {
        "task_family": dict(sorted(counts["task_family"].items())),
        "source_method": dict(sorted(counts["source_method"].items())),
        "repository": dict(sorted(counts["repository"].items())),
        "language": dict(sorted(counts["language"].items())),
        "verifier_type": dict(sorted(verifier_counts.items())),
    }


def _selection_shares_against_target(
    seeds: list[QuerySeed],
    *,
    target: int,
) -> dict[str, dict[str, float]]:
    counts = _selection_counts(seeds)
    denominator = max(1, target)
    return {
        dimension: {
            label: round(count / denominator, 6)
            for label, count in sorted(dimension_counts.items())
        }
        for dimension, dimension_counts in counts.items()
    }


def _selection_plan_issues(
    *,
    target: int,
    existing_target: int,
    selected_count: int,
    selected_audit: dict[str, Any],
    reserved: dict[str, Any],
) -> list[dict[str, Any]]:
    issues = []
    if selected_count < existing_target:
        issues.append(
            {
                "code": "existing_selection_shortfall",
                "message": (
                    f"Selected {selected_count} existing seeds but planned "
                    f"{existing_target} before reserved backfill slots"
                ),
                "severity": "error",
            }
        )
    if reserved["slots"]:
        issues.append(
            {
                "code": "reserved_backfill_required",
                "message": (
                    f"Selection needs at least {reserved['minimum_reserved_slots']} "
                    f"future backfill slots before reaching target {target}"
                ),
                "severity": "warning",
            }
        )
    if not reserved["slots"] and not bool(selected_audit.get("valid", False)):
        issues.append(
            {
                "code": "selected_seed_audit_failed",
                "message": "The selected seed slice does not satisfy the seed policy.",
                "severity": "error",
            }
        )
    return issues


def _remediation_allowlist_summary(
    allowlist_records: list[dict[str, Any]],
) -> dict[str, Any]:
    language_counts: Counter[str] = Counter()
    repositories: list[str] = []
    non_python: list[str] = []
    for record in allowlist_records:
        repository = str(record.get("repository", "")).strip()
        if not repository:
            continue
        language = _normalize_label(record.get("language"))
        repositories.append(repository)
        if language:
            language_counts[language] += 1
        if language and language != "python":
            non_python.append(repository)
    return {
        "repositories": len(repositories),
        "repository_names": sorted(repositories),
        "language_counts": dict(sorted(language_counts.items())),
        "non_python_repositories": sorted(non_python),
    }


def _remediation_requirement_for_slot(
    slot: dict[str, Any],
    *,
    allowlist_summary: dict[str, Any],
) -> dict[str, Any]:
    slot_type = str(slot.get("type", ""))
    target = str(slot.get("target", ""))
    minimum = _int(slot.get("minimum_count"), default=0)
    if slot_type == "share_cap_diversity":
        return _share_cap_remediation_requirement(
            slot,
            allowlist_summary=allowlist_summary,
        )
    if slot_type == "source_method_minimum" and target == "public_issue_workspace":
        return {
            "action": "collect_public_issue_sources",
            "target": target,
            "minimum_count": minimum,
            "accepted_source_types": ["public_issue"],
            "required_evidence": [
                "fixed source_revision",
                "public source_url",
                "candidate verifier command or explicit verifier evidence",
            ],
            "leakage_constraints": [
                "Do not use benchmark issue statements or hidden evaluator answers.",
            ],
        }
    if slot_type == "verifier_type_minimum" and target == "build_command":
        return {
            "action": "collect_build_command_evidence",
            "target": target,
            "minimum_count": minimum,
            "accepted_source_types": ["public_pr", "public_ci", "repository_synthetic"],
            "required_evidence": [
                "build_commands that run on the fixed source revision",
                "recorded command output or CI evidence for the build command",
            ],
            "leakage_constraints": [
                "Do not count benchmark commands as build commands unless they are "
                "project-native build checks.",
            ],
        }
    if slot_type == "verifier_type_minimum" and target == "hidden_test_patch":
        return {
            "action": "curate_hidden_test_patch_evidence",
            "target": target,
            "minimum_count": minimum,
            "accepted_source_types": ["public_issue", "public_pr"],
            "required_evidence": [
                "withheld hidden test patch derived from public behavior",
                "hidden evaluator kept out of the generation prompt",
            ],
            "leakage_constraints": [
                "Do not use benchmark oracle patches or benchmark test_patch fields.",
                "Do not expose hidden tests to the task-generation or rollout model.",
            ],
        }
    return {
        "action": "satisfy_reserved_backfill_slot",
        "target": target,
        "minimum_count": minimum,
        "slot_type": slot_type,
        "required_evidence": [],
        "leakage_constraints": [
            "Keep training sources separated from benchmark and holdout evaluator data.",
        ],
    }


def _share_cap_remediation_requirement(
    slot: dict[str, Any],
    *,
    allowlist_summary: dict[str, Any],
) -> dict[str, Any]:
    dimension = str(slot.get("dimension", ""))
    dominant = str(slot.get("dominant_label", ""))
    minimum = _int(slot.get("minimum_count"), default=0)
    if dimension == "language":
        return {
            "action": "collect_cross_language_sources",
            "target": str(slot.get("target", "")),
            "minimum_count": minimum,
            "exclude_language": dominant,
            "candidate_allowlist_repositories": allowlist_summary[
                "non_python_repositories"
            ],
            "required_evidence": [
                "permissive license",
                "fixed source_revision",
                "language-specific test or build command",
            ],
            "leakage_constraints": [
                "Do not satisfy language diversity with translated benchmark tasks.",
            ],
        }
    if dimension == "repository":
        candidates = [
            repository
            for repository in allowlist_summary["repository_names"]
            if repository != dominant
        ]
        return {
            "action": "collect_non_dominant_repository_sources",
            "target": str(slot.get("target", "")),
            "minimum_count": minimum,
            "exclude_repository": dominant,
            "candidate_allowlist_repositories": candidates,
            "required_evidence": [
                "permissive license",
                "fixed source_revision",
                "source-instance provenance",
            ],
            "leakage_constraints": [
                "Do not duplicate the dominant repository under alternate source IDs.",
            ],
        }
    return {
        "action": "collect_share_cap_diversity_sources",
        "target": str(slot.get("target", "")),
        "minimum_count": minimum,
        "dimension": dimension,
        "dominant_label": dominant,
        "required_evidence": [],
        "leakage_constraints": [
            "Keep train and evaluation source pools auditable and separated.",
        ],
    }


def _hidden_test_patch_requirement(remediation_plan: dict[str, Any]) -> dict[str, Any]:
    for requirement in _dict_list_or_empty(remediation_plan.get("requirements")):
        action = str(requirement.get("action", ""))
        target = str(requirement.get("target", ""))
        if action == "curate_hidden_test_patch_evidence" or target == "hidden_test_patch":
            return requirement
    return {}


def _hidden_test_patch_record_rejection(
    record: dict[str, Any],
    *,
    accepted_source_types: set[str],
) -> dict[str, str]:
    source_type = _curation_source_type(record)
    if source_type not in accepted_source_types:
        return {
            "code": "unsupported_source_type",
            "message": f"Unsupported source type for hidden test curation: {source_type}",
        }
    missing_fields = [
        field_name
        for field_name in (
            "repository",
            "source_revision",
            "source_instance_id",
            "source_url",
            "title",
        )
        if not str(record.get(field_name, "")).strip()
    ]
    if missing_fields:
        return {
            "code": "missing_required_public_source_fields",
            "message": "Missing required fields: " + ", ".join(sorted(missing_fields)),
        }
    source_revision = str(record.get("source_revision", "")).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_revision):
        return {
            "code": "invalid_source_revision",
            "message": "Hidden test curation requires a fixed 40-character commit SHA.",
        }
    if _record_has_benchmark_source(record):
        return {
            "code": "benchmark_source_rejected",
            "message": "Benchmark-derived records cannot seed hidden test curation.",
        }
    oracle_fields = _present_oracle_fields(record)
    if oracle_fields:
        return {
            "code": "oracle_fields_rejected",
            "message": "Record contains oracle fields: " + ", ".join(oracle_fields),
        }
    source_url = str(record.get("source_url", "")).strip()
    if not source_url.startswith(("https://github.com/", "http://github.com/")):
        return {
            "code": "non_github_public_source_url",
            "message": "Hidden test curation currently requires a public GitHub source URL.",
        }
    return {}


def _curation_rejection(
    record_index: int,
    record: dict[str, Any],
    rejection: dict[str, str],
) -> dict[str, Any]:
    return {
        "record_index": record_index,
        "source_instance_id": str(record.get("source_instance_id", "")),
        "repository": str(record.get("repository", "")),
        "source_type": _curation_source_type(record),
        "code": rejection["code"],
        "message": rejection["message"],
    }


def _select_hidden_test_patch_curation_records(
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return _select_balanced_source_records(records, limit=limit)


def _hidden_command_record_rejection(record: dict[str, Any]) -> dict[str, str]:
    source_type = _curation_source_type(record)
    if source_type not in {"public_issue", "public_pr", "public_ci"}:
        return {
            "code": "unsupported_source_type",
            "message": f"Unsupported source type for hidden command curation: {source_type}",
        }
    missing_fields = [
        field_name
        for field_name in (
            "repository",
            "source_uri",
            "source_revision",
            "source_instance_id",
        )
        if not str(record.get(field_name, "")).strip()
    ]
    if missing_fields:
        return {
            "code": "missing_required_hidden_command_fields",
            "message": "Missing required fields: " + ", ".join(sorted(missing_fields)),
        }
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(record.get("source_revision", ""))):
        return {
            "code": "invalid_source_revision",
            "message": "Hidden command curation requires a fixed 40-character commit SHA.",
        }
    if _record_has_benchmark_source(record):
        return {
            "code": "benchmark_source_rejected",
            "message": "Benchmark-derived records cannot seed hidden command curation.",
        }
    oracle_fields = _present_oracle_fields(record)
    if oracle_fields:
        return {
            "code": "oracle_fields_rejected",
            "message": "Record contains oracle fields: " + ", ".join(oracle_fields),
        }
    if not _candidate_verifier_commands(record):
        return {
            "code": "missing_candidate_verifier_commands",
            "message": "Hidden command curation requires existing command evidence.",
        }
    return {}


def _select_hidden_command_curation_records(
    failed_records: list[dict[str, Any]],
    remaining_records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected = _select_balanced_source_records(failed_records, limit=limit)
    remaining_limit = max(0, limit - len(selected))
    if remaining_limit:
        selected.extend(
            _select_balanced_source_records(
                remaining_records,
                limit=remaining_limit,
            )
        )
    return selected


def _hidden_command_failure_evidence_by_source_instance(
    rehearsal_summaries: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for summary in rehearsal_summaries:
        payload = _dict(summary)
        summary_hash = _stable_json_sha256(payload)
        scenario_to_source = _scenario_source_instance_map(payload)
        for result in _dict_list(_dict(payload.get("materialization")).get("results")):
            if result.get("valid", True):
                continue
            source_instance_id = str(result.get("source_instance_id", "")).strip()
            if not source_instance_id:
                source_instance_id = scenario_to_source.get(
                    str(result.get("scenario_id", "")),
                    "",
                )
            if not source_instance_id:
                continue
            evidence[source_instance_id] = {
                "summary_hash": summary_hash,
                "schema_version": str(payload.get("schema_version", "")),
                "registry_root": str(payload.get("registry_root", "")),
                "scenario_id": str(result.get("scenario_id", "")),
                "environment_id": str(result.get("environment_id", "")),
                "source_uri": str(result.get("source_uri", "")),
                "source_revision": str(result.get("source_revision", "")),
                "workspace": str(result.get("workspace", "")),
                "issues": _hidden_command_result_issues(payload, result),
                "command_results": _dict_list(result.get("command_results")),
            }
    return evidence


def _scenario_source_instance_map(summary: dict[str, Any]) -> dict[str, str]:
    registry_root = str(summary.get("registry_root", "")).strip()
    if not registry_root:
        return {}
    registry = ScenarioRegistry(registry_root)
    mapping: dict[str, str] = {}
    for scenario_id in _string_list(_dict(summary.get("import")).get("scenario_ids")):
        try:
            scenario = registry.get_scenario(scenario_id)
        except Exception:  # noqa: BLE001 - best-effort evidence mapping only.
            continue
        source_instance_id = str(
            scenario.query_seed.metadata.get("source_instance_id", "")
        )
        if source_instance_id:
            mapping[scenario_id] = source_instance_id
    return mapping


def _hidden_command_result_issues(
    summary: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    scenario_id = str(result.get("scenario_id", ""))
    return [
        issue
        for issue in _dict_list(_dict(summary.get("materialization")).get("issues"))
        if str(issue.get("scenario_id", "")) in {"", scenario_id}
    ]


def _select_balanced_source_records(
    records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            _normalize_label(record.get("language")),
            str(record.get("repository", "")),
            _curation_source_type(record),
        )
        buckets.setdefault(key, []).append(record)
    selected: list[dict[str, Any]] = []
    active_keys = sorted(buckets)
    while active_keys and len(selected) < limit:
        next_keys: list[tuple[str, str, str]] = []
        for key in active_keys:
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        active_keys = next_keys
    return selected


def _source_workspace_record_rejection(record: dict[str, Any]) -> dict[str, str]:
    source_type = _curation_source_type(record)
    if source_type not in {"public_issue", "public_pr", "public_ci"}:
        return {
            "code": "unsupported_source_type",
            "message": f"Unsupported source type for workspace materialization: {source_type}",
        }
    missing_fields = [
        field_name
        for field_name in (
            "repository",
            "source_uri",
            "source_revision",
            "source_instance_id",
        )
        if not str(record.get(field_name, "")).strip()
    ]
    if missing_fields:
        return {
            "code": "missing_required_workspace_fields",
            "message": "Missing required fields: " + ", ".join(sorted(missing_fields)),
        }
    source_revision = str(record.get("source_revision", "")).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_revision):
        return {
            "code": "invalid_source_revision",
            "message": "Workspace materialization requires a fixed 40-character commit SHA.",
        }
    source_uri = str(record.get("source_uri", "")).strip()
    if not (
        source_uri.startswith("https://github.com/")
        or source_uri.startswith("file://")
    ):
        return {
            "code": "unsupported_source_uri",
            "message": "Workspace materialization requires a public GitHub or file source URI.",
        }
    if _record_has_benchmark_source(record):
        return {
            "code": "benchmark_source_rejected",
            "message": "Benchmark-derived records cannot seed train workspace materialization.",
        }
    oracle_fields = _present_oracle_fields(record)
    if oracle_fields:
        return {
            "code": "oracle_fields_rejected",
            "message": "Record contains oracle fields: " + ", ".join(oracle_fields),
        }
    return {}


def _workspace_materialization_rejection(
    record_index: int,
    record: dict[str, Any],
    rejection: dict[str, str],
) -> dict[str, Any]:
    return {
        "record_index": record_index,
        "source_instance_id": str(record.get("source_instance_id", "")),
        "repository": str(record.get("repository", "")),
        "source_type": _curation_source_type(record),
        "code": rejection["code"],
        "message": rejection["message"],
    }


def _source_workspace_materialization_tasks(
    records: list[dict[str, Any]],
    workspace_root: Path,
) -> list[dict[str, Any]]:
    tasks_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        repository = str(record.get("repository", "")).strip()
        source_uri = str(record.get("source_uri", "")).strip()
        source_revision = str(record.get("source_revision", "")).strip()
        key = (repository, source_uri, source_revision)
        task = tasks_by_key.get(key)
        if task is None:
            task = _source_workspace_materialization_task(
                repository=repository,
                source_uri=source_uri,
                source_revision=source_revision,
                workspace_root=workspace_root,
            )
            tasks_by_key[key] = task
        task["source_instance_ids"].append(str(record.get("source_instance_id", "")))
        source_url = str(record.get("source_url", "")).strip()
        if source_url:
            task["source_urls"].append(source_url)
        task["record_count"] += 1
        source_type = _curation_source_type(record)
        task["source_type_counts"][source_type] = (
            task["source_type_counts"].get(source_type, 0) + 1
        )
        language = _normalize_label(record.get("language"))
        if language:
            task["language_counts"][language] = task["language_counts"].get(language, 0) + 1
    tasks = list(tasks_by_key.values())
    for task in tasks:
        task["source_instance_ids"] = sorted(_dedupe_strings(task["source_instance_ids"]))
        task["source_urls"] = sorted(_dedupe_strings(task["source_urls"]))
        task["source_type_counts"] = dict(sorted(task["source_type_counts"].items()))
        task["language_counts"] = dict(sorted(task["language_counts"].items()))
    return tasks


def _source_workspace_materialization_task(
    *,
    repository: str,
    source_uri: str,
    source_revision: str,
    workspace_root: Path,
) -> dict[str, Any]:
    digest = _stable_json_sha256(
        {
            "repository": repository,
            "source_uri": source_uri,
            "source_revision": source_revision,
        }
    )[:16]
    cache_path = (
        workspace_root
        / _safe_path_component(repository)
        / f"{source_revision[:12]}-{digest[:8]}"
    )
    cache_path_text = str(cache_path)
    return {
        "materialization_task_id": f"source-workspace-{digest}",
        "repository": repository,
        "source_uri": source_uri,
        "source_revision": source_revision,
        "cache_path": cache_path_text,
        "planned_file_source_uri": cache_path.resolve().as_uri(),
        "record_count": 0,
        "source_instance_ids": [],
        "source_urls": [],
        "source_type_counts": {},
        "language_counts": {},
        "resume_policy": (
            "Clone only when cache_path is absent; fetch the fixed revision when the "
            "commit object is missing; then checkout detached at source_revision."
        ),
        "materialization_args": {
            "clone": [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                source_uri,
                cache_path_text,
            ],
            "fetch_revision": [
                "git",
                "-C",
                cache_path_text,
                "fetch",
                "--filter=blob:none",
                "origin",
                source_revision,
            ],
            "checkout": [
                "git",
                "-C",
                cache_path_text,
                "checkout",
                "--detach",
                source_revision,
            ],
            "verify_revision": [
                "git",
                "-C",
                cache_path_text,
                "cat-file",
                "-e",
                f"{source_revision}^{{commit}}",
            ],
        },
        "record_update": {
            "source_uri": cache_path.resolve().as_uri(),
            "source_revision": source_revision,
        },
    }


def _source_workspace_materialization_shards(
    tasks: list[dict[str, Any]],
    shard_size: int,
) -> list[dict[str, Any]]:
    shards = []
    for shard_index, task_offset in enumerate(range(0, len(tasks), shard_size)):
        selected = tasks[task_offset : task_offset + shard_size]
        shard_id = f"source-workspace-shard-{shard_index:04d}"
        source_type_counts: Counter[str] = Counter()
        record_count = 0
        repositories = set()
        for task in selected:
            record_count += _int(task.get("record_count"), default=0)
            repositories.add(str(task.get("repository", "")))
            source_type_counts.update(_int_dict(task.get("source_type_counts")))
        shards.append(
            {
                "shard_id": shard_id,
                "task_offset": task_offset,
                "max_tasks": shard_size,
                "selected_tasks": len(selected),
                "record_count": record_count,
                "repositories": sorted(repositories),
                "source_type_counts": dict(sorted(source_type_counts.items())),
                "materialization_task_ids": [
                    str(task.get("materialization_task_id", "")) for task in selected
                ],
                "next_action": "materialize_workspaces",
            }
        )
    return shards


def _select_source_workspace_materialization_tasks(
    plan: dict[str, Any],
    *,
    shard_id: str,
    task_offset: int,
    max_tasks: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = _dict_list(plan.get("materialization_tasks"))
    task_by_id = {
        str(task.get("materialization_task_id", "")): task
        for task in tasks
        if str(task.get("materialization_task_id", ""))
    }
    issues: list[dict[str, Any]] = []
    if shard_id:
        shards = _dict_list(plan.get("shards"))
        shard = next(
            (
                candidate
                for candidate in shards
                if str(candidate.get("shard_id", "")) == shard_id
            ),
            None,
        )
        if shard is None:
            return [], [
                {
                    "code": "unknown_source_workspace_shard",
                    "message": f"Source workspace materialization shard not found: {shard_id}",
                    "severity": "error",
                }
            ]
        selected = []
        missing_ids = []
        for task_id in _string_list(shard.get("materialization_task_ids")):
            task = task_by_id.get(task_id)
            if task is None:
                missing_ids.append(task_id)
            else:
                selected.append(task)
        if missing_ids:
            issues.append(
                {
                    "code": "shard_references_missing_materialization_tasks",
                    "message": (
                        "Shard references missing materialization task IDs: "
                        + ", ".join(missing_ids)
                    ),
                    "severity": "error",
                }
            )
        return selected, issues

    offset = int(task_offset)
    if offset < 0:
        issues.append(
            {
                "code": "invalid_materialization_task_offset",
                "message": "Source workspace materialization task offset cannot be negative.",
                "severity": "error",
            }
        )
        offset = 0
    if max_tasks is None:
        return tasks[offset:], issues
    limit = int(max_tasks)
    if limit <= 0:
        issues.append(
            {
                "code": "invalid_materialization_max_tasks",
                "message": "Source workspace materialization max tasks must be positive.",
                "severity": "error",
            }
        )
        return [], issues
    return tasks[offset : offset + limit], issues


def _materialize_source_workspace_task(
    task: dict[str, Any],
    *,
    workspace_root: Path,
    timeout_seconds: float,
    dry_run: bool,
) -> dict[str, Any]:
    materialization_task_id = str(task.get("materialization_task_id", ""))
    source_uri = str(task.get("source_uri", "")).strip()
    source_revision = str(task.get("source_revision", "")).strip()
    cache_path = Path(str(task.get("cache_path", "")))
    planned_file_source_uri = str(task.get("planned_file_source_uri", "")).strip()
    result = {
        "materialization_task_id": materialization_task_id,
        "repository": str(task.get("repository", "")),
        "source_uri": source_uri,
        "source_revision": source_revision,
        "cache_path": str(cache_path),
        "planned_file_source_uri": planned_file_source_uri,
        "source_instance_ids": _string_list(task.get("source_instance_ids")),
        "dry_run": bool(dry_run),
        "commands": [],
        "error": "",
        "valid": True,
    }
    validation_error = _source_workspace_task_validation_error(
        task,
        workspace_root=workspace_root,
        cache_path=cache_path,
        source_uri=source_uri,
        source_revision=source_revision,
        planned_file_source_uri=planned_file_source_uri,
    )
    if validation_error:
        result["error"] = validation_error
        result["valid"] = False
        return result
    if dry_run:
        return result

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            if not (cache_path / ".git").is_dir():
                raise RuntimeError(f"Cache path is not a Git repository: {cache_path}")
        else:
            _run_checked_materialization_command(
                result,
                "clone",
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    source_uri,
                    str(cache_path),
                ],
                timeout_seconds=timeout_seconds,
            )
        _run_checked_materialization_command(
            result,
            "fetch_revision",
            [
                "git",
                "-C",
                str(cache_path),
                "fetch",
                "--filter=blob:none",
                "origin",
                source_revision,
            ],
            timeout_seconds=timeout_seconds,
        )
        _run_checked_materialization_command(
            result,
            "checkout",
            [
                "git",
                "-C",
                str(cache_path),
                "checkout",
                "--detach",
                source_revision,
            ],
            timeout_seconds=timeout_seconds,
        )
        _run_checked_materialization_command(
            result,
            "verify_revision",
            [
                "git",
                "-C",
                str(cache_path),
                "cat-file",
                "-e",
                f"{source_revision}^{{commit}}",
            ],
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - convert task failure into run summary.
        result["error"] = str(exc)
        result["valid"] = False
    return result


def _source_workspace_task_validation_error(
    task: dict[str, Any],
    *,
    workspace_root: Path,
    cache_path: Path,
    source_uri: str,
    source_revision: str,
    planned_file_source_uri: str,
) -> str:
    if not str(task.get("materialization_task_id", "")).strip():
        return "Materialization task is missing materialization_task_id."
    if not str(workspace_root).strip():
        return "Materialization plan is missing workspace_root."
    if not str(task.get("cache_path", "")).strip():
        return "Materialization task is missing cache_path."
    if not _string_list(task.get("source_instance_ids")):
        return "Materialization task is missing source_instance_ids."
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_revision):
        return "Materialization task source_revision is not a fixed 40-character SHA."
    if not (
        source_uri.startswith("https://github.com/")
        or source_uri.startswith("file://")
    ):
        return "Materialization task source_uri must be a public GitHub or file URI."
    if not planned_file_source_uri.startswith("file://"):
        return "Materialization task planned_file_source_uri must be a file URI."
    try:
        cache_path.resolve().relative_to(workspace_root.resolve())
    except ValueError:
        return "Materialization task cache_path is outside workspace_root."
    if Path(planned_file_source_uri[7:]).resolve() != cache_path.resolve():
        return "Materialization task planned_file_source_uri does not match cache_path."
    return ""


def _run_checked_materialization_command(
    task_result: dict[str, Any],
    label: str,
    arguments: list[str],
    *,
    timeout_seconds: float,
) -> None:
    command_result = _run_materialization_command(
        label,
        arguments,
        timeout_seconds=timeout_seconds,
    )
    task_result["commands"].append(command_result)
    if command_result["exit_code"] != 0:
        raise RuntimeError(
            f"Source workspace materialization command failed: {label} "
            f"(exit={command_result['exit_code']})"
        )


def _run_materialization_command(
    label: str,
    arguments: list[str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    completed = subprocess.run(
        arguments,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    result = {
        "label": label,
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "stdout_bytes": len(completed.stdout.encode("utf-8")),
        "stderr_bytes": len(completed.stderr.encode("utf-8")),
    }
    return result


def _source_type_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts[_curation_source_type(record)] += 1
    counts.pop("", None)
    return dict(sorted(counts.items()))


def _repository_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        repository = str(record.get("repository", "")).strip()
        if repository:
            counts[repository] += 1
    return dict(sorted(counts.items()))


def _hidden_test_patch_curation_record_errors(
    record: dict[str, Any],
    task: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_source_instance_id = str(task.get("source_instance_id", "")).strip()
    source_instance_id = str(record.get("source_instance_id", "")).strip()
    if not source_instance_id:
        errors.append("Curation record is missing source_instance_id.")
    elif source_instance_id != expected_source_instance_id:
        errors.append("Curation record source_instance_id does not match the plan task.")
    oracle_fields = _present_oracle_fields(record)
    if oracle_fields:
        errors.append("Curation record contains oracle fields: " + ", ".join(oracle_fields))

    summary_error = _validated_curation_text(
        record.get("public_behavior_summary"),
        field_name="public_behavior_summary",
        reject_hidden_terms=True,
    )
    if summary_error:
        errors.append(summary_error)
    test_patch_error = _validated_hidden_test_patch_text(
        record.get("hidden_test_patch")
    )
    if test_patch_error:
        errors.append(test_patch_error)
    command_validation = _validated_curation_command_list(
        record.get("hidden_test_commands"),
        field_name="hidden_test_commands",
    )
    if isinstance(command_validation, str):
        errors.append(command_validation)
    elif not command_validation:
        errors.append("hidden_test_commands must contain at least one command.")
    notes_error = _validated_curation_text(
        record.get("withheld_evaluator_notes"),
        field_name="withheld_evaluator_notes",
        reject_hidden_terms=False,
    )
    if notes_error:
        errors.append(notes_error)
    return errors


def _validated_curation_text(
    value: Any,
    *,
    field_name: str,
    reject_hidden_terms: bool,
) -> str:
    if not isinstance(value, str) or not value.strip():
        return f"{field_name} must be a non-empty string."
    if "\x00" in value:
        return f"{field_name} must not contain NUL bytes."
    lowered = value.lower()
    if "fail_to_pass" in lowered or "pass_to_pass" in lowered:
        return f"{field_name} must not reference benchmark oracle fields."
    if reject_hidden_terms and (
        "test_patch" in lowered
        or "hidden_test_patch" in lowered
        or "reference_patch" in lowered
    ):
        return f"{field_name} must not describe hidden patch artifacts."
    return ""


def _validated_hidden_test_patch_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "hidden_test_patch must be a non-empty unified diff string."
    if "\x00" in value:
        return "hidden_test_patch must not contain NUL bytes."
    lowered = value.lower()
    if "fail_to_pass" in lowered or "pass_to_pass" in lowered:
        return "hidden_test_patch must not reference benchmark oracle fields."
    if "diff --git " not in value:
        return "hidden_test_patch must contain a unified diff header."
    return ""


def _apply_hidden_test_patch_curation_to_source_record(
    source_record: dict[str, Any],
    curation_record: dict[str, Any],
    *,
    task_id: str,
) -> dict[str, Any]:
    command_validation = _validated_curation_command_list(
        curation_record.get("hidden_test_commands"),
        field_name="hidden_test_commands",
    )
    if isinstance(command_validation, str):
        raise ValueError("Validated hidden-test-patch curation record became invalid.")
    hidden_test_patch = str(curation_record.get("hidden_test_patch", ""))
    rewritten = dict(source_record)
    rewritten["test_patch"] = hidden_test_patch
    rewritten["test_commands"] = command_validation
    rewritten["candidate_verifier"] = {
        "type": "curated_hidden_test_patch",
        "commands": command_validation,
    }
    verifier_types = set(_optional_string_list(rewritten.get("verifier_types")))
    verifier_types.add("hidden_test_patch")
    rewritten["verifier_types"] = sorted(verifier_types)
    rewritten["hidden_test_patch_curation"] = {
        "curation_task_id": task_id,
        "curation_record_sha256": _stable_json_sha256(curation_record),
        "hidden_test_patch_sha256": hashlib.sha256(
            hidden_test_patch.encode("utf-8")
        ).hexdigest(),
        "hidden_test_commands": len(command_validation),
        "public_behavior_summary": str(
            curation_record.get("public_behavior_summary", "")
        ).strip(),
    }
    return rewritten


def _hidden_command_curation_record_errors(
    record: dict[str, Any],
    task: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_source_instance_id = str(task.get("source_instance_id", "")).strip()
    source_instance_id = str(record.get("source_instance_id", "")).strip()
    if not source_instance_id:
        errors.append("Curation record is missing source_instance_id.")
    elif source_instance_id != expected_source_instance_id:
        errors.append("Curation record source_instance_id does not match the plan task.")
    oracle_fields = _present_oracle_fields(record)
    if oracle_fields:
        errors.append("Curation record contains oracle fields: " + ", ".join(oracle_fields))

    hidden_commands = _validated_curation_command_list(
        record.get("curated_hidden_commands"),
        field_name="curated_hidden_commands",
    )
    setup_commands = _validated_curation_command_list(
        record.get("curated_setup_commands"),
        field_name="curated_setup_commands",
        allow_empty=True,
    )
    if isinstance(hidden_commands, str):
        errors.append(hidden_commands)
        hidden_command_list: list[str] = []
    else:
        hidden_command_list = hidden_commands
        if not hidden_command_list:
            errors.append("curated_hidden_commands must contain at least one command.")
    if isinstance(setup_commands, str):
        errors.append(setup_commands)

    command_runtime = str(record.get("command_runtime", "")).strip()
    if not command_runtime:
        errors.append("command_runtime is required.")
    expected_runtime = record.get("expected_runtime_seconds")
    if not isinstance(expected_runtime, int | float) or expected_runtime <= 0:
        errors.append("expected_runtime_seconds must be a positive number.")

    failed_command_hashes = {
        str(result.get("command_sha256", ""))
        for result in _dict_list(
            _dict(task.get("observed_failure")).get("command_results")
        )
        if _int(result.get("exit_code"), default=0) != 0
    }
    repeated_failed_commands = [
        command
        for command in hidden_command_list
        if hashlib.sha256(command.encode("utf-8")).hexdigest() in failed_command_hashes
    ]
    if repeated_failed_commands:
        errors.append(
            "curated_hidden_commands repeat commands that already failed in rehearsal."
        )
    return errors


def _validated_curation_command_list(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> list[str] | str:
    if value is None:
        return [] if allow_empty else f"{field_name} is required."
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return f"{field_name} must be a list of command strings."
    commands = []
    for index, command in enumerate(value):
        stripped = command.strip()
        if not stripped:
            return f"{field_name}[{index}] cannot be empty."
        if any(character in stripped for character in ("\x00", "\n", "\r")):
            return f"{field_name}[{index}] must be a single-line command."
        lowered = stripped.lower()
        if "fail_to_pass" in lowered or "pass_to_pass" in lowered:
            return f"{field_name}[{index}] must not reference benchmark oracle fields."
        if "test_patch" in lowered or "reference_patch" in lowered:
            return f"{field_name}[{index}] must not reference hidden patch artifacts."
        try:
            if not shlex.split(stripped):
                return f"{field_name}[{index}] cannot be empty."
        except ValueError as exc:
            return f"{field_name}[{index}] is not a valid shlex command: {exc}."
        commands.append(stripped)
    return commands


def _apply_hidden_command_curation_to_source_record(
    source_record: dict[str, Any],
    curation_record: dict[str, Any],
    *,
    task_id: str,
) -> dict[str, Any]:
    hidden_commands = _validated_curation_command_list(
        curation_record.get("curated_hidden_commands"),
        field_name="curated_hidden_commands",
    )
    setup_commands = _validated_curation_command_list(
        curation_record.get("curated_setup_commands"),
        field_name="curated_setup_commands",
        allow_empty=True,
    )
    if isinstance(hidden_commands, str) or isinstance(setup_commands, str):
        raise ValueError("Validated hidden-command curation record became invalid.")
    source_type = _curation_source_type(source_record)
    rewritten = dict(source_record)
    rewritten["setup_commands"] = setup_commands
    if source_type == "public_ci":
        rewritten["ci_commands"] = hidden_commands
    else:
        rewritten["test_commands"] = hidden_commands
    rewritten["candidate_verifier"] = {
        "type": "curated_hidden_commands",
        "commands": hidden_commands,
    }
    verifier_types = set(_optional_string_list(rewritten.get("verifier_types")))
    verifier_types.add("hidden_command")
    rewritten["verifier_types"] = sorted(verifier_types)
    rewritten["hidden_command_curation"] = {
        "curation_task_id": task_id,
        "curation_record_sha256": _stable_json_sha256(curation_record),
        "command_runtime": str(curation_record.get("command_runtime", "")).strip(),
        "expected_runtime_seconds": float(
            curation_record.get("expected_runtime_seconds", 0)
        ),
    }
    return rewritten


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "source"


def _hidden_command_curation_task(
    record: dict[str, Any],
    *,
    task_index: int,
    failure_evidence: dict[str, Any],
) -> dict[str, Any]:
    source_type = _curation_source_type(record)
    digest = _stable_json_sha256(
        {
            "repository": str(record.get("repository", "")),
            "source_instance_id": str(record.get("source_instance_id", "")),
            "source_revision": str(record.get("source_revision", "")),
            "source_type": source_type,
            "curation": "hidden_command",
        }
    )[:16]
    return {
        "curation_task_id": f"hidden-command-curation-{digest}",
        "task_index": task_index,
        "source_type": source_type,
        "repository": str(record.get("repository", "")),
        "source_uri": str(record.get("source_uri", "")),
        "workspace_original_source_uri": str(
            record.get("workspace_original_source_uri", "")
        ),
        "source_revision": str(record.get("source_revision", "")),
        "source_instance_id": str(record.get("source_instance_id", "")),
        "source_url": str(record.get("source_url", "")),
        "title": str(record.get("title", "")),
        "language": _normalize_label(record.get("language")),
        "license": str(record.get("license", "")),
        "task_family": _normalize_label(record.get("task_family")),
        "workspace_materialized": bool(record.get("workspace_materialized", False)),
        "workspace_cache_path": str(record.get("workspace_cache_path", "")),
        "current_setup_commands": _optional_string_list(record.get("setup_commands")),
        "current_candidate_verifier_commands": _candidate_verifier_commands(record),
        "observed_failure": failure_evidence,
        "required_curation_fields": [
            "curated_setup_commands",
            "curated_hidden_commands",
            "command_runtime",
            "expected_runtime_seconds",
            "withheld_curation_notes",
        ],
        "curation_output_contract": {
            "curated_setup_commands": (
                "Setup commands needed before hidden commands, stored outside public prompts."
            ),
            "curated_hidden_commands": (
                "Stable commands that exercise the intended behavior in the fixed workspace."
            ),
            "command_runtime": "Host, Docker image, or sandbox profile required to run commands.",
            "withheld_curation_notes": (
                "Auditor notes about command choice and failures, never for generation prompts."
            ),
        },
        "leakage_constraints": [
            "Do not use benchmark oracle patches or benchmark test_patch fields.",
            (
                "Do not expose hidden commands, setup commands, or failure evidence "
                "to the rollout model."
            ),
            "Keep public query text separate from withheld verifier commands and setup details.",
            "Prefer narrow repository-specific commands over broad full-suite commands.",
        ],
    }


def _hidden_test_patch_curation_task(
    record: dict[str, Any],
    *,
    task_index: int,
    leakage_constraints: list[str],
) -> dict[str, Any]:
    source_type = _curation_source_type(record)
    digest = _stable_json_sha256(
        {
            "repository": str(record.get("repository", "")),
            "source_instance_id": str(record.get("source_instance_id", "")),
            "source_revision": str(record.get("source_revision", "")),
            "source_type": source_type,
        }
    )[:16]
    return {
        "curation_task_id": f"hidden-test-patch-curation-{digest}",
        "task_index": task_index,
        "source_type": source_type,
        "repository": str(record.get("repository", "")),
        "source_uri": str(record.get("source_uri", "")),
        "source_revision": str(record.get("source_revision", "")),
        "source_instance_id": str(record.get("source_instance_id", "")),
        "source_url": str(record.get("source_url", "")),
        "title": str(record.get("title", "")),
        "labels": _optional_string_list(record.get("labels")),
        "language": _normalize_label(record.get("language")),
        "license": str(record.get("license", "")),
        "candidate_verifier_commands": _candidate_verifier_commands(record),
        "public_behavior_summary_required": True,
        "required_curation_fields": [
            "public_behavior_summary",
            "hidden_test_patch",
            "hidden_test_commands",
            "withheld_evaluator_notes",
        ],
        "curation_output_contract": {
            "hidden_test_patch": "Unified diff adding tests only, stored outside public prompts.",
            "hidden_test_commands": "Commands that exercise the hidden test patch.",
            "withheld_evaluator_notes": (
                "Notes for auditors and verifiers, never for generation prompts."
            ),
        },
        "leakage_constraints": _dedupe_strings(
            [
                *leakage_constraints,
                "Do not use benchmark oracle patches or benchmark test_patch fields.",
                "Do not expose hidden tests to the task-generation or rollout model.",
                "Store hidden evaluator payloads separately from public query text.",
            ]
        ),
    }


def _curation_source_type(record: dict[str, Any]) -> str:
    raw_type = _normalize_label(
        record.get("source_type")
        or record.get("type")
        or record.get("source_format")
        or record.get("collection_source")
    )
    if raw_type in {"issue", "issues", "public_issue"}:
        return "public_issue"
    if raw_type in {"pr", "prs", "pull_request", "pull_requests", "public_pr"}:
        return "public_pr"
    if raw_type in {"ci", "ci_failure", "ci_failures", "public_ci"}:
        return "public_ci"
    return raw_type


def _record_has_benchmark_source(record: dict[str, Any]) -> bool:
    benchmark_aliases = {_normalize_label(alias) for alias in DEFAULT_BENCHMARK_SOURCE_ALIASES}
    candidates = [
        record.get("source_name"),
        record.get("source_format"),
        record.get("dataset"),
        record.get("benchmark"),
        record.get("source"),
    ]
    metadata = _dict(record.get("metadata"))
    candidates.extend(
        [
            metadata.get("source_name"),
            metadata.get("source_format"),
            metadata.get("dataset"),
            metadata.get("benchmark"),
        ]
    )
    candidates.extend(_optional_string_list(record.get("contamination_tags")))
    for value in candidates:
        normalized = _normalize_label(value)
        if normalized in benchmark_aliases or "swe_bench" in normalized:
            return True
    return False


def _present_oracle_fields(record: dict[str, Any]) -> list[str]:
    oracle_fields = (
        "patch",
        "test_patch",
        "reference_patch",
        "oracle_patch",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "fail_to_pass",
        "pass_to_pass",
    )
    return sorted(field_name for field_name in oracle_fields if field_name in record)


def _candidate_verifier_commands(record: dict[str, Any]) -> list[str]:
    verifier = _dict(record.get("candidate_verifier"))
    commands = _optional_string_list(verifier.get("commands"))
    if commands:
        return commands
    return _optional_string_list(record.get("test_commands")) or _optional_string_list(
        record.get("ci_commands")
    )


def _optional_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _repository_synthetic_family_slots(
    selection_plan: dict[str, Any],
    backfill_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    slots = []
    seen: set[str] = set()
    for slot in _dict_list(_dict(selection_plan.get("reserved_backfill")).get("slots")):
        if slot.get("type") != "task_family_minimum":
            continue
        family = _normalize_label(slot.get("target"))
        if not family or family == "bug_repair":
            continue
        if family not in TASK_FAMILY_VERIFIER_TEMPLATES:
            continue
        amount = max(0, int(slot.get("minimum_count", 0) or 0))
        if amount <= 0:
            continue
        slots.append(
            {
                "task_family": family,
                "minimum_count": amount,
                "accepted_verifier_types": slot.get("accepted_verifier_types", []),
                "source": "selection_plan_reserved_backfill",
            }
        )
        seen.add(family)
    if slots:
        return sorted(slots, key=lambda item: item["task_family"])

    for gap in _dict_list(_dict(backfill_plan.get("gaps")).get("task_family")):
        family = _normalize_label(gap.get("target"))
        if not family or family == "bug_repair" or family in seen:
            continue
        if family not in TASK_FAMILY_VERIFIER_TEMPLATES:
            continue
        amount = max(0, int(gap.get("shortfall", 0) or 0))
        if amount <= 0:
            continue
        slots.append(
            {
                "task_family": family,
                "minimum_count": amount,
                "accepted_verifier_types": gap.get("accepted_verifier_types", []),
                "source": "backfill_plan_task_family_gap",
            }
        )
    return sorted(slots, key=lambda item: item["task_family"])


def _scenario_repository_snapshots(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario in scenarios:
        seed = scenario.query_seed
        environment = scenario.environment
        repository = _seed_repository_value(seed)
        revision = str(environment.source_revision or "").strip().lower()
        if not repository or not re.fullmatch(r"[0-9a-f]{40}", revision):
            continue
        source_uri = str(environment.source_uri or "").strip()
        if not source_uri:
            continue
        key = (repository, revision)
        snapshot = by_key.setdefault(
            key,
            {
                "repository": repository,
                "source_uri": source_uri,
                "source_revision": revision,
                "license": seed.license,
                "language": _seed_language_value(seed),
                "image_digest": environment.image_digest,
                "working_directory": environment.working_directory,
                "setup_commands": list(environment.setup_commands),
                "health_check": list(environment.health_check),
                "hidden_tests": [],
                "source_instance_ids": [],
                "source_urls": [],
            },
        )
        _extend_unique(snapshot["hidden_tests"], scenario.hidden_evaluator.hidden_tests)
        source_instance = seed.public.context.get("source_instance_id") or seed.metadata.get(
            "source_instance_id",
            "",
        )
        source_url = seed.public.context.get("source_url") or seed.metadata.get("source_url", "")
        if source_instance:
            _extend_unique(snapshot["source_instance_ids"], [str(source_instance)])
        if source_url:
            _extend_unique(snapshot["source_urls"], [str(source_url)])
    snapshots = list(by_key.values())
    for snapshot in snapshots:
        snapshot["hidden_tests"] = sorted(snapshot["hidden_tests"])
        snapshot["source_instance_ids"] = sorted(snapshot["source_instance_ids"])[:5]
        snapshot["source_urls"] = sorted(snapshot["source_urls"])[:5]
    return sorted(
        snapshots,
        key=lambda item: (
            item["repository"],
            item["source_revision"],
        ),
    )


def _build_synthetic_backfill_specs(
    slots: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ready_specs: list[dict[str, Any]] = []
    draft_specs: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    if not snapshots:
        for slot in slots:
            family = str(slot["task_family"])
            report[family] = {
                "planned": int(slot["minimum_count"]),
                "generator_ready": 0,
                "draft": int(slot["minimum_count"]),
                "missing_evidence": ["fixed_repository_snapshot"],
            }
        return ready_specs, draft_specs, report

    for slot in slots:
        family = str(slot["task_family"])
        count = int(slot["minimum_count"])
        ready_targets_by_repo: dict[str, list[dict[str, Any]]] = {}
        draft_targets_by_repo: dict[str, list[dict[str, Any]]] = {}
        missing: Counter[str] = Counter()
        for index in range(count):
            snapshot = snapshots[index % len(snapshots)]
            target, target_missing = _synthetic_backfill_target(
                snapshot,
                family=family,
                index=index,
            )
            spec_key = _snapshot_key(snapshot)
            if target_missing:
                missing.update(target_missing)
                draft_targets_by_repo.setdefault(spec_key, []).append(target)
            else:
                ready_targets_by_repo.setdefault(spec_key, []).append(target)
        ready_specs.extend(
            _specs_for_targets(family, snapshots, ready_targets_by_repo, generator_ready=True)
        )
        draft_specs.extend(
            _specs_for_targets(family, snapshots, draft_targets_by_repo, generator_ready=False)
        )
        report[family] = {
            "planned": count,
            "generator_ready": sum(len(targets) for targets in ready_targets_by_repo.values()),
            "draft": sum(len(targets) for targets in draft_targets_by_repo.values()),
            "missing_evidence": dict(sorted(missing.items())),
            "accepted_verifier_types": slot.get("accepted_verifier_types", []),
        }
    return ready_specs, draft_specs, report


def _synthetic_backfill_target(
    snapshot: dict[str, Any],
    *,
    family: str,
    index: int,
) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    paths = _synthetic_backfill_paths(snapshot, family)
    hidden_tests = list(snapshot.get("hidden_tests", []))
    target = {
        "name": f"{family}-{index:04d}-{snapshot['repository'].replace('/', '__')}",
        "paths": paths,
        "source_instances": list(snapshot.get("source_instance_ids", [])),
        "source_urls": list(snapshot.get("source_urls", [])),
        "difficulty": 3,
    }
    if hidden_tests:
        target["test_commands"] = hidden_tests
        target["ci_commands"] = hidden_tests
        target["build_commands"] = hidden_tests
        target["migration_commands"] = hidden_tests
        target["adversarial_tests"] = hidden_tests if family == "security_hardening" else []
    else:
        missing.append("hidden_command")

    if family == "docs_examples":
        missing.append("doctest_or_example_command")
    elif family == "performance":
        missing.append("benchmark_command_or_performance_threshold")
    elif family == "repo_understanding":
        target["retrieval_requirements"] = [
            f"cite {path}" for path in paths if path
        ] or ["cite inspected repository files"]
        target["trace_quality_rubric"] = [
            "The trace must cite inspected repository files before answering.",
            "The final answer must distinguish observed files from assumptions.",
        ]
    elif family == "code_review":
        target["diff_constraints"] = [
            "address only the requested review scope",
            "avoid unrelated formatting or dependency changes",
        ]
    elif family == "refactor":
        target["forbidden_state"] = {
            "forbidden_regex": ["public API rename without compatibility shim"]
        }
    return target, missing


def _synthetic_backfill_paths(snapshot: dict[str, Any], family: str) -> list[str]:
    repository = str(snapshot.get("repository", ""))
    default_paths = {
        "docs_examples": ["README.md", "docs/"],
        "performance": ["src/", "tests/", "benchmarks/"],
        "repo_understanding": ["README.md", "pyproject.toml"],
        "test_authoring": ["tests/"],
        "refactor": ["src/", "tests/"],
        "migration": ["pyproject.toml", "src/", "tests/"],
        "security_hardening": ["src/", "tests/"],
        "code_review": ["src/", "tests/"],
    }
    paths = list(default_paths.get(family, ["src/", "tests/"]))
    if repository.endswith("/rich") and family == "docs_examples":
        paths = ["README.md", "docs/"]
    return paths


def _specs_for_targets(
    family: str,
    snapshots: list[dict[str, Any]],
    targets_by_repo: dict[str, list[dict[str, Any]]],
    *,
    generator_ready: bool,
) -> list[dict[str, Any]]:
    snapshots_by_key = {_snapshot_key(snapshot): snapshot for snapshot in snapshots}
    specs = []
    for key, targets in sorted(targets_by_repo.items()):
        snapshot = snapshots_by_key[key]
        specs.append(
            {
                "repository": snapshot["repository"],
                "source_uri": snapshot["source_uri"],
                "source_revision": snapshot["source_revision"],
                "license": snapshot["license"],
                "language": snapshot["language"],
                "image_digest": snapshot["image_digest"],
                "working_directory": snapshot["working_directory"],
                "setup_commands": snapshot["setup_commands"],
                "health_check": snapshot["health_check"],
                "task_families": [family],
                "generator_ready": generator_ready,
                "targets": targets,
            }
        )
    return specs


def _synthetic_draft_specs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    draft_container = _dict(plan.get("draft_specs"))
    if draft_container:
        return _dict_list(draft_container.get("repositories"))
    if plan.get("repositories") is not None:
        return _dict_list(plan.get("repositories"))
    if plan.get("records") is not None:
        return _dict_list(plan.get("records"))
    if plan.get("specs") is not None:
        return _dict_list(plan.get("specs"))
    return []


def _synthetic_target_evidence_requirements(
    spec: dict[str, Any],
    target: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    template = TASK_FAMILY_VERIFIER_TEMPLATES.get(family)
    accepted = list(template.accepted_verifier_types) if template is not None else []
    missing: list[str] = []
    required_fields: list[str] = []
    examples: list[dict[str, str]] = []
    if family == "docs_examples" and not (
        _combined_string_list(spec, target, "doctest_commands")
        or _combined_string_list(spec, target, "example_commands")
    ):
        missing.append("doctest_or_example_command")
        required_fields.append("doctest_commands or example_commands")
        examples.extend(
            [
                {
                    "field": "doctest_commands",
                    "example": "python -m doctest README.md",
                },
                {
                    "field": "example_commands",
                    "example": "python examples/example_name.py",
                },
            ]
        )
    elif family == "performance" and not (
        _combined_string_list(spec, target, "benchmark_commands")
        or _combined_dict(spec, target, "performance_threshold")
    ):
        missing.append("benchmark_command_or_performance_threshold")
        required_fields.append("benchmark_commands or performance_threshold")
        examples.extend(
            [
                {
                    "field": "benchmark_commands",
                    "example": "python benchmarks/bench_name.py --max-ms 50",
                },
                {
                    "field": "performance_threshold",
                    "example": "{\"max_ms\": 50}",
                },
            ]
        )
    return {
        "accepted_verifier_types": accepted,
        "missing_evidence": missing,
        "required_fields": required_fields,
        "suggested_field_examples": examples,
    }


def _merge_synthetic_evidence_record(
    target: dict[str, Any],
    record: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    merged = dict(target)
    if family == "docs_examples":
        doctests = _string_list(record.get("doctest_commands"))
        examples = _string_list(record.get("example_commands"))
        if doctests:
            merged["doctest_commands"] = doctests
        if examples:
            merged["example_commands"] = examples
    elif family == "performance":
        benchmarks = _string_list(record.get("benchmark_commands"))
        threshold = _dict(record.get("performance_threshold"))
        if benchmarks:
            merged["benchmark_commands"] = benchmarks
        if threshold:
            merged["performance_threshold"] = threshold
    return merged


def _combined_string_list(
    spec: dict[str, Any],
    target: dict[str, Any],
    key: str,
) -> list[str]:
    values: list[str] = []
    _extend_unique(values, _string_list(spec.get(key)))
    _extend_unique(values, _string_list(target.get(key)))
    return values


def _combined_dict(spec: dict[str, Any], target: dict[str, Any], key: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    values.update(_dict(spec.get(key)))
    values.update(_dict(target.get(key)))
    return values


def _evidence_task_id(spec: dict[str, Any], target: dict[str, Any], family: str) -> str:
    digest = _stable_json_sha256(
        {
            "repository": spec.get("repository", ""),
            "source_revision": spec.get("source_revision", ""),
            "task_family": family,
            "target_name": target.get("name", ""),
            "paths": _string_list(target.get("paths")),
        }
    )
    return f"evidence_{digest[:16]}"


def _evidence_backfill_gap_summary(backfill_plan: dict[str, Any]) -> dict[str, Any]:
    gaps = _dict(backfill_plan.get("gaps"))
    dominance = _dict(gaps.get("dominance"))
    return {
        "train_eligible_shortfall": _int(
            _dict(gaps.get("train_eligible")).get("shortfall"),
            default=0,
        ),
        "task_family_shortfalls": _gap_shortfalls(gaps.get("task_family")),
        "verifier_type_shortfalls": _gap_shortfalls(gaps.get("verifier_type")),
        "source_method_shortfalls": _gap_shortfalls(gaps.get("source_method")),
        "language_dominance": _dict_list(dominance.get("language")),
        "repository_dominance": _dict_list(dominance.get("repository")),
    }


def _gap_shortfalls(value: Any) -> dict[str, int]:
    shortfalls: Counter[str] = Counter()
    for gap in _dict_list(value):
        target = _normalize_label(gap.get("target"))
        if target:
            shortfalls[target] += _int(gap.get("shortfall"), default=0)
    return dict(sorted(shortfalls.items()))


def _synthetic_evidence_next_actions(
    evidence_tasks: list[dict[str, Any]],
    backfill_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    family_counts: Counter[str] = Counter(
        str(task.get("task_family", "")) for task in evidence_tasks
    )
    actions: list[dict[str, Any]] = []
    if family_counts.get("docs_examples", 0):
        actions.append(
            {
                "action": "add_docs_example_evidence",
                "target": "docs_examples",
                "minimum_count": family_counts["docs_examples"],
                "required_fields": ["doctest_commands", "example_commands"],
            }
        )
    if family_counts.get("performance", 0):
        actions.append(
            {
                "action": "add_performance_evidence",
                "target": "performance",
                "minimum_count": family_counts["performance"],
                "required_fields": ["benchmark_commands", "performance_threshold"],
            }
        )
    for verifier, shortfall in _dict(
        backfill_summary.get("verifier_type_shortfalls")
    ).items():
        actions.append(
            {
                "action": "cover_verifier_type_gap",
                "target": verifier,
                "minimum_count": int(shortfall),
            }
        )
    if backfill_summary.get("language_dominance"):
        actions.append(
            {
                "action": "add_cross_language_sources_or_downsample",
                "target": "python",
                "minimum_count": int(
                    backfill_summary["language_dominance"][0].get(
                        "additional_non_target_if_no_downsampling",
                        0,
                    )
                ),
            }
        )
    if backfill_summary.get("repository_dominance"):
        first = backfill_summary["repository_dominance"][0]
        actions.append(
            {
                "action": "balance_or_sample_repository",
                "target": str(first.get("target", "")),
                "minimum_count": int(
                    first.get("additional_non_target_if_no_downsampling", 0)
                ),
            }
        )
    return actions


def _task_count(tasks: Iterable[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task in tasks:
        value = _normalize_label(task.get(field_name))
        if value:
            counts[value] += 1
    return dict(sorted(counts.items()))


def _multi_value_task_count(tasks: Iterable[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task in tasks:
        counts.update(_normalize_label(value) for value in _string_list(task.get(field_name)))
    counts.pop("", None)
    return dict(sorted(counts.items()))


def _record_template_output(records_output: str, shard_id: str) -> str:
    if not records_output:
        return f"{shard_id}-evidence-record-template.json"
    path = Path(records_output)
    if path.name.endswith("-evidence-records.json"):
        return str(
            path.with_name(
                path.name.replace(
                    "-evidence-records.json",
                    "-evidence-record-template.json",
                )
            )
        )
    suffix = path.suffix or ".json"
    return str(path.with_name(f"{path.stem}-template{suffix}"))


def _synthetic_evidence_record_template(task: dict[str, Any]) -> dict[str, Any]:
    family = _normalize_label(task.get("task_family"))
    record = {
        "evidence_task_id": str(task.get("evidence_task_id", "")),
        "task_family": family,
        "repository": str(task.get("repository", "")),
        "source_uri": str(task.get("source_uri", "")),
        "source_revision": str(task.get("source_revision", "")),
        "language": _normalize_label(task.get("language")),
        "target_name": str(task.get("target_name", "")),
        "paths": _string_list(task.get("paths")),
        "source_instances": _string_list(task.get("source_instances")),
        "source_urls": _string_list(task.get("source_urls")),
        "missing_evidence": _string_list(task.get("missing_evidence")),
        "required_fields": _string_list(task.get("required_fields")),
        "suggested_field_examples": _dict_list(task.get("suggested_field_examples")),
        "collector_notes": "",
    }
    if family == "docs_examples":
        record["doctest_commands"] = []
        record["example_commands"] = []
    elif family == "performance":
        record["benchmark_commands"] = []
        record["performance_threshold"] = {}
    return record


def _synthetic_evidence_shard_status(
    shard: dict[str, Any],
    shard_index: int,
) -> dict[str, Any]:
    shard_id = str(
        shard.get("shard_id") or f"synthetic-evidence-shard-{shard_index:04d}"
    )
    records_output = str(shard.get("records_output", ""))
    record_template_output = str(
        shard.get("record_template_output")
        or _record_template_output(records_output, shard_id)
    )
    template_status, _, template_issue = _read_json_artifact(
        record_template_output
    )
    records_status, records_payload, records_issue = _read_json_artifact(records_output)
    issues = _dedupe_strings([template_issue, records_issue])
    record_count = 0
    expected_ids = set(_string_list(shard.get("evidence_task_ids")))
    matching_record_count = 0
    missing_record_task_ids = sorted(expected_ids)
    extra_record_task_ids: list[str] = []
    if records_status == "missing":
        if template_status == "invalid":
            status = "blocked"
            next_action = "fix_evidence_template"
        else:
            status = "pending"
            next_action = (
                "fill_evidence_template"
                if template_status == "present"
                else "collect_evidence"
            )
        apply_status = "not_checked"
        spec_status = "not_checked"
        apply_ready_records = 0
        remaining_evidence_tasks = _int(shard.get("selected_tasks"), default=0)
    elif records_status == "invalid":
        status = "blocked"
        next_action = "fix_evidence_records"
        apply_status = "not_checked"
        spec_status = "not_checked"
        apply_ready_records = 0
        remaining_evidence_tasks = _int(shard.get("selected_tasks"), default=0)
    else:
        records, records_error = _synthetic_evidence_records_from_payload(records_payload)
        issues = _dedupe_strings([*issues, records_error])
        if records_error:
            status = "blocked"
            next_action = "fix_evidence_records"
            apply_status = "not_checked"
            spec_status = "not_checked"
            apply_ready_records = 0
            remaining_evidence_tasks = _int(shard.get("selected_tasks"), default=0)
        else:
            record_ids = {str(record.get("evidence_task_id", "")) for record in records}
            record_ids.discard("")
            record_count = len(records)
            matching_record_count = len(record_ids & expected_ids)
            missing_record_task_ids = sorted(expected_ids - record_ids)
            extra_record_task_ids = sorted(record_ids - expected_ids)
            (
                status,
                next_action,
                apply_status,
                spec_status,
                apply_ready_records,
                remaining_evidence_tasks,
                apply_issues,
            ) = _synthetic_evidence_apply_status(shard)
            issues = _dedupe_strings([*issues, *apply_issues])

    return {
        "shard_id": shard_id,
        "task_offset": _int(shard.get("task_offset"), default=0),
        "max_tasks": _int(shard.get("max_tasks"), default=0),
        "selected_tasks": _int(shard.get("selected_tasks"), default=0),
        "records_output": records_output,
        "record_template_output": record_template_output,
        "apply_output": str(shard.get("apply_output", "")),
        "spec_output": str(shard.get("spec_output", "")),
        "template_status": template_status,
        "records_status": records_status,
        "apply_status": apply_status,
        "spec_status": spec_status,
        "status": status,
        "next_action": next_action,
        "evidence_records": record_count,
        "matching_evidence_records": matching_record_count,
        "missing_record_task_ids": missing_record_task_ids,
        "extra_record_task_ids": extra_record_task_ids,
        "apply_ready_records": apply_ready_records,
        "remaining_evidence_tasks": remaining_evidence_tasks,
        "issues": issues,
    }


def _synthetic_evidence_apply_status(
    shard: dict[str, Any],
) -> tuple[str, str, str, str, int, int, list[str]]:
    apply_status, apply_payload, apply_issue = _read_json_artifact(shard.get("apply_output"))
    if apply_status == "missing":
        return (
            "ready_to_apply",
            "apply_evidence",
            "missing",
            "not_checked",
            0,
            _int(shard.get("selected_tasks"), default=0),
            [],
        )
    if apply_status == "invalid":
        return (
            "blocked",
            "inspect_apply_output",
            "invalid",
            "not_checked",
            0,
            0,
            _dedupe_strings([apply_issue]),
        )
    if not isinstance(apply_payload, dict):
        return (
            "blocked",
            "inspect_apply_output",
            "invalid",
            "not_checked",
            0,
            0,
            ["Apply output must contain a JSON object."],
        )

    apply_issues = _json_issue_texts(apply_payload.get("issues"))
    counts = _dict(apply_payload.get("counts"))
    ready_records = _int(counts.get("ready_records"), default=0)
    selected_ids = set(_string_list(shard.get("evidence_task_ids")))
    applied_ids = set(_string_list(apply_payload.get("applied_evidence_task_ids")))
    remaining_selected_ids = selected_ids - applied_ids
    remaining_tasks = (
        len(remaining_selected_ids)
        if selected_ids
        else _int(counts.get("remaining_evidence_tasks"), default=0)
    )
    if not bool(apply_payload.get("valid")):
        return (
            "blocked",
            "resolve_apply_errors",
            "failed",
            "not_checked",
            ready_records,
            remaining_tasks,
            apply_issues,
        )
    if remaining_tasks:
        return (
            "partial",
            "collect_remaining_evidence",
            "partial",
            "not_checked",
            ready_records,
            remaining_tasks,
            apply_issues,
        )

    spec_status, spec_payload, spec_issue = _read_json_artifact(shard.get("spec_output"))
    if spec_status == "missing":
        return (
            "ready_to_finalize",
            "write_spec_output",
            "complete",
            "missing",
            ready_records,
            remaining_tasks,
            apply_issues,
        )
    if spec_status == "invalid":
        return (
            "blocked",
            "fix_spec_output",
            "complete",
            "invalid",
            ready_records,
            remaining_tasks,
            _dedupe_strings([*apply_issues, spec_issue]),
        )
    if not _dict_list_or_empty(_dict(spec_payload).get("repositories")):
        return (
            "blocked",
            "fix_spec_output",
            "complete",
            "invalid",
            ready_records,
            remaining_tasks,
            _dedupe_strings(
                [*apply_issues, "Spec output must contain a repositories list."]
            ),
        )
    return (
        "complete",
        "none",
        "complete",
        "complete",
        ready_records,
        remaining_tasks,
        apply_issues,
    )


def _synthetic_evidence_shard_status_counts(
    shard_statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(str(shard.get("status", "")) for shard in shard_statuses)
    action_counts = Counter(str(shard.get("next_action", "")) for shard in shard_statuses)
    status_counts.pop("", None)
    action_counts.pop("", None)
    return {
        "schedule_shards": len(shard_statuses),
        "completed_shards": status_counts.get("complete", 0),
        "partial_shards": status_counts.get("partial", 0),
        "pending_shards": status_counts.get("pending", 0),
        "ready_to_apply_shards": status_counts.get("ready_to_apply", 0),
        "ready_to_finalize_shards": status_counts.get("ready_to_finalize", 0),
        "blocked_shards": status_counts.get("blocked", 0),
        "status": dict(sorted(status_counts.items())),
        "next_action": dict(sorted(action_counts.items())),
    }


def _synthetic_evidence_records_from_payload(
    payload: Any,
) -> tuple[list[dict[str, Any]], str]:
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return list(payload), ""
    if isinstance(payload, dict):
        records = payload.get("evidence_records", payload.get("records"))
        if isinstance(records, list) and all(isinstance(item, dict) for item in records):
            return list(records), ""
    return [], "Evidence records must be a JSON array or an object with records."


def _read_json_artifact(path_value: Any) -> tuple[str, Any, str]:
    path_text = str(path_value or "")
    if not path_text:
        return "missing", None, ""
    path = Path(path_text)
    if not path.exists():
        return "missing", None, ""
    try:
        return "present", json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", None, f"Artifact cannot be read as JSON: {path_text}: {exc}"


def _json_issue_texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    messages = []
    for item in value:
        if isinstance(item, str):
            message = item.strip()
        elif isinstance(item, dict):
            code = str(item.get("code", "")).strip()
            text = str(item.get("message") or item.get("issue") or "").strip()
            message = f"{code}: {text}" if code and text else text or code
        else:
            message = str(item).strip()
        if message:
            messages.append(message)
    return messages


def _dict_list_or_empty(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return list(value)
    return []


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    deduped = []
    seen = set()
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def _snapshot_key(snapshot: dict[str, Any]) -> str:
    return f"{snapshot['repository']}@{snapshot['source_revision']}"


def _planned_synthetic_count(specs: list[dict[str, Any]]) -> int:
    total = 0
    for spec in specs:
        total += len(spec.get("targets", [])) * len(spec.get("task_families", []))
    return total


def _synthetic_spec_task_family_counts(specs: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for spec in specs:
        target_count = len(_dict_list(spec.get("targets")))
        for family in _string_list(spec.get("task_families")):
            counts[family] += target_count
    return dict(sorted(counts.items()))


def _synthetic_target_key(
    repository_spec: dict[str, Any],
    target: dict[str, Any],
    family: str,
) -> str:
    return _stable_json_sha256(
        {
            "repository": str(repository_spec.get("repository", "")),
            "source_uri": str(repository_spec.get("source_uri", "")),
            "source_revision": str(repository_spec.get("source_revision", "")),
            "task_family": family,
            "target": {
                "name": str(target.get("name", "")),
                "paths": sorted(_string_list(target.get("paths"))),
            },
        }
    )


def _stable_json_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _extend_unique(values: list[str], additions: Iterable[str]) -> None:
    seen = set(values)
    for item in additions:
        if item and item not in seen:
            values.append(item)
            seen.add(item)


def _dimension_counts(
    seeds: Iterable[QuerySeed],
    value_fn: Any,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for seed in seeds:
        value = value_fn(seed)
        if value:
            counts[value] += 1
    return counts


def _seed_task_family(seed: QuerySeed) -> str:
    return _normalize_label(seed.task_family)


def _seed_source_method(seed: QuerySeed) -> str:
    return _normalize_label(seed.source_method)


def _seed_repository_value(seed: QuerySeed) -> str:
    value = seed.public.context.get("repository") or seed.metadata.get("repository", "")
    return _normalize_label(value).replace("__", "/")


def _seed_language_value(seed: QuerySeed) -> str:
    for tag in seed.coverage_tags:
        if tag.startswith("language:"):
            return _normalize_label(tag.split(":", 1)[1])
    return _normalize_label(seed.metadata.get("language", ""))


def _quarantine_summary(
    import_summaries: Iterable[RegistryImportSummary],
    synthetic_summaries: Iterable[RepositorySyntheticSummary],
    holdout_summaries: Iterable[dict[str, Any]],
    allowlist_filters: Iterable[AllowlistFilterSummary],
) -> dict[str, Any]:
    records = 0
    issues = []
    for summary in import_summaries:
        records += summary.skipped
        issues.extend(summary.issues)
    for summary in synthetic_summaries:
        records += summary.skipped
        issues.extend(summary.issues)
    for summary in holdout_summaries:
        records += int(summary.get("skipped", 0) or 0)
        issues.extend(summary.get("issues", []))
    for summary in allowlist_filters:
        records += summary.quarantined
        issues.extend(summary.issues)
    return {"records": records, "issues": issues}


def _train_verifier_type_counts(seeds: Iterable[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for seed in seeds:
        if seed.train_eligible:
            counts.update(seed.verifier_types)
    return dict(sorted(counts.items()))


def _seed_policy(value: Any) -> SeedLibraryPolicy:
    policy = _dict(value)
    return SeedLibraryPolicy(
        min_train_eligible=_int(policy.get("min_train_eligible"), default=0),
        required_task_families=_string_list(policy.get("required_task_families")),
        required_verifier_types=_string_list(policy.get("required_verifier_types")),
        max_task_family_share=_float(policy.get("max_task_family_share"), default=1.0),
        max_source_method_share=_float(policy.get("max_source_method_share"), default=1.0),
        max_repository_share=_float(policy.get("max_repository_share"), default=1.0),
        max_language_share=_float(policy.get("max_language_share"), default=1.0),
    )


def _source_snapshots(config: dict[str, Any], config_dir: Path) -> list[dict[str, str]]:
    snapshots = []
    section_defaults = {
        "public_issue_sources": "public_issue_pr",
        "public_ci_sources": "public_ci",
        "swe_style_sources": "auto",
        "repository_synthetic_sources": "repository_synthetic",
        "holdout_sources": "auto",
    }
    for section, default_format in section_defaults.items():
        for source in _dict_list(config.get(section)):
            path = _required_source_path(config_dir, source)
            snapshots.append(
                {
                    "section": section,
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "source_name": str(source.get("source_name", "")),
                    "format": _source_format(source, default_format),
                }
            )
    return snapshots


def _registry_validation_payload(validation: Any) -> dict[str, Any]:
    return {
        "valid": validation.valid,
        "issues": [asdict(issue) for issue in validation.issues],
    }


def _prepare_registry_root(root: Path, *, overwrite: bool) -> None:
    if not root.exists():
        return
    if overwrite:
        _ensure_safe_generated_root(root)
        shutil.rmtree(root)
        return
    if _registry_has_entries(root):
        raise ValueError(
            f"Registry root already contains entries; set overwrite_registries for {root}"
        )


def _ensure_safe_generated_root(root: Path) -> None:
    resolved = root.resolve()
    unsafe = {Path("/").resolve(), Path.home().resolve()}
    if resolved in unsafe or len(resolved.parts) < 4:
        raise ValueError(f"Refusing to overwrite unsafe registry root: {root}")


def _registry_has_entries(root: Path) -> bool:
    for name in ("seeds", "environments", "scenarios"):
        if any((root / name).glob("*.json")):
            return True
    return False


def _required_path(config_dir: Path, config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not value:
        raise ValueError(f"Seed corpus config requires {key}")
    return _resolve_path(config_dir, value)


def _optional_path(config_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    return _resolve_path(config_dir, value)


def _required_source_path(config_dir: Path, source: dict[str, Any]) -> Path:
    value = source.get("path") or source.get("source")
    if not value:
        raise ValueError("Seed corpus source requires a path")
    return _resolve_path(config_dir, value)


def _resolve_path(config_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _source_format(source: dict[str, Any], default: str) -> str:
    return str(source.get("format") or source.get("source_format") or default).replace("-", "_")


def _train_eligible(value: Any) -> bool | None:
    if value is None or value == "auto":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"Invalid train_eligible value: {value}")


def _write_optional_json(path: Path | None, value: dict[str, Any]) -> None:
    if path is not None:
        _write_json(path, value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Seed corpus config must contain a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("Seed corpus source sections must be lists of objects")
    return list(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("Expected a string or a list of strings")
    return [str(item) for item in value]


def _int_dict(value: Any) -> dict[str, int]:
    return {str(key): int(item) for key, item in _dict(value).items()}


def _int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
