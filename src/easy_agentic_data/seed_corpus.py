from __future__ import annotations

import hashlib
import json
import math
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
from easy_agentic_data.seed_library import (
    DEFAULT_BENCHMARK_SOURCE_ALIASES,
    SeedLibraryPolicy,
    TASK_FAMILY_VERIFIER_TEMPLATES,
    audit_seed_library,
)
from easy_agentic_data.seeds import QuerySeed
from easy_agentic_data.seed_review import build_seed_review_queue

SEED_CORPUS_SCHEMA_VERSION = "easy_agentic_data.seed_corpus.v1"
REGISTRY_IMPORT_REHEARSAL_SCHEMA_VERSION = "easy_agentic_data.registry_import_rehearsal.v1"
SEED_BACKFILL_PLAN_SCHEMA_VERSION = "easy_agentic_data.seed_backfill_plan.v1"
SEED_SELECTION_PLAN_SCHEMA_VERSION = "easy_agentic_data.seed_selection_plan.v1"


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
        arguments = shlex.split(command)
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
            raise RuntimeError(
                "Hidden verifier command failed during materialization rehearsal "
                f"(sha256={command_results[-1]['command_sha256']}, "
                f"exit={completed.returncode})"
            )
    result["hidden_commands_ran"] = True
    result["commands_run"] = len(command_results)
    result["command_results"] = command_results


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
