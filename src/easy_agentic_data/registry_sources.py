from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seed_library import (
    benchmark_contamination_tags,
    default_source_method_for_swe_source,
    default_task_family_for_swe_record,
    default_train_eligible_for_source,
)
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed

SUPPORTED_SOURCE_FORMATS = {
    "auto",
    "swe_bench",
    "swe_smith",
    "multi_swe",
    "public_issue",
    "public_pr",
    "public_issue_pr",
    "public_ci",
}
PUBLIC_ISSUE_PR_FORMATS = {"public_issue", "public_pr", "public_issue_pr"}
PUBLIC_CI_FORMATS = {"public_ci"}
DEFAULT_TRAIN_LICENSE_ALLOWLIST = {
    "0bsd",
    "apache_2.0",
    "bsd_2_clause",
    "bsd_3_clause",
    "cc0_1.0",
    "isc",
    "mit",
    "mpl_2.0",
    "python_2.0",
    "unlicense",
}


@dataclass
class RegistryImportSummary:
    """Summary of an external source import into the scenario registry."""

    source_format: str
    source_name: str
    imported: int = 0
    skipped: int = 0
    scenario_ids: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_source_records(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSON array/object or newline-delimited JSON file."""

    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    if source_path.suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            records.append(payload)
        return records

    payload = json.loads(text)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = _first_present(payload, ("records", "instances", "data", "tasks"))
        if records is None:
            records = [payload]
    else:
        raise ValueError("Source file must contain a JSON object, JSON array, or JSONL objects")
    if not isinstance(records, list):
        raise ValueError("Source file record container must be a JSON array")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every source record must be a JSON object")
    return list(records)


def import_swe_style_records(
    registry: ScenarioRegistry,
    records: Iterable[dict[str, Any]],
    *,
    source_format: str = "auto",
    source_name: str = "",
    split: str = "train",
    license_name: str = "",
    permitted_use: str = "research",
    limit: int | None = None,
    test_command_template: str = "",
    task_family: str = "",
    source_method: str = "",
    train_eligible: bool | None = None,
    contamination_tags: Iterable[str] | None = None,
    coverage_tags: Iterable[str] | None = None,
    strict: bool = False,
) -> RegistryImportSummary:
    """Import SWE-style issue records as query, environment, and scenario entries."""

    normalized_format = _normalize_source_format(source_format)
    summary = RegistryImportSummary(
        source_format=normalized_format,
        source_name=source_name or normalized_format,
    )
    for index, record in enumerate(records):
        if limit is not None and summary.imported >= limit:
            break
        try:
            scenario = scenario_from_swe_style_record(
                record,
                source_format=normalized_format,
                source_name=summary.source_name,
                split=split,
                license_name=license_name,
                permitted_use=permitted_use,
                test_command_template=test_command_template,
                task_family=task_family,
                source_method=source_method,
                train_eligible=train_eligible,
                contamination_tags=contamination_tags,
                coverage_tags=coverage_tags,
            )
        except ValueError as exc:
            message = f"record {index}: {exc}"
            if strict:
                raise ValueError(message) from exc
            summary.skipped += 1
            summary.issues.append(message)
            continue
        registry.add_scenario(scenario)
        summary.imported += 1
        summary.scenario_ids.append(scenario.scenario_id)
    return summary


def import_public_issue_pr_records(
    registry: ScenarioRegistry,
    records: Iterable[dict[str, Any]],
    *,
    source_format: str = "public_issue_pr",
    source_name: str = "",
    split: str = "train",
    license_name: str = "",
    permitted_use: str = "research",
    limit: int | None = None,
    test_command_template: str = "",
    task_family: str = "",
    source_method: str = "",
    train_eligible: bool | None = None,
    contamination_tags: Iterable[str] | None = None,
    coverage_tags: Iterable[str] | None = None,
    train_license_allowlist: Iterable[str] = DEFAULT_TRAIN_LICENSE_ALLOWLIST,
    strict: bool = False,
) -> RegistryImportSummary:
    """Import non-benchmark public issue and PR records with fixed workspaces."""

    normalized_format = _normalize_source_format(source_format)
    if normalized_format == "auto":
        normalized_format = "public_issue_pr"
    if normalized_format not in PUBLIC_ISSUE_PR_FORMATS:
        raise ValueError(f"Unsupported public issue/PR source format: {source_format}")
    summary = RegistryImportSummary(
        source_format=normalized_format,
        source_name=source_name or normalized_format,
    )
    for index, record in enumerate(records):
        if limit is not None and summary.imported >= limit:
            break
        try:
            scenario = scenario_from_public_issue_pr_record(
                record,
                source_format=normalized_format,
                source_name=summary.source_name,
                split=split,
                license_name=license_name,
                permitted_use=permitted_use,
                test_command_template=test_command_template,
                task_family=task_family,
                source_method=source_method,
                train_eligible=train_eligible,
                contamination_tags=contamination_tags,
                coverage_tags=coverage_tags,
                train_license_allowlist=train_license_allowlist,
            )
        except ValueError as exc:
            message = f"record {index}: {exc}"
            if strict:
                raise ValueError(message) from exc
            summary.skipped += 1
            summary.issues.append(message)
            continue
        registry.add_scenario(scenario)
        summary.imported += 1
        summary.scenario_ids.append(scenario.scenario_id)
    return summary


def import_public_ci_records(
    registry: ScenarioRegistry,
    records: Iterable[dict[str, Any]],
    *,
    source_format: str = "public_ci",
    source_name: str = "",
    split: str = "train",
    license_name: str = "",
    permitted_use: str = "research",
    limit: int | None = None,
    task_family: str = "",
    source_method: str = "",
    train_eligible: bool | None = None,
    contamination_tags: Iterable[str] | None = None,
    coverage_tags: Iterable[str] | None = None,
    train_license_allowlist: Iterable[str] = DEFAULT_TRAIN_LICENSE_ALLOWLIST,
    strict: bool = False,
) -> RegistryImportSummary:
    """Import public CI failure records with fixed workspace and CI verifier evidence."""

    normalized_format = _normalize_source_format(source_format)
    if normalized_format == "auto":
        normalized_format = "public_ci"
    if normalized_format not in PUBLIC_CI_FORMATS:
        raise ValueError(f"Unsupported public CI source format: {source_format}")
    summary = RegistryImportSummary(
        source_format=normalized_format,
        source_name=source_name or normalized_format,
    )
    for index, record in enumerate(records):
        if limit is not None and summary.imported >= limit:
            break
        try:
            scenario = scenario_from_public_ci_record(
                record,
                source_format=normalized_format,
                source_name=summary.source_name,
                split=split,
                license_name=license_name,
                permitted_use=permitted_use,
                task_family=task_family,
                source_method=source_method,
                train_eligible=train_eligible,
                contamination_tags=contamination_tags,
                coverage_tags=coverage_tags,
                train_license_allowlist=train_license_allowlist,
            )
        except ValueError as exc:
            message = f"record {index}: {exc}"
            if strict:
                raise ValueError(message) from exc
            summary.skipped += 1
            summary.issues.append(message)
            continue
        registry.add_scenario(scenario)
        summary.imported += 1
        summary.scenario_ids.append(scenario.scenario_id)
    return summary


def scenario_from_public_issue_pr_record(
    record: dict[str, Any],
    *,
    source_format: str = "public_issue_pr",
    source_name: str = "",
    split: str = "train",
    license_name: str = "",
    permitted_use: str = "research",
    test_command_template: str = "",
    task_family: str = "",
    source_method: str = "",
    train_eligible: bool | None = None,
    contamination_tags: Iterable[str] | None = None,
    coverage_tags: Iterable[str] | None = None,
    train_license_allowlist: Iterable[str] = DEFAULT_TRAIN_LICENSE_ALLOWLIST,
) -> Scenario:
    """Convert a public issue or PR export into a train-safe registry scenario."""

    normalized_format = _normalize_source_format(source_format)
    if normalized_format == "auto":
        normalized_format = "public_issue_pr"
    if normalized_format not in PUBLIC_ISSUE_PR_FORMATS:
        raise ValueError(f"Unsupported public issue/PR source format: {source_format}")
    source_type = _public_source_type(record, normalized_format)
    instance_id = _public_instance_id(record, source_type)
    query = _query_text(record)
    repo = _repo_name(record)
    source_uri = _source_uri(record, repo, source_name or normalized_format)
    if not source_uri:
        raise ValueError("public issue/PR import requires a repository source URI")
    revision = _fixed_source_revision(record)
    resolved_license = license_name or _text_field(record.get("license"))
    allowlist = _normalized_license_set(train_license_allowlist)
    license_allowed = _license_allowed_for_train(resolved_license, allowlist)
    default_train = default_train_eligible_for_source(
        source_name or normalized_format,
        normalized_format,
    )
    if train_eligible is True and not license_allowed:
        raise ValueError(
            f"license is not allowed for trainable public seeds: {resolved_license or '<missing>'}"
        )
    resolved_train_eligible = default_train if train_eligible is None else train_eligible
    if not license_allowed:
        resolved_train_eligible = False
    resolved_task_family = (
        _text_field(task_family).strip().lower().replace("-", "_")
        or _text_field(record.get("task_family")).strip().lower().replace("-", "_")
        or _infer_public_task_family(record, source_type)
    )
    resolved_source_method = (
        _text_field(source_method).strip().lower().replace("-", "_")
        or f"{source_type}_workspace"
    )
    command_groups = _public_command_groups(record, test_command_template)
    hidden_tests = _flatten_command_groups(command_groups)
    patch = _text_field(_first_present(record, ("patch", "reference_patch", "gold_patch")))
    test_patch = _text_field(_first_present(record, ("test_patch", "tests_patch")))
    verifier_types = _public_verifier_types(
        record,
        command_groups,
        patch=patch,
        test_patch=test_patch,
    )
    resolved_contamination_tags = sorted(
        set(_list_field(record.get("contamination_tags")))
        | set(contamination_tags or [])
        | set(benchmark_contamination_tags(source_name or normalized_format, normalized_format))
        | (set() if license_allowed else {"license_not_allowlisted"})
    )
    resolved_coverage_tags = _coverage_tags(
        record,
        resolved_task_family,
        normalized_format,
        repo,
        verifier_types,
        coverage_tags or [],
    )
    reference_artifacts = _public_reference_artifacts(
        record,
        source_name or normalized_format,
        instance_id,
        has_patch=bool(patch),
        has_test_patch=bool(test_patch),
    )
    seed = QuerySeed(
        public=PublicTaskContext(
            query=query,
            context=_public_issue_pr_context(record, instance_id, repo, source_type),
            constraints=_list_field(record.get("constraints")),
        ),
        category=_text_field(record.get("category")) or "software_engineering",
        difficulty=_difficulty(record),
        provenance=f"{source_name or normalized_format}:{instance_id}",
        license=resolved_license,
        split=split,
        task_family=resolved_task_family,
        source_method=resolved_source_method,
        train_eligible=resolved_train_eligible,
        contamination_tags=resolved_contamination_tags,
        verifier_types=verifier_types,
        coverage_tags=resolved_coverage_tags,
        metadata={
            "source_adapter": "public_issue_pr",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
            "source_type": source_type,
            "permitted_use": permitted_use,
            "repository": repo,
            "source_url": _public_source_url(record),
        },
    )
    environment = EnvironmentSpec(
        name=_environment_name(record, instance_id),
        version=_text_field(record.get("version")) or "1",
        description=f"Public {source_type.replace('_', ' ')} workspace for {instance_id}.",
        image_digest=_image_reference(record),
        source_uri=source_uri,
        source_revision=revision,
        working_directory=_text_field(record.get("working_directory")) or "/workspace",
        setup_commands=_list_field(record.get("setup_commands")),
        capability_packs=_list_field(record.get("capability_packs")),
        network_policy=_text_field(record.get("network_policy")) or "disabled",
        resource_limits=_dict_field(record.get("resource_limits")),
        health_check=_list_field(record.get("health_check")),
        reset_strategy=_text_field(record.get("reset_strategy")) or "recreate",
        evaluator_refs=reference_artifacts,
        metadata=_public_environment_metadata(
            record,
            normalized_format,
            source_name or normalized_format,
            source_type,
            instance_id,
            repo,
            permitted_use,
            resolved_license,
            patch,
            test_patch,
        ),
    )
    evaluator = HiddenEvaluatorContext(
        reference_artifacts=reference_artifacts,
        hidden_tests=hidden_tests,
        required_state=_dict_field(record.get("required_state")),
        forbidden_state=_dict_field(record.get("forbidden_state")),
        metadata={
            "source_adapter": "public_issue_pr",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
            "source_type": source_type,
            "command_groups": command_groups,
            "patch_sha256": _sha256(patch),
            "test_patch_sha256": _sha256(test_patch),
            "patch_stored_as_reference": bool(patch),
            "test_patch_stored_as_reference": bool(test_patch),
        },
    )
    return Scenario(
        query_seed=seed,
        environment=environment,
        hidden_evaluator=evaluator,
        metadata={
            "source_adapter": "public_issue_pr",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
            "source_type": source_type,
        },
    )


def scenario_from_public_ci_record(
    record: dict[str, Any],
    *,
    source_format: str = "public_ci",
    source_name: str = "",
    split: str = "train",
    license_name: str = "",
    permitted_use: str = "research",
    task_family: str = "",
    source_method: str = "",
    train_eligible: bool | None = None,
    contamination_tags: Iterable[str] | None = None,
    coverage_tags: Iterable[str] | None = None,
    train_license_allowlist: Iterable[str] = DEFAULT_TRAIN_LICENSE_ALLOWLIST,
) -> Scenario:
    """Convert a public CI failure export into a train-safe registry scenario."""

    normalized_format = _normalize_source_format(source_format)
    if normalized_format == "auto":
        normalized_format = "public_ci"
    if normalized_format not in PUBLIC_CI_FORMATS:
        raise ValueError(f"Unsupported public CI source format: {source_format}")
    source_type = _ci_source_type(record)
    instance_id = _ci_instance_id(record)
    query = _query_text(record)
    repo = _repo_name(record)
    source_uri = _workspace_source_uri(record, repo, source_name or normalized_format)
    if not source_uri:
        raise ValueError("public CI import requires a repository source URI")
    revision = _fixed_source_revision(record)
    ci_commands = _list_field(record.get("ci_commands"))
    if not ci_commands:
        raise ValueError("public CI import requires ci_commands verifier evidence")
    resolved_license = license_name or _text_field(record.get("license"))
    allowlist = _normalized_license_set(train_license_allowlist)
    license_allowed = _license_allowed_for_train(resolved_license, allowlist)
    default_train = default_train_eligible_for_source(
        source_name or normalized_format,
        normalized_format,
    )
    if train_eligible is True and not license_allowed:
        raise ValueError(
            f"license is not allowed for trainable public CI seeds: "
            f"{resolved_license or '<missing>'}"
        )
    resolved_train_eligible = default_train if train_eligible is None else train_eligible
    if not license_allowed:
        resolved_train_eligible = False
    resolved_task_family = (
        _text_field(task_family).strip().lower().replace("-", "_")
        or _text_field(record.get("task_family")).strip().lower().replace("-", "_")
        or "ci_build"
    )
    resolved_source_method = (
        _text_field(source_method).strip().lower().replace("-", "_")
        or "public_ci_workspace"
    )
    command_groups = {"hidden_command": ci_commands}
    hidden_tests = _flatten_command_groups(command_groups)
    verifier_types = _public_verifier_types(
        record,
        command_groups,
        patch="",
        test_patch="",
    )
    resolved_contamination_tags = sorted(
        set(_list_field(record.get("contamination_tags")))
        | set(contamination_tags or [])
        | set(benchmark_contamination_tags(source_name or normalized_format, normalized_format))
        | (set() if license_allowed else {"license_not_allowlisted"})
    )
    resolved_coverage_tags = _coverage_tags(
        record,
        resolved_task_family,
        normalized_format,
        repo,
        verifier_types,
        coverage_tags or [],
    )
    source_url = _public_source_url(record)
    seed = QuerySeed(
        public=PublicTaskContext(
            query=query,
            context=_public_ci_context(record, instance_id, repo, source_url),
            constraints=_list_field(record.get("constraints")),
        ),
        category=_text_field(record.get("category")) or "software_engineering",
        difficulty=_difficulty(record),
        provenance=f"{source_name or normalized_format}:{instance_id}",
        license=resolved_license,
        split=split,
        task_family=resolved_task_family,
        source_method=resolved_source_method,
        train_eligible=resolved_train_eligible,
        contamination_tags=resolved_contamination_tags,
        verifier_types=verifier_types,
        coverage_tags=resolved_coverage_tags,
        metadata={
            "source_adapter": "public_ci",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
            "source_type": source_type,
            "permitted_use": permitted_use,
            "repository": repo,
            "source_url": source_url,
        },
    )
    environment = EnvironmentSpec(
        name=_environment_name(record, instance_id),
        version=_text_field(record.get("version")) or "1",
        description=f"Public CI failure workspace for {instance_id}.",
        image_digest=_image_reference(record),
        source_uri=source_uri,
        source_revision=revision,
        working_directory=_text_field(record.get("working_directory")) or "/workspace",
        setup_commands=_list_field(record.get("setup_commands")),
        capability_packs=_list_field(record.get("capability_packs")),
        network_policy=_text_field(record.get("network_policy")) or "disabled",
        resource_limits=_dict_field(record.get("resource_limits")),
        health_check=_list_field(record.get("health_check")),
        reset_strategy=_text_field(record.get("reset_strategy")) or "recreate",
        metadata=_public_ci_environment_metadata(
            record,
            normalized_format,
            source_name or normalized_format,
            instance_id,
            repo,
            source_url,
            permitted_use,
            resolved_license,
        ),
    )
    evaluator = HiddenEvaluatorContext(
        hidden_tests=hidden_tests,
        required_state=_dict_field(record.get("required_state")),
        forbidden_state=_dict_field(record.get("forbidden_state")),
        metadata={
            "source_adapter": "public_ci",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
            "source_type": source_type,
            "ci_commands": ci_commands,
            "command_groups": command_groups,
            "candidate_verifier": _dict_field(record.get("candidate_verifier")),
        },
    )
    return Scenario(
        query_seed=seed,
        environment=environment,
        hidden_evaluator=evaluator,
        metadata={
            "source_adapter": "public_ci",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
            "source_type": source_type,
        },
    )


def scenario_from_swe_style_record(
    record: dict[str, Any],
    *,
    source_format: str = "auto",
    source_name: str = "",
    split: str = "train",
    license_name: str = "",
    permitted_use: str = "research",
    test_command_template: str = "",
    task_family: str = "",
    source_method: str = "",
    train_eligible: bool | None = None,
    contamination_tags: Iterable[str] | None = None,
    coverage_tags: Iterable[str] | None = None,
) -> Scenario:
    """Convert one SWE-bench-like record into a scenario without exposing gold patches."""

    normalized_format = _detect_source_format(record, _normalize_source_format(source_format))
    instance_id = _instance_id(record)
    query = _query_text(record)
    repo = _repo_name(record)
    provenance = f"{source_name or normalized_format}:{instance_id}"
    fail_to_pass = _list_field(_first_present(record, ("FAIL_TO_PASS", "fail_to_pass")))
    pass_to_pass = _list_field(_first_present(record, ("PASS_TO_PASS", "pass_to_pass")))
    hidden_tests = _hidden_test_commands(fail_to_pass, test_command_template)
    patch = _text_field(_first_present(record, ("patch", "fix_patch", "gold_patch")))
    test_patch = _text_field(_first_present(record, ("test_patch", "tests_patch")))
    resolved_task_family = (
        _text_field(task_family).strip().lower().replace("-", "_")
        or default_task_family_for_swe_record(record, normalized_format)
    )
    resolved_source_method = (
        _text_field(source_method).strip().lower().replace("-", "_")
        or default_source_method_for_swe_source(normalized_format)
    )
    resolved_train_eligible = (
        default_train_eligible_for_source(source_name or normalized_format, normalized_format)
        if train_eligible is None
        else train_eligible
    )
    resolved_contamination_tags = sorted(
        set(_list_field(record.get("contamination_tags")))
        | set(contamination_tags or [])
        | set(benchmark_contamination_tags(source_name or normalized_format, normalized_format))
    )
    verifier_types = _verifier_types(record, patch, test_patch, hidden_tests)
    resolved_coverage_tags = _coverage_tags(
        record,
        resolved_task_family,
        normalized_format,
        repo,
        verifier_types,
        coverage_tags or [],
    )
    reference_artifacts = _reference_artifacts(
        source_name or normalized_format,
        instance_id,
        has_patch=bool(patch),
        has_test_patch=bool(test_patch),
    )
    seed = QuerySeed(
        public=PublicTaskContext(
            query=query,
            context=_public_context(record, instance_id, repo),
            constraints=_list_field(record.get("constraints")),
        ),
        category=_text_field(record.get("category")) or "software_engineering",
        difficulty=_difficulty(record),
        provenance=provenance,
        license=license_name or _text_field(record.get("license")),
        split=split,
        task_family=resolved_task_family,
        source_method=resolved_source_method,
        train_eligible=resolved_train_eligible,
        contamination_tags=resolved_contamination_tags,
        verifier_types=verifier_types,
        coverage_tags=resolved_coverage_tags,
        metadata={
            "source_adapter": "swe_style",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
            "permitted_use": permitted_use,
        },
    )
    environment = EnvironmentSpec(
        name=_environment_name(record, instance_id),
        version=_text_field(record.get("version")) or "1",
        description=f"Imported workspace for {instance_id}.",
        image_digest=_image_reference(record),
        source_uri=_source_uri(record, repo, source_name or normalized_format),
        source_revision=_source_revision(record),
        working_directory=_text_field(record.get("working_directory")) or "/workspace",
        setup_commands=_list_field(record.get("setup_commands")),
        capability_packs=_list_field(record.get("capability_packs")),
        network_policy=_text_field(record.get("network_policy")) or "disabled",
        resource_limits=_dict_field(record.get("resource_limits")),
        health_check=_list_field(record.get("health_check")),
        reset_strategy=_text_field(record.get("reset_strategy")) or "recreate",
        evaluator_refs=reference_artifacts,
        metadata=_environment_metadata(
            record,
            normalized_format,
            source_name or normalized_format,
            instance_id,
            repo,
            patch,
            test_patch,
            permitted_use,
        ),
    )
    evaluator = HiddenEvaluatorContext(
        reference_artifacts=reference_artifacts,
        hidden_tests=hidden_tests,
        metadata={
            "source_adapter": "swe_style",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "patch_sha256": _sha256(patch),
            "test_patch_sha256": _sha256(test_patch),
            "patch_stored_as_reference": bool(patch),
            "test_patch_stored_as_reference": bool(test_patch),
            "test_patch": test_patch,
        },
    )
    return Scenario(
        query_seed=seed,
        environment=environment,
        hidden_evaluator=evaluator,
        metadata={
            "source_adapter": "swe_style",
            "source_format": normalized_format,
            "source_name": source_name or normalized_format,
            "source_instance_id": instance_id,
        },
    )


def _verifier_types(
    record: dict[str, Any],
    patch: str,
    test_patch: str,
    hidden_tests: list[str],
) -> list[str]:
    verifier_types = set(_list_field(record.get("verifier_types")))
    if hidden_tests:
        verifier_types.add("hidden_command")
    if test_patch:
        verifier_types.add("hidden_test_patch")
    if patch:
        verifier_types.add("reference_patch")
    return sorted(verifier_types)


def _coverage_tags(
    record: dict[str, Any],
    task_family: str,
    source_format: str,
    repo: str,
    verifier_types: list[str],
    extra_tags: Iterable[str],
) -> list[str]:
    tags = set(_list_field(record.get("coverage_tags"))) | set(extra_tags)
    tags.add(f"task_family:{task_family}")
    tags.add(f"source_format:{source_format}")
    for verifier_type in verifier_types:
        tags.add(f"verifier:{verifier_type}")
    if repo:
        tags.add(f"repo:{repo}")
    language = _text_field(record.get("language"))
    if language:
        tags.add(f"language:{language}")
    return sorted(tags)


def _public_source_type(record: dict[str, Any], source_format: str) -> str:
    explicit = _text_field(
        _first_present(record, ("source_type", "record_type", "type", "kind"))
    )
    normalized = explicit.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"issue", "public_issue"}:
        return "public_issue"
    if normalized in {"pr", "pull_request", "public_pr"}:
        return "public_pr"
    if normalized in {"ci", "ci_failure", "workflow_run", "public_ci"}:
        raise ValueError("CI source records require a CI-specific registry importer")
    if normalized:
        raise ValueError(f"unsupported public issue/PR source type: {explicit}")
    if source_format in {"public_issue", "public_pr"}:
        return source_format
    if "pull_request" in record or "pull_request_url" in record or "merged_at" in record:
        return "public_pr"
    return "public_issue"


def _public_instance_id(record: dict[str, Any], source_type: str) -> str:
    value = _first_present(
        record,
        (
            "source_instance_id",
            "instance_id",
            "task_id",
            "node_id",
            "issue_id",
            "pull_request_id",
            "id",
        ),
    )
    repo = _repo_name(record).replace("/", "__")
    number = _text_field(_first_present(record, ("number", "issue_number", "pr_number")))
    if value is None and repo and number:
        short_type = "pr" if source_type == "public_pr" else "issue"
        value = f"{repo}-{short_type}-{number}"
    if value is None:
        value = _public_source_url(record)
    instance_id = _text_field(value)
    if not instance_id:
        raise ValueError("missing public issue/PR source instance id")
    return instance_id


def _ci_source_type(record: dict[str, Any]) -> str:
    explicit = _text_field(
        _first_present(record, ("source_type", "record_type", "type", "kind"))
    )
    normalized = explicit.lower().replace("-", "_").replace(" ", "_")
    if normalized and normalized not in {"ci", "ci_failure", "workflow_run", "public_ci"}:
        raise ValueError(f"unsupported public CI source type: {explicit}")
    return "public_ci"


def _ci_instance_id(record: dict[str, Any]) -> str:
    value = _first_present(
        record,
        (
            "source_instance_id",
            "instance_id",
            "task_id",
            "workflow_run_id",
            "run_id",
            "id",
        ),
    )
    repo = _repo_name(record).replace("/", "__")
    if value is None and repo:
        run_number = _text_field(
            _first_present(record, ("run_number", "number", "workflow_run_number"))
        )
        if run_number:
            value = f"{repo}-ci-{run_number}"
    if value is None:
        value = _public_source_url(record)
    instance_id = _text_field(value)
    if not instance_id:
        raise ValueError("missing public CI source instance id")
    return instance_id


def _public_issue_pr_context(
    record: dict[str, Any],
    instance_id: str,
    repo: str,
    source_type: str,
) -> dict[str, Any]:
    context = _public_context(record, instance_id, repo)
    context["source_type"] = source_type
    labels = _list_field(record.get("labels"))
    if labels:
        context["labels"] = labels
    source_url = _public_source_url(record)
    if source_url:
        context["source_url"] = source_url
    return context


def _public_ci_context(
    record: dict[str, Any],
    instance_id: str,
    repo: str,
    source_url: str,
) -> dict[str, Any]:
    context = _public_context(record, instance_id, repo)
    context["source_type"] = "public_ci"
    if source_url:
        context["source_url"] = source_url
    labels = _list_field(record.get("labels"))
    if labels:
        context["labels"] = labels
    workflow_name = _text_field(record.get("workflow_name") or record.get("name"))
    if workflow_name:
        context["workflow_name"] = workflow_name
    event = _text_field(record.get("event"))
    if event:
        context["event"] = event
    head_branch = _text_field(record.get("head_branch"))
    if head_branch:
        context["head_branch"] = head_branch
    return context


def _public_source_url(record: dict[str, Any]) -> str:
    return _text_field(
        _first_present(
            record,
            (
                "source_url",
                "html_url",
                "issue_url",
                "pull_request_url",
                "url",
            ),
        )
    )


def _fixed_source_revision(record: dict[str, Any]) -> str:
    revision = _source_revision(record)
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("public issue/PR import requires a 40-character fixed source revision")
    return revision.lower()


def _normalized_license_set(values: Iterable[str]) -> set[str]:
    return {_normalize_license(value) for value in values if _normalize_license(value)}


def _normalize_license(value: Any) -> str:
    return _text_field(value).lower().replace("-", "_").replace(" ", "_")


def _license_allowed_for_train(license_name: str, allowlist: set[str]) -> bool:
    normalized = _normalize_license(license_name)
    return bool(normalized and normalized in allowlist)


def _infer_public_task_family(record: dict[str, Any], source_type: str) -> str:
    labels = {_normalize_task_label(value) for value in _list_field(record.get("labels"))}
    title_tokens = set(
        _normalize_task_label(value)
        for value in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", _text_field(record.get("title")))
    )
    signals = labels | title_tokens
    mapping = (
        ("code_review", {"review", "comments", "requested_changes"}),
        ("security_hardening", {"security", "vulnerability", "cve", "xss", "injection"}),
        ("performance", {"performance", "perf", "slow", "latency", "benchmark"}),
        ("dependency_upgrade", {"dependency", "dependencies", "upgrade", "deps"}),
        ("migration", {"migration", "migrate", "schema"}),
        ("ci_build", {"ci", "build", "packaging", "lint", "typing", "workflow"}),
        ("bug_repair", {"bug", "regression", "defect", "fix", "crash"}),
        ("test_authoring", {"test", "tests", "coverage", "flaky"}),
        ("refactor", {"refactor", "cleanup", "simplify"}),
        ("feature_implementation", {"feature", "enhancement", "api"}),
        ("repo_understanding", {"question", "help", "explain", "investigate"}),
    )
    for family, terms in mapping:
        if signals.intersection(terms):
            return family
    if signals.intersection({"docs", "documentation", "example", "examples", "readme"}):
        if _has_docs_example_verifier(record):
            return "docs_examples"
    if source_type == "public_pr":
        return "code_review"
    return "bug_repair"


def _has_docs_example_verifier(record: dict[str, Any]) -> bool:
    verifier_types = {
        _normalize_task_label(value)
        for value in _list_field(record.get("verifier_types"))
    }
    candidate = _dict_field(record.get("candidate_verifier"))
    candidate_type = _normalize_task_label(candidate.get("type"))
    if candidate_type:
        verifier_types.add(candidate_type)
    return bool(
        verifier_types.intersection({"doctest", "example_command"})
        or _list_field(record.get("example_commands"))
        or _list_field(record.get("doctest_commands"))
    )


def _normalize_task_label(value: Any) -> str:
    return _text_field(value).lower().replace("-", "_").replace(" ", "_")


def _public_command_groups(
    record: dict[str, Any],
    test_command_template: str,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    _add_command_group(
        groups,
        "hidden_command",
        _list_field(_first_present(record, ("hidden_tests", "test_commands", "eval_commands"))),
    )
    test_ids = _list_field(
        _first_present(record, ("FAIL_TO_PASS", "fail_to_pass", "test_ids"))
    )
    _add_command_group(
        groups,
        "hidden_command",
        _hidden_test_commands(test_ids, test_command_template),
    )
    _add_command_group(groups, "build_command", _list_field(record.get("build_commands")))
    _add_command_group(groups, "example_command", _list_field(record.get("example_commands")))
    _add_command_group(groups, "doctest", _list_field(record.get("doctest_commands")))
    _add_command_group(groups, "benchmark_command", _list_field(record.get("benchmark_commands")))
    _add_command_group(groups, "adversarial_test", _list_field(record.get("adversarial_tests")))
    return {key: value for key, value in sorted(groups.items()) if value}


def _add_command_group(
    groups: dict[str, list[str]],
    verifier_type: str,
    commands: Iterable[str],
) -> None:
    seen = set(groups.get(verifier_type, []))
    for command in commands:
        if command and command not in seen:
            groups.setdefault(verifier_type, []).append(command)
            seen.add(command)


def _flatten_command_groups(groups: dict[str, list[str]]) -> list[str]:
    commands: list[str] = []
    seen: set[str] = set()
    for group_commands in groups.values():
        for command in group_commands:
            if command not in seen:
                commands.append(command)
                seen.add(command)
    return commands


def _public_verifier_types(
    record: dict[str, Any],
    command_groups: dict[str, list[str]],
    *,
    patch: str,
    test_patch: str,
) -> list[str]:
    verifier_types = set(_list_field(record.get("verifier_types")))
    verifier_types.update(command_groups)
    if _dict_field(record.get("required_state")):
        verifier_types.add("required_state")
    if _dict_field(record.get("forbidden_state")):
        verifier_types.add("forbidden_state")
    if _list_field(record.get("diff_constraints")):
        verifier_types.add("diff_constraint")
    if record.get("performance_threshold") is not None:
        verifier_types.add("performance_threshold")
    if patch:
        verifier_types.add("reference_patch")
    if test_patch:
        verifier_types.add("hidden_test_patch")
    return sorted(_normalize_task_label(value) for value in verifier_types)


def _public_reference_artifacts(
    record: dict[str, Any],
    source_name: str,
    instance_id: str,
    *,
    has_patch: bool,
    has_test_patch: bool,
) -> list[str]:
    artifacts = set(_list_field(record.get("reference_artifacts")))
    artifacts.update(
        _reference_artifacts(
            source_name,
            instance_id,
            has_patch=has_patch,
            has_test_patch=has_test_patch,
        )
    )
    return sorted(artifacts)


def _public_environment_metadata(
    record: dict[str, Any],
    source_format: str,
    source_name: str,
    source_type: str,
    instance_id: str,
    repo: str,
    permitted_use: str,
    license_name: str,
    patch: str,
    test_patch: str,
) -> dict[str, Any]:
    metadata = {
        "source_adapter": "public_issue_pr",
        "source_format": source_format,
        "source_name": source_name,
        "source_instance_id": instance_id,
        "source_type": source_type,
        "permitted_use": permitted_use,
        "license": license_name,
        "source_url": _public_source_url(record),
        "patch_sha256": _sha256(patch),
        "test_patch_sha256": _sha256(test_patch),
    }
    if repo:
        metadata["repository"] = repo
    language = _text_field(record.get("language"))
    if language:
        metadata["language"] = language
    return {key: value for key, value in metadata.items() if value != ""}


def _public_ci_environment_metadata(
    record: dict[str, Any],
    source_format: str,
    source_name: str,
    instance_id: str,
    repo: str,
    source_url: str,
    permitted_use: str,
    license_name: str,
) -> dict[str, Any]:
    metadata = {
        "source_adapter": "public_ci",
        "source_format": source_format,
        "source_name": source_name,
        "source_instance_id": instance_id,
        "source_type": "public_ci",
        "permitted_use": permitted_use,
        "license": license_name,
        "source_url": source_url,
    }
    if repo:
        metadata["repository"] = repo
    language = _text_field(record.get("language"))
    if language:
        metadata["language"] = language
    workflow_name = _text_field(record.get("workflow_name") or record.get("name"))
    if workflow_name:
        metadata["workflow_name"] = workflow_name
    return {key: value for key, value in metadata.items() if value != ""}


def _normalize_source_format(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(f"Unsupported source format: {value}")
    return normalized


def _detect_source_format(record: dict[str, Any], source_format: str) -> str:
    if source_format != "auto":
        return source_format
    if "image_name" in record or str(record.get("repo", "")).startswith("swesmith/"):
        return "swe_smith"
    if {"org", "repo", "number"}.issubset(record):
        return "multi_swe"
    return "swe_bench"


def _query_text(record: dict[str, Any]) -> str:
    value = _first_present(record, ("problem_statement", "issue_text", "query", "prompt"))
    if value is None:
        title = _text_field(record.get("title"))
        body = _text_field(record.get("body"))
        value = f"{title}\n\n{body}".strip()
    query = _text_field(value)
    if not query:
        raise ValueError("missing problem statement or query text")
    return query


def _instance_id(record: dict[str, Any]) -> str:
    value = _first_present(record, ("instance_id", "task_id", "id"))
    if value is None and {"org", "repo", "number"}.issubset(record):
        value = f"{record['org']}__{record['repo']}-{record['number']}"
    instance_id = _text_field(value)
    if not instance_id:
        raise ValueError("missing source instance id")
    return instance_id


def _repo_name(record: dict[str, Any]) -> str:
    value = _first_present(record, ("repository", "full_repo", "repo"))
    if {"org", "repo"}.issubset(record) and "/" not in _text_field(value):
        value = f"{record['org']}/{record['repo']}"
    return _text_field(value)


def _public_context(record: dict[str, Any], instance_id: str, repo: str) -> dict[str, Any]:
    context: dict[str, Any] = {"source_instance_id": instance_id}
    if repo:
        context["repository"] = repo
    hints = _text_field(_first_present(record, ("hints_text", "hints")))
    if hints:
        context["hints"] = hints
    return context


def _environment_name(record: dict[str, Any], instance_id: str) -> str:
    value = _first_present(record, ("environment_name", "image_name", "repo", "repository"))
    name = _text_field(value) or instance_id
    return name.replace("/", "__")


def _source_uri(record: dict[str, Any], repo: str, source_name: str) -> str:
    explicit = _text_field(_first_present(record, ("source_uri", "clone_url", "html_url")))
    if explicit:
        return explicit
    if repo and "/" in repo and not repo.startswith("swesmith/"):
        return f"https://github.com/{repo}.git"
    if repo:
        return f"dataset://{quote(source_name, safe='')}/{quote(repo, safe='/')}"
    return ""


def _workspace_source_uri(record: dict[str, Any], repo: str, source_name: str) -> str:
    explicit = _text_field(_first_present(record, ("source_uri", "clone_url")))
    if explicit:
        return explicit
    if repo and "/" in repo and not repo.startswith("swesmith/"):
        return f"https://github.com/{repo}.git"
    if repo:
        return f"dataset://{quote(source_name, safe='')}/{quote(repo, safe='/')}"
    return ""


def _source_revision(record: dict[str, Any]) -> str:
    value = _first_present(
        record,
        (
            "base_commit",
            "source_revision",
            "base_sha",
            "commit",
            "environment_setup_commit",
        ),
    )
    return _text_field(value)


def _image_reference(record: dict[str, Any]) -> str:
    return _text_field(
        _first_present(record, ("image_digest", "docker_image_digest", "image_name", "image"))
    )


def _environment_metadata(
    record: dict[str, Any],
    source_format: str,
    source_name: str,
    instance_id: str,
    repo: str,
    patch: str,
    test_patch: str,
    permitted_use: str,
) -> dict[str, Any]:
    metadata = {
        "source_adapter": "swe_style",
        "source_format": source_format,
        "source_name": source_name,
        "source_instance_id": instance_id,
        "permitted_use": permitted_use,
        "patch_sha256": _sha256(patch),
        "test_patch_sha256": _sha256(test_patch),
    }
    if repo:
        metadata["repository"] = repo
    environment_setup_commit = _text_field(record.get("environment_setup_commit"))
    if environment_setup_commit:
        metadata["environment_setup_commit"] = environment_setup_commit
    image_name = _text_field(record.get("image_name"))
    if image_name:
        metadata["container_image"] = image_name
    return metadata


def _reference_artifacts(
    source_name: str,
    instance_id: str,
    *,
    has_patch: bool,
    has_test_patch: bool,
) -> list[str]:
    base = f"source://{quote(source_name, safe='')}/{quote(instance_id, safe='')}"
    artifacts = []
    if has_patch:
        artifacts.append(f"{base}/patch")
    if has_test_patch:
        artifacts.append(f"{base}/test_patch")
    return artifacts


def _hidden_test_commands(test_ids: list[str], template: str) -> list[str]:
    if not template:
        return []
    commands: list[str] = []
    seen: set[str] = set()
    for test_id in test_ids:
        command = template.format(test=shlex.quote(_stable_pytest_test_id(test_id)))
        if command not in seen:
            commands.append(command)
            seen.add(command)
    return commands


def _stable_pytest_test_id(test_id: str) -> str:
    if test_id.startswith("-"):
        raise ValueError(f"unsafe hidden test id: {test_id}")
    if any(character in test_id for character in ("\x00", "\n", "\r")):
        raise ValueError("hidden test id cannot contain control characters")
    if "[" in test_id:
        return test_id.split("[", 1)[0]
    return test_id


def _difficulty(record: dict[str, Any]) -> int:
    try:
        return int(record.get("difficulty", 3))
    except (TypeError, ValueError):
        return 3


def _text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _list_field(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            decoded = json.loads(stripped)
            return _list_field(decoded)
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [_text_field(item) for item in value if _text_field(item)]
    return [_text_field(value)]


def _dict_field(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("expected a JSON object")


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _sha256(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
