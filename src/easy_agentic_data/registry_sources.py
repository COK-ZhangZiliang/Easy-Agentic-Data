from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed

SUPPORTED_SOURCE_FORMATS = {"auto", "swe_bench", "swe_smith", "multi_swe"}


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


def scenario_from_swe_style_record(
    record: dict[str, Any],
    *,
    source_format: str = "auto",
    source_name: str = "",
    split: str = "train",
    license_name: str = "",
    permitted_use: str = "research",
    test_command_template: str = "",
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
    return [template.format(test=shlex.quote(_safe_test_id(test_id))) for test_id in test_ids]


def _safe_test_id(test_id: str) -> str:
    if test_id.startswith("-"):
        raise ValueError(f"unsafe hidden test id: {test_id}")
    if any(character in test_id for character in ("\x00", "\n", "\r")):
        raise ValueError("hidden test id cannot contain control characters")
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
