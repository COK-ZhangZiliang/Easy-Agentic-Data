from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.models import stable_id
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seed_library import (
    DEFAULT_BENCHMARK_SOURCE_ALIASES,
    SUPPORTED_TASK_FAMILIES,
    TASK_FAMILY_VERIFIER_TEMPLATES,
    benchmark_contamination_tags,
)
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed

DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES = (
    "test_authoring",
    "refactor",
    "dependency_upgrade",
    "migration",
    "docs_examples",
    "security_hardening",
    "performance",
    "ci_build",
    "code_review",
    "repo_understanding",
)

DEFAULT_SYNTHETIC_TRAIN_LICENSE_ALLOWLIST = {
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
class RepositorySyntheticSummary:
    """Summary for repository-grounded synthetic seed generation."""

    source_name: str
    generated: int = 0
    skipped: int = 0
    scenario_ids: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_repository_synthesis_specs(path: str | Path) -> list[dict[str, Any]]:
    """Load repository synthesis specs from JSON, JSONL, or a container object."""

    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    if source_path.suffix == ".jsonl":
        specs = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            specs.append(payload)
        return specs

    payload = json.loads(text)
    if isinstance(payload, list):
        specs = payload
    elif isinstance(payload, dict):
        specs = payload.get("repositories") or payload.get("records") or payload.get("specs")
        if specs is None:
            specs = [payload]
    else:
        raise ValueError("Synthesis spec must contain a JSON object, array, or JSONL objects")
    if not isinstance(specs, list):
        raise ValueError("Synthesis spec container must be a JSON array")
    if not all(isinstance(spec, dict) for spec in specs):
        raise ValueError("Every repository synthesis spec must be a JSON object")
    return list(specs)


def generate_repository_synthetic_scenarios(
    registry: ScenarioRegistry,
    specs: Iterable[dict[str, Any]],
    *,
    source_name: str = "repository_synthetic",
    split: str = "train",
    task_families: Iterable[str] | None = None,
    train_eligible: bool | None = None,
    train_license_allowlist: Iterable[str] = DEFAULT_SYNTHETIC_TRAIN_LICENSE_ALLOWLIST,
    limit: int | None = None,
    strict: bool = False,
) -> RepositorySyntheticSummary:
    """Generate deterministic repository-grounded synthetic scenarios."""

    summary = RepositorySyntheticSummary(source_name=source_name)
    for spec_index, spec in enumerate(specs):
        families = _resolved_task_families(spec, task_families)
        targets = _target_specs(spec)
        for target_index, target in enumerate(targets):
            for task_family in families:
                if limit is not None and summary.generated >= limit:
                    return summary
                try:
                    scenario = scenario_from_repository_synthetic_spec(
                        spec,
                        target=target,
                        task_family=task_family,
                        source_name=source_name,
                        split=split,
                        train_eligible=train_eligible,
                        train_license_allowlist=train_license_allowlist,
                    )
                except ValueError as exc:
                    message = (
                        f"spec {spec_index} target {target_index} "
                        f"family {task_family}: {exc}"
                    )
                    if strict:
                        raise ValueError(message) from exc
                    summary.skipped += 1
                    summary.issues.append(message)
                    continue
                registry.add_scenario(scenario)
                summary.generated += 1
                summary.scenario_ids.append(scenario.scenario_id)
    return summary


def scenario_from_repository_synthetic_spec(
    spec: dict[str, Any],
    *,
    target: dict[str, Any] | None = None,
    task_family: str,
    source_name: str = "repository_synthetic",
    split: str = "train",
    train_eligible: bool | None = None,
    train_license_allowlist: Iterable[str] = DEFAULT_SYNTHETIC_TRAIN_LICENSE_ALLOWLIST,
) -> Scenario:
    """Create one repository-grounded synthetic scenario with verifier evidence."""

    target_spec = target or _target_specs(spec)[0]
    family = _normalize_label(task_family)
    if family not in SUPPORTED_TASK_FAMILIES:
        raise ValueError(f"unsupported task family: {task_family}")
    if family == "bug_repair":
        raise ValueError("bug_repair should come from issue, commit, or benchmark-style sources")
    repo = _repo_name(spec)
    source_uri = _source_uri(spec, repo)
    if not source_uri:
        raise ValueError("repository-grounded synthesis requires a repository source URI")
    revision = _fixed_source_revision(spec)
    license_name = _text_field(spec.get("license"))
    allowlist = _normalized_license_set(train_license_allowlist)
    license_allowed = _license_allowed_for_train(license_name, allowlist)
    if train_eligible is True and not license_allowed:
        raise ValueError(
            f"license is not allowed for trainable synthetic seeds: {license_name or '<missing>'}"
        )
    resolved_train_eligible = (
        not _benchmark_contaminated(source_name) if train_eligible is None else train_eligible
    )
    if not license_allowed:
        resolved_train_eligible = False

    evidence = _evidence_for_family(spec, target_spec, family)
    template = TASK_FAMILY_VERIFIER_TEMPLATES[family]
    if not set(template.accepted_verifier_types).intersection(evidence.verifier_types):
        raise ValueError(
            f"missing verifier evidence for {family}; {template.minimum_evidence}"
        )

    target_name = _target_name(target_spec, repo)
    paths = _target_paths(target_spec)
    instance_id = stable_id(
        "synthetic",
        {
            "source_name": source_name,
            "repository": repo,
            "revision": revision,
            "task_family": family,
            "target": target_name,
            "paths": paths,
        },
    )
    contamination_tags = set(_list_field(spec.get("contamination_tags")))
    contamination_tags.update(benchmark_contamination_tags(source_name, "repository_synthetic"))
    if not license_allowed:
        contamination_tags.add("license_not_allowlisted")
    coverage_tags = _coverage_tags(
        spec,
        target_spec,
        family,
        repo,
        evidence.verifier_types,
    )
    seed = QuerySeed(
        public=PublicTaskContext(
            query=_query_for_family(spec, target_spec, family, repo),
            context={
                "source_instance_id": instance_id,
                "repository": repo,
                "target": target_name,
                "paths": paths,
                "synthesis_method": "repository_grounded_template",
            },
            constraints=_list_field(target_spec.get("constraints"))
            or _list_field(spec.get("constraints")),
        ),
        category=_text_field(spec.get("category")) or "software_engineering",
        difficulty=_difficulty(target_spec, spec),
        provenance=f"{source_name}:{instance_id}",
        license=license_name,
        split=split,
        task_family=family,
        source_method="repository_grounded_synthetic",
        train_eligible=resolved_train_eligible,
        contamination_tags=sorted(contamination_tags),
        verifier_types=evidence.verifier_types,
        coverage_tags=coverage_tags,
        metadata={
            "source_adapter": "repository_synthetic",
            "source_name": source_name,
            "source_instance_id": instance_id,
            "repository": repo,
            "language": _text_field(spec.get("language")),
            "target": target_name,
        },
    )
    environment = EnvironmentSpec(
        name=_environment_name(spec, repo),
        version=_text_field(spec.get("version")) or "1",
        description=f"Repository-grounded synthetic workspace for {repo}.",
        image_digest=_text_field(
            _first_present(spec, ("image_digest", "docker_image_digest", "image"))
        ),
        source_uri=source_uri,
        source_revision=revision,
        working_directory=_text_field(spec.get("working_directory")) or "/workspace",
        setup_commands=_list_field(spec.get("setup_commands")),
        capability_packs=_list_field(spec.get("capability_packs")),
        network_policy=_text_field(spec.get("network_policy")) or "disabled",
        resource_limits=_dict_field(spec.get("resource_limits")),
        health_check=_list_field(spec.get("health_check")),
        reset_strategy=_text_field(spec.get("reset_strategy")) or "recreate",
        evaluator_refs=evidence.reference_artifacts(source_name, instance_id),
        metadata={
            "source_adapter": "repository_synthetic",
            "source_name": source_name,
            "source_instance_id": instance_id,
            "task_family": family,
            "repository": repo,
            "license": license_name,
        },
    )
    evaluator = HiddenEvaluatorContext(
        reference_artifacts=evidence.reference_artifacts(source_name, instance_id),
        hidden_tests=evidence.hidden_tests,
        required_state=evidence.required_state,
        forbidden_state=evidence.forbidden_state,
        metadata={
            "source_adapter": "repository_synthetic",
            "source_name": source_name,
            "source_instance_id": instance_id,
            "task_family": family,
            "target": target_name,
            "paths": paths,
            "command_groups": evidence.command_groups,
            "diff_constraints": evidence.diff_constraints,
            "performance_threshold": evidence.performance_threshold,
            "retrieval_requirements": evidence.retrieval_requirements,
            "trace_quality_rubric": evidence.trace_quality_rubric,
        },
    )
    return Scenario(
        query_seed=seed,
        environment=environment,
        hidden_evaluator=evaluator,
        metadata={
            "source_adapter": "repository_synthetic",
            "source_name": source_name,
            "source_instance_id": instance_id,
            "task_family": family,
        },
    )


@dataclass
class _FamilyEvidence:
    verifier_types: list[str]
    command_groups: dict[str, list[str]] = field(default_factory=dict)
    required_state: dict[str, Any] = field(default_factory=dict)
    forbidden_state: dict[str, Any] = field(default_factory=dict)
    diff_constraints: list[str] = field(default_factory=list)
    performance_threshold: dict[str, Any] = field(default_factory=dict)
    retrieval_requirements: list[str] = field(default_factory=list)
    trace_quality_rubric: list[str] = field(default_factory=list)

    @property
    def hidden_tests(self) -> list[str]:
        commands: list[str] = []
        seen: set[str] = set()
        for group in self.command_groups.values():
            for command in group:
                if command not in seen:
                    commands.append(command)
                    seen.add(command)
        return commands

    def reference_artifacts(self, source_name: str, instance_id: str) -> list[str]:
        base = f"source://{quote(source_name, safe='')}/{quote(instance_id, safe='')}"
        artifacts = []
        if self.diff_constraints:
            artifacts.append(f"{base}/diff_constraints")
        if self.performance_threshold:
            artifacts.append(f"{base}/performance_threshold")
        if self.retrieval_requirements or self.trace_quality_rubric:
            artifacts.append(f"{base}/trace_requirements")
        return artifacts


def _resolved_task_families(
    spec: dict[str, Any],
    task_families: Iterable[str] | None,
) -> list[str]:
    values = list(task_families or [])
    if not values:
        values = _list_field(spec.get("task_families"))
    if not values:
        values = list(DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES)
    normalized = [_normalize_label(value) for value in values]
    seen: set[str] = set()
    output = []
    for family in normalized:
        if family and family not in seen:
            output.append(family)
            seen.add(family)
    return output


def _target_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    raw_targets = spec.get("targets")
    if raw_targets is None:
        return [dict(spec)]
    if not isinstance(raw_targets, list):
        raise ValueError("repository synthesis targets must be a JSON array")
    targets = []
    for target in raw_targets:
        if not isinstance(target, dict):
            raise ValueError("each repository synthesis target must be a JSON object")
        targets.append(dict(target))
    return targets or [dict(spec)]


def _evidence_for_family(
    spec: dict[str, Any],
    target: dict[str, Any],
    family: str,
) -> _FamilyEvidence:
    commands: dict[str, list[str]] = {}
    required_state = _merged_dict(spec, target, "required_state")
    forbidden_state = _merged_dict(spec, target, "forbidden_state")
    diff_constraints = _combined_list(spec, target, "diff_constraints")
    performance_threshold = _merged_dict(spec, target, "performance_threshold")
    retrieval_requirements = _combined_list(spec, target, "retrieval_requirements")
    trace_quality_rubric = _combined_list(spec, target, "trace_quality_rubric")

    if family in {"test_authoring", "refactor", "feature_implementation"}:
        _add_commands(commands, "hidden_command", _test_commands(spec, target))
    elif family == "dependency_upgrade":
        _add_commands(commands, "build_command", _combined_list(spec, target, "build_commands"))
        _add_commands(commands, "hidden_command", _test_commands(spec, target))
    elif family == "migration":
        _add_commands(
            commands,
            "migration_check",
            _combined_list(spec, target, "migration_commands"),
        )
        _add_commands(commands, "hidden_command", _test_commands(spec, target))
    elif family == "docs_examples":
        _add_commands(commands, "doctest", _combined_list(spec, target, "doctest_commands"))
        _add_commands(
            commands,
            "example_command",
            _combined_list(spec, target, "example_commands"),
        )
    elif family == "security_hardening":
        _add_commands(
            commands,
            "adversarial_test",
            _combined_list(spec, target, "adversarial_tests"),
        )
        _add_commands(commands, "hidden_command", _test_commands(spec, target))
    elif family == "performance":
        _add_commands(
            commands,
            "benchmark_command",
            _combined_list(spec, target, "benchmark_commands"),
        )
    elif family == "ci_build":
        _add_commands(commands, "build_command", _ci_commands(spec, target))
        _add_commands(commands, "hidden_command", _test_commands(spec, target))
    elif family == "code_review":
        _add_commands(commands, "hidden_command", _test_commands(spec, target))
    elif family == "repo_understanding":
        _add_commands(commands, "hidden_command", _test_commands(spec, target))
        if not retrieval_requirements:
            retrieval_requirements = _target_paths(target)
        if not trace_quality_rubric:
            trace_quality_rubric = [
                "The answer must cite inspected repository files or symbols."
            ]

    verifier_types = set(commands)
    if required_state:
        verifier_types.add("required_state")
    if forbidden_state:
        verifier_types.add("forbidden_state")
    if diff_constraints:
        verifier_types.add("diff_constraint")
    if performance_threshold:
        verifier_types.add("performance_threshold")
    if retrieval_requirements:
        verifier_types.add("retrieval_evidence")
    if trace_quality_rubric:
        verifier_types.add("trace_quality")

    return _FamilyEvidence(
        verifier_types=sorted(verifier_types),
        command_groups={key: value for key, value in sorted(commands.items()) if value},
        required_state=required_state,
        forbidden_state=forbidden_state,
        diff_constraints=diff_constraints,
        performance_threshold=performance_threshold,
        retrieval_requirements=retrieval_requirements,
        trace_quality_rubric=trace_quality_rubric,
    )


def _query_for_family(
    spec: dict[str, Any],
    target: dict[str, Any],
    family: str,
    repo: str,
) -> str:
    explicit = _text_field(target.get("query")) or _text_field(spec.get("query"))
    if explicit:
        return explicit
    target_name = _target_name(target, repo)
    paths = _target_paths(target)
    path_text = ", ".join(paths[:3]) if paths else "the relevant repository files"
    objective = _text_field(target.get("objective")) or _text_field(spec.get("objective"))
    if objective:
        return f"{objective} Focus on {target_name} in {repo}."
    templates = {
        "test_authoring": (
            "Add focused tests for {target} in {repo}. Preserve production behavior "
            "while covering {paths}."
        ),
        "feature_implementation": (
            "Implement the requested behavior for {target} in {repo}. "
            "Use {paths} as the main starting point and keep existing behavior compatible."
        ),
        "refactor": (
            "Refactor {target} in {repo} for clarity while preserving behavior. "
            "Use {paths} as the main starting point."
        ),
        "dependency_upgrade": (
            "Adapt {target} in {repo} for the declared dependency or runtime upgrade. "
            "Keep compatibility checks passing."
        ),
        "migration": (
            "Complete the repository migration for {target} in {repo}. "
            "Update the affected state around {paths}."
        ),
        "docs_examples": (
            "Repair or extend docs and examples for {target} in {repo}. "
            "Ensure the documented examples execute."
        ),
        "security_hardening": (
            "Harden {target} in {repo} against the declared security risk. "
            "Keep the public API behavior compatible."
        ),
        "performance": (
            "Improve the performance of {target} in {repo} without changing behavior. "
            "Use the benchmark evidence to confirm the result."
        ),
        "ci_build": (
            "Fix the CI, build, lint, packaging, or type-check path for {target} in {repo}."
        ),
        "code_review": (
            "Address the review constraints for {target} in {repo} without unrelated edits."
        ),
        "repo_understanding": (
            "Inspect {repo} and answer the repository-understanding request for {target}. "
            "Ground the answer in the relevant files."
        ),
    }
    return templates[family].format(target=target_name, repo=repo, paths=path_text)


def _coverage_tags(
    spec: dict[str, Any],
    target: dict[str, Any],
    family: str,
    repo: str,
    verifier_types: list[str],
) -> list[str]:
    tags = set(_list_field(spec.get("coverage_tags"))) | set(
        _list_field(target.get("coverage_tags"))
    )
    tags.add(f"task_family:{family}")
    tags.add("source_format:repository_synthetic")
    tags.add("source_method:repository_grounded_synthetic")
    if repo:
        tags.add(f"repo:{repo}")
    language = _text_field(spec.get("language"))
    if language:
        tags.add(f"language:{language}")
    for verifier_type in verifier_types:
        tags.add(f"verifier:{verifier_type}")
    target_name = _target_name(target, repo)
    if target_name:
        tags.add(f"target:{target_name}")
    return sorted(tags)


def _test_commands(spec: dict[str, Any], target: dict[str, Any]) -> list[str]:
    return _combined_list(spec, target, "test_commands") or _combined_list(
        spec,
        target,
        "hidden_tests",
    )


def _ci_commands(spec: dict[str, Any], target: dict[str, Any]) -> list[str]:
    return (
        _combined_list(spec, target, "ci_commands")
        or _combined_list(spec, target, "build_commands")
        or _combined_list(spec, target, "lint_commands")
        or _combined_list(spec, target, "typecheck_commands")
    )


def _add_commands(
    output: dict[str, list[str]],
    verifier_type: str,
    commands: Iterable[str],
) -> None:
    seen = set(output.get(verifier_type, []))
    for command in commands:
        if command and command not in seen:
            output.setdefault(verifier_type, []).append(command)
            seen.add(command)


def _combined_list(spec: dict[str, Any], target: dict[str, Any], key: str) -> list[str]:
    values = _list_field(spec.get(key))
    values.extend(_list_field(target.get(key)))
    seen: set[str] = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _merged_dict(spec: dict[str, Any], target: dict[str, Any], key: str) -> dict[str, Any]:
    merged = _dict_field(spec.get(key))
    merged.update(_dict_field(target.get(key)))
    return merged


def _repo_name(spec: dict[str, Any]) -> str:
    return _text_field(
        _first_present(spec, ("repository", "full_repo", "repo", "repository_name"))
    )


def _source_uri(spec: dict[str, Any], repo: str) -> str:
    explicit = _text_field(_first_present(spec, ("source_uri", "clone_url", "repo_url")))
    if explicit:
        return explicit
    if repo and "/" in repo:
        return f"https://github.com/{repo}.git"
    return ""


def _fixed_source_revision(spec: dict[str, Any]) -> str:
    revision = _text_field(
        _first_present(spec, ("source_revision", "base_commit", "commit", "base_sha"))
    )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("repository synthesis requires a 40-character fixed source revision")
    return revision.lower()


def _environment_name(spec: dict[str, Any], repo: str) -> str:
    value = _text_field(spec.get("environment_name")) or repo or "repository_synthetic"
    return value.replace("/", "__")


def _target_name(target: dict[str, Any], repo: str) -> str:
    return _text_field(target.get("name")) or _text_field(target.get("component")) or repo


def _target_paths(target: dict[str, Any]) -> list[str]:
    return (
        _list_field(target.get("paths"))
        or _list_field(target.get("files"))
        or _list_field(target.get("target_files"))
    )


def _difficulty(target: dict[str, Any], spec: dict[str, Any]) -> int:
    try:
        return int(_first_present(target, ("difficulty",)) or spec.get("difficulty", 3))
    except (TypeError, ValueError):
        return 3


def _benchmark_contaminated(source_name: str) -> bool:
    tags = benchmark_contamination_tags(
        source_name,
        "repository_synthetic",
        benchmark_sources=DEFAULT_BENCHMARK_SOURCE_ALIASES,
    )
    return bool(tags)


def _normalized_license_set(values: Iterable[str]) -> set[str]:
    return {_normalize_license(value) for value in values if _normalize_license(value)}


def _normalize_license(value: Any) -> str:
    return _text_field(value).lower().replace("-", "_").replace(" ", "_")


def _license_allowed_for_train(license_name: str, allowlist: set[str]) -> bool:
    normalized = _normalize_license(license_name)
    return bool(normalized and normalized in allowlist)


def _normalize_label(value: Any) -> str:
    return _text_field(value).lower().replace("-", "_").replace(" ", "_")


def _list_field(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            return _list_field(json.loads(stripped))
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [_text_field(item) for item in value if _text_field(item)]
    return [_text_field(value)]


def _dict_field(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("expected a JSON object")


def _text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None
