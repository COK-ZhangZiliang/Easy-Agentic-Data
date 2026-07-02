from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.scenarios import Scenario
from easy_agentic_data.seed_library import DEFAULT_BENCHMARK_SOURCE_ALIASES, is_benchmark_seed


@dataclass
class ScenarioDecontaminationIssue:
    """Scenario-level train/evaluation contamination issue."""

    code: str
    message: str
    scenario_id: str = ""
    severity: str = "warning"


@dataclass
class ScenarioDecontaminationAudit:
    """Summary of scenario-level oracle and benchmark contamination checks."""

    total: int = 0
    trainable: int = 0
    holdout: int = 0
    overlap_counts: dict[str, int] = field(default_factory=dict)
    issues: list[ScenarioDecontaminationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["valid"] = self.valid
        return value


def scenarios_from_registry(registry: ScenarioRegistry) -> list[Scenario]:
    """Load full scenario objects from a registry."""

    return [
        registry.get_scenario(row["scenario_id"]) for row in registry.list_scenarios()
    ]


def audit_scenario_decontamination(
    scenarios: Iterable[Scenario],
    *,
    holdout_scenarios: Iterable[Scenario] | None = None,
    benchmark_sources: Iterable[str] = DEFAULT_BENCHMARK_SOURCE_ALIASES,
) -> ScenarioDecontaminationAudit:
    """Compare trainable scenarios against held-out tests and evaluator artifacts."""

    scenario_list = list(scenarios)
    holdout_list = scenario_list if holdout_scenarios is None else list(holdout_scenarios)
    trainable = [
        scenario for scenario in scenario_list if scenario.query_seed.train_eligible
    ]
    holdouts = [
        scenario
        for scenario in holdout_list
        if not scenario.query_seed.train_eligible
        or is_benchmark_seed(scenario.query_seed, benchmark_sources=benchmark_sources)
    ]
    issues: list[ScenarioDecontaminationIssue] = []
    overlap_counts: Counter[str] = Counter()
    hidden_test_index = _index_holdouts(holdouts, _hidden_test_keys)
    artifact_index = _index_holdouts(holdouts, _reference_artifact_keys)
    oracle_hash_index = _index_holdouts(holdouts, _oracle_hash_keys)
    source_instance_index = _index_holdouts(holdouts, _source_instance_keys)

    for scenario in trainable:
        overlap_counts.update(
            _add_matches(
                scenario,
                hidden_test_index,
                _hidden_test_keys(scenario),
                code="holdout_hidden_test_overlap",
                message="Trainable scenario reuses held-out hidden test commands",
                severity="error",
                issues=issues,
            )
        )
        overlap_counts.update(
            _add_matches(
                scenario,
                artifact_index,
                _reference_artifact_keys(scenario),
                code="holdout_reference_artifact_overlap",
                message="Trainable scenario reuses held-out evaluator reference artifacts",
                severity="error",
                issues=issues,
            )
        )
        overlap_counts.update(
            _add_matches(
                scenario,
                oracle_hash_index,
                _oracle_hash_keys(scenario),
                code="holdout_oracle_hash_overlap",
                message="Trainable scenario reuses held-out evaluator oracle hashes",
                severity="error",
                issues=issues,
            )
        )
        overlap_counts.update(
            _add_matches(
                scenario,
                source_instance_index,
                _source_instance_keys(scenario),
                code="holdout_scenario_source_instance_overlap",
                message="Trainable scenario reuses held-out source instance metadata",
                severity="error",
                issues=issues,
            )
        )

    return ScenarioDecontaminationAudit(
        total=len(scenario_list),
        trainable=len(trainable),
        holdout=len(holdouts),
        overlap_counts=dict(sorted(overlap_counts.items())),
        issues=issues,
    )


def _index_holdouts(
    scenarios: Iterable[Scenario],
    key_fn: Any,
) -> dict[str, list[Scenario]]:
    index: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        for key in key_fn(scenario):
            index.setdefault(key, []).append(scenario)
    return index


def _add_matches(
    scenario: Scenario,
    index: dict[str, list[Scenario]],
    keys: Iterable[str],
    *,
    code: str,
    message: str,
    severity: str,
    issues: list[ScenarioDecontaminationIssue],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    matched_keys = []
    matched_scenarios: set[str] = set()
    for key in keys:
        matches = [
            holdout
            for holdout in index.get(key, [])
            if holdout.scenario_id != scenario.scenario_id
        ]
        if matches:
            matched_keys.append(key)
            matched_scenarios.update(match.scenario_id for match in matches)
    if not matched_keys:
        return counts
    counts[code] += len(matched_keys)
    issues.append(
        ScenarioDecontaminationIssue(
            code=code,
            message=(
                f"{message}; matches={len(matched_keys)}; "
                f"holdout_scenarios={len(matched_scenarios)}"
            ),
            scenario_id=scenario.scenario_id,
            severity=severity,
        )
    )
    return counts


def _hidden_test_keys(scenario: Scenario) -> list[str]:
    return sorted(
        {
            _normalized_command(command)
            for command in scenario.hidden_evaluator.hidden_tests
            if _normalized_command(command)
        }
    )


def _reference_artifact_keys(scenario: Scenario) -> list[str]:
    return sorted(
        {
            _normalize_text(artifact)
            for artifact in scenario.hidden_evaluator.reference_artifacts
            if _normalize_text(artifact)
        }
    )


def _oracle_hash_keys(scenario: Scenario) -> list[str]:
    metadata = scenario.hidden_evaluator.metadata
    keys = set()
    for field_name in ("patch_sha256", "test_patch_sha256", "reference_sha256"):
        value = _normalize_text(metadata.get(field_name, ""))
        if value:
            keys.add(f"{field_name}:{value}")
    return sorted(keys)


def _source_instance_keys(scenario: Scenario) -> list[str]:
    source_name = (
        scenario.metadata.get("source_name")
        or scenario.query_seed.metadata.get("source_name")
        or scenario.query_seed.provenance.split(":", 1)[0]
    )
    source_instance_id = (
        scenario.metadata.get("source_instance_id")
        or scenario.query_seed.metadata.get("source_instance_id")
    )
    if not source_name or not source_instance_id:
        return []
    return [f"{_normalize_source(source_name)}:{_normalize_text(source_instance_id)}"]


def _normalized_command(command: str) -> str:
    return " ".join(_normalize_text(command).split())


def _normalize_source(value: Any) -> str:
    return _normalize_text(value).replace("__", "/")


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return re.sub(r"\s+", "_", text)
