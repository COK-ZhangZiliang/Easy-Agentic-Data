from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from easy_agentic_data.models import stable_id
from easy_agentic_data.scenarios import Scenario


@dataclass
class SeedReviewQueue:
    """Stratified seed/scenario human-review queue."""

    total_scenarios: int = 0
    selected: int = 0
    stratum_counts: dict[str, int] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_seed_review_queue(
    scenarios: Iterable[Scenario],
    *,
    sample_per_stratum: int = 1,
    max_records: int | None = None,
) -> SeedReviewQueue:
    """Sample review records by task family, difficulty, source method, and verifier type."""

    scenario_list = sorted(
        list(scenarios),
        key=lambda scenario: scenario.scenario_id,
    )
    if sample_per_stratum < 1:
        raise ValueError("sample_per_stratum must be at least 1")
    strata = _strata(scenario_list)
    records: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    for stratum_key, candidates in sorted(strata.items()):
        for scenario in candidates[:sample_per_stratum]:
            if max_records is not None and len(records) >= max_records:
                return SeedReviewQueue(
                    total_scenarios=len(scenario_list),
                    selected=len(records),
                    stratum_counts=dict(sorted(stratum_counts.items())),
                    records=records,
                )
            records.append(_review_record(scenario, stratum_key))
            stratum_counts[stratum_key] += 1
    return SeedReviewQueue(
        total_scenarios=len(scenario_list),
        selected=len(records),
        stratum_counts=dict(sorted(stratum_counts.items())),
        records=records,
    )


def _strata(scenarios: list[Scenario]) -> dict[str, list[Scenario]]:
    strata: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        seed = scenario.query_seed
        verifier_types = seed.verifier_types or ["none"]
        for verifier_type in verifier_types:
            key = "|".join(
                [
                    f"family={seed.task_family}",
                    f"difficulty={seed.difficulty}",
                    f"source={seed.source_method}",
                    f"verifier={verifier_type}",
                ]
            )
            strata.setdefault(key, []).append(scenario)
    return strata


def _review_record(scenario: Scenario, stratum_key: str) -> dict[str, Any]:
    seed = scenario.query_seed
    record = {
        "review_id": stable_id(
            "review",
            {
                "scenario_id": scenario.scenario_id,
                "stratum": stratum_key,
            },
        ),
        "reason": "seed library stratified sample",
        "stratum": stratum_key,
        "scenario_id": scenario.scenario_id,
        "seed_id": seed.seed_id,
        "query": seed.public.query,
        "split": seed.split,
        "train_eligible": seed.train_eligible,
        "task_family": seed.task_family,
        "difficulty": seed.difficulty,
        "source_method": seed.source_method,
        "verifier_types": seed.verifier_types,
        "coverage_tags": seed.coverage_tags,
        "contamination_tags": seed.contamination_tags,
        "repository": seed.public.context.get("repository") or seed.metadata.get("repository", ""),
        "source_name": seed.metadata.get("source_name", ""),
        "source_instance_id": seed.metadata.get("source_instance_id", ""),
        "environment_id": scenario.environment.environment_id,
        "environment_source_uri": scenario.environment.source_uri,
        "environment_source_revision": scenario.environment.source_revision,
        "review_questions": [
            "Is the public query realistic for the declared task family?",
            "Is the seed free of benchmark, secret, or personal-data contamination?",
            "Do the verifier types and hidden evaluator metadata match the task?",
            "Is the workspace source reproducible and license-compatible?",
        ],
    }
    return record
