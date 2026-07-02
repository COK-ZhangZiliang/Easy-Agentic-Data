from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

from easy_agentic_data.seeds import QuerySeed

SUPPORTED_TASK_FAMILIES = {
    "bug_repair",
    "feature_implementation",
    "test_authoring",
    "refactor",
    "dependency_upgrade",
    "migration",
    "security_hardening",
    "performance",
    "docs_examples",
    "ci_build",
    "code_review",
    "repo_understanding",
}

DEFAULT_BENCHMARK_SOURCE_ALIASES = {
    "multi_swe_bench",
    "swe_bench",
    "swe_bench_lite",
    "swe_bench_multimodal",
    "swe_bench_verified",
    "multi_swe_bench/multi_swe_bench",
    "princeton_nlp/swe_bench",
    "princeton_nlp/swe_bench_lite",
    "princeton_nlp/swe_bench_verified",
}


@dataclass(frozen=True)
class TaskFamilyVerifierTemplate:
    """Minimum executable evidence expected for one task family."""

    task_family: str
    accepted_verifier_types: tuple[str, ...]
    minimum_evidence: str


TASK_FAMILY_VERIFIER_TEMPLATES = {
    "bug_repair": TaskFamilyVerifierTemplate(
        task_family="bug_repair",
        accepted_verifier_types=("hidden_command", "hidden_test_patch"),
        minimum_evidence="A failing behavior must be checked by a hidden command or test patch.",
    ),
    "feature_implementation": TaskFamilyVerifierTemplate(
        task_family="feature_implementation",
        accepted_verifier_types=("hidden_command", "required_state"),
        minimum_evidence="The new behavior must be checked by executable tests or state checks.",
    ),
    "test_authoring": TaskFamilyVerifierTemplate(
        task_family="test_authoring",
        accepted_verifier_types=("hidden_command", "hidden_test_patch"),
        minimum_evidence="The added tests must be inspected or run against the target behavior.",
    ),
    "refactor": TaskFamilyVerifierTemplate(
        task_family="refactor",
        accepted_verifier_types=("hidden_command", "forbidden_state"),
        minimum_evidence="Preserved behavior and forbidden rewrites must be checked.",
    ),
    "dependency_upgrade": TaskFamilyVerifierTemplate(
        task_family="dependency_upgrade",
        accepted_verifier_types=("build_command", "hidden_command"),
        minimum_evidence="Build, import, or compatibility commands must validate the upgrade.",
    ),
    "migration": TaskFamilyVerifierTemplate(
        task_family="migration",
        accepted_verifier_types=("hidden_command", "migration_check", "required_state"),
        minimum_evidence="The migrated state must be validated by commands or state checks.",
    ),
    "security_hardening": TaskFamilyVerifierTemplate(
        task_family="security_hardening",
        accepted_verifier_types=("adversarial_test", "hidden_command"),
        minimum_evidence="An adversarial or hidden regression test must exercise the risk.",
    ),
    "performance": TaskFamilyVerifierTemplate(
        task_family="performance",
        accepted_verifier_types=("benchmark_command", "performance_threshold"),
        minimum_evidence="A bounded benchmark or metric threshold must validate the improvement.",
    ),
    "docs_examples": TaskFamilyVerifierTemplate(
        task_family="docs_examples",
        accepted_verifier_types=("doctest", "example_command"),
        minimum_evidence="The documented example must execute or be checked as a doctest.",
    ),
    "ci_build": TaskFamilyVerifierTemplate(
        task_family="ci_build",
        accepted_verifier_types=("build_command", "hidden_command"),
        minimum_evidence="The broken CI, build, lint, packaging, or type command must pass.",
    ),
    "code_review": TaskFamilyVerifierTemplate(
        task_family="code_review",
        accepted_verifier_types=("diff_constraint", "hidden_command"),
        minimum_evidence="Review constraints and targeted behavior must be checked.",
    ),
    "repo_understanding": TaskFamilyVerifierTemplate(
        task_family="repo_understanding",
        accepted_verifier_types=("retrieval_evidence", "trace_quality"),
        minimum_evidence="The trace must include retrieval or reasoning evidence for the answer.",
    ),
}


@dataclass
class SeedLibraryIssue:
    """Actionable seed-library hygiene issue."""

    code: str
    message: str
    seed_id: str = ""
    severity: str = "warning"


@dataclass
class SeedLibraryAudit:
    """Coverage and contamination summary for a set of query seeds."""

    total: int = 0
    train_eligible: int = 0
    train_blocked: int = 0
    benchmark_blocked: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    task_family_counts: dict[str, int] = field(default_factory=dict)
    source_method_counts: dict[str, int] = field(default_factory=dict)
    verifier_type_counts: dict[str, int] = field(default_factory=dict)
    coverage_tag_counts: dict[str, int] = field(default_factory=dict)
    contamination_tag_counts: dict[str, int] = field(default_factory=dict)
    train_task_family_counts: dict[str, int] = field(default_factory=dict)
    train_source_method_counts: dict[str, int] = field(default_factory=dict)
    train_repository_counts: dict[str, int] = field(default_factory=dict)
    train_language_counts: dict[str, int] = field(default_factory=dict)
    decontamination_counts: dict[str, int] = field(default_factory=dict)
    issues: list[SeedLibraryIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["valid"] = self.valid
        return value


@dataclass
class SeedLibraryPolicy:
    """Scale-up gates for the trainable seed pool."""

    min_train_eligible: int = 0
    required_task_families: list[str] = field(default_factory=list)
    required_verifier_types: list[str] = field(default_factory=list)
    max_task_family_share: float = 1.0
    max_source_method_share: float = 1.0
    max_repository_share: float = 1.0
    max_language_share: float = 1.0

    def __post_init__(self) -> None:
        self.min_train_eligible = max(0, int(self.min_train_eligible))
        self.required_task_families = _normalize_labels(self.required_task_families)
        self.required_verifier_types = _normalize_labels(self.required_verifier_types)
        for field_name in (
            "max_task_family_share",
            "max_source_method_share",
            "max_repository_share",
            "max_language_share",
        ):
            value = float(getattr(self, field_name))
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"{field_name} must be in the range (0, 1]")
            setattr(self, field_name, value)


def audit_seed_library(
    seeds: Iterable[QuerySeed],
    *,
    benchmark_sources: Iterable[str] = DEFAULT_BENCHMARK_SOURCE_ALIASES,
    policy: SeedLibraryPolicy | None = None,
    holdout_seeds: Iterable[QuerySeed] | None = None,
) -> SeedLibraryAudit:
    """Summarize task coverage and flag train/evaluation contamination risks."""

    seed_list = list(seeds)
    benchmark_aliases = _benchmark_aliases(benchmark_sources)
    split_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    source_method_counts: Counter[str] = Counter()
    verifier_counts: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    contamination_counts: Counter[str] = Counter()
    train_family_counts: Counter[str] = Counter()
    train_source_method_counts: Counter[str] = Counter()
    train_repository_counts: Counter[str] = Counter()
    train_language_counts: Counter[str] = Counter()
    train_verifier_counts: Counter[str] = Counter()
    issues: list[SeedLibraryIssue] = []
    total = 0
    train_eligible = 0
    train_blocked = 0
    benchmark_blocked = 0

    for seed in seed_list:
        total += 1
        split_counts[seed.split] += 1
        family_counts[seed.task_family] += 1
        source_method_counts[seed.source_method] += 1
        verifier_counts.update(seed.verifier_types)
        coverage_counts.update(seed.coverage_tags)
        contamination_counts.update(seed.contamination_tags)
        benchmark = _is_benchmark_seed(seed, benchmark_aliases)

        if seed.train_eligible:
            train_eligible += 1
            train_family_counts[seed.task_family] += 1
            train_source_method_counts[seed.source_method] += 1
            train_verifier_counts.update(seed.verifier_types)
            repository = _seed_repository(seed)
            if repository:
                train_repository_counts[repository] += 1
            language = _seed_language(seed)
            if language:
                train_language_counts[language] += 1
        else:
            train_blocked += 1
        if benchmark and not seed.train_eligible:
            benchmark_blocked += 1
        if benchmark and seed.train_eligible:
            issues.append(
                SeedLibraryIssue(
                    code="benchmark_train_eligible",
                    message="Benchmark-derived seed is marked train-eligible",
                    seed_id=seed.seed_id,
                    severity="error",
                )
            )
        if seed.task_family == "general":
            issues.append(
                SeedLibraryIssue(
                    code="missing_task_family",
                    message="Seed uses the generic task family",
                    seed_id=seed.seed_id,
                )
            )
        if seed.source_method == "unspecified":
            issues.append(
                SeedLibraryIssue(
                    code="missing_source_method",
                    message="Seed source method is unspecified",
                    seed_id=seed.seed_id,
                )
            )
        if not seed.verifier_types:
            issues.append(
                SeedLibraryIssue(
                    code="missing_verifier",
                    message="Seed does not declare any executable verifier type",
                    seed_id=seed.seed_id,
                )
            )
        if seed.task_family not in SUPPORTED_TASK_FAMILIES and seed.train_eligible:
            issues.append(
                SeedLibraryIssue(
                    code="unknown_task_family",
                    message=(
                        "Train-eligible seed uses an unsupported task family: "
                        f"{seed.task_family}"
                    ),
                    seed_id=seed.seed_id,
                )
            )
        template = TASK_FAMILY_VERIFIER_TEMPLATES.get(seed.task_family)
        if template is not None and seed.train_eligible:
            accepted = set(template.accepted_verifier_types)
            if not accepted.intersection(seed.verifier_types):
                issues.append(
                    SeedLibraryIssue(
                        code="family_verifier_gap",
                        message=(
                            f"Task family {seed.task_family} requires one of "
                            f"{sorted(accepted)}; {template.minimum_evidence}"
                        ),
                        seed_id=seed.seed_id,
                        severity="error",
                    )
                )
        if seed.train_eligible and not seed.license:
            issues.append(
                SeedLibraryIssue(
                    code="train_without_license",
                    message="Train-eligible seed does not declare a license",
                    seed_id=seed.seed_id,
                    severity="error",
                )
            )

    if holdout_seeds is None:
        holdout_list = seed_list
    else:
        holdout_list = list(holdout_seeds)
    decontamination_counts = _add_decontamination_issues(
        seed_list,
        holdout_list,
        benchmark_aliases,
        issues,
    )
    _add_policy_issues(
        SeedLibraryPolicy() if policy is None else policy,
        train_eligible,
        train_family_counts,
        train_source_method_counts,
        train_repository_counts,
        train_language_counts,
        train_verifier_counts,
        issues,
    )

    return SeedLibraryAudit(
        total=total,
        train_eligible=train_eligible,
        train_blocked=train_blocked,
        benchmark_blocked=benchmark_blocked,
        split_counts=dict(sorted(split_counts.items())),
        task_family_counts=dict(sorted(family_counts.items())),
        source_method_counts=dict(sorted(source_method_counts.items())),
        verifier_type_counts=dict(sorted(verifier_counts.items())),
        coverage_tag_counts=dict(sorted(coverage_counts.items())),
        contamination_tag_counts=dict(sorted(contamination_counts.items())),
        train_task_family_counts=dict(sorted(train_family_counts.items())),
        train_source_method_counts=dict(sorted(train_source_method_counts.items())),
        train_repository_counts=dict(sorted(train_repository_counts.items())),
        train_language_counts=dict(sorted(train_language_counts.items())),
        decontamination_counts=dict(sorted(decontamination_counts.items())),
        issues=issues,
    )


def is_benchmark_seed(
    seed: QuerySeed,
    *,
    benchmark_sources: Iterable[str] = DEFAULT_BENCHMARK_SOURCE_ALIASES,
) -> bool:
    """Return true when a seed comes from a known evaluation benchmark source."""

    return _is_benchmark_seed(seed, _benchmark_aliases(benchmark_sources))


def _is_benchmark_seed(seed: QuerySeed, aliases: set[str]) -> bool:
    source_name = seed.metadata.get("source_name", "") or seed.provenance.split(":", 1)[0]
    candidates = [
        source_name,
        seed.metadata.get("dataset", ""),
    ]
    if not source_name:
        candidates.append(seed.metadata.get("source_format", ""))
    return any(_normalize_source(value) in aliases for value in candidates) or any(
        tag == "benchmark_source" or tag.startswith("benchmark:")
        for tag in seed.contamination_tags
    )


def _add_decontamination_issues(
    seeds: Iterable[QuerySeed],
    holdout_seeds: Iterable[QuerySeed],
    benchmark_aliases: set[str],
    issues: list[SeedLibraryIssue],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    trainable = [seed for seed in seeds if seed.train_eligible]
    holdouts = [
        seed
        for seed in holdout_seeds
        if not seed.train_eligible or _is_benchmark_seed(seed, benchmark_aliases)
    ]
    query_index = _index_holdouts(holdouts, _normalized_query)
    provenance_index = _index_holdouts(holdouts, lambda seed: seed.provenance)
    instance_index = _index_holdouts(holdouts, _seed_source_instance_key)
    repository_index = _index_holdouts(holdouts, _seed_repository)

    for seed in trainable:
        counts.update(
            _add_index_matches(
                seed,
                query_index,
                _normalized_query(seed),
                code="holdout_query_overlap",
                message="Train-eligible seed has the same normalized query as a holdout seed",
                severity="error",
                issues=issues,
            )
        )
        counts.update(
            _add_index_matches(
                seed,
                provenance_index,
                seed.provenance,
                code="holdout_provenance_overlap",
                message="Train-eligible seed reuses holdout provenance",
                severity="error",
                issues=issues,
            )
        )
        counts.update(
            _add_index_matches(
                seed,
                instance_index,
                _seed_source_instance_key(seed),
                code="holdout_source_instance_overlap",
                message="Train-eligible seed reuses a holdout source instance",
                severity="error",
                issues=issues,
            )
        )
        counts.update(
            _add_index_matches(
                seed,
                repository_index,
                _seed_repository(seed),
                code="holdout_repository_overlap",
                message="Train-eligible seed shares a repository with holdout seeds",
                severity="warning",
                issues=issues,
            )
        )
    return dict(counts)


def _add_policy_issues(
    policy: SeedLibraryPolicy,
    train_total: int,
    train_family_counts: Counter[str],
    train_source_method_counts: Counter[str],
    train_repository_counts: Counter[str],
    train_language_counts: Counter[str],
    train_verifier_counts: Counter[str],
    issues: list[SeedLibraryIssue],
) -> None:
    if train_total < policy.min_train_eligible:
        issues.append(
            SeedLibraryIssue(
                code="min_train_eligible_not_met",
                message=(
                    f"Trainable seed count {train_total} is below the required "
                    f"minimum {policy.min_train_eligible}"
                ),
                severity="error",
            )
        )
    for task_family in policy.required_task_families:
        if train_family_counts.get(task_family, 0) == 0:
            issues.append(
                SeedLibraryIssue(
                    code="missing_required_task_family",
                    message=f"Required task family is absent from trainable seeds: {task_family}",
                    severity="error",
                )
            )
    for verifier_type in policy.required_verifier_types:
        if train_verifier_counts.get(verifier_type, 0) == 0:
            issues.append(
                SeedLibraryIssue(
                    code="missing_required_verifier",
                    message=(
                        "Required verifier type is absent from trainable seeds: "
                        f"{verifier_type}"
                    ),
                    severity="error",
                )
            )
    _add_share_issue(
        train_family_counts,
        train_total,
        policy.max_task_family_share,
        "task_family_dominance",
        "Task family",
        issues,
    )
    _add_share_issue(
        train_source_method_counts,
        train_total,
        policy.max_source_method_share,
        "source_method_dominance",
        "Source method",
        issues,
    )
    _add_share_issue(
        train_repository_counts,
        train_total,
        policy.max_repository_share,
        "repository_dominance",
        "Repository",
        issues,
    )
    _add_share_issue(
        train_language_counts,
        train_total,
        policy.max_language_share,
        "language_dominance",
        "Language",
        issues,
    )


def benchmark_contamination_tags(
    source_name: str,
    source_format: str = "",
    *,
    benchmark_sources: Iterable[str] = DEFAULT_BENCHMARK_SOURCE_ALIASES,
) -> list[str]:
    """Create contamination tags for known benchmark sources."""

    aliases = {_normalize_source(value) for value in benchmark_sources}
    tags: set[str] = set()
    candidates = [source_name] if source_name else [source_format]
    for candidate in candidates:
        normalized = _normalize_source(candidate)
        if normalized in aliases:
            tags.add("benchmark_source")
            tags.add(f"benchmark:{normalized.replace('/', ':')}")
    return sorted(tags)


def default_train_eligible_for_source(source_name: str, source_format: str = "") -> bool:
    """Default to non-train for known benchmark sources."""

    return not benchmark_contamination_tags(source_name, source_format)


def default_task_family_for_swe_record(record: dict[str, Any], source_format: str) -> str:
    explicit = _normalize_label(record.get("task_family"))
    if explicit:
        return explicit
    if source_format in {"swe_bench", "swe_smith", "multi_swe"}:
        return "bug_repair"
    return "software_engineering"


def default_source_method_for_swe_source(source_format: str) -> str:
    if source_format == "swe_smith":
        return "synthetic_issue_workspace"
    return "external_issue_workspace"


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_labels(values: Iterable[str]) -> list[str]:
    return sorted({_normalize_label(value) for value in values if _normalize_label(value)})


def _normalize_source(value: Any) -> str:
    return _normalize_label(value).replace("__", "/")


def _benchmark_aliases(values: Iterable[str]) -> set[str]:
    return {_normalize_source(value) for value in values}


def _normalized_query(seed: QuerySeed) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", seed.public.query.lower()))


def _seed_repository(seed: QuerySeed) -> str:
    value = seed.public.context.get("repository") or seed.metadata.get("repository", "")
    return _normalize_source(value)


def _seed_language(seed: QuerySeed) -> str:
    for tag in seed.coverage_tags:
        if tag.startswith("language:"):
            return _normalize_label(tag.split(":", 1)[1])
    return _normalize_label(seed.metadata.get("language", ""))


def _seed_source_instance_key(seed: QuerySeed) -> str:
    source_name = seed.metadata.get("source_name", "") or seed.provenance.split(":", 1)[0]
    source_instance_id = seed.metadata.get("source_instance_id", "")
    if not source_name or not source_instance_id:
        return ""
    return f"{_normalize_source(source_name)}:{_normalize_label(source_instance_id)}"


def _index_holdouts(
    seeds: Iterable[QuerySeed],
    key_fn: Any,
) -> dict[str, list[QuerySeed]]:
    index: dict[str, list[QuerySeed]] = {}
    for seed in seeds:
        key = key_fn(seed)
        if key:
            index.setdefault(key, []).append(seed)
    return index


def _add_index_matches(
    seed: QuerySeed,
    index: dict[str, list[QuerySeed]],
    key: str,
    *,
    code: str,
    message: str,
    severity: str,
    issues: list[SeedLibraryIssue],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not key:
        return counts
    matches = [
        holdout
        for holdout in index.get(key, [])
        if holdout.seed_id != seed.seed_id or not seed.train_eligible
    ]
    if not matches:
        return counts
    counts[code] += len(matches)
    issues.append(
        SeedLibraryIssue(
            code=code,
            message=f"{message}; matches={len(matches)}; key={key}",
            seed_id=seed.seed_id,
            severity=severity,
        )
    )
    return counts


def _add_share_issue(
    counts: Counter[str],
    total: int,
    max_share: float,
    code: str,
    label: str,
    issues: list[SeedLibraryIssue],
) -> None:
    if total <= 0 or max_share >= 1.0 or not counts:
        return
    value, count = counts.most_common(1)[0]
    share = count / total
    if share > max_share:
        issues.append(
            SeedLibraryIssue(
                code=code,
                message=(
                    f"{label} '{value}' accounts for {share:.3f} of trainable seeds, "
                    f"above the maximum share {max_share:.3f}"
                ),
                severity="error",
            )
        )
