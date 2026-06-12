from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import tarfile
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


@dataclass
class RegistryIssue:
    code: str
    message: str
    entry_id: str = ""


@dataclass
class RegistryValidation:
    issues: List[RegistryIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues


class ScenarioRegistry:
    """Git-friendly JSON registry with a disposable SQLite discovery index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.seed_dir = self.root / "seeds"
        self.environment_dir = self.root / "environments"
        self.scenario_dir = self.root / "scenarios"
        self.database = self.root / "registry.sqlite3"

    def initialize(self) -> None:
        for directory in (self.seed_dir, self.environment_dir, self.scenario_dir):
            directory.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS seeds (
                    seed_id TEXT PRIMARY KEY, category TEXT, difficulty INTEGER,
                    split TEXT, provenance TEXT, query_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS environments (
                    environment_id TEXT PRIMARY KEY, name TEXT, version TEXT,
                    source_uri TEXT, source_revision TEXT
                );
                CREATE TABLE IF NOT EXISTS scenarios (
                    scenario_id TEXT PRIMARY KEY, seed_id TEXT, environment_id TEXT
                );
                CREATE TABLE IF NOT EXISTS rollouts (
                    job_id TEXT PRIMARY KEY, scenario_id TEXT, status TEXT,
                    trace_id TEXT, attempts INTEGER DEFAULT 0, error TEXT DEFAULT ''
                );
                """
            )

    def add_seed(self, seed: QuerySeed) -> None:
        self.initialize()
        _write_json(self.seed_dir / f"{seed.seed_id}.json", seed.to_dict())
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO seeds VALUES (?, ?, ?, ?, ?, ?)",
                (
                    seed.seed_id, seed.category, seed.difficulty, seed.split,
                    seed.provenance, _normalized_query(seed.public.query),
                ),
            )

    def add_environment(self, environment: EnvironmentSpec) -> None:
        self.initialize()
        _write_json(
            self.environment_dir / f"{environment.environment_id}.json",
            environment.to_dict(),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO environments VALUES (?, ?, ?, ?, ?)",
                (
                    environment.environment_id, environment.name, environment.version,
                    environment.source_uri, environment.source_revision,
                ),
            )

    def add_scenario(self, scenario: Scenario) -> None:
        self.add_seed(scenario.query_seed)
        self.add_environment(scenario.environment)
        _write_json(self.scenario_dir / f"{scenario.scenario_id}.json", scenario.to_dict())
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO scenarios VALUES (?, ?, ?)",
                (
                    scenario.scenario_id, scenario.query_seed.seed_id,
                    scenario.environment.environment_id,
                ),
            )

    def get_scenario(self, scenario_id: str) -> Scenario:
        return Scenario.from_dict(_read_json(self.scenario_dir / f"{scenario_id}.json"))

    def list_scenarios(self) -> List[Dict[str, str]]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT scenario_id, seed_id, environment_id FROM scenarios ORDER BY scenario_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def materialize(
        self,
        scenario_id: str,
        *,
        random_seed: int,
        parameters: Dict[str, Any] | None = None,
        initial_state_hash: str = "",
    ) -> ScenarioInstance:
        return ScenarioInstance.materialize(
            self.get_scenario(scenario_id),
            random_seed=random_seed,
            parameters=parameters,
            initial_state_hash=initial_state_hash,
        )

    def rebuild_index(self) -> None:
        if self.database.exists():
            self.database.unlink()
        self.initialize()
        for path in sorted(self.scenario_dir.glob("*.json")):
            self.add_scenario(Scenario.from_dict(_read_json(path)))

    def validate(self) -> RegistryValidation:
        issues: List[RegistryIssue] = []
        seeds = [
            QuerySeed.from_dict(_read_json(path)) for path in sorted(self.seed_dir.glob("*.json"))
        ]
        seen_ids: set[str] = set()
        query_splits: Dict[str, set[str]] = {}
        provenance_splits: Dict[str, set[str]] = {}
        for seed in seeds:
            if seed.seed_id in seen_ids:
                issues.append(RegistryIssue("duplicate_id", "Duplicate seed ID", seed.seed_id))
            seen_ids.add(seed.seed_id)
            query_splits.setdefault(_normalized_query(seed.public.query), set()).add(seed.split)
            if seed.provenance:
                provenance_splits.setdefault(seed.provenance, set()).add(seed.split)
        for value, splits in query_splits.items():
            if "train" in splits and "evaluation" in splits:
                issues.append(RegistryIssue("split_leakage", f"Query appears across splits: {value}"))
        for source, splits in provenance_splits.items():
            if "train" in splits and "evaluation" in splits:
                issues.append(RegistryIssue("source_leakage", f"Source appears across splits: {source}"))
        for path in sorted(self.environment_dir.glob("*.json")):
            environment = EnvironmentSpec.from_dict(_read_json(path))
            if environment.image_digest and "@sha256:" not in environment.image_digest:
                issues.append(
                    RegistryIssue(
                        "mutable_image", "Environment image is not pinned by digest",
                        environment.environment_id,
                    )
                )
            if environment.source_uri.startswith("file://") and not Path(
                environment.source_uri[7:]
            ).exists():
                issues.append(
                    RegistryIssue("missing_source", "Environment source is missing", environment.environment_id)
                )
            elif environment.source_uri.startswith("file://") and environment.source_revision:
                completed = subprocess.run(
                    [
                        "git", "-C", environment.source_uri[7:], "cat-file", "-e",
                        f"{environment.source_revision}^{{commit}}",
                    ],
                    capture_output=True,
                )
                if completed.returncode != 0:
                    issues.append(
                        RegistryIssue(
                            "invalid_revision",
                            "Environment source revision is not a valid commit",
                            environment.environment_id,
                        )
                    )
        return RegistryValidation(issues)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection


def import_repository_environment(
    repository: str | Path,
    revision: str,
    *,
    name: str,
    version: str = "1",
) -> EnvironmentSpec:
    repository_path = Path(repository).resolve()
    completed = subprocess.run(
        ["git", "-C", str(repository_path), "rev-parse", f"{revision}^{{commit}}"],
        text=True, capture_output=True, check=True,
    )
    commit = completed.stdout.strip()
    return EnvironmentSpec(
        name=name,
        version=version,
        source_uri=repository_path.as_uri(),
        source_revision=commit,
    )


def materialize_environment_source(environment: EnvironmentSpec, destination: str | Path) -> Path:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    if not environment.source_uri:
        return destination_path
    if not environment.source_uri.startswith("file://"):
        raise ValueError(f"Unsupported source URI for local materialization: {environment.source_uri}")
    repository = Path(environment.source_uri[7:])
    if environment.source_revision:
        archive = subprocess.run(
            ["git", "-C", str(repository), "archive", environment.source_revision],
            capture_output=True, check=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            members = bundle.getmembers()
            for member in members:
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe repository archive path: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"Repository archive links are not supported: {member.name}")
            bundle.extractall(destination_path, members=members)
    else:
        shutil.copytree(repository, destination_path, dirs_exist_ok=True)
    if environment.fixture_patch:
        subprocess.run(
            ["patch", "-p1", "-i", str(Path(environment.fixture_patch).resolve())],
            cwd=destination_path,
            check=True,
            text=True,
            capture_output=True,
        )
    return destination_path


def mutation_seed(
    *,
    query: str,
    failing_test: str,
    provenance: str,
    category: str = "software_engineering",
) -> QuerySeed:
    return QuerySeed(
        public=PublicTaskContext(query=query, context={"failing_test": failing_test}),
        category=category,
        provenance=provenance,
        metadata={"source_adapter": "mutation"},
    )


def issue_commit_scenario(
    *,
    issue_text: str,
    environment: EnvironmentSpec,
    reference_patch_artifact: str,
    hidden_tests: Iterable[str],
    provenance: str,
) -> Scenario:
    seed = QuerySeed(
        public=PublicTaskContext(query=issue_text),
        category="software_engineering",
        provenance=provenance,
        metadata={"source_adapter": "issue_commit"},
    )
    return Scenario(
        seed,
        environment,
        HiddenEvaluatorContext(
            reference_artifacts=[reference_patch_artifact],
            hidden_tests=list(hidden_tests),
        ),
    )


def exact_duplicate_groups(seeds: Iterable[QuerySeed]) -> List[List[str]]:
    groups: Dict[str, List[str]] = {}
    for seed in seeds:
        groups.setdefault(_normalized_query(seed.public.query), []).append(seed.seed_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def semantic_duplicate_candidates(seeds: Iterable[QuerySeed]) -> List[tuple[str, str]]:
    items = list(seeds)
    pairs = []
    for index, left in enumerate(items):
        left_tokens = set(_semantic_tokens(left.public.query))
        for right in items[index + 1:]:
            right_tokens = set(_semantic_tokens(right.public.query))
            union = left_tokens | right_tokens
            if union and len(left_tokens & right_tokens) / len(union) >= 0.8:
                pairs.append((left.seed_id, right.seed_id))
    return pairs


def _tokens(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _semantic_tokens(value: str) -> List[str]:
    return [token for token in _tokens(value) if not token.isdigit()]


def _normalized_query(value: str) -> str:
    return " ".join(_tokens(value))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
