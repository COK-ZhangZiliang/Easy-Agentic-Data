from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from easy_agentic_data.environments import EnvironmentSpec, is_immutable_image_reference
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


@dataclass
class RegistryIssue:
    code: str
    message: str
    entry_id: str = ""


@dataclass
class RegistryValidation:
    issues: list[RegistryIssue] = field(default_factory=list)

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
            _ensure_seed_columns(connection)

    def add_seed(self, seed: QuerySeed) -> None:
        self.initialize()
        _write_json(self.seed_dir / f"{seed.seed_id}.json", seed.to_dict())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO seeds (
                    seed_id, category, difficulty, split, provenance, query_hash,
                    task_family, source_method, train_eligible, contamination_tags,
                    verifier_types, coverage_tags
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seed.seed_id,
                    seed.category,
                    seed.difficulty,
                    seed.split,
                    seed.provenance,
                    _normalized_query(seed.public.query),
                    seed.task_family,
                    seed.source_method,
                    int(seed.train_eligible),
                    json.dumps(seed.contamination_tags, sort_keys=True),
                    json.dumps(seed.verifier_types, sort_keys=True),
                    json.dumps(seed.coverage_tags, sort_keys=True),
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
                    environment.environment_id,
                    environment.name,
                    environment.version,
                    environment.source_uri,
                    environment.source_revision,
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
                    scenario.scenario_id,
                    scenario.query_seed.seed_id,
                    scenario.environment.environment_id,
                ),
            )

    def get_scenario(self, scenario_id: str) -> Scenario:
        return Scenario.from_dict(_read_json(self.scenario_dir / f"{scenario_id}.json"))

    def list_scenarios(self) -> list[dict[str, str]]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT scenario_id, seed_id, environment_id FROM scenarios ORDER BY scenario_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_seeds(self) -> list[QuerySeed]:
        self.initialize()
        return [
            QuerySeed.from_dict(_read_json(path)) for path in sorted(self.seed_dir.glob("*.json"))
        ]

    def materialize(
        self,
        scenario_id: str,
        *,
        random_seed: int,
        parameters: dict[str, Any] | None = None,
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
        issues: list[RegistryIssue] = []
        seeds = [
            QuerySeed.from_dict(_read_json(path)) for path in sorted(self.seed_dir.glob("*.json"))
        ]
        seen_ids: set[str] = set()
        query_splits: dict[str, set[str]] = {}
        provenance_splits: dict[str, set[str]] = {}
        for seed in seeds:
            if seed.seed_id in seen_ids:
                issues.append(RegistryIssue("duplicate_id", "Duplicate seed ID", seed.seed_id))
            seen_ids.add(seed.seed_id)
            query_splits.setdefault(_normalized_query(seed.public.query), set()).add(seed.split)
            if seed.provenance:
                provenance_splits.setdefault(seed.provenance, set()).add(seed.split)
        for value, splits in query_splits.items():
            if "train" in splits and "evaluation" in splits:
                issues.append(
                    RegistryIssue("split_leakage", f"Query appears across splits: {value}")
                )
        for source, splits in provenance_splits.items():
            if "train" in splits and "evaluation" in splits:
                issues.append(
                    RegistryIssue("source_leakage", f"Source appears across splits: {source}")
                )
        for path in sorted(self.environment_dir.glob("*.json")):
            environment = EnvironmentSpec.from_dict(_read_json(path))
            if environment.image_digest and not is_immutable_image_reference(
                environment.image_digest
            ):
                issues.append(
                    RegistryIssue(
                        "mutable_image",
                        "Environment image is not content-addressed by digest",
                        environment.environment_id,
                    )
                )
            if (
                environment.source_uri.startswith("file://")
                and not Path(environment.source_uri[7:]).exists()
            ):
                issues.append(
                    RegistryIssue(
                        "missing_source",
                        "Environment source is missing",
                        environment.environment_id,
                    )
                )
            elif environment.source_uri.startswith("file://") and environment.source_revision:
                completed = subprocess.run(
                    [
                        "git",
                        "-C",
                        environment.source_uri[7:],
                        "cat-file",
                        "-e",
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
        text=True,
        capture_output=True,
        check=True,
    )
    commit = completed.stdout.strip()
    return EnvironmentSpec(
        name=name,
        version=version,
        source_uri=repository_path.as_uri(),
        source_revision=commit,
    )


def materialize_environment_source(
    environment: EnvironmentSpec,
    destination: str | Path,
    *,
    run_health_checks: bool = True,
) -> Path:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    if not environment.source_uri:
        if run_health_checks:
            run_environment_health_checks(environment, destination_path)
        return destination_path
    if not environment.source_uri.startswith("file://"):
        raise ValueError(
            f"Unsupported source URI for local materialization: {environment.source_uri}"
        )
    repository = Path(environment.source_uri[7:])
    if environment.source_revision:
        archive = subprocess.run(
            ["git", "-C", str(repository), "archive", environment.source_revision],
            capture_output=True,
            check=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            members = bundle.getmembers()
            for member in members:
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe repository archive path: {member.name}")
                if member.issym() or member.islnk():
                    _validate_archive_link(member)
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
    if run_health_checks:
        run_environment_health_checks(environment, destination_path)
    return destination_path


def run_environment_health_checks(environment: EnvironmentSpec, workspace: str | Path) -> None:
    workspace_path = Path(workspace)
    for command in environment.health_check:
        arguments = _command_arguments(command, field_name="health_check")
        completed = subprocess.run(
            arguments,
            cwd=workspace_path,
            text=True,
            capture_output=True,
            timeout=_environment_timeout_seconds(environment),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Environment health check failed "
                f"({command!r}, exit={completed.returncode}): "
                f"stdout={_bounded_output(completed.stdout)!r} "
                f"stderr={_bounded_output(completed.stderr)!r}"
            )


def validate_environment_resets(
    environment: EnvironmentSpec,
    workspace_root: str | Path,
    *,
    attempts: int = 2,
) -> list[str]:
    if attempts < 2:
        raise ValueError("Reset validation requires at least two attempts")
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    with tempfile.TemporaryDirectory(dir=root) as directory:
        for index in range(attempts):
            destination = Path(directory) / f"reset-{index}"
            materialize_environment_source(environment, destination)
            hashes.append(_workspace_tree_hash(destination))
    if len(set(hashes)) != 1:
        raise RuntimeError("Environment reset validation produced inconsistent workspace state")
    return hashes


def _validate_archive_link(member: tarfile.TarInfo) -> None:
    link = PurePosixPath(member.linkname)
    if link.is_absolute():
        raise ValueError(
            f"Unsafe repository archive link target: {member.name} -> {member.linkname}"
        )
    member_parent = PurePosixPath(member.name).parent
    resolved_parts = []
    for part in (*member_parent.parts, *link.parts):
        if part in {"", "."}:
            continue
        if part == "..":
            if not resolved_parts:
                raise ValueError(
                    f"Unsafe repository archive link target: {member.name} -> {member.linkname}"
                )
            resolved_parts.pop()
        else:
            resolved_parts.append(part)


def _command_arguments(command: str, *, field_name: str) -> list[str]:
    arguments = shlex.split(command)
    if not arguments:
        raise ValueError(f"Environment {field_name} command cannot be empty")
    return arguments


def _environment_timeout_seconds(environment: EnvironmentSpec) -> float:
    value = environment.resource_limits.get("timeout_seconds", 30.0)
    return float(value)


def _bounded_output(value: str, limit: int = 4000) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="replace")


def _workspace_tree_hash(root: Path) -> str:
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if _is_git_path(relative):
            continue
        if path.is_symlink():
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "target": path.readlink().as_posix(),
                }
            )
        elif path.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        elif path.is_dir():
            entries.append({"path": relative, "type": "dir"})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_git_path(relative: str) -> bool:
    return relative == ".git" or relative.startswith(".git/")


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


def exact_duplicate_groups(seeds: Iterable[QuerySeed]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for seed in seeds:
        groups.setdefault(_normalized_query(seed.public.query), []).append(seed.seed_id)
    return [ids for ids in groups.values() if len(ids) > 1]


def semantic_duplicate_candidates(seeds: Iterable[QuerySeed]) -> list[tuple[str, str]]:
    items = list(seeds)
    pairs = []
    for index, left in enumerate(items):
        left_tokens = set(_semantic_tokens(left.public.query))
        for right in items[index + 1 :]:
            right_tokens = set(_semantic_tokens(right.public.query))
            union = left_tokens | right_tokens
            if union and len(left_tokens & right_tokens) / len(union) >= 0.8:
                pairs.append((left.seed_id, right.seed_id))
    return pairs


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _semantic_tokens(value: str) -> list[str]:
    return [token for token in _tokens(value) if not token.isdigit()]


def _normalized_query(value: str) -> str:
    return " ".join(_tokens(value))


def _ensure_seed_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in connection.execute("PRAGMA table_info(seeds)").fetchall()
    }
    columns = {
        "task_family": "TEXT DEFAULT 'general'",
        "source_method": "TEXT DEFAULT 'unspecified'",
        "train_eligible": "INTEGER DEFAULT 1",
        "contamination_tags": "TEXT DEFAULT '[]'",
        "verifier_types": "TEXT DEFAULT '[]'",
        "coverage_tags": "TEXT DEFAULT '[]'",
    }
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE seeds ADD COLUMN {name} {definition}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
