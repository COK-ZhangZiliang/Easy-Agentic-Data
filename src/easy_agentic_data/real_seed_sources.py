from __future__ import annotations

import json
import os
import re
import ssl
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_sources import (
    RegistryImportSummary,
    import_swe_style_records,
    load_source_records,
)

SWE_BENCH_LITE_DATASET = "princeton-nlp/SWE-bench_Lite"
HF_DATASET_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DEFAULT_DEMO_IMAGE_DIGEST = (
    "python@sha256:f417205fec4ccb0d5023fdb5ecb4c8eba31c1834f94dcbcd1a2e8325fa7a7b89"
)
DEFAULT_CAPABILITY_PACKS = [
    "list_files",
    "read_file",
    "search_files",
    "apply_patch",
    "run_command",
    "git_status",
    "git_diff",
    "ask_user",
]
DEFAULT_RESOURCE_LIMITS: dict[str, Any] = {
    "timeout_seconds": 60,
    "max_output_bytes": 200_000,
    "max_workspace_bytes": 500_000_000,
    "memory": "1g",
    "cpus": 1.0,
    "pids": 256,
}


@dataclass
class PreparedRepository:
    """A real source repository cloned for a seed record."""

    instance_id: str
    source_uri: str
    local_path: str
    revision: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class RealSeedPreparationSummary:
    """Summary for a real seed import and repository preparation run."""

    seed_source: str
    dataset: str
    split: str
    offset: int
    requested: int
    registry_root: str
    cache_root: str
    import_summary: RegistryImportSummary
    repositories: list[PreparedRepository] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["import_summary"] = self.import_summary.to_dict()
        value["repositories"] = [repository.to_dict() for repository in self.repositories]
        return value


def load_real_seed_records(
    *,
    source: str | Path | None = None,
    dataset: str = SWE_BENCH_LITE_DATASET,
    split: str = "dev",
    offset: int = 0,
    length: int = 1,
    timeout_seconds: float = 60.0,
    ca_bundle_env: str | None = "SSL_CERT_FILE",
) -> list[dict[str, Any]]:
    """Load real seed records from a local file or Hugging Face dataset rows."""

    if source:
        return load_source_records(source)
    return fetch_huggingface_dataset_rows(
        dataset=dataset,
        split=split,
        offset=offset,
        length=length,
        timeout_seconds=timeout_seconds,
        ca_bundle_env=ca_bundle_env,
    )


def fetch_huggingface_dataset_rows(
    *,
    dataset: str,
    split: str,
    offset: int = 0,
    length: int = 1,
    config: str = "default",
    timeout_seconds: float = 60.0,
    ca_bundle_env: str | None = "SSL_CERT_FILE",
) -> list[dict[str, Any]]:
    """Fetch a small page of rows from the Hugging Face dataset server."""

    if offset < 0:
        raise ValueError("offset cannot be negative")
    if length <= 0:
        raise ValueError("length must be positive")
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(f"{HF_DATASET_ROWS_ENDPOINT}?{query}", method="GET")
    context = _ssl_context(ca_bundle_env)
    with urllib.request.urlopen(request, timeout=timeout_seconds, context=context) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return records_from_huggingface_rows_payload(payload)


def records_from_huggingface_rows_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract record objects from the Hugging Face rows API response shape."""

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Hugging Face rows response is missing rows")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"Hugging Face row {index} must be an object")
        row = item.get("row")
        if not isinstance(row, dict):
            raise ValueError(f"Hugging Face row {index} is missing an object row payload")
        records.append(dict(row))
    return records


def prepare_real_seed_registry(
    *,
    registry_root: str | Path,
    cache_root: str | Path,
    records: Iterable[dict[str, Any]] | None = None,
    source: str | Path | None = None,
    dataset: str = SWE_BENCH_LITE_DATASET,
    split: str = "dev",
    offset: int = 0,
    limit: int = 1,
    source_name: str = "",
    source_format: str = "swe-bench",
    image_digest: str = DEFAULT_DEMO_IMAGE_DIGEST,
    setup_commands: Iterable[str] | None = None,
    network_policy: str = "disabled",
    pull_repositories: bool = True,
    test_command_template: str = "python -m pytest {test}",
    license_name: str = "",
    permitted_use: str = "research",
    timeout_seconds: float = 60.0,
    strict: bool = True,
) -> RealSeedPreparationSummary:
    """Create registry entries from real seed records and local repository checkouts."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    registry = ScenarioRegistry(registry_root)
    cache_path = Path(cache_root)
    cache_path.mkdir(parents=True, exist_ok=True)
    source_records = (
        list(records)
        if records is not None
        else load_real_seed_records(
            source=source,
            dataset=dataset,
            split=split,
            offset=offset,
            length=limit,
            timeout_seconds=timeout_seconds,
        )
    )
    source_records = source_records[:limit]

    prepared_records: list[dict[str, Any]] = []
    repositories: list[PreparedRepository] = []
    setup_command_list = list(setup_commands or [])
    for record in source_records:
        prepared = _with_runtime_defaults(
            record,
            image_digest=image_digest,
            setup_commands=setup_command_list,
            network_policy=network_policy,
        )
        if pull_repositories:
            repository = clone_record_repository(
                prepared,
                cache_path,
                timeout_seconds=timeout_seconds,
            )
            prepared["source_uri"] = Path(repository.local_path).as_uri()
            repositories.append(repository)
        prepared_records.append(prepared)

    import_summary = import_swe_style_records(
        registry,
        prepared_records,
        source_format=source_format,
        source_name=source_name or dataset,
        split=_registry_split(split),
        license_name=license_name,
        permitted_use=permitted_use,
        limit=limit,
        test_command_template=test_command_template,
        strict=strict,
    )
    return RealSeedPreparationSummary(
        seed_source=str(source) if source else "huggingface_rows",
        dataset=dataset,
        split=split,
        offset=offset,
        requested=limit,
        registry_root=str(Path(registry_root)),
        cache_root=str(cache_path),
        import_summary=import_summary,
        repositories=repositories,
    )


def clone_record_repository(
    record: dict[str, Any],
    cache_root: str | Path,
    *,
    timeout_seconds: float = 300.0,
) -> PreparedRepository:
    """Clone the repository named by a SWE-style record and check out its base commit."""

    instance_id = _text(_first_present(record, ("instance_id", "task_id", "id")))
    if not instance_id:
        raise ValueError("record is missing instance_id")
    source_uri = _record_repository_uri(record)
    revision = _text(
        _first_present(record, ("base_commit", "source_revision", "base_sha", "commit"))
    )
    if not revision:
        raise ValueError(f"{instance_id} is missing a source revision")

    destination = Path(cache_root) / _safe_path_component(instance_id)
    if not destination.exists():
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                source_uri,
                str(destination),
            ],
            timeout_seconds=timeout_seconds,
        )
    _ensure_commit_available(destination, revision, timeout_seconds=timeout_seconds)
    _run(["git", "-C", str(destination), "checkout", "--detach", revision], timeout_seconds=60)
    return PreparedRepository(
        instance_id=instance_id,
        source_uri=source_uri,
        local_path=str(destination.resolve()),
        revision=revision,
    )


def _with_runtime_defaults(
    record: dict[str, Any],
    *,
    image_digest: str,
    setup_commands: list[str] | None = None,
    network_policy: str = "disabled",
) -> dict[str, Any]:
    prepared = dict(record)
    prepared.setdefault("image_digest", image_digest)
    if setup_commands:
        prepared.setdefault("setup_commands", list(setup_commands))
    prepared.setdefault("capability_packs", list(DEFAULT_CAPABILITY_PACKS))
    prepared.setdefault("resource_limits", dict(DEFAULT_RESOURCE_LIMITS))
    prepared.setdefault("network_policy", network_policy)
    prepared.setdefault("working_directory", "/workspace")
    return prepared


def _registry_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"dev", "validation", "valid"}:
        return "validation"
    if normalized in {"test", "evaluation", "eval"}:
        return "evaluation"
    if normalized == "train":
        return "train"
    raise ValueError(f"Unsupported seed split for registry import: {value}")


def _record_repository_uri(record: dict[str, Any]) -> str:
    explicit = _text(_first_present(record, ("source_uri", "clone_url", "html_url")))
    if explicit:
        return explicit
    repository = _text(_first_present(record, ("repository", "full_repo", "repo")))
    if {"org", "repo"}.issubset(record) and "/" not in repository:
        repository = f"{record['org']}/{record['repo']}"
    if not repository or "/" not in repository:
        raise ValueError("record is missing a cloneable repository name")
    return f"https://github.com/{repository}.git"


def _ensure_commit_available(
    repository: Path,
    revision: str,
    *,
    timeout_seconds: float,
) -> None:
    check = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{revision}^{{commit}}"],
        text=True,
        capture_output=True,
    )
    if check.returncode == 0:
        return
    fetch = subprocess.run(
        ["git", "-C", str(repository), "fetch", "--filter=blob:none", "origin", revision],
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if fetch.returncode != 0:
        _run(
            ["git", "-C", str(repository), "fetch", "--filter=blob:none", "origin"],
            timeout_seconds=timeout_seconds,
        )
    _run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{revision}^{{commit}}"],
        timeout_seconds=30,
    )


def _run(command: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout_seconds,
    )


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "seed"


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ssl_context(ca_bundle_env: str | None) -> ssl.SSLContext:
    ca_bundle = os.environ.get(ca_bundle_env, "") if ca_bundle_env else ""
    if ca_bundle:
        if not os.path.isfile(ca_bundle):
            raise ValueError(f"CA bundle from {ca_bundle_env} does not exist: {ca_bundle}")
        return ssl.create_default_context(cafile=ca_bundle)
    return ssl.create_default_context()
