from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from easy_agentic_data.registry_sources import DEFAULT_TRAIN_LICENSE_ALLOWLIST


@dataclass
class RepositoryAllowlistIssue:
    """Actionable issue for a candidate repository allowlist."""

    code: str
    message: str
    repository: str = ""
    severity: str = "warning"


@dataclass
class RepositoryAllowlistAudit:
    """Summary of repository allowlist readiness for public seed collection."""

    total: int = 0
    approved: int = 0
    blocked: int = 0
    license_counts: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    collection_source_counts: dict[str, int] = field(default_factory=dict)
    issues: list[RepositoryAllowlistIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["valid"] = self.valid
        return value


@dataclass
class AllowlistFilterSummary:
    """Summary for train-source records filtered against an allowlist."""

    source_name: str
    checked: int = 0
    allowed: int = 0
    quarantined: int = 0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_repository_allowlist(path: str | Path) -> list[dict[str, Any]]:
    """Load repository allowlist records from JSON or JSONL."""

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
                raise ValueError(f"Allowlist JSONL line {line_number} must contain an object")
            records.append(payload)
        return records

    payload = json.loads(text)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = (
            payload.get("repositories")
            or payload.get("allowlist")
            or payload.get("records")
        )
        if records is None:
            records = [payload]
    else:
        raise ValueError("Repository allowlist must contain a JSON object, array, or JSONL")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Repository allowlist records must be JSON objects")
    return list(records)


def audit_repository_allowlist(
    records: Iterable[dict[str, Any]],
    *,
    license_allowlist: Iterable[str] = DEFAULT_TRAIN_LICENSE_ALLOWLIST,
    benchmark_repositories: Iterable[str] = (),
) -> RepositoryAllowlistAudit:
    """Validate public repository collection candidates before source ingestion."""

    allowlisted_licenses = {_normalize_label(value) for value in license_allowlist}
    benchmark_set = {_normalize_repository(value) for value in benchmark_repositories}
    issues: list[RepositoryAllowlistIssue] = []
    license_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    collection_counts: Counter[str] = Counter()
    total = 0
    approved = 0

    for record in records:
        total += 1
        repository = _record_repository(record)
        license_name = _normalize_label(record.get("license"))
        language = _normalize_label(record.get("language"))
        collection_sources = _collection_sources(record)
        if license_name:
            license_counts[license_name] += 1
        if language:
            language_counts[language] += 1
        collection_counts.update(collection_sources)
        before = len(issues)

        if not repository:
            issues.append(
                RepositoryAllowlistIssue(
                    code="missing_repository",
                    message="Allowlist record is missing repository",
                    severity="error",
                )
            )
        if repository in benchmark_set or bool(record.get("benchmark_overlap", False)):
            issues.append(
                RepositoryAllowlistIssue(
                    code="benchmark_overlap",
                    message="Repository is marked as benchmark-overlapping",
                    repository=repository,
                    severity="error",
                )
            )
        if not _text(record.get("source_uri")):
            issues.append(
                RepositoryAllowlistIssue(
                    code="missing_source_uri",
                    message="Repository source URI is required",
                    repository=repository,
                    severity="error",
                )
            )
        elif not _public_source_uri(_text(record.get("source_uri"))):
            issues.append(
                RepositoryAllowlistIssue(
                    code="non_public_source_uri",
                    message="Repository source URI must be public https or file URI",
                    repository=repository,
                    severity="error",
                )
            )
        if not license_name:
            issues.append(
                RepositoryAllowlistIssue(
                    code="missing_license",
                    message="Repository license is required",
                    repository=repository,
                    severity="error",
                )
            )
        elif license_name not in allowlisted_licenses:
            issues.append(
                RepositoryAllowlistIssue(
                    code="license_not_allowlisted",
                    message=f"Repository license is not train-allowlisted: {license_name}",
                    repository=repository,
                    severity="error",
                )
            )
        if not language:
            issues.append(
                RepositoryAllowlistIssue(
                    code="missing_language",
                    message="Repository language is required for coverage budgets",
                    repository=repository,
                    severity="error",
                )
            )
        if not collection_sources:
            issues.append(
                RepositoryAllowlistIssue(
                    code="missing_collection_source",
                    message="Allowlist record must declare issue, PR, CI, or review sources",
                    repository=repository,
                    severity="error",
                )
            )
        if not _list_field(record.get("test_commands")) and not _list_field(
            record.get("ci_commands")
        ):
            issues.append(
                RepositoryAllowlistIssue(
                    code="missing_stable_command",
                    message="Allowlist record must declare stable test or CI commands",
                    repository=repository,
                    severity="error",
                )
            )
        if len(issues) == before:
            approved += 1

    return RepositoryAllowlistAudit(
        total=total,
        approved=approved,
        blocked=total - approved,
        license_counts=dict(sorted(license_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        collection_source_counts=dict(sorted(collection_counts.items())),
        issues=issues,
    )


def filter_records_by_allowlist(
    records: Iterable[dict[str, Any]],
    allowlist_records: Iterable[dict[str, Any]],
    *,
    source_name: str,
) -> tuple[list[dict[str, Any]], AllowlistFilterSummary]:
    """Quarantine train-source records that are outside the repository allowlist."""

    allowed_repositories = {
        _record_repository(record): record
        for record in allowlist_records
        if _record_repository(record)
    }
    allowed = []
    summary = AllowlistFilterSummary(source_name=source_name)
    for index, record in enumerate(records):
        summary.checked += 1
        repository = _record_repository(record)
        allowlist = allowed_repositories.get(repository)
        issue = _record_allowlist_issue(record, allowlist, index=index)
        if issue:
            summary.quarantined += 1
            summary.issues.append(issue)
            continue
        allowed.append(record)
        summary.allowed += 1
    return allowed, summary


def _record_allowlist_issue(
    record: dict[str, Any],
    allowlist: dict[str, Any] | None,
    *,
    index: int,
) -> str:
    repository = _record_repository(record)
    if not repository:
        return f"record {index}: missing repository"
    if allowlist is None:
        return f"record {index}: repository is not allowlisted: {repository}"
    if not _fixed_revision(record):
        return f"record {index}: source revision must be a 40-character fixed commit"
    source_uri = _record_source_uri(record)
    if not source_uri:
        return f"record {index}: missing source URI"
    allowlist_source_uri = _record_allowlist_source_uri(record, source_uri)
    allowed_source_uri = _text(allowlist.get("source_uri"))
    if allowed_source_uri and _normalize_source_uri(
        allowlist_source_uri
    ) != _normalize_source_uri(allowed_source_uri):
        return f"record {index}: source URI does not match allowlist for {repository}"
    license_name = _normalize_label(record.get("license") or allowlist.get("license"))
    allowlist_license = _normalize_label(allowlist.get("license"))
    if allowlist_license and license_name != allowlist_license:
        return f"record {index}: license does not match allowlist for {repository}"
    if _contains_private_url(record):
        return f"record {index}: record contains private or local URL"
    return ""


def _record_repository(record: dict[str, Any]) -> str:
    repository = record.get("repository") or record.get("repo") or record.get("full_name")
    org = record.get("org") or record.get("owner")
    name = record.get("name")
    if not repository and org and name:
        repository = f"{org}/{name}"
    return _normalize_repository(repository)


def _record_source_uri(record: dict[str, Any]) -> str:
    source_uri = _text(record.get("source_uri") or record.get("repo_url"))
    if source_uri:
        return source_uri
    repository = _record_repository(record)
    if repository:
        return f"https://github.com/{repository}.git"
    return ""


def _record_allowlist_source_uri(record: dict[str, Any], source_uri: str) -> str:
    if source_uri.startswith("file://") and bool(record.get("workspace_materialized")):
        return _text(record.get("workspace_original_source_uri")) or source_uri
    return source_uri


def _collection_sources(record: dict[str, Any]) -> list[str]:
    values = []
    values.extend(_list_field(record.get("collection_sources")))
    if _list_field(record.get("issue_queries")) or _list_field(record.get("issue_labels")):
        values.append("issues")
    if _list_field(record.get("pr_queries")) or _list_field(record.get("pr_labels")):
        values.append("pull_requests")
    if _list_field(record.get("ci_queries")):
        values.append("ci")
    if _list_field(record.get("review_queries")):
        values.append("reviews")
    return sorted({_normalize_label(value) for value in values if _normalize_label(value)})


def _fixed_revision(record: dict[str, Any]) -> str:
    revision = _text(
        record.get("source_revision")
        or record.get("base_commit")
        or record.get("base_sha")
        or record.get("commit")
    )
    return revision if re.fullmatch(r"[0-9a-fA-F]{40}", revision) else ""


def _public_source_uri(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return True
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    if not host or host in {"localhost", "127.0.0.1"}:
        return False
    if "." not in host:
        return False
    return True


def _normalize_source_uri(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        path = (parsed.path or "").rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        return f"{parsed.scheme.lower()}://{(parsed.hostname or '').lower()}{path.lower()}"
    return value.rstrip("/")


def _contains_private_url(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_private_url(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_private_url(item) for item in value)
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "http://localhost",
            "https://localhost",
            "http://127.0.0.1",
            "https://127.0.0.1",
            "http://go/",
            "https://go/",
            "ssh://",
            "git@",
        )
    )


def _normalize_repository(value: Any) -> str:
    text = _text(value).lower()
    if text.endswith(".git"):
        text = text[:-4]
    return text.strip("/")


def _normalize_label(value: Any) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _text(value: Any) -> str:
    return str(value or "").strip()
