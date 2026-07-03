from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from easy_agentic_data.repository_allowlist import (
    AllowlistFilterSummary,
    audit_repository_allowlist,
    filter_records_by_allowlist,
)

PUBLIC_SOURCE_REQUIRED_FIELDS = (
    "repository",
    "source_uri",
    "source_revision",
    "source_instance_id",
    "source_url",
    "title",
    "body",
    "labels",
    "license",
    "language",
    "candidate_verifier",
)


@dataclass
class SourceCollectionIssue:
    """Actionable issue for a locally exported public source record."""

    code: str
    message: str
    record_index: int = -1
    source_instance_id: str = ""
    repository: str = ""
    severity: str = "error"


@dataclass
class SourceCollectionAudit:
    """Audit summary for local public issue/PR exports before registry import."""

    total: int = 0
    accepted: int = 0
    quarantined: int = 0
    repository_counts: dict[str, int] = field(default_factory=dict)
    language_counts: dict[str, int] = field(default_factory=dict)
    source_type_counts: dict[str, int] = field(default_factory=dict)
    allowlist_filter: dict[str, Any] = field(default_factory=dict)
    issues: list[SourceCollectionIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["valid"] = self.valid
        return value


def build_source_collection_plan(
    allowlist_records: Iterable[dict[str, Any]],
    *,
    output_root: str | Path,
    source_name: str = "curated-public-sources",
) -> dict[str, Any]:
    """Create local collection tasks from a repository allowlist."""

    records = list(allowlist_records)
    allowlist_audit = audit_repository_allowlist(records)
    output = Path(output_root)
    tasks = []
    for record in records:
        repository = _repository(record)
        source_uri = _text(record.get("source_uri"))
        if not repository or not source_uri:
            continue
        for collection_source in _collection_sources(record):
            tasks.append(
                {
                    "task_id": _stable_id(
                        "collection-task",
                        {
                            "source_name": source_name,
                            "repository": repository,
                            "collection_source": collection_source,
                        },
                    ),
                    "source_name": source_name,
                    "repository": repository,
                    "source_uri": source_uri,
                    "license": _text(record.get("license")),
                    "language": _text(record.get("language")),
                    "collection_source": collection_source,
                    "labels": _labels_for_source(record, collection_source),
                    "queries": _queries_for_source(record, collection_source),
                    "stable_commands": _stable_commands(record),
                    "required_record_fields": list(PUBLIC_SOURCE_REQUIRED_FIELDS),
                    "output_path": str(
                        output
                        / f"{repository.replace('/', '__')}-{collection_source}.jsonl"
                    ),
                }
            )
    return {
        "schema_version": "easy_agentic_data.source_collection_plan.v1",
        "source_name": source_name,
        "output_root": str(output),
        "allowlist_audit": allowlist_audit.to_dict(),
        "tasks": tasks,
        "total_tasks": len(tasks),
        "valid": allowlist_audit.valid and bool(tasks),
    }


def audit_public_source_records(
    records: Iterable[dict[str, Any]],
    allowlist_records: Iterable[dict[str, Any]],
    *,
    source_name: str,
) -> SourceCollectionAudit:
    """Audit local public issue/PR source exports before registry import."""

    record_list = list(records)
    filtered, allowlist_summary = filter_records_by_allowlist(
        record_list,
        allowlist_records,
        source_name=source_name,
    )
    allowed_ids = {id(record) for record in filtered}
    issues: list[SourceCollectionIssue] = []
    repository_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    seen_instances: dict[str, int] = {}
    accepted = 0

    for index, record in enumerate(record_list):
        repository = _repository(record)
        instance_id = _source_instance_id(record)
        before = len(issues)
        _add_record_issues(record, index, issues)
        if instance_id:
            if instance_id in seen_instances:
                issues.append(
                    SourceCollectionIssue(
                        code="duplicate_source_instance_id",
                        message="Source instance ID appears more than once",
                        record_index=index,
                        source_instance_id=instance_id,
                        repository=repository,
                    )
                )
            else:
                seen_instances[instance_id] = index
        if id(record) in allowed_ids and len(issues) == before:
            accepted += 1
            repository_counts[repository] += 1
            language_counts[_normalize_label(record.get("language"))] += 1
            source_type_counts[_source_type(record)] += 1

    for message in allowlist_summary.issues:
        issues.append(
            SourceCollectionIssue(
                code="allowlist_quarantine",
                message=message,
            )
        )
    quarantined = len(record_list) - accepted
    return SourceCollectionAudit(
        total=len(record_list),
        accepted=accepted,
        quarantined=quarantined,
        repository_counts=dict(sorted(repository_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        source_type_counts=dict(sorted(source_type_counts.items())),
        allowlist_filter=allowlist_summary.to_dict(),
        issues=issues,
    )


def _add_record_issues(
    record: dict[str, Any],
    index: int,
    issues: list[SourceCollectionIssue],
) -> None:
    repository = _repository(record)
    instance_id = _source_instance_id(record)
    checks = (
        ("missing_repository", "Record must include a repository", bool(repository)),
        ("missing_title", "Record must include a public title", bool(_text(record.get("title")))),
        ("missing_body", "Record must include a public body", bool(_text(record.get("body")))),
        ("missing_labels", "Record must include labels", bool(_list_field(record.get("labels")))),
        ("missing_license", "Record must include a license", bool(_text(record.get("license")))),
        ("missing_language", "Record must include a language", bool(_text(record.get("language")))),
        (
            "missing_source_instance_id",
            "Record must include or derive a stable source instance ID",
            bool(instance_id),
        ),
        (
            "missing_candidate_verifier",
            "Record must include candidate verifier evidence",
            _has_candidate_verifier(record),
        ),
        (
            "missing_source_url",
            "Record must include a public issue or PR URL",
            bool(_source_url(record)),
        ),
        (
            "missing_fixed_revision",
            "Record must include a 40-character fixed source revision",
            bool(_fixed_revision(record)),
        ),
    )
    for code, message, passed in checks:
        if not passed:
            issues.append(
                SourceCollectionIssue(
                    code=code,
                    message=message,
                    record_index=index,
                    source_instance_id=instance_id,
                    repository=repository,
                )
            )
    if _source_url(record) and not _public_url(_source_url(record)):
        issues.append(
            SourceCollectionIssue(
                code="non_public_source_url",
                message="Source URL must be public http(s), not local or private",
                record_index=index,
                source_instance_id=instance_id,
                repository=repository,
            )
        )
    if _contains_private_url(record):
        issues.append(
            SourceCollectionIssue(
                code="private_url",
                message="Record contains a private, local, or SSH URL",
                record_index=index,
                source_instance_id=instance_id,
                repository=repository,
            )
        )


def _collection_sources(record: dict[str, Any]) -> list[str]:
    values = []
    values.extend(_list_field(record.get("collection_sources")))
    if _list_field(record.get("issue_labels")) or _list_field(record.get("issue_queries")):
        values.append("issues")
    if _list_field(record.get("pr_labels")) or _list_field(record.get("pr_queries")):
        values.append("pull_requests")
    if _list_field(record.get("ci_queries")):
        values.append("ci")
    if _list_field(record.get("review_queries")):
        values.append("reviews")
    return sorted({_normalize_label(value) for value in values if _normalize_label(value)})


def _labels_for_source(record: dict[str, Any], collection_source: str) -> list[str]:
    if collection_source == "issues":
        return _list_field(record.get("issue_labels")) or _list_field(record.get("labels"))
    if collection_source == "pull_requests":
        return _list_field(record.get("pr_labels")) or _list_field(record.get("labels"))
    return _list_field(record.get("labels"))


def _queries_for_source(record: dict[str, Any], collection_source: str) -> list[str]:
    if collection_source == "issues":
        return _list_field(record.get("issue_queries"))
    if collection_source == "pull_requests":
        return _list_field(record.get("pr_queries"))
    if collection_source == "ci":
        return _list_field(record.get("ci_queries"))
    if collection_source == "reviews":
        return _list_field(record.get("review_queries"))
    return []


def _stable_commands(record: dict[str, Any]) -> list[str]:
    commands = []
    commands.extend(_list_field(record.get("test_commands")))
    commands.extend(_list_field(record.get("ci_commands")))
    return sorted(dict.fromkeys(commands))


def _has_candidate_verifier(record: dict[str, Any]) -> bool:
    for field_name in (
        "test_commands",
        "build_commands",
        "ci_commands",
        "doctest_commands",
        "benchmark_commands",
        "adversarial_tests",
        "diff_constraints",
        "hidden_tests",
    ):
        if _list_field(record.get(field_name)):
            return True
    return any(
        isinstance(record.get(field_name), dict) and bool(record.get(field_name))
        for field_name in ("required_state", "forbidden_state", "performance_threshold")
    )


def _source_instance_id(record: dict[str, Any]) -> str:
    value = (
        record.get("source_instance_id")
        or record.get("instance_id")
        or record.get("issue_id")
        or record.get("pull_request_id")
        or record.get("id")
    )
    if value:
        return _text(value)
    repository = _repository(record).replace("/", "__")
    number = _text(record.get("number") or record.get("issue_number") or record.get("pr_number"))
    if repository and number:
        prefix = "pr" if _source_type(record) == "public_pr" else "issue"
        return f"{repository}-{prefix}-{number}"
    return _source_url(record)


def _source_type(record: dict[str, Any]) -> str:
    explicit = _normalize_label(record.get("source_type") or record.get("type") or record.get("kind"))
    if explicit in {"pr", "pull_request", "public_pr"}:
        return "public_pr"
    return "public_issue"


def _source_url(record: dict[str, Any]) -> str:
    return _text(
        record.get("source_url")
        or record.get("html_url")
        or record.get("issue_url")
        or record.get("pull_request_url")
        or record.get("url")
    )


def _fixed_revision(record: dict[str, Any]) -> str:
    value = _text(
        record.get("source_revision")
        or record.get("base_commit")
        or record.get("base_sha")
        or record.get("commit")
    )
    return value.lower() if re.fullmatch(r"[0-9a-fA-F]{40}", value) else ""


def _repository(record: dict[str, Any]) -> str:
    repository = record.get("repository") or record.get("repo") or record.get("full_name")
    org = record.get("org") or record.get("owner")
    name = record.get("name")
    if not repository and org and name:
        repository = f"{org}/{name}"
    return _text(repository).lower().strip("/")


def _public_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and "." in parsed.hostname


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


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:16]}"


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
