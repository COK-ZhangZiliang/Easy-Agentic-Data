from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
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


@dataclass
class SourceExportSummary:
    """Summary of public source records exported from a collection plan."""

    plan_tasks: int = 0
    selected_tasks: int = 0
    processed_tasks: int = 0
    exported: int = 0
    new_records: int = 0
    existing_records: int = 0
    duplicate_records: int = 0
    skipped_tasks: int = 0
    skipped_records: int = 0
    allow_partial: bool = False
    output_path: str = ""
    source_type_counts: dict[str, int] = field(default_factory=dict)
    repository_counts: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.exported > 0 and (self.allow_partial or not self.issues)

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


def export_public_source_records(
    collection_plan: dict[str, Any],
    *,
    output_path: str | Path,
    limit_per_task: int = 10,
    task_offset: int = 0,
    max_tasks: int | None = None,
    sleep_seconds: float = 0.0,
    resume: bool = False,
    allow_partial: bool = False,
    fixture_root: str | Path | None = None,
    github_token_env: str = "",
    timeout_seconds: float = 30.0,
) -> SourceExportSummary:
    """Export public issue/PR records from a collection plan into JSONL."""

    tasks = list(collection_plan.get("tasks", []))
    output = Path(output_path)
    selected_tasks = _selected_tasks(tasks, task_offset=task_offset, max_tasks=max_tasks)
    client: _SourceClient
    if fixture_root:
        client = _FixtureSourceClient(Path(fixture_root))
    else:
        client = _GitHubSourceClient(
            token=os.environ.get(github_token_env, "") if github_token_env else "",
            timeout_seconds=timeout_seconds,
        )
    records = _existing_export_records(output) if resume else []
    seen_instance_ids = {_source_instance_id(record) for record in records}
    issues: list[str] = []
    source_type_counts: Counter[str] = Counter()
    repository_counts: Counter[str] = Counter()
    for record in records:
        source_type_counts[_source_type(record)] += 1
        repository_counts[_repository(record)] += 1
    existing_records = len(records)
    new_records = 0
    duplicate_records = 0
    skipped_tasks = 0
    skipped_records = 0
    processed_tasks = 0

    for task_index, task in enumerate(selected_tasks):
        collection_source = _normalize_label(task.get("collection_source"))
        if collection_source not in {"issues", "pull_requests"}:
            skipped_tasks += 1
            continue
        processed_tasks += 1
        try:
            exported, skipped = _export_task_records(
                task,
                client=client,
                limit=max(0, limit_per_task),
            )
        except Exception as exc:
            issues.append(f"{task.get('task_id', '<unknown>')}: {exc}")
            _sleep_between_tasks(task_index, selected_tasks, sleep_seconds)
            continue
        skipped_records += skipped
        for record in exported:
            instance_id = _source_instance_id(record)
            if instance_id in seen_instance_ids:
                duplicate_records += 1
                continue
            seen_instance_ids.add(instance_id)
            records.append(record)
            new_records += 1
            source_type_counts[_source_type(record)] += 1
            repository_counts[_repository(record)] += 1
        _sleep_between_tasks(task_index, selected_tasks, sleep_seconds)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return SourceExportSummary(
        plan_tasks=len(tasks),
        selected_tasks=len(selected_tasks),
        processed_tasks=processed_tasks,
        exported=len(records),
        new_records=new_records,
        existing_records=existing_records,
        duplicate_records=duplicate_records,
        skipped_tasks=skipped_tasks,
        skipped_records=skipped_records,
        allow_partial=allow_partial,
        output_path=str(output),
        source_type_counts=dict(sorted(source_type_counts.items())),
        repository_counts=dict(sorted(repository_counts.items())),
        issues=issues,
    )


def _selected_tasks(
    tasks: list[dict[str, Any]],
    *,
    task_offset: int,
    max_tasks: int | None,
) -> list[dict[str, Any]]:
    start = max(0, task_offset)
    selected = tasks[start:]
    if max_tasks is not None:
        selected = selected[: max(0, max_tasks)]
    return selected


def _existing_export_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"Existing export line {line_number} must contain an object")
        records.append(payload)
    return records


def _sleep_between_tasks(
    task_index: int,
    selected_tasks: list[dict[str, Any]],
    sleep_seconds: float,
) -> None:
    if sleep_seconds > 0 and task_index < len(selected_tasks) - 1:
        time.sleep(sleep_seconds)


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


def _export_task_records(
    task: dict[str, Any],
    *,
    client: "_SourceClient",
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    if limit <= 0:
        return [], 0
    collection_source = _normalize_label(task.get("collection_source"))
    if collection_source == "issues":
        raw_records = [
            item
            for item in client.issues(task, limit=limit)
            if not isinstance(item.get("pull_request"), dict)
        ][:limit]
    elif collection_source == "pull_requests":
        raw_records = client.pull_requests(task, limit=limit)[:limit]
    else:
        return [], 0
    records = []
    skipped = 0
    for raw_record in raw_records:
        record = _normalize_exported_record(
            task,
            raw_record,
            source_type="pull_request" if collection_source == "pull_requests" else "issue",
            client=client,
        )
        if _export_record_ready(record):
            records.append(record)
        else:
            skipped += 1
    return records, skipped


def _normalize_exported_record(
    task: dict[str, Any],
    raw_record: dict[str, Any],
    *,
    source_type: str,
    client: "_SourceClient",
) -> dict[str, Any]:
    repository = _repository(task)
    number = int(raw_record.get("number") or 0)
    if number <= 0:
        raise ValueError(f"record from {repository} is missing a positive number")
    labels = _label_names(raw_record.get("labels")) or _list_field(task.get("labels"))
    stable_commands = _list_field(task.get("stable_commands"))
    source_revision = _source_revision_for_record(
        task,
        raw_record,
        source_type=source_type,
        client=client,
    )
    source_instance_id = (
        f"{repository.replace('/', '__')}-"
        f"{'pr' if source_type == 'pull_request' else 'issue'}-{number}"
    )
    return {
        "id": source_instance_id,
        "type": "pull_request" if source_type == "pull_request" else "issue",
        "repository": repository,
        "source_uri": _text(task.get("source_uri")),
        "source_revision": source_revision,
        "source_instance_id": source_instance_id,
        "source_url": _text(raw_record.get("html_url")),
        "title": _text(raw_record.get("title")),
        "body": _text(raw_record.get("body")),
        "labels": labels,
        "license": _text(task.get("license")),
        "language": _text(task.get("language")),
        "test_commands": stable_commands,
        "candidate_verifier": {
            "type": "stable_commands",
            "commands": stable_commands,
        },
        "collection_source": _normalize_label(task.get("collection_source")),
        "source_name": _text(task.get("source_name")),
    }


def _export_record_ready(record: dict[str, Any]) -> bool:
    return all(
        (
            _repository(record),
            _text(record.get("source_uri")),
            _fixed_sha(record.get("source_revision")),
            _text(record.get("source_url")),
            _text(record.get("title")),
            _text(record.get("body")),
            _list_field(record.get("labels")),
            _text(record.get("license")),
            _text(record.get("language")),
            _has_candidate_verifier(record),
        )
    )


def _source_revision_for_record(
    task: dict[str, Any],
    raw_record: dict[str, Any],
    *,
    source_type: str,
    client: "_SourceClient",
) -> str:
    if source_type == "pull_request":
        base = raw_record.get("base")
        if isinstance(base, dict):
            sha = _fixed_sha(base.get("sha"))
            if sha:
                return sha
    task_revision = _fixed_sha(task.get("source_revision"))
    if task_revision:
        return task_revision
    return client.default_branch_sha(_repository(task))


def _label_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels = []
    for item in value:
        if isinstance(item, dict):
            name = _text(item.get("name"))
        else:
            name = _text(item)
        if name:
            labels.append(name)
    return labels


def _fixed_sha(value: Any) -> str:
    text = _text(value)
    return text.lower() if re.fullmatch(r"[0-9a-fA-F]{40}", text) else ""


class _SourceClient:
    def issues(self, task: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def pull_requests(self, task: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        raise NotImplementedError

    def default_branch_sha(self, repository: str) -> str:
        raise NotImplementedError


class _FixtureSourceClient(_SourceClient):
    def __init__(self, root: Path) -> None:
        self.root = root

    def issues(self, task: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        del limit
        return self._records(_repository(task), "issues.json")

    def pull_requests(self, task: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        del limit
        return self._records(_repository(task), "pull_requests.json")

    def default_branch_sha(self, repository: str) -> str:
        metadata = self._payload(repository, "repository.json")
        sha = _fixed_sha(metadata.get("default_branch_sha"))
        if sha:
            return sha
        branch = _text(metadata.get("default_branch")) or "main"
        branch_payload = self._payload(repository, f"branches/{branch}.json")
        commit = branch_payload.get("commit")
        if isinstance(commit, dict):
            sha = _fixed_sha(commit.get("sha"))
        else:
            sha = _fixed_sha(branch_payload.get("sha"))
        if not sha:
            raise ValueError(f"fixture branch for {repository} is missing a fixed commit SHA")
        return sha

    def _records(self, repository: str, filename: str) -> list[dict[str, Any]]:
        payload = self._payload(repository, filename)
        records = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            raise ValueError(f"fixture {filename} for {repository} must contain records")
        return list(records)

    def _payload(self, repository: str, filename: str) -> Any:
        path = self.root / repository.replace("/", "__") / filename
        return json.loads(path.read_text(encoding="utf-8"))


class _GitHubSourceClient(_SourceClient):
    def __init__(self, *, token: str = "", timeout_seconds: float = 30.0) -> None:
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._repo_cache: dict[str, dict[str, Any]] = {}
        self._branch_cache: dict[str, str] = {}

    def issues(self, task: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        repository = _repository(task)
        params = {
            "state": "open",
            "per_page": str(min(max(limit, 1), 100)),
        }
        payload = self._get_json(f"/repos/{repository}/issues", params=params)
        if not isinstance(payload, list):
            raise ValueError(f"GitHub issues response for {repository} is not a list")
        return payload

    def pull_requests(self, task: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        repository = _repository(task)
        params = {
            "state": "open",
            "per_page": str(min(max(limit, 1), 100)),
        }
        payload = self._get_json(f"/repos/{repository}/pulls", params=params)
        if not isinstance(payload, list):
            raise ValueError(f"GitHub pulls response for {repository} is not a list")
        return payload

    def default_branch_sha(self, repository: str) -> str:
        if repository not in self._branch_cache:
            metadata = self._repository_metadata(repository)
            branch = _text(metadata.get("default_branch")) or "main"
            branch_payload = self._get_json(f"/repos/{repository}/branches/{branch}")
            commit = branch_payload.get("commit")
            sha = _fixed_sha(commit.get("sha")) if isinstance(commit, dict) else ""
            if not sha:
                raise ValueError(f"GitHub branch metadata for {repository} lacks a fixed SHA")
            self._branch_cache[repository] = sha
        return self._branch_cache[repository]

    def _repository_metadata(self, repository: str) -> dict[str, Any]:
        if repository not in self._repo_cache:
            payload = self._get_json(f"/repos/{repository}")
            if not isinstance(payload, dict):
                raise ValueError(f"GitHub repository metadata for {repository} is not an object")
            self._repo_cache[repository] = payload
        return self._repo_cache[repository]

    def _get_json(
        self,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> Any:
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        request = urllib.request.Request(
            f"https://api.github.com{path}{query}",
            headers=self._headers(),
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "easy-agentic-data-source-collection",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


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
