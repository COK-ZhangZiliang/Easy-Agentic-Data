import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.repository_allowlist import (
    audit_repository_allowlist,
    filter_records_by_allowlist,
    load_repository_allowlist,
)


class RepositoryAllowlistTests(unittest.TestCase):
    def test_allowlist_audit_accepts_public_trainable_repository(self) -> None:
        audit = audit_repository_allowlist([_allowlist_record()])

        self.assertTrue(audit.valid)
        self.assertEqual(audit.approved, 1)
        self.assertEqual(audit.license_counts, {"mit": 1})
        self.assertEqual(audit.language_counts, {"python": 1})
        self.assertEqual(
            audit.collection_source_counts,
            {"issues": 1, "pull_requests": 1},
        )

    def test_allowlist_audit_rejects_missing_or_unsafe_metadata(self) -> None:
        audit = audit_repository_allowlist(
            [
                {
                    "repository": "example/private",
                    "source_uri": "ssh://git@example.com/private.git",
                    "license": "GPL-3.0",
                    "language": "",
                    "benchmark_overlap": True,
                }
            ]
        )

        codes = {issue.code for issue in audit.issues}
        self.assertFalse(audit.valid)
        self.assertIn("benchmark_overlap", codes)
        self.assertIn("non_public_source_uri", codes)
        self.assertIn("license_not_allowlisted", codes)
        self.assertIn("missing_language", codes)
        self.assertIn("missing_collection_source", codes)
        self.assertIn("missing_stable_command", codes)

    def test_filter_records_quarantines_non_allowlisted_or_mutable_sources(self) -> None:
        allowed, summary = filter_records_by_allowlist(
            [
                _source_record(),
                {**_source_record(), "repository": "other/tool", "id": "issue-2"},
                {**_source_record(), "source_revision": "main", "id": "issue-3"},
            ],
            [_allowlist_record()],
            source_name="curated-public-issues",
        )

        self.assertEqual(len(allowed), 1)
        self.assertEqual(summary.checked, 3)
        self.assertEqual(summary.allowed, 1)
        self.assertEqual(summary.quarantined, 2)
        self.assertIn("not allowlisted", summary.issues[0])
        self.assertIn("40-character fixed commit", summary.issues[1])

    def test_filter_records_accepts_swe_style_repo_and_fixed_commit(self) -> None:
        allowed, summary = filter_records_by_allowlist(
            [
                {
                    "instance_id": "example__tool-1",
                    "repo": "example/tool",
                    "base_commit": "b" * 40,
                    "license": "MIT",
                }
            ],
            [_allowlist_record()],
            source_name="curated-swe-style",
        )

        self.assertEqual(len(allowed), 1)
        self.assertEqual(summary.quarantined, 0)

    def test_filter_records_normalizes_source_uri_shape(self) -> None:
        allowed, summary = filter_records_by_allowlist(
            [
                {
                    **_source_record(),
                    "source_uri": "https://github.com/example/tool/",
                }
            ],
            [_allowlist_record()],
            source_name="curated-public-issues",
        )

        self.assertEqual(len(allowed), 1)
        self.assertEqual(summary.quarantined, 0)

    def test_filter_records_uses_original_uri_for_materialized_workspaces(self) -> None:
        allowed, summary = filter_records_by_allowlist(
            [
                {
                    **_source_record(),
                    "source_uri": "file:///tmp/example-tool-workspace",
                    "workspace_materialized": True,
                    "workspace_original_source_uri": "https://github.com/example/tool.git",
                },
                {
                    **_source_record(),
                    "id": "issue-2",
                    "source_uri": "file:///tmp/other-tool-workspace",
                    "workspace_materialized": True,
                    "workspace_original_source_uri": "https://github.com/other/tool.git",
                },
            ],
            [_allowlist_record()],
            source_name="curated-materialized-public-issues",
        )

        self.assertEqual(len(allowed), 1)
        self.assertEqual(summary.allowed, 1)
        self.assertEqual(summary.quarantined, 1)
        self.assertIn("source URI does not match allowlist", summary.issues[0])

    def test_cli_allowlist_audit_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "allowlist.json"
            output = Path(directory) / "audit.json"
            source.write_text(json.dumps({"repositories": [_allowlist_record()]}), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "allowlist-audit",
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["valid"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["approved"], 1)
            self.assertEqual(load_repository_allowlist(source)[0]["repository"], "example/tool")


def _allowlist_record() -> dict[str, object]:
    return {
        "repository": "example/tool",
        "source_uri": "https://github.com/example/tool.git",
        "license": "MIT",
        "language": "Python",
        "collection_sources": ["issues", "pull_requests"],
        "test_commands": ["python -m pytest tests/test_parser.py"],
    }


def _source_record() -> dict[str, object]:
    return {
        "id": "issue-1",
        "repository": "example/tool",
        "source_uri": "https://github.com/example/tool.git",
        "source_revision": "a" * 40,
        "license": "MIT",
        "title": "Fix parser whitespace handling",
        "body": "The parser drops significant whitespace.",
    }


if __name__ == "__main__":
    unittest.main()
