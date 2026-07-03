import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.source_collection import (
    audit_public_source_records,
    build_source_collection_plan,
)


class SourceCollectionTests(unittest.TestCase):
    def test_build_source_collection_plan_from_allowlist(self) -> None:
        plan = build_source_collection_plan(
            [_allowlist_record()],
            output_root="runs/source-exports",
            source_name="curated-public-sources",
        )

        self.assertTrue(plan["valid"])
        self.assertEqual(plan["total_tasks"], 2)
        self.assertEqual(
            {task["collection_source"] for task in plan["tasks"]},
            {"issues", "pull_requests"},
        )
        self.assertTrue(
            all("source_revision" in task["required_record_fields"] for task in plan["tasks"])
        )

    def test_audit_public_source_records_accepts_complete_exports(self) -> None:
        audit = audit_public_source_records(
            [_source_record()],
            [_allowlist_record()],
            source_name="curated-public-sources",
        )

        self.assertTrue(audit.valid)
        self.assertEqual(audit.accepted, 1)
        self.assertEqual(audit.quarantined, 0)
        self.assertEqual(audit.repository_counts, {"example/tool": 1})
        self.assertEqual(audit.source_type_counts, {"public_issue": 1})

    def test_audit_public_source_records_flags_missing_contract_fields(self) -> None:
        audit = audit_public_source_records(
            [
                {
                    **_source_record(),
                    "id": "issue-101",
                    "body": "",
                    "labels": [],
                    "source_revision": "main",
                    "source_url": "http://go/private",
                    "test_commands": [],
                },
                {**_source_record(), "id": "issue-100"},
            ],
            [_allowlist_record()],
            source_name="curated-public-sources",
        )

        codes = {issue.code for issue in audit.issues}
        self.assertFalse(audit.valid)
        self.assertEqual(audit.accepted, 1)
        self.assertEqual(audit.quarantined, 1)
        self.assertIn("missing_body", codes)
        self.assertIn("missing_labels", codes)
        self.assertIn("missing_fixed_revision", codes)
        self.assertIn("missing_candidate_verifier", codes)
        self.assertIn("non_public_source_url", codes)
        self.assertIn("private_url", codes)

    def test_cli_collection_plan_and_audit_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            source = root / "public.jsonl"
            plan_output = root / "plan.json"
            audit_output = root / "audit.json"
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )
            source.write_text(json.dumps(_source_record()) + "\n", encoding="utf-8")
            plan_stdout = io.StringIO()

            with redirect_stdout(plan_stdout):
                plan_exit_code = main(
                    [
                        "registry",
                        "collection-plan",
                        "--allowlist",
                        str(allowlist),
                        "--output",
                        str(plan_output),
                        "--output-root",
                        str(root / "exports"),
                    ]
                )

            audit_stdout = io.StringIO()
            with redirect_stdout(audit_stdout):
                audit_exit_code = main(
                    [
                        "registry",
                        "collection-audit",
                        "--source",
                        str(source),
                        "--allowlist",
                        str(allowlist),
                        "--output",
                        str(audit_output),
                    ]
                )

            self.assertEqual(plan_exit_code, 0)
            self.assertEqual(audit_exit_code, 0)
            self.assertTrue(json.loads(plan_output.read_text(encoding="utf-8"))["valid"])
            self.assertTrue(json.loads(audit_output.read_text(encoding="utf-8"))["valid"])


def _allowlist_record() -> dict[str, object]:
    return {
        "repository": "example/tool",
        "source_uri": "https://github.com/example/tool.git",
        "license": "MIT",
        "language": "Python",
        "collection_sources": ["issues", "pull_requests"],
        "issue_labels": ["bug", "parser"],
        "pr_labels": ["review"],
        "test_commands": ["python -m pytest tests/test_parser.py"],
    }


def _source_record() -> dict[str, object]:
    return {
        "id": "issue-100",
        "type": "issue",
        "repository": "example/tool",
        "source_uri": "https://github.com/example/tool.git",
        "source_revision": "a" * 40,
        "source_url": "https://github.com/example/tool/issues/100",
        "title": "Fix parser whitespace handling",
        "body": "The parser drops significant whitespace around quoted values.",
        "labels": ["bug", "parser"],
        "license": "MIT",
        "language": "Python",
        "test_commands": ["python -m pytest tests/test_parser.py::test_whitespace"],
    }


if __name__ == "__main__":
    unittest.main()
