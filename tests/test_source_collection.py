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

    def test_cli_collection_export_writes_auditable_public_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            plan_output = root / "plan.json"
            export_output = root / "exports.jsonl"
            audit_output = root / "audit.json"
            fixture_repo = root / "fixtures" / "example__tool"
            (fixture_repo / "branches").mkdir(parents=True)
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )
            (fixture_repo / "repository.json").write_text(
                json.dumps({"default_branch": "main"}),
                encoding="utf-8",
            )
            (fixture_repo / "branches" / "main.json").write_text(
                json.dumps({"commit": {"sha": "a" * 40}}),
                encoding="utf-8",
            )
            (fixture_repo / "issues.json").write_text(
                json.dumps(
                    [
                        {
                            "number": 100,
                            "html_url": "https://github.com/example/tool/issues/100",
                            "title": "Fix parser whitespace handling",
                            "body": "The parser drops significant whitespace.",
                            "labels": [{"name": "bug"}, {"name": "parser"}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (fixture_repo / "pull_requests.json").write_text(
                json.dumps(
                    [
                        {
                            "number": 101,
                            "html_url": "https://github.com/example/tool/pull/101",
                            "title": "Add parser regression coverage",
                            "body": "This PR adds tests for quoted whitespace.",
                            "labels": [{"name": "review"}],
                            "base": {"sha": "b" * 40},
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                plan_exit_code = main(
                    [
                        "registry",
                        "collection-plan",
                        "--allowlist",
                        str(allowlist),
                        "--output",
                        str(plan_output),
                    ]
                )
            export_stdout = io.StringIO()
            with redirect_stdout(export_stdout):
                export_exit_code = main(
                    [
                        "registry",
                        "collection-export",
                        "--plan",
                        str(plan_output),
                        "--output",
                        str(export_output),
                        "--fixture-root",
                        str(root / "fixtures"),
                        "--limit-per-task",
                        "1",
                    ]
                )
            with redirect_stdout(io.StringIO()):
                audit_exit_code = main(
                    [
                        "registry",
                        "collection-audit",
                        "--source",
                        str(export_output),
                        "--allowlist",
                        str(allowlist),
                        "--output",
                        str(audit_output),
                    ]
                )

            exported_records = [
                json.loads(line) for line in export_output.read_text(encoding="utf-8").splitlines()
            ]
            export_summary = json.loads(export_stdout.getvalue())
            audit = json.loads(audit_output.read_text(encoding="utf-8"))
            self.assertEqual(plan_exit_code, 0)
            self.assertEqual(export_exit_code, 0)
            self.assertEqual(audit_exit_code, 0)
            self.assertEqual(export_summary["exported"], 2)
            self.assertEqual(
                export_summary["source_type_counts"],
                {"public_issue": 1, "public_pr": 1},
            )
            self.assertEqual(audit["accepted"], 2)
            self.assertEqual(audit["quarantined"], 0)
            self.assertEqual(
                {record["source_revision"] for record in exported_records},
                {"a" * 40, "b" * 40},
            )
            self.assertTrue(all(record["candidate_verifier"] for record in exported_records))


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
