import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.source_collection import (
    audit_public_source_records,
    build_source_collection_plan,
    merge_source_export_summaries,
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

    def test_merge_source_export_summaries_rejects_malformed_summary_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON objects"):
            merge_source_export_summaries([_source_record()], [[]])

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

    def test_cli_collection_audit_writes_accepted_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            source = root / "public.jsonl"
            audit_output = root / "audit.json"
            accepted_output = root / "accepted.jsonl"
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )
            source.write_text(
                json.dumps(_source_record())
                + "\n"
                + json.dumps(
                    {
                        **_source_record(),
                        "id": "issue-101",
                        "source_instance_id": "example__tool-issue-101",
                        "source_url": "https://github.com/example/tool/issues/101",
                        "body": "The docs mention http://127.0.0.1:8000.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "registry",
                        "collection-audit",
                        "--source",
                        str(source),
                        "--allowlist",
                        str(allowlist),
                        "--output",
                        str(audit_output),
                        "--accepted-output",
                        str(accepted_output),
                    ]
                )

            audit = json.loads(audit_output.read_text(encoding="utf-8"))
            accepted = [
                json.loads(line)
                for line in accepted_output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(exit_code, 2)
            self.assertFalse(audit["valid"])
            self.assertEqual(audit["accepted"], 1)
            self.assertEqual(audit["quarantined"], 1)
            self.assertEqual([record["id"] for record in accepted], ["issue-100"])

    def test_cli_collection_shards_writes_deterministic_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            schedule_output = root / "schedule.json"
            source_output = root / "source.jsonl"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            plan_output.write_text(json.dumps(plan), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "registry",
                        "collection-shards",
                        "--plan",
                        str(plan_output),
                        "--source-output",
                        str(source_output),
                        "--summary-output-dir",
                        str(root / "summaries"),
                        "--preflight-output-dir",
                        str(root / "preflight"),
                        "--shard-size",
                        "1",
                        "--limit-per-task",
                        "3",
                        "--sleep-seconds",
                        "0.5",
                        "--resume",
                        "--allow-partial",
                        "--github-token-env",
                        "GITHUB_TOKEN",
                        "--require-github-token",
                        "--output",
                        str(schedule_output),
                    ]
                )

            schedule = json.loads(schedule_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertTrue(schedule["valid"])
            self.assertEqual(schedule["plan_tasks"], 2)
            self.assertEqual(schedule["shard_size"], 1)
            self.assertEqual(schedule["shard_count"], 2)
            self.assertEqual(
                [shard["collection_source_counts"] for shard in schedule["shards"]],
                [{"issues": 1}, {"pull_requests": 1}],
            )
            first_shard = schedule["shards"][0]
            self.assertEqual(first_shard["task_offset"], 0)
            self.assertIn("collection-preflight", first_shard["preflight_args"])
            self.assertIn("collection-export", first_shard["export_args"])
            self.assertIn("--require-github-token", first_shard["preflight_args"])
            self.assertIn("--resume", first_shard["export_args"])
            self.assertIn("--allow-partial", first_shard["export_args"])
            self.assertFalse(source_output.exists())

    def test_cli_collection_shard_status_reports_pending_and_blocked_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            schedule_output = root / "schedule.json"
            status_output = root / "status.json"
            source_output = root / "source.jsonl"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            plan_output.write_text(json.dumps(plan), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                schedule_exit_code = main(
                    [
                        "registry",
                        "collection-shards",
                        "--plan",
                        str(plan_output),
                        "--source-output",
                        str(source_output),
                        "--summary-output-dir",
                        str(root / "summaries"),
                        "--preflight-output-dir",
                        str(root / "preflight"),
                        "--shard-size",
                        "1",
                        "--github-token-env",
                        "GITHUB_TOKEN",
                        "--require-github-token",
                        "--output",
                        str(schedule_output),
                    ]
                )

            schedule = json.loads(schedule_output.read_text(encoding="utf-8"))
            first_preflight = Path(schedule["shards"][0]["preflight_output"])
            first_preflight.parent.mkdir(parents=True, exist_ok=True)
            first_preflight.write_text(
                json.dumps(
                    {
                        "valid": False,
                        "ready_for_collection": False,
                        "selected_tasks": 1,
                        "issues": [
                            {
                                "code": "missing_github_token",
                                "message": "environment variable GITHUB_TOKEN is not set",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                status_exit_code = main(
                    [
                        "registry",
                        "collection-shard-status",
                        "--schedule",
                        str(schedule_output),
                        "--source",
                        str(source_output),
                        "--output",
                        str(status_output),
                    ]
                )

            status = json.loads(status_output.read_text(encoding="utf-8"))
            self.assertEqual(schedule_exit_code, 0)
            self.assertEqual(status_exit_code, 2)
            self.assertFalse(status["ready_for_summary"])
            self.assertEqual(status["blocked_shards"], 1)
            self.assertEqual(status["pending_shards"], 1)
            self.assertEqual(status["source_records"], 0)
            self.assertEqual(status["shards"][0]["preflight_status"], "blocked")
            self.assertEqual(status["shards"][0]["next_action"], "resolve_preflight")
            self.assertEqual(status["shards"][1]["preflight_status"], "missing")
            self.assertEqual(status["shards"][1]["next_action"], "run_preflight")

    def test_cli_collection_shard_status_accepts_complete_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            schedule_output = root / "schedule.json"
            status_output = root / "status.json"
            source_output = root / "source.jsonl"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            plan_output.write_text(json.dumps(plan), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                schedule_exit_code = main(
                    [
                        "registry",
                        "collection-shards",
                        "--plan",
                        str(plan_output),
                        "--source-output",
                        str(source_output),
                        "--summary-output-dir",
                        str(root / "summaries"),
                        "--preflight-output-dir",
                        str(root / "preflight"),
                        "--shard-size",
                        "1",
                        "--output",
                        str(schedule_output),
                    ]
                )

            source_output.write_text(
                json.dumps(_source_record()) + "\n" + json.dumps(_pr_source_record()) + "\n",
                encoding="utf-8",
            )
            schedule = json.loads(schedule_output.read_text(encoding="utf-8"))
            for index, shard in enumerate(schedule["shards"]):
                preflight_output = Path(shard["preflight_output"])
                summary_output = Path(shard["summary_output"])
                preflight_output.parent.mkdir(parents=True, exist_ok=True)
                summary_output.parent.mkdir(parents=True, exist_ok=True)
                preflight_output.write_text(
                    json.dumps(
                        {
                            "valid": True,
                            "ready_for_collection": True,
                            "selected_tasks": 1,
                            "issues": [],
                        }
                    ),
                    encoding="utf-8",
                )
                summary_output.write_text(
                    json.dumps(
                        {
                            "valid": True,
                            "plan_tasks": 2,
                            "task_offset": index,
                            "selected_tasks": 1,
                            "processed_tasks": 1,
                            "exported": 1,
                            "allow_partial": True,
                            "issues": [],
                            "blocking_issues": [],
                        }
                    ),
                    encoding="utf-8",
                )

            with redirect_stdout(io.StringIO()):
                status_exit_code = main(
                    [
                        "registry",
                        "collection-shard-status",
                        "--schedule",
                        str(schedule_output),
                        "--source",
                        str(source_output),
                        "--output",
                        str(status_output),
                    ]
                )

            status = json.loads(status_output.read_text(encoding="utf-8"))
            self.assertEqual(schedule_exit_code, 0)
            self.assertEqual(status_exit_code, 0)
            self.assertTrue(status["ready_for_summary"])
            self.assertEqual(status["completed_shards"], 2)
            self.assertEqual(status["source_records"], 2)
            self.assertTrue(all(shard["next_action"] == "none" for shard in status["shards"]))

    def test_cli_collection_shard_status_preserves_partial_export_issues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            schedule_output = root / "schedule.json"
            status_output = root / "status.json"
            source_output = root / "source.jsonl"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            plan_output.write_text(json.dumps(plan), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                schedule_exit_code = main(
                    [
                        "registry",
                        "collection-shards",
                        "--plan",
                        str(plan_output),
                        "--source-output",
                        str(source_output),
                        "--summary-output-dir",
                        str(root / "summaries"),
                        "--preflight-output-dir",
                        str(root / "preflight"),
                        "--shard-size",
                        "1",
                        "--output",
                        str(schedule_output),
                    ]
                )

            source_output.write_text(json.dumps(_source_record()) + "\n", encoding="utf-8")
            schedule = json.loads(schedule_output.read_text(encoding="utf-8"))
            first_shard = schedule["shards"][0]
            preflight_output = Path(first_shard["preflight_output"])
            summary_output = Path(first_shard["summary_output"])
            preflight_output.parent.mkdir(parents=True, exist_ok=True)
            summary_output.parent.mkdir(parents=True, exist_ok=True)
            preflight_output.write_text(
                json.dumps({"valid": True, "ready_for_collection": True, "issues": []}),
                encoding="utf-8",
            )
            summary_output.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "selected_tasks": 1,
                        "processed_tasks": 1,
                        "exported": 1,
                        "allow_partial": True,
                        "issues": ["collection-task-1: ssl failure"],
                        "blocking_issues": [],
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                status_exit_code = main(
                    [
                        "registry",
                        "collection-shard-status",
                        "--schedule",
                        str(schedule_output),
                        "--source",
                        str(source_output),
                        "--output",
                        str(status_output),
                    ]
                )

            status = json.loads(status_output.read_text(encoding="utf-8"))
            self.assertEqual(schedule_exit_code, 0)
            self.assertEqual(status_exit_code, 2)
            self.assertFalse(status["ready_for_summary"])
            self.assertEqual(status["partial_shards"], 1)
            self.assertEqual(status["completed_shards"], 0)
            self.assertEqual(status["shards"][0]["status"], "partial")
            self.assertEqual(status["shards"][0]["next_action"], "plan_retry")
            self.assertEqual(status["shards"][0]["issues"], ["collection-task-1: ssl failure"])

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

    def test_cli_collection_export_writes_auditable_ci_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            plan_output = root / "plan.json"
            export_output = root / "exports.jsonl"
            audit_output = root / "audit.json"
            readiness_output = root / "readiness.json"
            _write_fixture_source(root / "fixtures", include_pull_requests=False, include_ci=True)
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record_with_ci()]}),
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

            records = [
                json.loads(line) for line in export_output.read_text(encoding="utf-8").splitlines()
            ]
            export_summary = json.loads(export_stdout.getvalue())
            (root / "summary.json").write_text(
                json.dumps(export_summary),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                readiness_exit_code = main(
                    [
                        "registry",
                        "collection-readiness",
                        "--plan",
                        str(plan_output),
                        "--export-summary",
                        str(root / "summary.json"),
                        "--audit",
                        str(audit_output),
                        "--min-accepted",
                        "2",
                        "--require-source-type",
                        "public_issue",
                        "--require-source-type",
                        "public_ci",
                        "--output",
                        str(readiness_output),
                    ]
                )
            audit = json.loads(audit_output.read_text(encoding="utf-8"))
            ci_record = next(record for record in records if record["type"] == "ci_failure")
            self.assertEqual(plan_exit_code, 0)
            self.assertEqual(export_exit_code, 0)
            self.assertEqual(audit_exit_code, 0)
            self.assertEqual(readiness_exit_code, 0)
            self.assertEqual(export_summary["source_type_counts"]["public_ci"], 1)
            self.assertEqual(audit["source_type_counts"]["public_ci"], 1)
            self.assertEqual(ci_record["source_revision"], "c" * 40)
            self.assertEqual(ci_record["ci_commands"], ["python -m build", "python -m pytest"])
            self.assertEqual(ci_record["candidate_verifier"]["type"], "ci_commands")

    def test_cli_collection_export_resumes_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            plan_output = root / "plan.json"
            export_output = root / "exports.jsonl"
            summary_output = root / "summary.json"
            _write_fixture_source(root / "fixtures", include_pull_requests=True)
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )
            existing_record = {
                **_source_record(),
                "id": "example__tool-issue-100",
                "source_instance_id": "example__tool-issue-100",
            }
            export_output.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "registry",
                        "collection-plan",
                        "--allowlist",
                        str(allowlist),
                        "--output",
                        str(plan_output),
                    ]
                )
            with redirect_stdout(io.StringIO()):
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
                        "--resume",
                        "--summary-output",
                        str(summary_output),
                    ]
                )

            records = [
                json.loads(line) for line in export_output.read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(export_exit_code, 0)
            self.assertEqual(len(records), 2)
            self.assertEqual(summary["existing_records"], 1)
            self.assertEqual(summary["new_records"], 1)
            self.assertEqual(summary["duplicate_records"], 1)
            self.assertEqual(summary["exported"], 2)

    def test_cli_collection_export_allows_partial_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            plan_output = root / "plan.json"
            export_output = root / "exports.jsonl"
            _write_fixture_source(root / "fixtures", include_pull_requests=False)
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                main(
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
                        "--allow-partial",
                    ]
                )

            summary = json.loads(export_stdout.getvalue())
            records = [
                json.loads(line) for line in export_output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(export_exit_code, 0)
            self.assertEqual(len(records), 1)
            self.assertTrue(summary["valid"])
            self.assertTrue(summary["allow_partial"])
            self.assertEqual(summary["exported"], 1)
            self.assertEqual(summary["new_records"], 1)
            self.assertEqual(len(summary["issues"]), 1)

    def test_cli_collection_export_records_task_outcomes_for_retry_planning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowlist = root / "allowlist.json"
            plan_output = root / "plan.json"
            export_output = root / "exports.jsonl"
            _write_fixture_source(root / "fixtures", include_pull_requests=False)
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                main(
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
                exit_code = main(
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
                        "--allow-partial",
                    ]
                )

            summary = json.loads(export_stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["task_offset"], 0)
            self.assertIsNone(summary["max_tasks"])
            self.assertEqual(
                [item["status"] for item in summary["task_outcomes"]],
                ["processed", "failed"],
            )
            self.assertEqual(summary["task_outcomes"][0]["new_records"], 1)
            self.assertIn("collection-task_", summary["task_outcomes"][1]["issue"])

    def test_cli_collection_export_requires_github_token_when_requested(self) -> None:
        env_name = "EAD_TEST_MISSING_GITHUB_TOKEN"
        old_value = os.environ.pop(env_name, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan_output = root / "plan.json"
                export_output = root / "exports.jsonl"
                summary_output = root / "summary.json"
                plan = build_source_collection_plan(
                    [_allowlist_record()],
                    output_root=root / "exports",
                    source_name="curated-public-sources",
                )
                plan_output.write_text(json.dumps(plan), encoding="utf-8")

                with redirect_stdout(io.StringIO()):
                    exit_code = main(
                        [
                            "registry",
                            "collection-export",
                            "--plan",
                            str(plan_output),
                            "--output",
                            str(export_output),
                            "--summary-output",
                            str(summary_output),
                            "--github-token-env",
                            env_name,
                            "--require-github-token",
                        ]
                    )

                summary = json.loads(summary_output.read_text(encoding="utf-8"))
                self.assertEqual(exit_code, 2)
                self.assertFalse(summary["valid"])
                self.assertEqual(summary["exported"], 0)
                self.assertEqual(
                    summary["blocking_issues"],
                    [f"missing_github_token: environment variable {env_name} is not set"],
                )
                self.assertFalse(export_output.exists())
        finally:
            if old_value is not None:
                os.environ[env_name] = old_value

    def test_cli_collection_preflight_blocks_missing_required_github_token(self) -> None:
        env_name = "EAD_TEST_MISSING_GITHUB_TOKEN"
        old_value = os.environ.pop(env_name, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan_output = root / "plan.json"
                source_output = root / "source.jsonl"
                preflight_output = root / "preflight.json"
                plan = build_source_collection_plan(
                    [_allowlist_record()],
                    output_root=root / "exports",
                    source_name="curated-public-sources",
                )
                plan_output.write_text(json.dumps(plan), encoding="utf-8")

                with redirect_stdout(io.StringIO()):
                    exit_code = main(
                        [
                            "registry",
                            "collection-preflight",
                            "--plan",
                            str(plan_output),
                            "--source",
                            str(source_output),
                            "--github-token-env",
                            env_name,
                            "--require-github-token",
                            "--task-offset",
                            "0",
                            "--max-tasks",
                            "1",
                            "--output",
                            str(preflight_output),
                        ]
                    )

                preflight = json.loads(preflight_output.read_text(encoding="utf-8"))
                issue_codes = {issue["code"] for issue in preflight["issues"]}
                self.assertEqual(exit_code, 2)
                self.assertFalse(preflight["ready_for_collection"])
                self.assertEqual(preflight["selected_tasks"], 1)
                self.assertFalse(preflight["github_token_configured"])
                self.assertFalse(source_output.exists())
                self.assertIn("missing_github_token", issue_codes)
        finally:
            if old_value is not None:
                os.environ[env_name] = old_value

    def test_cli_collection_preflight_accepts_existing_artifacts(self) -> None:
        env_name = "EAD_TEST_GITHUB_TOKEN"
        old_value = os.environ.get(env_name)
        os.environ[env_name] = "value-not-emitted"
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan_output = root / "plan.json"
                source_output = root / "source.jsonl"
                summary_output = root / "summary.json"
                preflight_output = root / "preflight.json"
                plan = build_source_collection_plan(
                    [_allowlist_record()],
                    output_root=root / "exports",
                    source_name="curated-public-sources",
                )
                plan_output.write_text(json.dumps(plan), encoding="utf-8")
                source_output.write_text(json.dumps(_source_record()) + "\n", encoding="utf-8")
                summary_output.write_text(
                    json.dumps(
                        {
                            "valid": True,
                            "plan_tasks": 2,
                            "selected_tasks": 1,
                            "processed_tasks": 1,
                            "exported": 1,
                            "issues": [],
                        }
                    ),
                    encoding="utf-8",
                )

                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "registry",
                            "collection-preflight",
                            "--plan",
                            str(plan_output),
                            "--source",
                            str(source_output),
                            "--summary",
                            str(summary_output),
                            "--github-token-env",
                            env_name,
                            "--require-github-token",
                            "--require-source",
                            "--output",
                            str(preflight_output),
                        ]
                    )

                preflight = json.loads(preflight_output.read_text(encoding="utf-8"))
                self.assertEqual(exit_code, 0)
                self.assertTrue(preflight["ready_for_collection"])
                self.assertTrue(preflight["github_token_configured"])
                self.assertEqual(preflight["source_records"], 1)
                self.assertEqual(preflight["existing_summaries"], 1)
                self.assertEqual(preflight["issues"], [])
                self.assertNotIn("value-not-emitted", stdout.getvalue())
                self.assertNotIn(
                    "value-not-emitted",
                    preflight_output.read_text(encoding="utf-8"),
                )
        finally:
            if old_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = old_value

    def test_cli_collection_retry_plan_assigns_explicit_retry_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            summary_output = root / "summary.json"
            retry_output = root / "retry.json"
            legacy_summary_output = root / "legacy-summary.json"
            legacy_retry_output = root / "legacy-retry.json"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            failed_task = plan["tasks"][0]
            plan_output.write_text(json.dumps(plan), encoding="utf-8")
            summary_output.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "plan_tasks": 2,
                        "task_offset": 0,
                        "selected_tasks": 1,
                        "processed_tasks": 1,
                        "skipped_tasks": 0,
                        "output_path": str(root / "records.jsonl"),
                        "issues": [f"{failed_task['task_id']}: HTTP Error 403"],
                        "task_outcomes": [
                            {
                                "task_id": failed_task["task_id"],
                                "task_index": 0,
                                "status": "failed",
                                "issue": f"{failed_task['task_id']}: HTTP Error 403",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "registry",
                        "collection-retry-plan",
                        "--plan",
                        str(plan_output),
                        "--export-summary",
                        str(summary_output),
                        "--output",
                        str(retry_output),
                    ]
                )

            retry_plan = json.loads(retry_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertTrue(retry_plan["ready_for_retry"])
            self.assertEqual(retry_plan["reason_counts"], {"failed": 1, "not_selected": 1})
            self.assertEqual(
                [(item["task_index"], item["repository"]) for item in retry_plan["retry_tasks"]],
                [(0, "example/tool"), (1, "example/tool")],
            )
            self.assertEqual(
                retry_plan["retry_tasks"][0]["collection_export_args"],
                ["--task-offset", "0", "--max-tasks", "1"],
            )

            legacy_summary_output.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "plan_tasks": 2,
                        "selected_tasks": 1,
                        "processed_tasks": 0,
                        "skipped_tasks": 1,
                        "output_path": str(root / "records.jsonl"),
                        "issues": [],
                    }
                ),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                legacy_exit_code = main(
                    [
                        "registry",
                        "collection-retry-plan",
                        "--plan",
                        str(plan_output),
                        "--export-summary",
                        str(legacy_summary_output),
                        "--output",
                        str(legacy_retry_output),
                    ]
                )

            legacy_retry_plan = json.loads(legacy_retry_output.read_text(encoding="utf-8"))
            self.assertEqual(legacy_exit_code, 0)
            self.assertEqual(
                legacy_retry_plan["reason_counts"],
                {"not_selected": 1, "skipped_or_unverified": 1},
            )

    def test_cli_collection_retry_plan_keeps_selected_tasks_without_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            summary_output = root / "summary.json"
            retry_output = root / "retry.json"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            plan_output.write_text(json.dumps(plan), encoding="utf-8")
            summary_output.write_text(
                json.dumps(
                    {
                        "valid": False,
                        "plan_tasks": 2,
                        "task_offset": 0,
                        "selected_tasks": 1,
                        "processed_tasks": 0,
                        "skipped_tasks": 0,
                        "output_path": str(root / "records.jsonl"),
                        "blocking_issues": [
                            "missing_github_token: environment variable GITHUB_TOKEN is not set"
                        ],
                        "issues": [],
                        "task_outcomes": [],
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "registry",
                        "collection-retry-plan",
                        "--plan",
                        str(plan_output),
                        "--export-summary",
                        str(summary_output),
                        "--output",
                        str(retry_output),
                    ]
                )

            retry_plan = json.loads(retry_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                retry_plan["reason_counts"],
                {"missing_task_outcome": 1, "not_selected": 1},
            )
            self.assertEqual(
                [task["task_index"] for task in retry_plan["retry_tasks"]],
                [0, 1],
            )
            self.assertEqual(
                retry_plan["retry_tasks"][0]["issue"],
                "Selected collection task has no recorded outcome",
            )

    def test_cli_collection_retry_run_executes_retry_plan_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            retry_plan_output = root / "retry.json"
            export_output = root / "exports.jsonl"
            summary_output = root / "retry-run.json"
            _write_fixture_source(root / "fixtures", include_pull_requests=True)
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            plan_output.write_text(json.dumps(plan), encoding="utf-8")
            retry_plan_output.write_text(
                json.dumps(
                    {
                        "retry_tasks": [
                            {
                                "task_id": task["task_id"],
                                "task_index": index,
                                "repository": task["repository"],
                                "collection_source": task["collection_source"],
                            }
                            for index, task in enumerate(plan["tasks"])
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "registry",
                        "collection-retry-run",
                        "--plan",
                        str(plan_output),
                        "--retry-plan",
                        str(retry_plan_output),
                        "--output",
                        str(export_output),
                        "--summary-output",
                        str(summary_output),
                        "--fixture-root",
                        str(root / "fixtures"),
                        "--limit-per-task",
                        "1",
                    ]
                )

            records = [
                json.loads(line) for line in export_output.read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["attempted_tasks"], 2)
            self.assertEqual(summary["completed_tasks"], 2)
            self.assertEqual(summary["failed_tasks"], 0)
            self.assertEqual(summary["new_records"], 2)
            self.assertEqual(
                [attempt["status"] for attempt in summary["attempts"]],
                ["processed", "processed"],
            )
            self.assertEqual([record["type"] for record in records], ["issue", "pull_request"])

    def test_cli_collection_retry_run_requires_github_token_when_requested(self) -> None:
        env_name = "EAD_TEST_MISSING_GITHUB_TOKEN"
        old_value = os.environ.pop(env_name, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan_output = root / "plan.json"
                retry_plan_output = root / "retry.json"
                export_output = root / "exports.jsonl"
                summary_output = root / "retry-run.json"
                plan = build_source_collection_plan(
                    [_allowlist_record()],
                    output_root=root / "exports",
                    source_name="curated-public-sources",
                )
                plan_output.write_text(json.dumps(plan), encoding="utf-8")
                retry_plan_output.write_text(
                    json.dumps(
                        {
                            "retry_tasks": [
                                {
                                    "task_id": plan["tasks"][0]["task_id"],
                                    "task_index": 0,
                                    "repository": "example/tool",
                                    "collection_source": "issues",
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

                with redirect_stdout(io.StringIO()):
                    exit_code = main(
                        [
                            "registry",
                            "collection-retry-run",
                            "--plan",
                            str(plan_output),
                            "--retry-plan",
                            str(retry_plan_output),
                            "--output",
                            str(export_output),
                            "--summary-output",
                            str(summary_output),
                            "--github-token-env",
                            env_name,
                            "--require-github-token",
                        ]
                    )

                summary = json.loads(summary_output.read_text(encoding="utf-8"))
                self.assertEqual(exit_code, 2)
                self.assertFalse(summary["valid"])
                self.assertEqual(
                    summary["blocking_issues"],
                    [f"missing_github_token: environment variable {env_name} is not set"],
                )
                self.assertFalse(export_output.exists())
        finally:
            if old_value is not None:
                os.environ[env_name] = old_value

    def test_cli_collection_summary_merges_export_and_retry_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            source_output = root / "source.jsonl"
            export_summary_output = root / "export-summary.json"
            retry_summary_output = root / "retry-summary.json"
            merged_summary_output = root / "merged-summary.json"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            issue_task = plan["tasks"][0]
            pr_task = plan["tasks"][1]
            plan_output.write_text(json.dumps(plan), encoding="utf-8")
            source_output.write_text(
                json.dumps(_source_record()) + "\n" + json.dumps(_pr_source_record()) + "\n",
                encoding="utf-8",
            )
            export_summary_output.write_text(
                json.dumps(
                    {
                        "valid": False,
                        "plan_tasks": 2,
                        "task_offset": 0,
                        "selected_tasks": 2,
                        "processed_tasks": 2,
                        "exported": 1,
                        "issues": [f"{pr_task['task_id']}: HTTP Error 403"],
                        "task_outcomes": [
                            {
                                "task_id": issue_task["task_id"],
                                "task_index": 0,
                                "repository": "example/tool",
                                "collection_source": "issues",
                                "status": "processed",
                                "new_records": 1,
                            },
                            {
                                "task_id": pr_task["task_id"],
                                "task_index": 1,
                                "repository": "example/tool",
                                "collection_source": "pull_requests",
                                "status": "failed",
                                "issue": f"{pr_task['task_id']}: HTTP Error 403",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            retry_summary_output.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "retry_tasks": 1,
                        "selected_retry_tasks": 1,
                        "attempted_tasks": 1,
                        "completed_tasks": 1,
                        "attempts": [
                            {
                                "task_id": pr_task["task_id"],
                                "task_index": 1,
                                "repository": "example/tool",
                                "collection_source": "pull_requests",
                                "status": "processed",
                                "new_records": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "registry",
                        "collection-summary",
                        "--source",
                        str(source_output),
                        "--summary",
                        str(export_summary_output),
                        "--summary",
                        str(retry_summary_output),
                        "--plan",
                        str(plan_output),
                        "--output",
                        str(merged_summary_output),
                    ]
                )

            summary = json.loads(merged_summary_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["exported"], 2)
            self.assertEqual(summary["selected_tasks"], 2)
            self.assertEqual(summary["processed_tasks"], 2)
            self.assertEqual(summary["issues"], [])
            self.assertEqual(
                summary["source_type_counts"],
                {"public_issue": 1, "public_pr": 1},
            )
            self.assertEqual(
                [outcome["status"] for outcome in summary["task_outcomes"]],
                ["processed", "processed"],
            )

    def test_cli_collection_summary_blocks_unresolved_issues_without_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            source_output = root / "source.jsonl"
            export_summary_output = root / "export-summary.json"
            merged_summary_output = root / "merged-summary.json"
            partial_summary_output = root / "partial-summary.json"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            failed_task = plan["tasks"][1]
            issue = f"{failed_task['task_id']}: HTTP Error 403"
            plan_output.write_text(json.dumps(plan), encoding="utf-8")
            source_output.write_text(json.dumps(_source_record()) + "\n", encoding="utf-8")
            export_summary_output.write_text(
                json.dumps(
                    {
                        "valid": False,
                        "plan_tasks": 2,
                        "task_offset": 0,
                        "selected_tasks": 2,
                        "processed_tasks": 2,
                        "exported": 1,
                        "issues": [issue],
                        "task_outcomes": [
                            {
                                "task_id": failed_task["task_id"],
                                "task_index": 1,
                                "repository": "example/tool",
                                "collection_source": "pull_requests",
                                "status": "failed",
                                "issue": issue,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "registry",
                        "collection-summary",
                        "--source",
                        str(source_output),
                        "--summary",
                        str(export_summary_output),
                        "--plan",
                        str(plan_output),
                        "--output",
                        str(merged_summary_output),
                    ]
                )
            with redirect_stdout(io.StringIO()):
                partial_exit_code = main(
                    [
                        "registry",
                        "collection-summary",
                        "--source",
                        str(source_output),
                        "--summary",
                        str(export_summary_output),
                        "--plan",
                        str(plan_output),
                        "--output",
                        str(partial_summary_output),
                        "--allow-partial",
                    ]
                )

            summary = json.loads(merged_summary_output.read_text(encoding="utf-8"))
            partial_summary = json.loads(partial_summary_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 2)
            self.assertFalse(summary["valid"])
            self.assertEqual(summary["issues"], [issue])
            self.assertEqual(partial_exit_code, 0)
            self.assertTrue(partial_summary["valid"])
            self.assertTrue(partial_summary["allow_partial"])

    def test_cli_collection_split_routes_mixed_records_by_source_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mixed.jsonl"
            issue_pr_output = root / "issue-pr.jsonl"
            issue_pr_summary_output = root / "issue-pr-summary.json"
            ci_output = root / "ci.jsonl"
            ci_summary_output = root / "ci-summary.json"
            source.write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in (_source_record(), _pr_source_record(), _ci_source_record())
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                issue_pr_exit_code = main(
                    [
                        "registry",
                        "collection-split",
                        "--source",
                        str(source),
                        "--output",
                        str(issue_pr_output),
                        "--summary-output",
                        str(issue_pr_summary_output),
                        "--include-source-type",
                        "public_issue",
                        "--include-source-type",
                        "public_pr",
                    ]
                )
            with redirect_stdout(io.StringIO()):
                ci_exit_code = main(
                    [
                        "registry",
                        "collection-split",
                        "--source",
                        str(source),
                        "--output",
                        str(ci_output),
                        "--summary-output",
                        str(ci_summary_output),
                        "--include-source-type",
                        "public-ci",
                    ]
                )

            issue_pr_records = [
                json.loads(line)
                for line in issue_pr_output.read_text(encoding="utf-8").splitlines()
            ]
            ci_records = [
                json.loads(line) for line in ci_output.read_text(encoding="utf-8").splitlines()
            ]
            issue_pr_summary = json.loads(issue_pr_summary_output.read_text(encoding="utf-8"))
            ci_summary = json.loads(ci_summary_output.read_text(encoding="utf-8"))
            self.assertEqual(issue_pr_exit_code, 0)
            self.assertEqual(ci_exit_code, 0)
            self.assertEqual(
                [record["type"] for record in issue_pr_records],
                ["issue", "pull_request"],
            )
            self.assertEqual([record["type"] for record in ci_records], ["ci_failure"])
            self.assertEqual(issue_pr_summary["selected_records"], 2)
            self.assertEqual(issue_pr_summary["skipped_records"], 1)
            self.assertEqual(
                issue_pr_summary["source_type_counts"],
                {"public_issue": 1, "public_pr": 1},
            )
            self.assertEqual(ci_summary["included_source_types"], ["public_ci"])
            self.assertEqual(ci_summary["source_type_counts"], {"public_ci": 1})

    def test_cli_collection_split_blocks_empty_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "issue-only.jsonl"
            output = root / "ci.jsonl"
            source.write_text(json.dumps(_source_record()) + "\n", encoding="utf-8")

            split_stdout = io.StringIO()
            with redirect_stdout(split_stdout):
                exit_code = main(
                    [
                        "registry",
                        "collection-split",
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--include-source-type",
                        "public_ci",
                    ]
                )

            summary = json.loads(split_stdout.getvalue())
            codes = {issue["code"] for issue in summary["issues"]}
            self.assertEqual(exit_code, 2)
            self.assertFalse(summary["valid"])
            self.assertEqual(summary["selected_records"], 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "")
            self.assertIn("no_source_records_selected", codes)

    def test_cli_collection_readiness_accepts_complete_source_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            summary_output = root / "summary.json"
            audit_output = root / "audit.json"
            readiness_output = root / "readiness.json"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            audit = audit_public_source_records(
                [_source_record(), _pr_source_record()],
                [_allowlist_record()],
                source_name="curated-public-sources",
            )
            plan_output.write_text(json.dumps(plan), encoding="utf-8")
            summary_output.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "plan_tasks": 2,
                        "selected_tasks": 2,
                        "processed_tasks": 2,
                        "exported": 2,
                        "duplicate_records": 0,
                        "skipped_tasks": 0,
                        "skipped_records": 0,
                        "source_type_counts": {"public_issue": 1, "public_pr": 1},
                        "repository_counts": {"example/tool": 2},
                        "issues": [],
                    }
                ),
                encoding="utf-8",
            )
            audit_output.write_text(json.dumps(audit.to_dict()), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "registry",
                        "collection-readiness",
                        "--plan",
                        str(plan_output),
                        "--export-summary",
                        str(summary_output),
                        "--audit",
                        str(audit_output),
                        "--min-accepted",
                        "2",
                        "--max-quarantined",
                        "0",
                        "--require-source-type",
                        "public_issue",
                        "--require-source-type",
                        "public_pr",
                        "--require-clean-export",
                        "--require-all-plan-tasks",
                        "--output",
                        str(readiness_output),
                    ]
                )

            readiness = json.loads(readiness_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertTrue(readiness["ready_for_import"])
            self.assertEqual(readiness["accepted_records"], 2)
            self.assertEqual(readiness["present_source_types"], {"public_issue": 1, "public_pr": 1})
            self.assertEqual(readiness["issues"], [])

    def test_cli_collection_readiness_blocks_incomplete_source_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_output = root / "plan.json"
            summary_output = root / "summary.json"
            audit_output = root / "audit.json"
            plan = build_source_collection_plan(
                [_allowlist_record()],
                output_root=root / "exports",
                source_name="curated-public-sources",
            )
            audit = audit_public_source_records(
                [
                    _source_record(),
                    {
                        **_source_record(),
                        "id": "issue-101",
                        "source_instance_id": "example__tool-issue-101",
                        "body": "",
                    },
                ],
                [_allowlist_record()],
                source_name="curated-public-sources",
            )
            plan_output.write_text(json.dumps(plan), encoding="utf-8")
            summary_output.write_text(
                json.dumps(
                    {
                        "valid": True,
                        "allow_partial": True,
                        "plan_tasks": 2,
                        "selected_tasks": 1,
                        "processed_tasks": 1,
                        "exported": 2,
                        "duplicate_records": 0,
                        "skipped_tasks": 0,
                        "skipped_records": 0,
                        "source_type_counts": {"public_issue": 2},
                        "repository_counts": {"example/tool": 2},
                        "issues": ["collection-task: rate limit"],
                    }
                ),
                encoding="utf-8",
            )
            audit_output.write_text(json.dumps(audit.to_dict()), encoding="utf-8")

            readiness_stdout = io.StringIO()
            with redirect_stdout(readiness_stdout):
                exit_code = main(
                    [
                        "registry",
                        "collection-readiness",
                        "--plan",
                        str(plan_output),
                        "--export-summary",
                        str(summary_output),
                        "--audit",
                        str(audit_output),
                        "--min-accepted",
                        "2",
                        "--max-quarantined",
                        "0",
                        "--require-source-type",
                        "public_pr",
                        "--require-clean-export",
                        "--require-all-plan-tasks",
                    ]
                )

            readiness = json.loads(readiness_stdout.getvalue())
            codes = {issue["code"] for issue in readiness["issues"]}
            self.assertEqual(exit_code, 2)
            self.assertFalse(readiness["ready_for_import"])
            self.assertIn("accepted_records_below_minimum", codes)
            self.assertIn("quarantine_budget_exceeded", codes)
            self.assertIn("missing_required_source_type", codes)
            self.assertIn("source_export_issues", codes)
            self.assertIn("partial_plan_selection", codes)


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


def _allowlist_record_with_ci() -> dict[str, object]:
    return {
        **_allowlist_record(),
        "collection_sources": ["issues", "ci"],
        "labels": ["ci", "failure"],
        "issue_labels": ["bug", "parser"],
        "pr_labels": [],
        "test_commands": ["python -m pytest"],
        "ci_commands": ["python -m build"],
    }


def _write_fixture_source(
    root: Path,
    *,
    include_pull_requests: bool,
    include_ci: bool = False,
) -> None:
    fixture_repo = root / "example__tool"
    (fixture_repo / "branches").mkdir(parents=True)
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
    if include_pull_requests:
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
    if include_ci:
        (fixture_repo / "workflow_runs.json").write_text(
            json.dumps(
                {
                    "workflow_runs": [
                        {
                            "id": 202,
                            "html_url": "https://github.com/example/tool/actions/runs/202",
                            "name": "tests",
                            "display_title": "CI failure on parser change",
                            "status": "completed",
                            "conclusion": "failure",
                            "event": "pull_request",
                            "head_branch": "parser-whitespace",
                            "head_sha": "c" * 40,
                            "head_commit": {
                                "message": "Fix parser whitespace handling",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )


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


def _pr_source_record() -> dict[str, object]:
    return {
        **_source_record(),
        "id": "pr-101",
        "type": "pull_request",
        "source_revision": "b" * 40,
        "source_instance_id": "example__tool-pr-101",
        "source_url": "https://github.com/example/tool/pull/101",
        "title": "Add parser regression coverage",
        "body": "This PR adds tests for quoted whitespace.",
        "labels": ["review"],
    }


def _ci_source_record() -> dict[str, object]:
    return {
        **_source_record(),
        "id": "ci-202",
        "type": "ci_failure",
        "source_revision": "c" * 40,
        "source_instance_id": "example__tool-ci-202",
        "source_url": "https://github.com/example/tool/actions/runs/202",
        "title": "CI failure on parser change",
        "body": "Workflow: tests\nConclusion: failure\nHead SHA: " + "c" * 40,
        "labels": ["ci", "failure"],
        "ci_commands": ["python -m build", "python -m pytest"],
        "candidate_verifier": {
            "type": "ci_commands",
            "commands": ["python -m build", "python -m pytest"],
        },
    }


if __name__ == "__main__":
    unittest.main()
