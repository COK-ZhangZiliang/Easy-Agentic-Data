import hashlib
import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from easy_agentic_data.cli import main
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_sources import load_source_records
from easy_agentic_data.repository_synthetic import DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seed_corpus import (
    _hidden_rehearsal_command_arguments,
    _run_git_apply_for_rehearsal,
    _run_hidden_commands_for_rehearsal,
    apply_hidden_command_curation_records,
    apply_hidden_test_patch_curation_records,
    apply_synthetic_evidence_records,
    assemble_seed_candidate_registry,
    build_hidden_command_curation_plan,
    build_hidden_command_curation_record_template,
    build_hidden_test_patch_curation_plan,
    build_hidden_test_patch_curation_record_template,
    build_seed_backfill_plan,
    build_seed_corpus,
    build_seed_remediation_plan,
    build_seed_selection_plan,
    build_source_workspace_materialization_plan,
    build_synthetic_backfill_spec_plan,
    build_synthetic_evidence_backfill_plan,
    build_synthetic_evidence_record_templates,
    build_synthetic_evidence_shard_schedule,
    combine_synthetic_generator_ready_specs,
    materialize_source_workspaces,
    rehearse_registry_import,
    summarize_synthetic_evidence_shard_status,
)
from easy_agentic_data.seed_library import SUPPORTED_TASK_FAMILIES, SeedLibraryPolicy
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed

PINNED_IMAGE = "python@sha256:" + ("c" * 64)


class SeedCorpusTests(unittest.TestCase):
    def test_build_seed_corpus_freezes_manifest_and_review_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_corpus_inputs(root)

            manifest = build_seed_corpus(config_path, overwrite_outputs=True)

            self.assertTrue(manifest["valid"])
            self.assertFalse(manifest["approved_for_scale"])
            self.assertEqual(manifest["seed_audit"]["train_eligible"], 12)
            self.assertGreater(
                manifest["seed_audit"]["train_verifier_type_counts"]["hidden_command"],
                0,
            )
            self.assertEqual(manifest["scenario_audit"]["trainable"], 12)
            self.assertEqual(manifest["quarantine"], {"records": 0, "issues": []})
            self.assertTrue(manifest["repository_allowlist"]["audit"]["valid"])
            self.assertEqual(manifest["repository_allowlist"]["filters"][0]["allowed"], 1)
            self.assertEqual(len(manifest["source_snapshots"]), 3)
            self.assertIn(
                "repository_synthetic",
                {snapshot["format"] for snapshot in manifest["source_snapshots"]},
            )
            self.assertTrue(Path(manifest["manifest_output"]).is_file())
            disk_manifest = json.loads(
                Path(manifest["manifest_output"]).read_text(encoding="utf-8")
            )
            self.assertEqual(disk_manifest["manifest_output"], manifest["manifest_output"])
            self.assertTrue((root / "seed-audit.json").is_file())
            self.assertTrue((root / "scenario-audit.json").is_file())
            review_lines = (root / "seed-review.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(review_lines), 0)
            self.assertEqual(len(ScenarioRegistry(root / "train").list_scenarios()), 12)
            self.assertEqual(len(ScenarioRegistry(root / "holdout").list_scenarios()), 1)

    def test_seed_backfill_plan_quantifies_coverage_and_dominance_gaps(self) -> None:
        plan = build_seed_backfill_plan(_backfill_audit(), _backfill_policy())

        family_gaps = {gap["target"]: gap for gap in plan["gaps"]["task_family"]}
        verifier_gaps = {gap["target"]: gap for gap in plan["gaps"]["verifier_type"]}
        source_method_gaps = {
            gap["target"]: gap for gap in plan["gaps"]["source_method"]
        }
        dominance = plan["gaps"]["dominance"]
        actions = {(action["action"], action["target"]) for action in plan["recommended_actions"]}

        self.assertTrue(plan["valid"])
        self.assertTrue(plan["requires_backfill"])
        self.assertEqual(family_gaps["docs_examples"]["shortfall"], 5)
        self.assertEqual(family_gaps["test_authoring"]["shortfall"], 10)
        self.assertIn("doctest", family_gaps["docs_examples"]["accepted_verifier_types"])
        self.assertEqual(verifier_gaps["doctest"]["shortfall"], 5)
        self.assertEqual(
            source_method_gaps["repository_grounded_synthetic"]["shortfall"],
            10,
        )
        self.assertEqual(
            dominance["task_family"][0]["additional_non_target_if_no_downsampling"],
            50,
        )
        self.assertEqual(
            dominance["language"][0]["additional_non_target_if_no_downsampling"],
            25,
        )
        self.assertIn(("generate_repository_grounded_synthetic_family", "docs_examples"), actions)
        self.assertIn(
            ("generate_repository_grounded_synthetic_records", "repository_grounded_synthetic"),
            actions,
        )
        self.assertIn(("add_cross_language_sources_or_downsample", "python"), actions)
        self.assertIn(("refresh_holdout_and_decontamination", "holdout_registry"), actions)

    def test_cli_seed_backfill_plan_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / "seed-audit.json"
            policy_path = root / "policy.json"
            output_path = root / "backfill-plan.json"
            audit_path.write_text(json.dumps(_backfill_audit()), encoding="utf-8")
            policy_path.write_text(json.dumps(_backfill_policy()), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-backfill-plan",
                        "--audit",
                        str(audit_path),
                        "--policy",
                        str(policy_path),
                        "--output",
                        str(output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["requires_backfill"])

    def test_seed_selection_plan_reserves_backfill_slots(self) -> None:
        seeds = [
            _selection_seed("bug-1", "bug_repair", "public_ci_workspace", "repo/a"),
            _selection_seed("bug-2", "bug_repair", "public_ci_workspace", "repo/a"),
            _selection_seed("bug-3", "bug_repair", "public_ci_workspace", "repo/a"),
            _selection_seed("ci-1", "ci_build", "public_ci_workspace", "repo/b"),
            _selection_seed("ci-2", "ci_build", "public_issue_workspace", "repo/c"),
            _selection_seed("review-1", "code_review", "public_pr_workspace", "repo/d"),
        ]

        plan = build_seed_selection_plan(
            seeds,
            _selection_policy(),
            target_train_eligible=6,
        )
        slot_keys = {(slot["type"], slot["target"]) for slot in plan["reserved_backfill"]["slots"]}

        self.assertTrue(plan["valid"])
        self.assertFalse(plan["ready_for_rollout"])
        self.assertTrue(plan["requires_backfill"])
        self.assertEqual(plan["target_train_eligible"], 6)
        self.assertEqual(plan["reserved_backfill"]["minimum_reserved_slots"], 2)
        self.assertEqual(plan["existing_selection_target"], 4)
        self.assertEqual(plan["selected_existing_count"], 4)
        self.assertLessEqual(plan["selected_counts"]["task_family"]["bug_repair"], 3)
        self.assertLessEqual(plan["selected_counts"]["repository"]["repo/a"], 3)
        self.assertLessEqual(
            plan["selected_shares_against_target"]["task_family"]["bug_repair"],
            0.5,
        )
        self.assertIn(("task_family_minimum", "docs_examples"), slot_keys)
        self.assertIn(("source_method_minimum", "repository_grounded_synthetic"), slot_keys)
        self.assertIn(("share_cap_diversity", "non_python"), slot_keys)
        self.assertEqual(len(plan["selected_seed_ids_sha256"]), 64)

    def test_cli_seed_selection_plan_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_root = root / "registry"
            policy_path = root / "policy.json"
            output_path = root / "selection-plan.json"
            registry = ScenarioRegistry(registry_root)
            registry.initialize()
            for seed in [
                _selection_seed("bug-1", "bug_repair", "public_ci_workspace", "repo/a"),
                _selection_seed("ci-1", "ci_build", "public_issue_workspace", "repo/b"),
            ]:
                registry.add_seed(seed)
            policy_path.write_text(json.dumps(_selection_policy()), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-selection-plan",
                        "--root",
                        str(registry_root),
                        "--policy",
                        str(policy_path),
                        "--target-train-eligible",
                        "4",
                        "--output",
                        str(output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertEqual(payload["candidate_train_eligible"], 2)
            self.assertTrue(payload["requires_backfill"])

    def test_seed_remediation_plan_maps_reserved_slots_to_requirements(self) -> None:
        selection_plan = _remediation_selection_plan()
        allowlist = [
            _allowlist_record(repository="repo/b", language="Python"),
            _allowlist_record(repository="repo/ts", language="TypeScript"),
        ]

        plan = build_seed_remediation_plan(
            selection_plan,
            _remediation_policy(),
            allowlist_records=allowlist,
        )
        actions = {requirement["action"] for requirement in plan["requirements"]}

        self.assertTrue(plan["valid"])
        self.assertTrue(plan["ready_for_collection"])
        self.assertEqual(plan["target"]["minimum_future_slots"], 4)
        self.assertIn("collect_cross_language_sources", actions)
        self.assertIn("collect_non_dominant_repository_sources", actions)
        self.assertIn("collect_public_issue_sources", actions)
        self.assertIn("collect_build_command_evidence", actions)
        self.assertIn("curate_hidden_test_patch_evidence", actions)
        cross_language = next(
            item
            for item in plan["requirements"]
            if item["action"] == "collect_cross_language_sources"
        )
        self.assertEqual(cross_language["candidate_allowlist_repositories"], ["repo/ts"])
        hidden_patch = next(
            item
            for item in plan["requirements"]
            if item["action"] == "curate_hidden_test_patch_evidence"
        )
        self.assertIn("Do not use benchmark oracle patches", hidden_patch["leakage_constraints"][0])

    def test_seed_remediation_plan_blocks_missing_cross_language_allowlist(
        self,
    ) -> None:
        plan = build_seed_remediation_plan(
            _remediation_selection_plan(),
            _remediation_policy(),
            allowlist_records=[_allowlist_record(repository="repo/b", language="Python")],
        )

        self.assertFalse(plan["valid"])
        self.assertFalse(plan["ready_for_collection"])
        self.assertEqual(
            [issue["code"] for issue in plan["issues"]],
            ["missing_cross_language_allowlist_candidate"],
        )

    def test_cli_seed_remediation_plan_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "selection-plan.json"
            policy_path = root / "policy.json"
            allowlist_path = root / "allowlist.json"
            output_path = root / "remediation-plan.json"
            selection_path.write_text(
                json.dumps(_remediation_selection_plan()),
                encoding="utf-8",
            )
            policy_path.write_text(json.dumps(_remediation_policy()), encoding="utf-8")
            allowlist_path.write_text(
                json.dumps(
                    {
                        "repositories": [
                            _allowlist_record(repository="repo/b", language="Python"),
                            _allowlist_record(
                                repository="repo/ts",
                                language="TypeScript",
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-remediation-plan",
                        "--selection-plan",
                        str(selection_path),
                        "--policy",
                        str(policy_path),
                        "--allowlist",
                        str(allowlist_path),
                        "--output",
                        str(output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["ready_for_collection"])

    def test_hidden_test_patch_curation_plan_rejects_leaky_records(self) -> None:
        remediation = build_seed_remediation_plan(
            _remediation_selection_plan(),
            _remediation_policy(),
            allowlist_records=[
                _allowlist_record(repository="repo/b", language="Python"),
                _allowlist_record(repository="repo/ts", language="TypeScript"),
            ],
        )
        records = [
            _curation_source_record("issue-1", "issue"),
            _curation_source_record("pr-1", "pull_request"),
            _curation_source_record("ci-1", "ci_failure"),
            {
                **_curation_source_record("bench-1", "issue"),
                "source_name": "swe_bench_lite",
            },
            {
                **_curation_source_record("oracle-1", "issue"),
                "test_patch": "diff --git a/test.py b/test.py",
            },
        ]

        plan = build_hidden_test_patch_curation_plan(records, remediation)
        rejection_codes = {item["code"] for item in plan["rejected_records"]}

        self.assertTrue(plan["valid"])
        self.assertTrue(plan["ready_for_curation"])
        self.assertEqual(plan["target"]["required_hidden_test_patch_records"], 2)
        self.assertEqual(plan["counts"]["selected_curation_tasks"], 2)
        self.assertEqual(plan["counts"]["eligible_records"], 2)
        self.assertEqual(plan["counts"]["rejected_records"], 3)
        self.assertEqual(
            {task["source_type"] for task in plan["curation_tasks"]},
            {"public_issue", "public_pr"},
        )
        self.assertIn("unsupported_source_type", rejection_codes)
        self.assertIn("benchmark_source_rejected", rejection_codes)
        self.assertIn("oracle_fields_rejected", rejection_codes)
        self.assertIn(
            "hidden_test_patch",
            plan["curation_tasks"][0]["required_curation_fields"],
        )
        self.assertIn(
            "Do not use benchmark oracle patches",
            plan["curation_tasks"][0]["leakage_constraints"][0],
        )

    def test_hidden_test_patch_curation_plan_balances_repositories(self) -> None:
        remediation = build_seed_remediation_plan(
            _remediation_selection_plan(),
            _remediation_policy(),
            allowlist_records=[
                _allowlist_record(repository="repo/b", language="Python"),
                _allowlist_record(repository="repo/ts", language="TypeScript"),
            ],
        )
        records = [
            _curation_source_record("issue-1", "issue", repository="example/a"),
            _curation_source_record("issue-2", "issue", repository="example/a"),
            _curation_source_record("issue-3", "issue", repository="example/b"),
            _curation_source_record("issue-4", "issue", repository="example/b"),
        ]

        plan = build_hidden_test_patch_curation_plan(records, remediation, max_records=2)

        self.assertEqual(
            {task["repository"] for task in plan["curation_tasks"]},
            {"example/a", "example/b"},
        )

    def test_hidden_test_patch_curation_plan_requires_remediation_requirement(
        self,
    ) -> None:
        plan = build_hidden_test_patch_curation_plan(
            [_curation_source_record("issue-1", "issue")],
            {"requirements": []},
            max_records=1,
        )

        self.assertTrue(plan["valid"])
        self.assertFalse(plan["ready_for_curation"])
        self.assertEqual(plan["counts"]["selected_curation_tasks"], 0)
        self.assertEqual(
            [issue["code"] for issue in plan["issues"]],
            ["hidden_test_patch_requirement_missing"],
        )

    def test_cli_hidden_test_patch_curation_plan_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            remediation_path = root / "remediation.json"
            output_path = root / "curation-plan.json"
            remediation = build_seed_remediation_plan(
                _remediation_selection_plan(),
                _remediation_policy(),
                allowlist_records=[
                    _allowlist_record(repository="repo/b", language="Python"),
                    _allowlist_record(repository="repo/ts", language="TypeScript"),
                ],
            )
            source_path.write_text(
                json.dumps(_curation_source_record("issue-1", "issue")) + "\n",
                encoding="utf-8",
            )
            remediation_path.write_text(json.dumps(remediation), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "hidden-test-patch-curation-plan",
                        "--source-records",
                        str(source_path),
                        "--remediation-plan",
                        str(remediation_path),
                        "--max-records",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["ready_for_curation"])
            self.assertEqual(payload["counts"]["selected_curation_tasks"], 1)

    def test_hidden_test_patch_curation_template_and_empty_apply(self) -> None:
        source_record = _curation_source_record("issue-1", "issue")
        plan = build_hidden_test_patch_curation_plan(
            [source_record],
            build_seed_remediation_plan(
                _remediation_selection_plan(),
                _remediation_policy(),
                allowlist_records=[_allowlist_record(repository="example/tool")],
            ),
            max_records=1,
        )

        template = build_hidden_test_patch_curation_record_template(plan)
        payload, rewritten_records = apply_hidden_test_patch_curation_records(
            plan,
            [source_record],
            template["records"],
        )

        self.assertTrue(template["valid"])
        self.assertEqual(template["counts"]["template_records"], 1)
        self.assertEqual(template["records"][0]["hidden_test_patch"], "")
        self.assertEqual(template["records"][0]["hidden_test_commands"], [])
        self.assertFalse(payload["ready_for_import_rehearsal"])
        self.assertFalse(payload["valid"])
        self.assertEqual(payload["counts"]["invalid_curation_records"], 1)
        self.assertNotIn("test_patch", rewritten_records[0])

    def test_hidden_test_patch_curation_apply_rewrites_source_records(self) -> None:
        source_record = _curation_source_record("issue-1", "issue")
        plan = build_hidden_test_patch_curation_plan(
            [source_record],
            build_seed_remediation_plan(
                _remediation_selection_plan(),
                _remediation_policy(),
                allowlist_records=[_allowlist_record(repository="example/tool")],
            ),
            max_records=1,
        )
        curation_records = _complete_hidden_test_patch_curation_records(
            plan["curation_tasks"]
        )

        payload, rewritten_records = apply_hidden_test_patch_curation_records(
            plan,
            [source_record],
            curation_records,
        )

        self.assertTrue(payload["valid"])
        self.assertTrue(payload["ready_for_import_rehearsal"])
        self.assertEqual(payload["counts"]["applied_curation_records"], 1)
        self.assertEqual(payload["counts"]["remaining_curation_tasks"], 0)
        rewritten = rewritten_records[0]
        self.assertIn("hidden_test_patch", rewritten["verifier_types"])
        self.assertIn("diff --git", rewritten["test_patch"])
        self.assertEqual(rewritten["test_patch"], curation_records[0]["hidden_test_patch"])
        self.assertEqual(
            rewritten["test_commands"],
            ["python -m pytest tests/test_hidden_parser.py"],
        )
        self.assertEqual(
            rewritten["candidate_verifier"]["type"],
            "curated_hidden_test_patch",
        )
        self.assertEqual(
            rewritten["hidden_test_patch_curation"]["curation_task_id"],
            plan["curation_tasks"][0]["curation_task_id"],
        )

    def test_cli_hidden_test_patch_template_and_apply_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            plan_path = root / "curation-plan.json"
            records_path = root / "records.json"
            template_path = root / "template-summary.json"
            filled_records_path = root / "filled-records.json"
            output_records_path = root / "curated-source.jsonl"
            apply_path = root / "apply.json"
            source_record = _curation_source_record("issue-1", "issue")
            plan = build_hidden_test_patch_curation_plan(
                [source_record],
                build_seed_remediation_plan(
                    _remediation_selection_plan(),
                    _remediation_policy(),
                    allowlist_records=[_allowlist_record(repository="example/tool")],
                ),
                max_records=1,
            )
            source_path.write_text(json.dumps(source_record) + "\n", encoding="utf-8")
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            filled_records_path.write_text(
                json.dumps(
                    {
                        "records": _complete_hidden_test_patch_curation_records(
                            plan["curation_tasks"]
                        )
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                template_exit_code = main(
                    [
                        "registry",
                        "hidden-test-patch-curation-record-template",
                        "--curation-plan",
                        str(plan_path),
                        "--records-output",
                        str(records_path),
                        "--output",
                        str(template_path),
                    ]
                )
            with redirect_stdout(io.StringIO()):
                apply_exit_code = main(
                    [
                        "registry",
                        "hidden-test-patch-curation-apply",
                        "--curation-plan",
                        str(plan_path),
                        "--source-records",
                        str(source_path),
                        "--curation-records",
                        str(filled_records_path),
                        "--output-records",
                        str(output_records_path),
                        "--output",
                        str(apply_path),
                    ]
                )

            self.assertEqual(template_exit_code, 0)
            self.assertEqual(apply_exit_code, 0)
            self.assertTrue(records_path.exists())
            self.assertTrue(template_path.exists())
            self.assertTrue(apply_path.exists())
            rewritten_records = load_source_records(output_records_path)
            self.assertEqual(len(rewritten_records), 1)
            self.assertIn("diff --git", rewritten_records[0]["test_patch"])
            self.assertEqual(
                json.loads(apply_path.read_text(encoding="utf-8"))["counts"][
                    "applied_curation_records"
                ],
                1,
            )

    def test_hidden_command_curation_plan_prioritizes_failed_records(self) -> None:
        failed = _workspace_source_record("issue-1", "issue", repository="example/a")
        clean = _workspace_source_record("ci-1", "ci_failure", repository="example/b")
        benchmark = {
            **_workspace_source_record("bench-1", "issue", repository="example/c"),
            "source_name": "swe_bench_lite",
        }
        oracle = {
            **_workspace_source_record("oracle-1", "issue", repository="example/d"),
            "test_patch": "diff --git a/test.py b/test.py",
        }
        missing_commands = {
            **_workspace_source_record("missing-1", "issue", repository="example/e"),
            "candidate_verifier": {},
            "test_commands": [],
            "ci_commands": [],
        }
        summary = {
            "schema_version": "easy_agentic_data.registry_import_rehearsal.v1",
            "registry_root": "",
            "materialization": {
                "issues": [
                    {
                        "code": "materialization_rehearsal_failed",
                        "message": "command failed",
                        "scenario_id": "scenario-1",
                        "severity": "error",
                    }
                ],
                "results": [
                    {
                        "valid": False,
                        "scenario_id": "scenario-1",
                        "source_instance_id": failed["source_instance_id"],
                        "command_results": [
                            {
                                "command_sha256": "abc",
                                "exit_code": 1,
                                "stdout_sha256": "out",
                                "stderr_sha256": "err",
                            }
                        ],
                    }
                ],
            },
        }

        plan = build_hidden_command_curation_plan(
            [clean, benchmark, failed, oracle, missing_commands],
            rehearsal_summaries=[summary],
            max_records=2,
        )
        rejection_codes = {item["code"] for item in plan["rejected_records"]}
        first_task = plan["curation_tasks"][0]

        self.assertTrue(plan["valid"])
        self.assertTrue(plan["ready_for_curation"])
        self.assertEqual(plan["counts"]["eligible_records"], 2)
        self.assertEqual(plan["counts"]["records_with_observed_failures"], 1)
        self.assertEqual(plan["counts"]["selected_curation_tasks"], 2)
        self.assertEqual(first_task["source_instance_id"], failed["source_instance_id"])
        self.assertEqual(first_task["observed_failure"]["command_results"][0]["exit_code"], 1)
        self.assertIn("curated_hidden_commands", first_task["required_curation_fields"])
        self.assertIn("benchmark_source_rejected", rejection_codes)
        self.assertIn("oracle_fields_rejected", rejection_codes)
        self.assertIn("missing_candidate_verifier_commands", rejection_codes)

    def test_cli_hidden_command_curation_plan_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            summary_path = root / "summary.json"
            output_path = root / "hidden-command-curation-plan.json"
            record = _workspace_source_record("issue-1", "issue")
            source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "materialization": {
                            "results": [
                                {
                                    "valid": False,
                                    "source_instance_id": record["source_instance_id"],
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "hidden-command-curation-plan",
                        "--source-records",
                        str(source_path),
                        "--rehearsal-summary",
                        str(summary_path),
                        "--output",
                        str(output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["ready_for_curation"])
            self.assertEqual(payload["counts"]["selected_curation_tasks"], 1)

    def test_hidden_command_curation_record_template_maps_tasks(self) -> None:
        records = [
            _workspace_source_record("issue-1", "issue"),
            _workspace_source_record("ci-1", "ci_failure"),
        ]
        plan = build_hidden_command_curation_plan(records, max_records=2)

        payload = build_hidden_command_curation_record_template(plan)

        self.assertTrue(payload["valid"])
        self.assertEqual(payload["counts"]["template_records"], 2)
        first_record = payload["records"][0]
        self.assertIn("curation_task_id", first_record)
        self.assertIn("source_instance_id", first_record)
        self.assertEqual(first_record["curated_hidden_commands"], [])
        self.assertIn("current_candidate_verifier_commands", first_record)

    def test_apply_hidden_command_curation_records_rewrites_source_view(self) -> None:
        issue = _workspace_source_record("issue-1", "issue")
        ci = _workspace_source_record("ci-1", "ci_failure")
        plan = build_hidden_command_curation_plan([issue, ci])
        curation_records = _complete_hidden_command_curation_records(
            plan["curation_tasks"],
            {
                issue["source_instance_id"]: ["python -m pytest tests/test_parser.py -q"],
                ci["source_instance_id"]: ["python -m pytest tests/test_ci.py -q"],
            },
        )

        payload, rewritten = apply_hidden_command_curation_records(
            plan,
            [issue, ci],
            curation_records,
        )

        rewritten_by_id = {record["source_instance_id"]: record for record in rewritten}
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["ready_for_import_rehearsal"])
        self.assertEqual(payload["counts"]["applied_curation_records"], 2)
        self.assertEqual(payload["counts"]["remaining_curation_tasks"], 0)
        self.assertEqual(
            rewritten_by_id[issue["source_instance_id"]]["test_commands"],
            ["python -m pytest tests/test_parser.py -q"],
        )
        self.assertEqual(
            rewritten_by_id[ci["source_instance_id"]]["ci_commands"],
            ["python -m pytest tests/test_ci.py -q"],
        )
        self.assertEqual(
            rewritten_by_id[issue["source_instance_id"]]["setup_commands"],
            ["python -m pip install -e ."],
        )
        self.assertEqual(
            rewritten_by_id[ci["source_instance_id"]]["candidate_verifier"]["type"],
            "curated_hidden_commands",
        )
        self.assertIn(
            "hidden_command_curation",
            rewritten_by_id[issue["source_instance_id"]],
        )

    def test_apply_hidden_command_curation_rejects_known_failed_command(self) -> None:
        record = _workspace_source_record("issue-1", "issue")
        failed_command = record["candidate_verifier"]["commands"][0]
        summary = {
            "materialization": {
                "results": [
                    {
                        "valid": False,
                        "source_instance_id": record["source_instance_id"],
                        "command_results": [
                            {
                                "command_sha256": hashlib.sha256(
                                    failed_command.encode("utf-8")
                                ).hexdigest(),
                                "exit_code": 1,
                            }
                        ],
                    }
                ]
            }
        }
        plan = build_hidden_command_curation_plan(
            [record],
            rehearsal_summaries=[summary],
        )
        curation_records = _complete_hidden_command_curation_records(
            plan["curation_tasks"],
            {record["source_instance_id"]: [failed_command]},
        )

        payload, _rewritten = apply_hidden_command_curation_records(
            plan,
            [record],
            curation_records,
        )

        self.assertFalse(payload["valid"])
        self.assertFalse(payload["ready_for_import_rehearsal"])
        self.assertEqual(payload["counts"]["invalid_curation_records"], 1)
        self.assertIn(
            "repeat commands",
            payload["invalid_curation_records"][0]["errors"][-1],
        )

    def test_cli_hidden_command_curation_template_and_apply_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            plan_path = root / "curation-plan.json"
            template_summary_path = root / "template-summary.json"
            template_records_path = root / "template-records.json"
            completed_records_path = root / "completed-records.json"
            apply_summary_path = root / "apply-summary.json"
            output_records_path = root / "source-curated.jsonl"
            record = _workspace_source_record("issue-1", "issue")
            source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            plan = build_hidden_command_curation_plan([record])
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            completed_records_path.write_text(
                json.dumps(
                    {
                        "records": _complete_hidden_command_curation_records(
                            plan["curation_tasks"],
                            {
                                record["source_instance_id"]: [
                                    "python -m pytest tests/test_parser.py -q"
                                ]
                            },
                        )
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                template_exit = main(
                    [
                        "registry",
                        "hidden-command-curation-record-template",
                        "--curation-plan",
                        str(plan_path),
                        "--records-output",
                        str(template_records_path),
                        "--output",
                        str(template_summary_path),
                    ]
                )
            with redirect_stdout(io.StringIO()):
                apply_exit = main(
                    [
                        "registry",
                        "hidden-command-curation-apply",
                        "--curation-plan",
                        str(plan_path),
                        "--source-records",
                        str(source_path),
                        "--curation-records",
                        str(completed_records_path),
                        "--output-records",
                        str(output_records_path),
                        "--output",
                        str(apply_summary_path),
                    ]
                )

            template_payload = json.loads(
                template_summary_path.read_text(encoding="utf-8")
            )
            apply_payload = json.loads(apply_summary_path.read_text(encoding="utf-8"))
            curated_record = json.loads(
                output_records_path.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(template_exit, 0)
            self.assertEqual(apply_exit, 0)
            self.assertTrue(template_payload["valid"])
            self.assertTrue(apply_payload["ready_for_import_rehearsal"])
            self.assertTrue(template_records_path.is_file())
            self.assertEqual(
                curated_record["test_commands"],
                ["python -m pytest tests/test_parser.py -q"],
            )

    def test_source_workspace_materialization_plan_groups_fixed_revisions(self) -> None:
        records = [
            _workspace_source_record("issue-1", "issue", repository="example/a"),
            _workspace_source_record("issue-2", "issue", repository="example/a"),
            _workspace_source_record("pr-1", "pull_request", repository="example/b"),
            {
                **_workspace_source_record("bench-1", "issue", repository="example/c"),
                "source_name": "swe_bench_lite",
            },
            {
                **_workspace_source_record("oracle-1", "issue", repository="example/d"),
                "patch": "diff --git a/app.py b/app.py",
            },
            {
                **_workspace_source_record("bad-revision", "issue", repository="example/e"),
                "source_revision": "main",
            },
        ]

        plan = build_source_workspace_materialization_plan(
            records,
            workspace_root="runs/source-workspaces",
            max_records=3,
            shard_size=2,
        )
        rejection_codes = {item["code"] for item in plan["rejected_records"]}
        first_task = plan["materialization_tasks"][0]

        self.assertTrue(plan["valid"])
        self.assertTrue(plan["ready_for_materialization"])
        self.assertEqual(plan["counts"]["eligible_records"], 3)
        self.assertEqual(plan["counts"]["selected_records"], 3)
        self.assertEqual(plan["counts"]["materialization_tasks"], 2)
        self.assertEqual(plan["counts"]["rejected_records"], 3)
        self.assertEqual(plan["shard_count"], 1)
        self.assertIn("benchmark_source_rejected", rejection_codes)
        self.assertIn("oracle_fields_rejected", rejection_codes)
        self.assertIn("invalid_source_revision", rejection_codes)
        self.assertEqual(first_task["record_count"], 2)
        self.assertEqual(
            first_task["materialization_args"]["checkout"][-1],
            "f" * 40,
        )
        self.assertTrue(first_task["planned_file_source_uri"].startswith("file://"))
        self.assertEqual(
            plan["selected_source_type_counts"],
            {"public_issue": 2, "public_pr": 1},
        )

    def test_source_workspace_materialization_plan_avoids_uri_cache_collisions(self) -> None:
        same_revision = "f" * 40
        records = [
            {
                **_workspace_source_record("issue-1", "issue", repository="example/a"),
                "source_revision": same_revision,
                "source_uri": "https://github.com/example/a.git",
            },
            {
                **_workspace_source_record("issue-2", "issue", repository="example/a"),
                "source_revision": same_revision,
                "source_uri": "file:///tmp/example-a.git",
            },
        ]

        plan = build_source_workspace_materialization_plan(
            records,
            workspace_root="runs/source-workspaces",
        )
        cache_paths = {
            task["cache_path"] for task in plan["materialization_tasks"]
        }
        planned_file_source_uris = {
            task["planned_file_source_uri"]
            for task in plan["materialization_tasks"]
        }

        self.assertTrue(plan["valid"])
        self.assertEqual(plan["counts"]["materialization_tasks"], 2)
        self.assertEqual(len(cache_paths), 2)
        self.assertEqual(len(planned_file_source_uris), 2)

    def test_cli_source_workspace_materialization_plan_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.jsonl"
            output_path = root / "workspace-plan.json"
            source_path.write_text(
                json.dumps(_workspace_source_record("issue-1", "issue")) + "\n",
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "source-workspace-materialization-plan",
                        "--source-records",
                        str(source_path),
                        "--workspace-root",
                        str(root / "workspaces"),
                        "--max-records",
                        "1",
                        "--output",
                        str(output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["ready_for_materialization"])
            self.assertEqual(payload["counts"]["materialization_tasks"], 1)

    def test_materialize_source_workspaces_rewrites_local_file_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            records = [
                {
                    **_workspace_source_record("issue-1", "issue"),
                    "source_uri": repository.as_uri(),
                    "source_revision": commit,
                }
            ]
            plan = build_source_workspace_materialization_plan(
                records,
                workspace_root=root / "workspaces",
            )

            payload, rewritten_records = materialize_source_workspaces(
                plan,
                records,
                timeout_seconds=60,
            )

            rewritten = rewritten_records[0]
            task = payload["tasks"][0]
            self.assertTrue(payload["valid"])
            self.assertTrue(payload["ready_for_import_rehearsal"])
            self.assertEqual(payload["counts"]["rewritten_records"], 1)
            self.assertTrue(rewritten["source_uri"].startswith("file://"))
            self.assertTrue(rewritten["workspace_materialized"])
            self.assertEqual(
                rewritten["workspace_materialization_task_id"],
                task["materialization_task_id"],
            )
            self.assertTrue((Path(task["cache_path"]) / ".git").is_dir())
            self.assertEqual(
                [command["label"] for command in task["commands"]],
                ["clone", "fetch_revision", "checkout", "verify_revision"],
            )
            self.assertNotIn("stdout", task["commands"][0])

    def test_materialize_source_workspaces_rejects_escaped_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            records = [
                {
                    **_workspace_source_record("issue-1", "issue"),
                    "source_uri": repository.as_uri(),
                    "source_revision": commit,
                }
            ]
            plan = build_source_workspace_materialization_plan(
                records,
                workspace_root=root / "workspaces",
            )
            plan["materialization_tasks"][0]["cache_path"] = str(root / "escaped")

            payload, rewritten_records = materialize_source_workspaces(plan, records)

            self.assertFalse(payload["valid"])
            self.assertFalse(payload["ready_for_import_rehearsal"])
            self.assertEqual(rewritten_records, [])
            self.assertIn(
                "outside workspace_root",
                payload["tasks"][0]["error"],
            )

    def test_materialize_source_workspaces_records_failed_git_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            records = [
                {
                    **_workspace_source_record("issue-1", "issue"),
                    "source_uri": repository.as_uri(),
                    "source_revision": commit,
                }
            ]
            plan = build_source_workspace_materialization_plan(
                records,
                workspace_root=root / "workspaces",
            )
            plan["materialization_tasks"][0]["source_uri"] = (
                root / "missing-upstream"
            ).as_uri()

            payload, rewritten_records = materialize_source_workspaces(
                plan,
                records,
                timeout_seconds=5,
            )
            command = payload["tasks"][0]["commands"][0]

            self.assertFalse(payload["valid"])
            self.assertEqual(rewritten_records, [])
            self.assertEqual(command["label"], "clone")
            self.assertNotEqual(command["exit_code"], 0)
            self.assertIn("stderr_sha256", command)
            self.assertNotIn("stderr", command)

    def test_cli_materialize_source_workspaces_writes_rewritten_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            source_path = root / "source.jsonl"
            plan_path = root / "workspace-plan.json"
            records_output = root / "materialized-source.jsonl"
            summary_output = root / "materialization-summary.json"
            source_path.write_text(
                json.dumps(
                    {
                        **_workspace_source_record("issue-1", "issue"),
                        "source_uri": repository.as_uri(),
                        "source_revision": commit,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            plan = build_source_workspace_materialization_plan(
                load_source_records(source_path),
                workspace_root=root / "workspaces",
            )
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "materialize-source-workspaces",
                        "--plan",
                        str(plan_path),
                        "--source-records",
                        str(source_path),
                        "--output-records",
                        str(records_output),
                        "--output",
                        str(summary_output),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(summary_output.read_text(encoding="utf-8"))
            rewritten = load_source_records(records_output)[0]
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["outputs"]["records_output_written"])
            self.assertTrue(rewritten["source_uri"].startswith("file://"))
            self.assertTrue(rewritten["workspace_materialized"])

    def test_synthetic_backfill_spec_plan_splits_ready_and_draft_specs(self) -> None:
        selection_plan = _synthetic_backfill_selection_plan()
        backfill_plan = _synthetic_backfill_plan()

        plan = build_synthetic_backfill_spec_plan(
            [_synthetic_source_scenario("repo/a"), _synthetic_source_scenario("repo/b")],
            selection_plan,
            backfill_plan,
            max_repositories=2,
        )
        report = plan["evidence_report"]

        self.assertTrue(plan["valid"])
        self.assertFalse(plan["ready_to_generate"])
        self.assertEqual(plan["planned_ready_records"], 3)
        self.assertEqual(plan["planned_draft_records"], 3)
        self.assertEqual(report["code_review"]["generator_ready"], 1)
        self.assertEqual(report["test_authoring"]["generator_ready"], 2)
        self.assertEqual(report["docs_examples"]["draft"], 2)
        self.assertIn("doctest_or_example_command", report["docs_examples"]["missing_evidence"])
        self.assertEqual(report["performance"]["draft"], 1)
        ready_families = {
            spec["task_families"][0]
            for spec in plan["generator_ready_specs"]["repositories"]
        }
        self.assertEqual(ready_families, {"code_review", "test_authoring"})

    def test_synthetic_backfill_spec_plan_can_use_backfill_plan_without_selection_slots(
        self,
    ) -> None:
        plan = build_synthetic_backfill_spec_plan(
            [_synthetic_source_scenario("repo/a")],
            {"reserved_backfill": {"slots": []}},
            {
                "gaps": {
                    "task_family": [
                        {"target": "test_authoring", "shortfall": 2},
                    ]
                }
            },
        )

        self.assertTrue(plan["valid"])
        self.assertEqual(plan["planned_ready_records"], 2)
        self.assertEqual(plan["family_slots"][0]["source"], "backfill_plan_task_family_gap")

    def test_cli_synthetic_backfill_spec_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry_root = root / "registry"
            output = root / "synthetic-backfill.json"
            spec_output = root / "ready-spec.json"
            selection_path = root / "selection.json"
            backfill_path = root / "backfill.json"
            registry = ScenarioRegistry(registry_root)
            registry.add_scenario(_synthetic_source_scenario("repo/a"))
            selection_path.write_text(
                json.dumps(_synthetic_backfill_selection_plan()),
                encoding="utf-8",
            )
            backfill_path.write_text(json.dumps(_synthetic_backfill_plan()), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-synthetic-backfill-spec",
                        "--root",
                        str(registry_root),
                        "--selection-plan",
                        str(selection_path),
                        "--backfill-plan",
                        str(backfill_path),
                        "--output",
                        str(output),
                        "--spec-output",
                        str(spec_output),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output.read_text(encoding="utf-8"))
            ready_spec = json.loads(spec_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertEqual(ready_spec, payload["generator_ready_specs"])
            self.assertGreater(payload["planned_ready_records"], 0)

    def test_synthetic_evidence_backfill_plan_lists_draft_targets(self) -> None:
        synthetic_plan = build_synthetic_backfill_spec_plan(
            [_synthetic_source_scenario("repo/a"), _synthetic_source_scenario("repo/b")],
            _synthetic_backfill_selection_plan(),
            _synthetic_backfill_plan(),
            max_repositories=2,
        )
        evidence_plan = build_synthetic_evidence_backfill_plan(
            synthetic_plan,
            build_seed_backfill_plan(_backfill_audit(), _backfill_policy()),
        )
        task_counts = evidence_plan["counts"]["task_family"]
        missing_counts = evidence_plan["counts"]["missing_evidence"]
        actions = {
            (action["action"], action["target"])
            for action in evidence_plan["recommended_next_actions"]
        }

        self.assertTrue(evidence_plan["valid"])
        self.assertFalse(evidence_plan["ready_for_generation"])
        self.assertEqual(evidence_plan["counts"]["evidence_tasks"], 3)
        self.assertEqual(task_counts["docs_examples"], 2)
        self.assertEqual(task_counts["performance"], 1)
        self.assertEqual(missing_counts["doctest_or_example_command"], 2)
        self.assertEqual(missing_counts["benchmark_command_or_performance_threshold"], 1)
        self.assertIn(("add_docs_example_evidence", "docs_examples"), actions)
        self.assertIn(("add_performance_evidence", "performance"), actions)
        self.assertEqual(
            evidence_plan["backfill_gap_summary"]["verifier_type_shortfalls"]["doctest"],
            5,
        )

    def test_cli_synthetic_evidence_plan_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_path = root / "synthetic-backfill.json"
            backfill_path = root / "backfill.json"
            output_path = root / "synthetic-evidence-plan.json"
            synthetic_plan = build_synthetic_backfill_spec_plan(
                [_synthetic_source_scenario("repo/a")],
                _synthetic_backfill_selection_plan(),
                _synthetic_backfill_plan(),
            )
            synthetic_path.write_text(json.dumps(synthetic_plan), encoding="utf-8")
            backfill_path.write_text(
                json.dumps(build_seed_backfill_plan(_backfill_audit(), _backfill_policy())),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-synthetic-evidence-plan",
                        "--synthetic-backfill-plan",
                        str(synthetic_path),
                        "--backfill-plan",
                        str(backfill_path),
                        "--output",
                        str(output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertGreater(payload["counts"]["evidence_tasks"], 0)
            self.assertFalse(payload["ready_for_generation"])

    def test_synthetic_evidence_shard_schedule_partitions_tasks(self) -> None:
        synthetic_plan = build_synthetic_backfill_spec_plan(
            [_synthetic_source_scenario("repo/a"), _synthetic_source_scenario("repo/b")],
            _synthetic_backfill_selection_plan(),
            _synthetic_backfill_plan(),
            max_repositories=2,
        )
        evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)

        schedule = build_synthetic_evidence_shard_schedule(
            evidence_plan,
            synthetic_backfill_plan_path="runs/synthetic-backfill.json",
            output_dir="runs/evidence-shards",
            shard_size=2,
        )

        self.assertTrue(schedule["valid"])
        self.assertEqual(schedule["evidence_tasks"], 3)
        self.assertEqual(schedule["shard_count"], 2)
        self.assertEqual(schedule["shards"][0]["selected_tasks"], 2)
        self.assertEqual(schedule["shards"][1]["selected_tasks"], 1)
        self.assertIn(
            "seed-synthetic-evidence-apply",
            schedule["shards"][0]["apply_args"],
        )
        self.assertEqual(
            schedule["shards"][0]["missing_evidence_counts"]["doctest_or_example_command"],
            2,
        )
        self.assertEqual(
            schedule["shards"][1]["missing_evidence_counts"][
                "benchmark_command_or_performance_threshold"
            ],
            1,
        )

    def test_cli_synthetic_evidence_shards_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_path = root / "synthetic-backfill.json"
            evidence_path = root / "synthetic-evidence-plan.json"
            schedule_path = root / "synthetic-evidence-shards.json"
            shard_dir = root / "evidence-shards"
            synthetic_plan = build_synthetic_backfill_spec_plan(
                [_synthetic_source_scenario("repo/a")],
                _synthetic_backfill_selection_plan(),
                _synthetic_backfill_plan(),
            )
            evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)
            synthetic_path.write_text(json.dumps(synthetic_plan), encoding="utf-8")
            evidence_path.write_text(json.dumps(evidence_plan), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-synthetic-evidence-shards",
                        "--evidence-plan",
                        str(evidence_path),
                        "--synthetic-backfill-plan",
                        str(synthetic_path),
                        "--output-dir",
                        str(shard_dir),
                        "--shard-size",
                        "2",
                        "--output",
                        str(schedule_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["valid"])
            self.assertGreater(payload["shard_count"], 0)

    def test_synthetic_evidence_record_templates_map_shard_tasks(self) -> None:
        synthetic_plan = build_synthetic_backfill_spec_plan(
            [_synthetic_source_scenario("repo/a"), _synthetic_source_scenario("repo/b")],
            _synthetic_backfill_selection_plan(),
            _synthetic_backfill_plan(),
            max_repositories=2,
        )
        evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)
        schedule = build_synthetic_evidence_shard_schedule(
            evidence_plan,
            synthetic_backfill_plan_path="runs/synthetic-backfill.json",
            output_dir="runs/evidence-shards",
            shard_size=2,
        )

        payload = build_synthetic_evidence_record_templates(evidence_plan, schedule)

        self.assertTrue(payload["valid"])
        self.assertEqual(payload["counts"]["template_shards"], 2)
        self.assertEqual(payload["counts"]["template_records"], 3)
        first_template = payload["shard_templates"][0]
        first_record = first_template["template_payload"]["records"][0]
        self.assertEqual(
            first_template["record_template_output"],
            schedule["shards"][0]["record_template_output"],
        )
        self.assertEqual(first_record["task_family"], "docs_examples")
        self.assertEqual(first_record["doctest_commands"], [])
        self.assertEqual(first_record["example_commands"], [])
        self.assertIn("source_revision", first_record)
        self.assertIn("required_fields", first_record)

    def test_cli_synthetic_evidence_record_templates_writes_shard_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_path = root / "synthetic-backfill.json"
            evidence_path = root / "synthetic-evidence-plan.json"
            schedule_path = root / "synthetic-evidence-shards.json"
            summary_path = root / "synthetic-evidence-record-templates.json"
            synthetic_plan = build_synthetic_backfill_spec_plan(
                [_synthetic_source_scenario("repo/a")],
                _synthetic_backfill_selection_plan(),
                _synthetic_backfill_plan(),
            )
            evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)
            schedule = build_synthetic_evidence_shard_schedule(
                evidence_plan,
                synthetic_backfill_plan_path=synthetic_path,
                output_dir=root / "evidence-shards",
                shard_size=20,
            )
            synthetic_path.write_text(json.dumps(synthetic_plan), encoding="utf-8")
            evidence_path.write_text(json.dumps(evidence_plan), encoding="utf-8")
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-synthetic-evidence-record-templates",
                        "--evidence-plan",
                        str(evidence_path),
                        "--schedule",
                        str(schedule_path),
                        "--output",
                        str(summary_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            template_path = Path(payload["shard_templates"][0]["record_template_output"])
            template_payload = json.loads(template_path.read_text(encoding="utf-8"))
            status = summarize_synthetic_evidence_shard_status(schedule)
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(template_path.exists())
            self.assertEqual(template_payload["schema_version"], payload["schema_version"])
            self.assertEqual(status["shards"][0]["template_status"], "present")
            self.assertEqual(status["shards"][0]["next_action"], "fill_evidence_template")

    def test_synthetic_evidence_shard_status_reports_pending_and_apply_ready(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_plan = build_synthetic_backfill_spec_plan(
                [_synthetic_source_scenario("repo/a"), _synthetic_source_scenario("repo/b")],
                _synthetic_backfill_selection_plan(),
                _synthetic_backfill_plan(),
                max_repositories=2,
            )
            evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)
            schedule = build_synthetic_evidence_shard_schedule(
                evidence_plan,
                synthetic_backfill_plan_path=root / "synthetic-backfill.json",
                output_dir=root / "evidence-shards",
                shard_size=2,
            )
            task_by_id = {
                str(task["evidence_task_id"]): task
                for task in evidence_plan["evidence_tasks"]
            }
            first_shard_tasks = [
                task_by_id[task_id]
                for task_id in schedule["shards"][0]["evidence_task_ids"]
            ]
            records_output = Path(schedule["shards"][0]["records_output"])
            records_output.parent.mkdir(parents=True, exist_ok=True)
            records_output.write_text(
                json.dumps({"records": _complete_evidence_records(first_shard_tasks)}),
                encoding="utf-8",
            )

            payload = summarize_synthetic_evidence_shard_status(schedule)

            self.assertTrue(payload["valid"])
            self.assertEqual(payload["counts"]["ready_to_apply_shards"], 1)
            self.assertEqual(payload["counts"]["pending_shards"], 1)
            self.assertEqual(payload["shards"][0]["status"], "ready_to_apply")
            self.assertEqual(payload["shards"][0]["next_action"], "apply_evidence")
            self.assertEqual(payload["shards"][0]["evidence_records"], 2)
            self.assertEqual(payload["shards"][1]["status"], "pending")
            self.assertEqual(payload["shards"][1]["next_action"], "collect_evidence")

    def test_synthetic_evidence_shard_status_treats_sharded_apply_as_complete(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_plan = build_synthetic_backfill_spec_plan(
                [_synthetic_source_scenario("repo/a"), _synthetic_source_scenario("repo/b")],
                _synthetic_backfill_selection_plan(),
                _synthetic_backfill_plan(),
                max_repositories=2,
            )
            evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)
            schedule = build_synthetic_evidence_shard_schedule(
                evidence_plan,
                synthetic_backfill_plan_path=root / "synthetic-backfill.json",
                output_dir=root / "evidence-shards",
                shard_size=2,
            )
            task_by_id = {
                str(task["evidence_task_id"]): task
                for task in evidence_plan["evidence_tasks"]
            }
            first_shard_tasks = [
                task_by_id[task_id]
                for task_id in schedule["shards"][0]["evidence_task_ids"]
            ]
            records = _complete_evidence_records(first_shard_tasks)
            apply_payload = apply_synthetic_evidence_records(synthetic_plan, records)
            shard = schedule["shards"][0]
            Path(shard["records_output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(shard["records_output"]).write_text(
                json.dumps({"records": records}),
                encoding="utf-8",
            )
            Path(shard["apply_output"]).write_text(
                json.dumps(apply_payload),
                encoding="utf-8",
            )
            Path(shard["spec_output"]).write_text(
                json.dumps(apply_payload["generator_ready_specs"]),
                encoding="utf-8",
            )

            payload = summarize_synthetic_evidence_shard_status(schedule)

            self.assertFalse(apply_payload["ready_for_generation"])
            self.assertEqual(apply_payload["counts"]["remaining_evidence_tasks"], 1)
            self.assertEqual(payload["counts"]["completed_shards"], 1)
            self.assertEqual(payload["counts"]["pending_shards"], 1)
            self.assertEqual(payload["shards"][0]["status"], "complete")
            self.assertEqual(payload["shards"][0]["next_action"], "none")
            self.assertEqual(payload["shards"][0]["remaining_evidence_tasks"], 0)

    def test_cli_synthetic_evidence_shard_status_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_path = root / "synthetic-backfill.json"
            schedule_path = root / "synthetic-evidence-shards.json"
            status_path = root / "synthetic-evidence-shard-status.json"
            synthetic_plan = build_synthetic_backfill_spec_plan(
                [_synthetic_source_scenario("repo/a")],
                _synthetic_backfill_selection_plan(),
                _synthetic_backfill_plan(),
            )
            evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)
            schedule = build_synthetic_evidence_shard_schedule(
                evidence_plan,
                synthetic_backfill_plan_path=synthetic_path,
                output_dir=root / "evidence-shards",
                shard_size=20,
            )
            records = _complete_evidence_records(evidence_plan["evidence_tasks"])
            apply_payload = apply_synthetic_evidence_records(synthetic_plan, records)
            shard = schedule["shards"][0]
            Path(shard["records_output"]).parent.mkdir(parents=True, exist_ok=True)
            Path(shard["records_output"]).write_text(
                json.dumps({"records": records}),
                encoding="utf-8",
            )
            Path(shard["apply_output"]).write_text(
                json.dumps(apply_payload),
                encoding="utf-8",
            )
            Path(shard["spec_output"]).write_text(
                json.dumps(apply_payload["generator_ready_specs"]),
                encoding="utf-8",
            )
            schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-synthetic-evidence-shard-status",
                        "--schedule",
                        str(schedule_path),
                        "--output",
                        str(status_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertEqual(payload["counts"]["completed_shards"], 1)
            self.assertEqual(payload["shards"][0]["status"], "complete")
            self.assertEqual(payload["shards"][0]["next_action"], "none")

    def test_apply_synthetic_evidence_records_emits_generator_ready_specs(self) -> None:
        synthetic_plan = build_synthetic_backfill_spec_plan(
            [_synthetic_source_scenario("repo/a"), _synthetic_source_scenario("repo/b")],
            _synthetic_backfill_selection_plan(),
            _synthetic_backfill_plan(),
            max_repositories=2,
        )
        evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)
        records = _complete_evidence_records(evidence_plan["evidence_tasks"])

        payload = apply_synthetic_evidence_records(synthetic_plan, records)

        ready_specs = payload["generator_ready_specs"]["repositories"]
        self.assertTrue(payload["valid"])
        self.assertTrue(payload["ready_for_generation"])
        self.assertEqual(payload["counts"]["applied_evidence_records"], 3)
        self.assertEqual(payload["counts"]["ready_records"], 3)
        self.assertEqual(payload["counts"]["remaining_evidence_tasks"], 0)
        self.assertEqual({spec["generator_ready"] for spec in ready_specs}, {True})
        ready_targets = [target for spec in ready_specs for target in spec["targets"]]
        self.assertTrue(
            any("doctest_commands" in target for target in ready_targets),
        )
        self.assertTrue(
            any("benchmark_commands" in target for target in ready_targets),
        )

    def test_cli_synthetic_evidence_apply_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_path = root / "synthetic-backfill.json"
            records_path = root / "evidence-records.json"
            output_path = root / "synthetic-evidence-apply.json"
            spec_output = root / "generator-ready-from-evidence.json"
            synthetic_plan = build_synthetic_backfill_spec_plan(
                [_synthetic_source_scenario("repo/a")],
                _synthetic_backfill_selection_plan(),
                _synthetic_backfill_plan(),
            )
            evidence_plan = build_synthetic_evidence_backfill_plan(synthetic_plan)
            synthetic_path.write_text(json.dumps(synthetic_plan), encoding="utf-8")
            records = {"records": _complete_evidence_records(evidence_plan["evidence_tasks"])}
            records_path.write_text(
                json.dumps(records),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-synthetic-evidence-apply",
                        "--synthetic-backfill-plan",
                        str(synthetic_path),
                        "--evidence-records",
                        str(records_path),
                        "--output",
                        str(output_path),
                        "--spec-output",
                        str(spec_output),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            ready_spec = json.loads(spec_output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertEqual(ready_spec, payload["generator_ready_specs"])
            self.assertTrue(payload["ready_for_generation"])

    def test_combine_synthetic_generator_ready_specs_merges_counts(self) -> None:
        spec_a = _ready_synthetic_spec(
            repository="example/docs",
            target_name="docs",
            task_families=["docs_examples"],
        )
        spec_b = _ready_synthetic_spec(
            repository="example/perf",
            target_name="limiter",
            task_families=["performance"],
        )

        payload = combine_synthetic_generator_ready_specs(
            [{"repositories": [spec_a]}, {"repositories": [spec_b]}]
        )

        self.assertTrue(payload["valid"])
        self.assertTrue(payload["ready_for_generation"])
        self.assertEqual(payload["counts"]["input_specs"], 2)
        self.assertEqual(payload["counts"]["input_repositories"], 2)
        self.assertEqual(payload["counts"]["combined_repositories"], 2)
        self.assertEqual(payload["counts"]["planned_ready_records"], 2)
        self.assertEqual(
            payload["counts"]["task_family"],
            {"docs_examples": 1, "performance": 1},
        )
        self.assertEqual(
            payload["combined_generator_ready_specs"],
            {"repositories": [spec_a, spec_b]},
        )

    def test_combine_synthetic_generator_ready_specs_rejects_duplicates(self) -> None:
        spec = _ready_synthetic_spec(
            repository="example/docs",
            target_name="docs",
            task_families=["docs_examples"],
        )

        payload = combine_synthetic_generator_ready_specs(
            [{"repositories": [spec]}, {"repositories": [spec]}]
        )

        self.assertFalse(payload["valid"])
        self.assertFalse(payload["ready_for_generation"])
        self.assertEqual(
            [issue["code"] for issue in payload["issues"]],
            ["duplicate_generator_ready_target"],
        )

    def test_cli_synthetic_ready_spec_combine_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_a_path = root / "ready-a.json"
            spec_b_path = root / "ready-b.json"
            output_path = root / "combined-summary.json"
            spec_output_path = root / "combined-spec.json"
            spec_a = _ready_synthetic_spec(
                repository="example/docs",
                target_name="docs",
                task_families=["docs_examples"],
            )
            spec_b = _ready_synthetic_spec(
                repository="example/perf",
                target_name="limiter",
                task_families=["performance"],
            )
            spec_a_path.write_text(
                json.dumps({"repositories": [spec_a]}),
                encoding="utf-8",
            )
            spec_b_path.write_text(
                json.dumps({"repositories": [spec_b]}),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-synthetic-ready-spec-combine",
                        "--spec",
                        str(spec_a_path),
                        "--spec",
                        str(spec_b_path),
                        "--output",
                        str(output_path),
                        "--spec-output",
                        str(spec_output_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output_path.read_text(encoding="utf-8"))
            ready_spec = json.loads(spec_output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["valid"])
            self.assertEqual(
                ready_spec,
                payload["combined_generator_ready_specs"],
            )

    def test_assemble_seed_candidate_registry_copies_selection_and_generates_backfill(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            output_root = root / "assembled"
            selected_scenario = _synthetic_source_scenario("repo/a")
            skipped_scenario = _synthetic_source_scenario("repo/b")
            registry = ScenarioRegistry(source_root)
            registry.add_scenario(selected_scenario)
            registry.add_scenario(skipped_scenario)

            payload = assemble_seed_candidate_registry(
                source_root=source_root,
                output_root=output_root,
                selection_plan={
                    "selected_seed_ids": [selected_scenario.query_seed.seed_id],
                },
                synthetic_specs=[_assembly_synthetic_spec()],
                source_name="unit-synthetic-backfill",
                policy_config=_assembly_policy(),
            )

            assembled_registry = ScenarioRegistry(output_root)
            assembled_seeds = assembled_registry.list_seeds()
            task_families = {seed.task_family for seed in assembled_seeds}
            source_methods = {seed.source_method for seed in assembled_seeds}

            self.assertTrue(payload["valid"])
            self.assertTrue(payload["ready_for_rollout"])
            self.assertEqual(payload["selection"]["copied_existing_count"], 1)
            self.assertEqual(payload["synthetic_generation"]["generated"], 1)
            self.assertEqual(payload["output_registry"]["scenario_count"], 2)
            self.assertEqual(len(assembled_registry.list_scenarios()), 2)
            self.assertEqual(task_families, {"bug_repair", "test_authoring"})
            self.assertIn("repository_grounded_synthetic", source_methods)

    def test_assemble_seed_candidate_registry_ignores_unselected_duplicate_seeds(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            output_root = root / "assembled"
            selected_scenario = _synthetic_source_scenario("repo/a")
            duplicate_seed_scenario = _synthetic_source_scenario("repo/b")
            duplicate_environment_scenario = Scenario(
                query_seed=duplicate_seed_scenario.query_seed,
                environment=replace(duplicate_seed_scenario.environment, version="2"),
                hidden_evaluator=duplicate_seed_scenario.hidden_evaluator,
            )
            registry = ScenarioRegistry(source_root)
            registry.add_scenario(selected_scenario)
            registry.add_scenario(duplicate_seed_scenario)
            registry.add_scenario(duplicate_environment_scenario)

            payload = assemble_seed_candidate_registry(
                source_root=source_root,
                output_root=output_root,
                selection_plan={
                    "selected_seed_ids": [selected_scenario.query_seed.seed_id],
                },
                synthetic_specs=[_assembly_synthetic_spec()],
                source_name="unit-synthetic-backfill",
                policy_config=_assembly_policy(),
            )

            self.assertTrue(payload["valid"])
            self.assertEqual(payload["selection"]["duplicate_source_seed_ids"], [])
            self.assertEqual(payload["selection"]["copied_existing_count"], 1)
            self.assertEqual(payload["synthetic_generation"]["generated"], 1)

    def test_cli_assemble_candidate_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            output_root = root / "assembled"
            summary_path = root / "assembly-summary.json"
            selection_path = root / "selection-plan.json"
            synthetic_path = root / "generator-ready-spec.json"
            policy_path = root / "policy.json"
            scenario = _synthetic_source_scenario("repo/a")
            registry = ScenarioRegistry(source_root)
            registry.add_scenario(scenario)
            selection_path.write_text(
                json.dumps({"selected_seed_ids": [scenario.query_seed.seed_id]}),
                encoding="utf-8",
            )
            synthetic_path.write_text(
                json.dumps({"repositories": [_assembly_synthetic_spec()]}),
                encoding="utf-8",
            )
            policy_path.write_text(json.dumps(_assembly_policy()), encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "assemble-candidate",
                        "--source-root",
                        str(source_root),
                        "--selection-plan",
                        str(selection_path),
                        "--synthetic-spec",
                        str(synthetic_path),
                        "--output-root",
                        str(output_root),
                        "--policy",
                        str(policy_path),
                        "--output",
                        str(summary_path),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, disk_payload)
            self.assertTrue(payload["valid"])
            self.assertEqual(payload["selection"]["copied_existing_count"], 1)
            self.assertEqual(payload["synthetic_generation"]["generated"], 1)
            self.assertEqual(len(ScenarioRegistry(output_root).list_scenarios()), 2)

    def test_cli_build_seed_corpus_fails_when_budget_is_unmet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_corpus_inputs(
                root,
                coverage_budgets={
                    "min_language_counts": {"ruby": 1},
                    "max_quarantined_records": 0,
                },
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "build-corpus",
                        "--config",
                        str(config_path),
                        "--overwrite-outputs",
                    ]
                )

            manifest = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertFalse(manifest["valid"])
            self.assertFalse(manifest["coverage_budget"]["valid"])
            self.assertEqual(
                manifest["coverage_budget"]["issues"][0]["code"],
                "min_language_count_not_met",
            )
            self.assertTrue((root / "manifest.json").is_file())

    def test_build_seed_corpus_refuses_stale_registry_entries_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_corpus_inputs(root)

            self.assertTrue(build_seed_corpus(config_path)["valid"])
            with self.assertRaisesRegex(ValueError, "already contains entries"):
                build_seed_corpus(config_path)

    def test_build_seed_corpus_quarantines_records_outside_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _write_corpus_inputs(
                root,
                extra_public_records=[
                    {**_public_issue_record(), "id": "issue-200", "repository": "other/tool"}
                ],
                coverage_budgets={
                    "min_task_family_counts": {
                        family: 1 for family in sorted(SUPPORTED_TASK_FAMILIES)
                    },
                    "min_language_counts": {"python": 1},
                    "max_quarantined_records": 1,
                },
            )

            manifest = build_seed_corpus(config_path, overwrite_outputs=True)

            self.assertTrue(manifest["valid"])
            self.assertEqual(manifest["quarantine"]["records"], 1)
            self.assertIn("not allowlisted", manifest["quarantine"]["issues"][0])

    def test_rehearse_registry_import_gates_public_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "public.jsonl"
            allowlist = root / "allowlist.json"
            source.write_text(json.dumps(_public_issue_record()) + "\n", encoding="utf-8")
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )

            rehearsal = rehearse_registry_import(
                registry_root=root / "rehearsal",
                source_path=source,
                source_format="public-issue",
                source_name="curated-public-issues",
                allowlist_path=allowlist,
                min_imported=1,
                max_quarantined=0,
                seed_policy=_seed_policy_for_public_probe(),
            )

            self.assertTrue(rehearsal["valid"])
            self.assertEqual(rehearsal["import"]["imported"], 1)
            self.assertEqual(rehearsal["allowlist_filter"]["allowed"], 1)
            self.assertTrue(rehearsal["registry_validation"]["valid"])
            self.assertTrue(rehearsal["seed_audit"]["valid"])
            self.assertEqual(
                len(ScenarioRegistry(root / "rehearsal").list_scenarios()),
                1,
            )

    def test_rehearse_registry_import_gates_public_ci_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ci.jsonl"
            allowlist = root / "allowlist.json"
            source.write_text(json.dumps(_public_ci_record()) + "\n", encoding="utf-8")
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )

            rehearsal = rehearse_registry_import(
                registry_root=root / "rehearsal",
                source_path=source,
                source_format="public-ci",
                source_name="curated-public-ci",
                allowlist_path=allowlist,
                min_imported=1,
                max_quarantined=0,
                seed_policy=SeedLibraryPolicy(
                    min_train_eligible=1,
                    required_task_families=["ci_build"],
                    required_verifier_types=["hidden_command"],
                ),
            )

            self.assertTrue(rehearsal["valid"])
            self.assertEqual(rehearsal["import"]["source_format"], "public_ci")
            self.assertEqual(rehearsal["import"]["imported"], 1)
            scenario = ScenarioRegistry(root / "rehearsal").get_scenario(
                rehearsal["import"]["scenario_ids"][0]
            )
            self.assertEqual(scenario.query_seed.task_family, "ci_build")
            self.assertEqual(scenario.hidden_evaluator.hidden_tests, ["python -m pytest"])

    def test_rehearse_registry_import_materializes_public_ci_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            source = root / "ci.jsonl"
            allowlist = root / "allowlist.json"
            ci_command = _python_command(
                "from pathlib import Path; "
                "assert Path('app.py').read_text(encoding='utf-8') == 'value = 1\\n'"
            )
            source.write_text(
                json.dumps(
                    {
                        **_public_ci_record(),
                        "source_uri": repository.as_uri(),
                        "source_revision": commit,
                        "ci_commands": [ci_command],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record(source_uri=repository.as_uri())]}),
                encoding="utf-8",
            )

            rehearsal = rehearse_registry_import(
                registry_root=root / "rehearsal",
                source_path=source,
                source_format="public-ci",
                source_name="curated-public-ci",
                allowlist_path=allowlist,
                min_imported=1,
                max_quarantined=0,
                seed_policy=SeedLibraryPolicy(
                    min_train_eligible=1,
                    required_task_families=["ci_build"],
                    required_verifier_types=["hidden_command"],
                ),
                materialize_sample_count=1,
                materialize_root=root / "materialized",
                run_hidden_commands=True,
            )

            materialization = rehearsal["materialization"]
            result = materialization["results"][0]
            self.assertTrue(rehearsal["valid"])
            self.assertTrue(rehearsal["validation"]["materialization_valid"])
            self.assertTrue(materialization["enabled"])
            self.assertTrue(materialization["valid"])
            self.assertEqual(materialization["sampled"], 1)
            self.assertTrue(result["hidden_commands_ran"])
            self.assertEqual(result["commands_run"], 1)
            self.assertIn("command_sha256", result["command_results"][0])
            self.assertNotIn("command", result["command_results"][0])
            self.assertTrue((Path(result["workspace"]) / "app.py").is_file())

    def test_rehearse_registry_import_rehearses_hidden_test_patch_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            source = root / "public.jsonl"
            allowlist = root / "allowlist.json"
            hidden_command = _python_command(
                "from pathlib import Path; "
                "assert Path('hidden_test_marker.py').read_text(encoding='utf-8') "
                "== 'hidden = True\\n'"
            )
            record = {
                **_public_issue_record(),
                "source_uri": repository.as_uri(),
                "source_revision": commit,
                "patch": "",
                "test_patch": _hidden_marker_test_patch(),
                "test_commands": [hidden_command],
            }
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record(source_uri=repository.as_uri())]}),
                encoding="utf-8",
            )

            rehearsal = rehearse_registry_import(
                registry_root=root / "rehearsal",
                source_path=source,
                source_format="public-issue",
                source_name="curated-public-issues",
                allowlist_path=allowlist,
                min_imported=1,
                max_quarantined=0,
                seed_policy=SeedLibraryPolicy(
                    min_train_eligible=1,
                    required_task_families=["bug_repair"],
                    required_verifier_types=["hidden_test_patch"],
                ),
                hidden_test_patch_sample_count=1,
                hidden_test_patch_root=root / "hidden-patch-rehearsal",
                hidden_test_patch_expected_outcome="pass",
            )

            patch_rehearsal = rehearsal["hidden_test_patch_rehearsal"]
            result = patch_rehearsal["results"][0]
            scenario = ScenarioRegistry(root / "rehearsal").get_scenario(
                rehearsal["import"]["scenario_ids"][0]
            )
            self.assertTrue(rehearsal["valid"])
            self.assertTrue(rehearsal["validation"]["hidden_test_patch_rehearsal_valid"])
            self.assertTrue(patch_rehearsal["enabled"])
            self.assertTrue(patch_rehearsal["valid"])
            self.assertEqual(patch_rehearsal["sampled"], 1)
            self.assertEqual(result["patch_check_exit_code"], 0)
            self.assertEqual(result["patch_apply_exit_code"], 0)
            self.assertTrue(result["hidden_commands_ran"])
            self.assertEqual(result["commands_run"], 1)
            self.assertIn("test_patch_sha256", result)
            self.assertNotIn("test_patch", result)
            self.assertIn("command_sha256", result["command_results"][0])
            self.assertTrue((Path(result["workspace"]) / "hidden_test_marker.py").is_file())
            self.assertIn("test_patch", scenario.hidden_evaluator.metadata)

    def test_rehearse_registry_import_accepts_expected_hidden_test_patch_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            source = root / "public.jsonl"
            allowlist = root / "allowlist.json"
            record = {
                **_public_issue_record(),
                "source_uri": repository.as_uri(),
                "source_revision": commit,
                "patch": "",
                "test_patch": _hidden_marker_test_patch(),
                "test_commands": [_python_command("import sys; sys.exit(7)")],
            }
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record(source_uri=repository.as_uri())]}),
                encoding="utf-8",
            )

            rehearsal = rehearse_registry_import(
                registry_root=root / "rehearsal",
                source_path=source,
                source_format="public-issue",
                source_name="curated-public-issues",
                allowlist_path=allowlist,
                min_imported=1,
                max_quarantined=0,
                seed_policy=SeedLibraryPolicy(
                    min_train_eligible=1,
                    required_task_families=["bug_repair"],
                    required_verifier_types=["hidden_test_patch"],
                ),
                hidden_test_patch_sample_count=1,
                hidden_test_patch_root=root / "hidden-patch-rehearsal",
            )

            result = rehearsal["hidden_test_patch_rehearsal"]["results"][0]
            self.assertTrue(rehearsal["valid"])
            self.assertEqual(
                rehearsal["hidden_test_patch_rehearsal"]["expected_outcome"],
                "fail",
            )
            self.assertTrue(result["hidden_commands_ran"])
            self.assertEqual(result["command_outcome"], "fail")
            self.assertEqual(result["command_results"][0]["exit_code"], 7)

    def test_rehearse_registry_import_reports_hidden_test_patch_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            source = root / "public.jsonl"
            allowlist = root / "allowlist.json"
            record = {
                **_public_issue_record(),
                "source_uri": repository.as_uri(),
                "source_revision": commit,
                "patch": "",
                "test_commands": [_python_command("print('ok')")],
            }
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record(source_uri=repository.as_uri())]}),
                encoding="utf-8",
            )

            rehearsal = rehearse_registry_import(
                registry_root=root / "rehearsal",
                source_path=source,
                source_format="public-issue",
                source_name="curated-public-issues",
                allowlist_path=allowlist,
                min_imported=1,
                max_quarantined=0,
                seed_policy=_seed_policy_for_public_probe(),
                hidden_test_patch_sample_count=1,
                hidden_test_patch_root=root / "hidden-patch-rehearsal",
            )

            patch_rehearsal = rehearsal["hidden_test_patch_rehearsal"]
            self.assertFalse(rehearsal["valid"])
            self.assertFalse(rehearsal["validation"]["hidden_test_patch_rehearsal_valid"])
            self.assertFalse(patch_rehearsal["valid"])
            self.assertEqual(patch_rehearsal["sampled"], 0)
            self.assertEqual(
                patch_rehearsal["issues"][0]["code"],
                "hidden_test_patch_sample_shortfall",
            )

    def test_cli_import_rehearsal_materializes_public_ci_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            source = root / "ci.jsonl"
            allowlist = root / "allowlist.json"
            output = root / "rehearsal.json"
            source.write_text(
                json.dumps(
                    {
                        **_public_ci_record(),
                        "source_uri": repository.as_uri(),
                        "source_revision": commit,
                        "ci_commands": [
                            _python_command(
                                "from pathlib import Path; assert Path('app.py').is_file()"
                            )
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record(source_uri=repository.as_uri())]}),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "import-rehearsal",
                        "--root",
                        str(root / "rehearsal"),
                        "--source",
                        str(source),
                        "--format",
                        "public-ci",
                        "--source-name",
                        "curated-public-ci",
                        "--allowlist",
                        str(allowlist),
                        "--min-imported",
                        "1",
                        "--max-quarantined",
                        "0",
                        "--require-task-family",
                        "ci-build",
                        "--require-verifier-type",
                        "hidden-command",
                        "--materialize-sample-count",
                        "1",
                        "--materialize-root",
                        str(root / "materialized"),
                        "--run-hidden-commands",
                        "--output",
                        str(output),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, json.loads(output.read_text(encoding="utf-8")))
            self.assertTrue(payload["materialization"]["valid"])
            self.assertTrue(payload["materialization"]["results"][0]["hidden_commands_ran"])

    def test_cli_import_rehearsal_runs_hidden_test_patch_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = _write_git_repository(root)
            source = root / "public.jsonl"
            allowlist = root / "allowlist.json"
            output = root / "rehearsal.json"
            record = {
                **_public_issue_record(),
                "source_uri": repository.as_uri(),
                "source_revision": commit,
                "patch": "",
                "test_patch": _hidden_marker_test_patch(),
                "test_commands": [
                    _python_command(
                        "from pathlib import Path; "
                        "assert Path('hidden_test_marker.py').is_file()"
                    )
                ],
            }
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record(source_uri=repository.as_uri())]}),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "import-rehearsal",
                        "--root",
                        str(root / "rehearsal"),
                        "--source",
                        str(source),
                        "--format",
                        "public-issue",
                        "--source-name",
                        "curated-public-issues",
                        "--allowlist",
                        str(allowlist),
                        "--min-imported",
                        "1",
                        "--max-quarantined",
                        "0",
                        "--require-task-family",
                        "bug-repair",
                        "--require-verifier-type",
                        "hidden-test-patch",
                        "--hidden-test-patch-sample-count",
                        "1",
                        "--hidden-test-patch-root",
                        str(root / "hidden-patch-rehearsal"),
                        "--hidden-test-patch-expected-outcome",
                        "pass",
                        "--output",
                        str(output),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload, json.loads(output.read_text(encoding="utf-8")))
            self.assertTrue(payload["hidden_test_patch_rehearsal"]["valid"])
            self.assertEqual(
                payload["hidden_test_patch_rehearsal"]["results"][0]["patch_apply_exit_code"],
                0,
            )

    def test_hidden_rehearsal_command_arguments_falls_back_to_python3(self) -> None:
        def fake_which(name: str) -> str | None:
            if name == "python":
                return None
            if name == "python3":
                return sys.executable
            return None

        with patch("easy_agentic_data.seed_corpus.shutil.which", side_effect=fake_which):
            arguments = _hidden_rehearsal_command_arguments("python -m pytest tests")

        self.assertEqual(arguments[0], sys.executable)
        self.assertEqual(arguments[1:], ["-m", "pytest", "tests"])

    def test_hidden_rehearsal_records_failed_command_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result: dict[str, object] = {}

            with self.assertRaises(RuntimeError):
                _run_hidden_commands_for_rehearsal(
                    [
                        _python_command(
                            "import sys; sys.stderr.write('boom'); sys.exit(7)"
                        )
                    ],
                    Path(directory),
                    result,
                )

            self.assertEqual(result["commands_run"], 1)
            command_result = result["command_results"][0]
            self.assertEqual(command_result["exit_code"], 7)
            self.assertIn("command_sha256", command_result)
            self.assertIn("stderr_sha256", command_result)
            self.assertNotIn("command", command_result)
            self.assertNotIn("stderr", command_result)

    def test_git_apply_rehearsal_does_not_escape_nested_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            workspace = root / "workspace"
            workspace.mkdir()
            patch_text = (
                "diff --git a/tests/hidden_marker.py b/tests/hidden_marker.py\n"
                "new file mode 100644\n"
                "index 0000000..f222992\n"
                "--- /dev/null\n"
                "+++ b/tests/hidden_marker.py\n"
                "@@ -0,0 +1 @@\n"
                "+VALUE = 1\n"
            )

            check = _run_git_apply_for_rehearsal(patch_text, workspace, check=True)
            apply = _run_git_apply_for_rehearsal(patch_text, workspace, check=False)

            self.assertEqual(check["exit_code"], 0)
            self.assertEqual(apply["exit_code"], 0)
            self.assertTrue((workspace / "tests" / "hidden_marker.py").is_file())
            self.assertFalse((root / "tests" / "hidden_marker.py").exists())

    def test_build_seed_corpus_imports_public_ci_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_source = root / "public.jsonl"
            ci_source = root / "ci.jsonl"
            allowlist_source = root / "allowlist.json"
            public_source.write_text(
                json.dumps(_public_issue_record()) + "\n",
                encoding="utf-8",
            )
            ci_source.write_text(json.dumps(_public_ci_record()) + "\n", encoding="utf-8")
            allowlist_source.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )
            config = {
                "train_registry_root": "train",
                "manifest_output": "manifest.json",
                "overwrite_outputs": True,
                "overwrite_registries": True,
                "repository_allowlist": allowlist_source.name,
                "public_issue_sources": [
                    {
                        "path": public_source.name,
                        "format": "public-issue",
                        "source_name": "curated-public-issues",
                    }
                ],
                "public_ci_sources": [
                    {
                        "path": ci_source.name,
                        "format": "public-ci",
                        "source_name": "curated-public-ci",
                    }
                ],
                "seed_policy": {
                    "min_train_eligible": 2,
                    "required_task_families": ["bug_repair", "ci_build"],
                    "required_verifier_types": ["hidden_command"],
                },
                "coverage_budgets": {
                    "min_task_family_counts": {"bug_repair": 1, "ci_build": 1},
                    "min_language_counts": {"python": 1},
                    "max_quarantined_records": 0,
                },
                "review": {"required": False},
                "scale_decision": {
                    "approved": False,
                    "reason": "Unit-test corpus requires pilot review before scale-up.",
                },
            }
            config_path = root / "seed-corpus.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            manifest = build_seed_corpus(config_path, overwrite_outputs=True)

            self.assertTrue(manifest["valid"])
            self.assertEqual(manifest["seed_audit"]["train_eligible"], 2)
            self.assertEqual(
                manifest["seed_audit"]["train_task_family_counts"],
                {"bug_repair": 1, "ci_build": 1},
            )
            self.assertIn(
                "public_ci",
                {snapshot["format"] for snapshot in manifest["source_snapshots"]},
            )

    def test_cli_import_rehearsal_returns_nonzero_when_policy_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "public.jsonl"
            allowlist = root / "allowlist.json"
            output = root / "rehearsal.json"
            source.write_text(json.dumps(_public_issue_record()) + "\n", encoding="utf-8")
            allowlist.write_text(
                json.dumps({"repositories": [_allowlist_record()]}),
                encoding="utf-8",
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "import-rehearsal",
                        "--root",
                        str(root / "rehearsal"),
                        "--source",
                        str(source),
                        "--format",
                        "public-issue",
                        "--source-name",
                        "curated-public-issues",
                        "--allowlist",
                        str(allowlist),
                        "--min-imported",
                        "1",
                        "--max-quarantined",
                        "0",
                        "--min-train-eligible",
                        "1",
                        "--require-task-family",
                        "code-review",
                        "--require-verifier-type",
                        "hidden-command",
                        "--output",
                        str(output),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            disk_payload = json.loads(output.read_text(encoding="utf-8"))
            codes = {issue["code"] for issue in payload["seed_audit"]["issues"]}
            self.assertEqual(exit_code, 2)
            self.assertFalse(payload["valid"])
            self.assertFalse(payload["validation"]["seed_audit_valid"])
            self.assertEqual(payload, disk_payload)
            self.assertIn("missing_required_task_family", codes)


def _write_corpus_inputs(
    root: Path,
    *,
    coverage_budgets: dict[str, object] | None = None,
    extra_public_records: list[dict[str, object]] | None = None,
) -> Path:
    public_source = root / "public.jsonl"
    synthetic_source = root / "synthetic.json"
    holdout_source = root / "holdout.jsonl"
    public_records = [_public_issue_record(), *(extra_public_records or [])]
    public_source.write_text(
        "\n".join(json.dumps(record) for record in public_records) + "\n",
        encoding="utf-8",
    )
    synthetic_source.write_text(json.dumps(_repository_synthesis_spec()) + "\n", encoding="utf-8")
    holdout_source.write_text(json.dumps(_swe_bench_record()) + "\n", encoding="utf-8")
    allowlist_source = root / "allowlist.json"
    allowlist_source.write_text(
        json.dumps({"repositories": [_allowlist_record()]}),
        encoding="utf-8",
    )
    config = {
        "train_registry_root": "train",
        "holdout_registry_root": "holdout",
        "manifest_output": "manifest.json",
        "seed_audit_output": "seed-audit.json",
        "scenario_audit_output": "scenario-audit.json",
        "review_queue_output": "seed-review.jsonl",
        "overwrite_outputs": True,
        "repository_allowlist": allowlist_source.name,
        "public_issue_sources": [
            {
                "path": public_source.name,
                "format": "public-issue",
                "source_name": "curated-public-issues",
            }
        ],
        "repository_synthetic_sources": [
            {
                "path": synthetic_source.name,
                "source_name": "curated-repository-synthetic",
                "task_families": _synthetic_task_families(),
            }
        ],
        "holdout_sources": [
            {
                "path": holdout_source.name,
                "format": "swe-bench",
                "source_name": "princeton-nlp/SWE-bench_Lite",
                "license": "MIT",
                "test_command_template": "python -m pytest {test}",
            }
        ],
        "seed_policy": {
            "min_train_eligible": 12,
            "required_task_families": sorted(SUPPORTED_TASK_FAMILIES),
            "required_verifier_types": [
                "hidden_command",
                "doctest",
                "adversarial_test",
                "benchmark_command",
                "retrieval_evidence",
            ],
        },
        "coverage_budgets": coverage_budgets
        or {
            "min_task_family_counts": {family: 1 for family in sorted(SUPPORTED_TASK_FAMILIES)},
            "min_language_counts": {"python": 1},
            "max_quarantined_records": 0,
        },
        "review": {"sample_per_stratum": 1, "required": True},
        "scale_decision": {
            "approved": False,
            "reason": "Unit-test corpus requires pilot review before scale-up.",
        },
    }
    config_path = root / "seed-corpus.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _synthetic_task_families() -> list[str]:
    return sorted(set(DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES) | {"feature_implementation"})


def _seed_policy_for_public_probe():
    return SeedLibraryPolicy(
        min_train_eligible=1,
        required_task_families=["bug_repair"],
        required_verifier_types=["hidden_command"],
    )


def _backfill_audit() -> dict[str, object]:
    return {
        "valid": False,
        "total": 100,
        "train_eligible": 100,
        "train_task_family_counts": {
            "bug_repair": 30,
            "ci_build": 60,
            "test_authoring": 10,
        },
        "verifier_type_counts": {"hidden_command": 100},
        "train_source_method_counts": {
            "public_ci_workspace": 60,
            "public_issue_workspace": 40,
        },
        "train_language_counts": {"Python": 100},
        "train_repository_counts": {
            "example/tool": 70,
            "other/tool": 30,
        },
        "issues": [
            {
                "code": "missing_required_task_family",
                "message": "Required task family is absent from trainable seeds: docs_examples",
                "severity": "error",
            }
        ],
    }


def _backfill_policy() -> dict[str, object]:
    return {
        "target_train_eligible": 100,
        "seed_policy": {
            "min_train_eligible": 100,
            "required_task_families": ["bug_repair", "docs_examples", "ci_build"],
            "required_verifier_types": ["hidden_command", "doctest"],
            "max_task_family_share": 0.40,
            "max_source_method_share": 0.70,
            "max_repository_share": 0.50,
            "max_language_share": 0.80,
        },
        "coverage_budgets": {
            "min_task_family_counts": {
                "docs_examples": 5,
                "test_authoring": 20,
            },
            "min_source_method_counts": {
                "repository_grounded_synthetic": 10,
            },
            "min_verifier_type_counts": {
                "doctest": 5,
            },
        },
    }


def _selection_seed(
    suffix: str,
    task_family: str,
    source_method: str,
    repository: str,
    *,
    language: str = "python",
) -> QuerySeed:
    return QuerySeed(
        PublicTaskContext(
            f"Handle {suffix}.",
            context={"repository": repository},
        ),
        license="MIT",
        task_family=task_family,
        source_method=source_method,
        verifier_types=["hidden_command"],
        coverage_tags=[f"language:{language}"],
        metadata={"repository": repository, "language": language},
    )


def _selection_policy() -> dict[str, object]:
    return {
        "target_train_eligible": 6,
        "seed_policy": {
            "min_train_eligible": 6,
            "required_task_families": ["bug_repair", "ci_build", "docs_examples"],
            "required_verifier_types": ["hidden_command", "doctest"],
            "max_task_family_share": 0.50,
            "max_source_method_share": 0.75,
            "max_repository_share": 0.50,
            "max_language_share": 0.80,
        },
        "coverage_budgets": {
            "min_task_family_counts": {"docs_examples": 2},
            "min_source_method_counts": {"repository_grounded_synthetic": 2},
            "min_verifier_type_counts": {"doctest": 2},
        },
    }


def _remediation_selection_plan() -> dict[str, object]:
    return {
        "target_train_eligible": 10,
        "selected_existing_count": 6,
        "reserved_backfill": {
            "minimum_reserved_slots": 4,
            "slots": [
                {
                    "type": "share_cap_diversity",
                    "dimension": "language",
                    "dominant_label": "python",
                    "target": "non_python",
                    "minimum_count": 2,
                },
                {
                    "type": "share_cap_diversity",
                    "dimension": "repository",
                    "dominant_label": "repo/a",
                    "target": "non_repo/a",
                    "minimum_count": 4,
                },
                {
                    "type": "source_method_minimum",
                    "target": "public_issue_workspace",
                    "minimum_count": 1,
                },
                {
                    "type": "verifier_type_minimum",
                    "target": "build_command",
                    "minimum_count": 2,
                },
                {
                    "type": "verifier_type_minimum",
                    "target": "hidden_test_patch",
                    "minimum_count": 2,
                },
            ],
        },
    }


def _remediation_policy() -> dict[str, object]:
    return {
        "target_train_eligible": 10,
        "seed_policy": {
            "min_train_eligible": 10,
            "required_task_families": ["bug_repair"],
            "required_verifier_types": ["hidden_command"],
            "max_task_family_share": 1.0,
            "max_source_method_share": 1.0,
            "max_repository_share": 0.50,
            "max_language_share": 0.80,
        },
    }


def _assembly_policy() -> dict[str, object]:
    return {
        "seed_policy": {
            "min_train_eligible": 2,
            "required_task_families": ["bug_repair", "test_authoring"],
            "required_verifier_types": ["hidden_command"],
            "max_task_family_share": 1.0,
            "max_source_method_share": 1.0,
            "max_repository_share": 1.0,
            "max_language_share": 1.0,
        }
    }


def _synthetic_source_scenario(repository: str) -> Scenario:
    seed = QuerySeed(
        PublicTaskContext(
            f"Improve {repository}.",
            context={
                "repository": repository,
                "source_instance_id": f"{repository.replace('/', '__')}-issue-1",
                "source_url": f"https://github.com/{repository}/issues/1",
            },
        ),
        license="MIT",
        task_family="bug_repair",
        source_method="public_issue_workspace",
        verifier_types=["hidden_command"],
        coverage_tags=["language:python", f"repo:{repository}"],
        metadata={"repository": repository, "language": "Python"},
    )
    environment = EnvironmentSpec(
        name=repository.replace("/", "__"),
        version="1",
        source_uri=f"https://github.com/{repository}.git",
        source_revision="a" * 40,
        image_digest=PINNED_IMAGE,
        working_directory="/workspace",
    )
    return Scenario(
        query_seed=seed,
        environment=environment,
        hidden_evaluator=HiddenEvaluatorContext(hidden_tests=["python -m pytest tests"]),
    )


def _synthetic_backfill_selection_plan() -> dict[str, object]:
    return {
        "reserved_backfill": {
            "slots": [
                {"type": "task_family_minimum", "target": "code_review", "minimum_count": 1},
                {
                    "type": "task_family_minimum",
                    "target": "docs_examples",
                    "minimum_count": 2,
                },
                {"type": "task_family_minimum", "target": "performance", "minimum_count": 1},
                {
                    "type": "task_family_minimum",
                    "target": "test_authoring",
                    "minimum_count": 2,
                },
            ]
        }
    }


def _synthetic_backfill_plan() -> dict[str, object]:
    return {
        "gaps": {
            "task_family": [
                {"target": "code_review", "shortfall": 1},
                {"target": "docs_examples", "shortfall": 2},
                {"target": "performance", "shortfall": 1},
                {"target": "test_authoring", "shortfall": 2},
            ]
        }
    }


def _complete_evidence_records(tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task in tasks:
        record: dict[str, object] = {"evidence_task_id": task["evidence_task_id"]}
        if task["task_family"] == "docs_examples":
            record["doctest_commands"] = ["python -m doctest README.md"]
        elif task["task_family"] == "performance":
            record["benchmark_commands"] = ["python benchmarks/bench_target.py --max-ms 50"]
            record["performance_threshold"] = {"max_ms": 50}
        records.append(record)
    return records


def _complete_hidden_test_patch_curation_records(
    tasks: list[dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task in tasks:
        records.append(
            {
                "curation_task_id": task["curation_task_id"],
                "source_instance_id": task["source_instance_id"],
                "public_behavior_summary": "Parser preserves whitespace around quoted values.",
                "hidden_test_patch": (
                    "diff --git a/tests/test_hidden_parser.py "
                    "b/tests/test_hidden_parser.py\n"
                    "new file mode 100644\n"
                    "--- /dev/null\n"
                    "+++ b/tests/test_hidden_parser.py\n"
                    "@@\n"
                    "+def test_preserves_quoted_whitespace():\n"
                    "+    assert True\n"
                ),
                "hidden_test_commands": ["python -m pytest tests/test_hidden_parser.py"],
                "withheld_evaluator_notes": "Validated by unit test fixture.",
            }
        )
    return records


def _complete_hidden_command_curation_records(
    tasks: list[dict[str, object]],
    commands_by_source_instance: dict[str, list[str]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task in tasks:
        source_instance_id = str(task["source_instance_id"])
        records.append(
            {
                "curation_task_id": task["curation_task_id"],
                "source_instance_id": source_instance_id,
                "curated_setup_commands": ["python -m pip install -e ."],
                "curated_hidden_commands": commands_by_source_instance[
                    source_instance_id
                ],
                "command_runtime": "local-python",
                "expected_runtime_seconds": 30,
                "withheld_curation_notes": "Validated by unit test fixture.",
            }
        )
    return records


def _allowlist_record(
    *,
    repository: str = "example/tool",
    language: str = "Python",
    source_uri: str = "https://github.com/example/tool.git",
) -> dict[str, object]:
    return {
        "repository": repository,
        "source_uri": source_uri,
        "license": "MIT",
        "language": language,
        "collection_sources": ["issues", "pull_requests"],
        "test_commands": ["python -m pytest tests/test_parser.py"],
    }


def _public_issue_record() -> dict[str, object]:
    return {
        "id": "issue-100",
        "type": "issue",
        "repository": "example/tool",
        "source_uri": "https://github.com/example/tool.git",
        "source_revision": "e" * 40,
        "title": "Fix parser whitespace handling",
        "body": "The parser drops significant whitespace around quoted values.",
        "labels": ["bug", "parser"],
        "license": "MIT",
        "language": "Python",
        "image_digest": PINNED_IMAGE,
        "test_commands": ["python -m pytest tests/test_parser.py::test_whitespace"],
        "patch": "HIDDEN_ORACLE_PATCH",
    }


def _public_ci_record() -> dict[str, object]:
    return {
        **_public_issue_record(),
        "id": "ci-100",
        "type": "ci_failure",
        "source_instance_id": "example__tool-ci-100",
        "source_url": "https://github.com/example/tool/actions/runs/100",
        "title": "CI failure on parser workflow",
        "body": "The parser workflow failed on a fixed commit.",
        "labels": ["ci", "failure"],
        "ci_commands": ["python -m pytest"],
        "candidate_verifier": {
            "type": "ci_commands",
            "commands": ["python -m pytest"],
        },
    }


def _curation_source_record(
    source_id: str,
    source_type: str,
    *,
    repository: str = "example/tool",
) -> dict[str, object]:
    source_number = source_id.rsplit("-", 1)[-1]
    github_path = "pull" if source_type in {"pr", "pull_request"} else "issues"
    instance_prefix = repository.replace("/", "__")
    return {
        "id": source_id,
        "type": source_type,
        "repository": repository,
        "source_uri": f"https://github.com/{repository}.git",
        "source_revision": "f" * 40,
        "source_instance_id": f"{instance_prefix}-{source_id}",
        "source_url": f"https://github.com/{repository}/{github_path}/{source_number}",
        "title": f"Curate hidden tests for {source_id}",
        "body": "Public behavior says this path should preserve parser whitespace.",
        "labels": ["bug", "parser"],
        "license": "MIT",
        "language": "Python",
        "candidate_verifier": {
            "type": "stable_commands",
            "commands": ["python -m pytest tests/test_parser.py"],
        },
    }


def _workspace_source_record(
    source_id: str,
    source_type: str,
    *,
    repository: str = "example/tool",
) -> dict[str, object]:
    return {
        **_curation_source_record(
            source_id,
            source_type,
            repository=repository,
        ),
        "source_name": "curated_public_sources",
    }


def _repository_synthesis_spec() -> dict[str, object]:
    return {
        "repository": "example/tool",
        "source_uri": "https://github.com/example/tool.git",
        "source_revision": "f" * 40,
        "license": "MIT",
        "language": "Python",
        "image_digest": PINNED_IMAGE,
        "working_directory": "/workspace",
        "setup_commands": ["python -m pip install -e ."],
        "targets": [
            {
                "name": "parser",
                "paths": ["src/tool/parser.py", "tests/test_parser.py"],
                "test_commands": ["python -m pytest tests/test_parser.py"],
                "build_commands": ["python -m build"],
                "ci_commands": ["python -m pytest", "python -m build"],
                "doctest_commands": ["python -m doctest README.md"],
                "example_commands": ["python examples/parser_demo.py"],
                "benchmark_commands": ["python benchmarks/parser_bench.py --max-ms 50"],
                "adversarial_tests": ["python -m pytest tests/security/test_parser.py"],
                "migration_commands": ["python scripts/check_migration.py"],
                "required_state": {"files": ["pyproject.toml"]},
                "forbidden_state": {"forbidden_regex": ["eval("]},
                "diff_constraints": ["do not rename the public Parser API"],
                "performance_threshold": {"max_ms": 50},
                "retrieval_requirements": ["cite src/tool/parser.py"],
                "trace_quality_rubric": ["cite the parser entrypoint before answering"],
                "difficulty": 3,
            }
        ],
    }


def _ready_synthetic_spec(
    *,
    repository: str,
    target_name: str,
    task_families: list[str],
) -> dict[str, object]:
    spec = json.loads(json.dumps(_repository_synthesis_spec()))
    spec["repository"] = repository
    spec["source_uri"] = f"https://github.com/{repository}.git"
    spec["generator_ready"] = True
    spec["task_families"] = task_families
    spec["targets"][0]["name"] = target_name
    spec["targets"][0]["paths"] = [
        f"src/{target_name}.py",
        f"tests/test_{target_name}.py",
    ]
    return spec


def _assembly_synthetic_spec() -> dict[str, object]:
    spec = dict(_repository_synthesis_spec())
    spec["task_families"] = ["test_authoring"]
    return spec


def _swe_bench_record() -> dict[str, object]:
    return {
        "instance_id": "sample__parser-1",
        "repo": "sample/parser",
        "base_commit": "b" * 40,
        "environment_setup_commit": "d" * 40,
        "problem_statement": "Fix parser handling for quoted values.",
        "patch": "diff --git a/parser.py b/parser.py\n+return parsed\n",
        "test_patch": "diff --git a/tests/test_parser.py\n+def test_quoted_values(): ...\n",
        "FAIL_TO_PASS": ["tests/test_parser.py::test_quoted_values"],
        "PASS_TO_PASS": ["tests/test_parser.py::test_existing_values"],
        "version": "1",
        "image_digest": PINNED_IMAGE,
        "difficulty": 2,
    }


def _write_git_repository(root: Path) -> tuple[Path, str]:
    repository = root / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    (repository / "app.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "app.py"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True)
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return repository, completed.stdout.strip()


def _hidden_marker_test_patch() -> str:
    return (
        "diff --git a/hidden_test_marker.py b/hidden_test_marker.py\n"
        "new file mode 100644\n"
        "index 0000000..b5e1e91\n"
        "--- /dev/null\n"
        "+++ b/hidden_test_marker.py\n"
        "@@ -0,0 +1 @@\n"
        "+hidden = True\n"
    )


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


if __name__ == "__main__":
    unittest.main()
