import io
import json
import shlex
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.registry_sources import (
    import_public_ci_records,
    import_public_issue_pr_records,
    import_swe_style_records,
    scenario_from_public_ci_record,
    scenario_from_public_issue_pr_record,
    scenario_from_swe_style_record,
)
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed

PINNED_IMAGE = "python@sha256:" + ("a" * 64)


class RegistrySourceTests(unittest.TestCase):
    def test_swe_bench_record_imports_query_workspace_and_hidden_oracle_refs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            summary = import_swe_style_records(
                registry,
                [_swe_bench_record()],
                source_format="swe-bench",
                source_name="princeton-nlp/SWE-bench_Lite",
                split="validation",
                license_name="MIT",
                permitted_use="research",
                test_command_template="python -m pytest {test}",
            )

            self.assertEqual(summary.imported, 1)
            self.assertEqual(summary.skipped, 0)
            scenario = registry.get_scenario(summary.scenario_ids[0])
            public = scenario.to_dict(include_hidden=False)
            encoded_public = json.dumps(public, sort_keys=True)

            self.assertEqual(scenario.query_seed.public.query, "Fix the quiet flag behavior.")
            self.assertEqual(scenario.query_seed.public.context["repository"], "example/tool")
            self.assertEqual(scenario.query_seed.split, "validation")
            self.assertEqual(scenario.query_seed.license, "MIT")
            self.assertEqual(scenario.query_seed.task_family, "bug_repair")
            self.assertEqual(scenario.query_seed.source_method, "external_issue_workspace")
            self.assertFalse(scenario.query_seed.train_eligible)
            self.assertIn("benchmark_source", scenario.query_seed.contamination_tags)
            self.assertIn(
                "benchmark:princeton_nlp:swe_bench_lite",
                scenario.query_seed.contamination_tags,
            )
            self.assertIn("hidden_command", scenario.query_seed.verifier_types)
            self.assertIn("hidden_test_patch", scenario.query_seed.verifier_types)
            self.assertIn("reference_patch", scenario.query_seed.verifier_types)
            self.assertIn("task_family:bug_repair", scenario.query_seed.coverage_tags)
            self.assertEqual(scenario.environment.source_uri, "https://github.com/example/tool.git")
            self.assertEqual(scenario.environment.source_revision, "b" * 40)
            self.assertEqual(scenario.environment.image_digest, PINNED_IMAGE)
            self.assertEqual(
                scenario.hidden_evaluator.hidden_tests,
                ["python -m pytest tests/test_cli.py::test_quiet"],
            )
            self.assertEqual(
                scenario.hidden_evaluator.metadata["fail_to_pass"],
                ["tests/test_cli.py::test_quiet"],
            )
            self.assertIn("def test_quiet", scenario.hidden_evaluator.metadata["test_patch"])
            self.assertNotIn("return False", encoded_public)
            self.assertNotIn("def test_quiet", encoded_public)
            self.assertNotIn("test_quiet", encoded_public)
            self.assertTrue(registry.validate().valid)

    def test_auto_detects_multi_swe_records_and_preserves_full_repo_name(self) -> None:
        scenario = scenario_from_swe_style_record(
            {
                "org": "zeromicro",
                "repo": "go-zero",
                "number": 2787,
                "problem_statement": "Fix request validation.",
                "base_sha": "c" * 40,
                "fix_patch": "diff --git a/server.go b/server.go\n",
                "test_patch": "diff --git a/server_test.go b/server_test.go\n",
                "fail_to_pass": ["server_test.go::TestValidation"],
                "image_digest": PINNED_IMAGE,
            },
            source_format="auto",
            source_name="Multi-SWE-bench",
        )

        self.assertEqual(scenario.metadata["source_format"], "multi_swe")
        self.assertEqual(scenario.query_seed.public.context["repository"], "zeromicro/go-zero")
        self.assertEqual(
            scenario.environment.source_uri, "https://github.com/zeromicro/go-zero.git"
        )
        self.assertEqual(scenario.environment.source_revision, "c" * 40)

    def test_cli_import_reads_jsonl_source_into_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            root = Path(directory) / "registry"
            source.write_text(json.dumps(_swe_bench_record()) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "import",
                        "--root",
                        str(root),
                        "--source",
                        str(source),
                        "--format",
                        "swe-bench",
                        "--source-name",
                        "sample",
                        "--license",
                        "MIT",
                        "--task-family",
                        "test-authoring",
                        "--source-method",
                        "curated_issue_workspace",
                        "--train-eligible",
                        "true",
                        "--coverage-tag",
                        "language:python",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["imported"], 1)
            registry = ScenarioRegistry(root)
            self.assertEqual(len(registry.list_scenarios()), 1)
            scenario = registry.get_scenario(payload["scenario_ids"][0])
            self.assertTrue(scenario.query_seed.train_eligible)
            self.assertEqual(scenario.query_seed.task_family, "test_authoring")
            self.assertEqual(scenario.query_seed.source_method, "curated_issue_workspace")
            self.assertIn("language:python", scenario.query_seed.coverage_tags)

            audit_stdout = io.StringIO()
            with redirect_stdout(audit_stdout):
                audit_exit_code = main(
                    [
                        "registry",
                        "seed-audit",
                        "--root",
                        str(root),
                    ]
                )

            audit = json.loads(audit_stdout.getvalue())
            self.assertEqual(audit_exit_code, 0)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["task_family_counts"], {"test_authoring": 1})
            self.assertEqual(audit["train_eligible"], 1)

    def test_cli_seed_audit_extra_benchmark_source_keeps_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "registry"
            registry = ScenarioRegistry(root)
            registry.add_seed(
                QuerySeed(
                    PublicTaskContext("Fix a held-out benchmark issue."),
                    license="MIT",
                    task_family="bug_repair",
                    source_method="external_issue_workspace",
                    train_eligible=True,
                    verifier_types=["hidden_command"],
                    metadata={"source_name": "princeton-nlp/SWE-bench_Lite"},
                )
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-audit",
                        "--root",
                        str(root),
                        "--benchmark-source",
                        "internal-holdout",
                    ]
                )

            audit = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(
                [issue["code"] for issue in audit["issues"]],
                ["benchmark_train_eligible"],
            )

    def test_cli_seed_audit_initializes_empty_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "empty-registry"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-audit",
                        "--root",
                        str(root),
                    ]
                )

            audit = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(audit["total"], 0)
            self.assertTrue((root / "seeds").is_dir())
            self.assertTrue((root / "registry.sqlite3").is_file())

    def test_cli_seed_audit_checks_policy_and_holdout_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "registry"
            holdout_root = Path(directory) / "holdout"
            ScenarioRegistry(root).add_seed(
                QuerySeed(
                    PublicTaskContext(
                        "Fix parser whitespace handling.",
                        context={"repository": "example/tool"},
                    ),
                    license="MIT",
                    provenance="curated:example__tool-1",
                    task_family="bug_repair",
                    source_method="curated_issue_workspace",
                    verifier_types=["hidden_command"],
                    metadata={
                        "source_name": "curated",
                        "source_instance_id": "example__tool-1",
                    },
                )
            )
            ScenarioRegistry(holdout_root).add_seed(
                QuerySeed(
                    PublicTaskContext(
                        "Fix parser whitespace handling!",
                        context={"repository": "example/tool"},
                    ),
                    split="evaluation",
                    provenance="curated:example__tool-1",
                    task_family="bug_repair",
                    source_method="external_issue_workspace",
                    verifier_types=["hidden_command"],
                    metadata={
                        "source_name": "curated",
                        "source_instance_id": "example__tool-1",
                    },
                )
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "seed-audit",
                        "--root",
                        str(root),
                        "--holdout-root",
                        str(holdout_root),
                        "--require-task-family",
                        "test-authoring",
                    ]
                )

            audit = json.loads(stdout.getvalue())
            codes = {issue["code"] for issue in audit["issues"]}
            self.assertEqual(exit_code, 2)
            self.assertIn("holdout_query_overlap", codes)
            self.assertIn("missing_required_task_family", codes)

    def test_public_issue_import_requires_fixed_revision_and_hides_oracles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            summary = import_public_issue_pr_records(
                registry,
                [_public_issue_record()],
                source_format="public-issue",
                source_name="curated-public-issues",
            )

            self.assertEqual(summary.imported, 1)
            self.assertEqual(summary.skipped, 0)
            scenario = registry.get_scenario(summary.scenario_ids[0])
            encoded_public = json.dumps(scenario.to_dict(include_hidden=False), sort_keys=True)

            self.assertEqual(scenario.query_seed.task_family, "bug_repair")
            self.assertEqual(scenario.query_seed.source_method, "public_issue_workspace")
            self.assertTrue(scenario.query_seed.train_eligible)
            self.assertEqual(scenario.query_seed.license, "MIT")
            self.assertIn("hidden_command", scenario.query_seed.verifier_types)
            self.assertIn("task_family:bug_repair", scenario.query_seed.coverage_tags)
            self.assertIn("repo:example/tool", scenario.query_seed.coverage_tags)
            self.assertEqual(scenario.environment.source_revision, "e" * 40)
            self.assertEqual(
                scenario.hidden_evaluator.hidden_tests,
                ["python -m pytest tests/test_parser.py::test_whitespace"],
            )
            self.assertNotIn("SECRET_ORACLE_PATCH", encoded_public)
            self.assertNotIn("test_whitespace", encoded_public)
            self.assertTrue(registry.validate().valid)

    def test_public_pr_import_infers_review_family_and_verifier_types(self) -> None:
        scenario = scenario_from_public_issue_pr_record(
            {
                **_public_issue_record(),
                "type": "pull_request",
                "number": 42,
                "title": "Address review comments on parser cleanup",
                "labels": ["review"],
                "build_commands": ["python -m build"],
                "diff_constraints": ["do not modify public API names"],
            },
            source_format="public-pr",
            source_name="curated-public-prs",
        )

        self.assertEqual(scenario.query_seed.task_family, "code_review")
        self.assertEqual(scenario.query_seed.source_method, "public_pr_workspace")
        self.assertIn("build_command", scenario.query_seed.verifier_types)
        self.assertIn("diff_constraint", scenario.query_seed.verifier_types)
        self.assertIn("python -m build", scenario.hidden_evaluator.hidden_tests)

    def test_public_pr_import_ignores_body_checklist_noise_for_family(self) -> None:
        scenario = scenario_from_public_issue_pr_record(
            {
                **_public_issue_record(),
                "type": "pull_request",
                "number": 43,
                "title": "fix: add explicit stacklevel to warnings",
                "body": (
                    "Syntax verified.\n\n"
                    "Checklist:\n"
                    "- [x] You've added tests.\n"
                    "- [ ] You've updated the documentation."
                ),
                "labels": ["bug", "documentation", "tests"],
            },
            source_format="public-pr",
            source_name="curated-public-prs",
        )

        self.assertEqual(scenario.query_seed.task_family, "bug_repair")
        self.assertIn("hidden_command", scenario.query_seed.verifier_types)
        self.assertIn("task_family:bug_repair", scenario.query_seed.coverage_tags)

    def test_public_pr_docs_label_needs_example_verifier_for_docs_family(self) -> None:
        scenario = scenario_from_public_issue_pr_record(
            {
                **_public_issue_record(),
                "type": "pull_request",
                "number": 44,
                "title": "docs: add guide for readiness probes",
                "labels": ["docs"],
                "test_commands": ["python -m pytest tests"],
            },
            source_format="public-pr",
            source_name="curated-public-prs",
        )

        self.assertEqual(scenario.query_seed.task_family, "code_review")
        self.assertIn("hidden_command", scenario.query_seed.verifier_types)
        self.assertNotIn("task_family:docs_examples", scenario.query_seed.coverage_tags)

    def test_public_pr_docs_label_uses_docs_family_with_example_verifier(self) -> None:
        scenario = scenario_from_public_issue_pr_record(
            {
                **_public_issue_record(),
                "type": "pull_request",
                "number": 45,
                "title": "docs: add guide for readiness probes",
                "labels": ["docs"],
                "example_commands": ["python docs/examples/readiness.py"],
            },
            source_format="public-pr",
            source_name="curated-public-prs",
        )

        self.assertEqual(scenario.query_seed.task_family, "docs_examples")
        self.assertIn("example_command", scenario.query_seed.verifier_types)
        self.assertIn("task_family:docs_examples", scenario.query_seed.coverage_tags)

    def test_public_issue_pr_import_skips_ci_records_without_ci_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            summary = import_public_issue_pr_records(
                registry,
                [_public_ci_record()],
                source_format="public-issue-pr",
                source_name="curated-public-sources",
            )

            self.assertEqual(summary.imported, 0)
            self.assertEqual(summary.skipped, 1)
            self.assertIn("CI source records require", summary.issues[0])
            self.assertEqual(registry.list_scenarios(), [])

    def test_public_ci_import_uses_ci_commands_as_hidden_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            summary = import_public_ci_records(
                registry,
                [_public_ci_record()],
                source_format="public-ci",
                source_name="curated-public-ci",
            )

            self.assertEqual(summary.imported, 1)
            self.assertEqual(summary.skipped, 0)
            scenario = registry.get_scenario(summary.scenario_ids[0])
            encoded_public = json.dumps(scenario.to_dict(include_hidden=False), sort_keys=True)

            self.assertEqual(scenario.query_seed.task_family, "ci_build")
            self.assertEqual(scenario.query_seed.source_method, "public_ci_workspace")
            self.assertTrue(scenario.query_seed.train_eligible)
            self.assertIn("hidden_command", scenario.query_seed.verifier_types)
            self.assertIn("task_family:ci_build", scenario.query_seed.coverage_tags)
            self.assertIn("source_format:public_ci", scenario.query_seed.coverage_tags)
            self.assertEqual(scenario.query_seed.public.context["source_type"], "public_ci")
            self.assertEqual(scenario.environment.source_uri, "https://github.com/example/tool.git")
            self.assertEqual(scenario.environment.source_revision, "e" * 40)
            self.assertEqual(scenario.environment.metadata["source_adapter"], "public_ci")
            self.assertEqual(scenario.hidden_evaluator.hidden_tests, ["python -m pytest"])
            self.assertEqual(
                scenario.hidden_evaluator.metadata["ci_commands"],
                ["python -m pytest"],
            )
            self.assertNotIn("python -m pytest", encoded_public)
            self.assertTrue(registry.validate().valid)

    def test_public_ci_import_requires_ci_commands(self) -> None:
        record = {**_public_ci_record(), "ci_commands": []}

        with self.assertRaisesRegex(ValueError, "ci_commands verifier evidence"):
            scenario_from_public_ci_record(record, source_format="public-ci")

    def test_cli_public_ci_import_and_seed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ci.jsonl"
            root = Path(directory) / "registry"
            source.write_text(json.dumps(_public_ci_record()) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "import",
                        "--root",
                        str(root),
                        "--source",
                        str(source),
                        "--format",
                        "public-ci",
                        "--source-name",
                        "curated-public-ci",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["imported"], 1)

            audit_stdout = io.StringIO()
            with redirect_stdout(audit_stdout):
                audit_exit_code = main(
                    [
                        "registry",
                        "seed-audit",
                        "--root",
                        str(root),
                        "--require-task-family",
                        "ci-build",
                        "--require-verifier-type",
                        "hidden-command",
                    ]
                )

            audit = json.loads(audit_stdout.getvalue())
            self.assertEqual(audit_exit_code, 0)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["train_task_family_counts"], {"ci_build": 1})

    def test_public_issue_auto_blocks_non_allowlisted_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            record = {**_public_issue_record(), "license": "GPL-3.0"}
            summary = import_public_issue_pr_records(
                registry,
                [record],
                source_format="public-issue",
                source_name="curated-public-issues",
            )

            scenario = registry.get_scenario(summary.scenario_ids[0])
            self.assertEqual(summary.imported, 1)
            self.assertFalse(scenario.query_seed.train_eligible)
            self.assertIn("license_not_allowlisted", scenario.query_seed.contamination_tags)

    def test_public_issue_import_skips_explicit_train_disallowed_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            record = {**_public_issue_record(), "license": "GPL-3.0"}
            summary = import_public_issue_pr_records(
                registry,
                [record],
                source_format="public-issue",
                source_name="curated-public-issues",
                train_eligible=True,
            )

            self.assertEqual(summary.imported, 0)
            self.assertEqual(summary.skipped, 1)
            self.assertIn("license is not allowed", summary.issues[0])

    def test_public_issue_import_rejects_mutable_revision(self) -> None:
        record = {**_public_issue_record(), "source_revision": "main"}

        with self.assertRaisesRegex(ValueError, "40-character fixed source revision"):
            scenario_from_public_issue_pr_record(record, source_format="public-issue")

    def test_cli_public_issue_import_and_seed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "public.jsonl"
            root = Path(directory) / "registry"
            source.write_text(json.dumps(_public_issue_record()) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "import",
                        "--root",
                        str(root),
                        "--source",
                        str(source),
                        "--format",
                        "public-issue",
                        "--source-name",
                        "curated-public-issues",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["imported"], 1)

            audit_stdout = io.StringIO()
            with redirect_stdout(audit_stdout):
                audit_exit_code = main(
                    [
                        "registry",
                        "seed-audit",
                        "--root",
                        str(root),
                        "--require-task-family",
                        "bug-repair",
                        "--require-verifier-type",
                        "hidden-command",
                    ]
                )

            audit = json.loads(audit_stdout.getvalue())
            self.assertEqual(audit_exit_code, 0)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["train_task_family_counts"], {"bug_repair": 1})

    def test_test_command_template_quotes_test_id_as_one_argument(self) -> None:
        record = _swe_bench_record()
        record["FAIL_TO_PASS"] = ["tests/test parser.py::test quoted"]

        scenario = scenario_from_swe_style_record(
            record,
            source_format="swe-bench",
            source_name="sample",
            test_command_template="python -m pytest {test}",
        )
        command = scenario.hidden_evaluator.hidden_tests[0]

        self.assertEqual(
            shlex.split(command),
            ["python", "-m", "pytest", "tests/test parser.py::test quoted"],
        )

    def test_parameterized_pytest_ids_fall_back_to_stable_node_id(self) -> None:
        record = _swe_bench_record()
        record["FAIL_TO_PASS"] = [
            'tests/test_cli.py::TestCLI::test_quiet["noisy value',
            "tests/test_cli.py::TestCLI::test_quiet[other-value]",
        ]

        scenario = scenario_from_swe_style_record(
            record,
            source_format="swe-bench",
            source_name="sample",
            test_command_template="python -m pytest {test}",
        )

        self.assertEqual(
            scenario.hidden_evaluator.hidden_tests,
            ["python -m pytest tests/test_cli.py::TestCLI::test_quiet"],
        )

    def test_test_command_template_rejects_option_like_test_ids(self) -> None:
        record = _swe_bench_record()
        record["FAIL_TO_PASS"] = ["--maxfail=1"]

        with self.assertRaisesRegex(ValueError, "unsafe hidden test id"):
            scenario_from_swe_style_record(
                record,
                source_format="swe-bench",
                source_name="sample",
                test_command_template="python -m pytest {test}",
            )


def _swe_bench_record() -> dict[str, object]:
    return {
        "instance_id": "example__tool-1",
        "repo": "example/tool",
        "base_commit": "b" * 40,
        "environment_setup_commit": "d" * 40,
        "problem_statement": "Fix the quiet flag behavior.",
        "hints_text": "The CLI should reject incompatible flags.",
        "patch": "diff --git a/tool.py b/tool.py\n+return False\n",
        "test_patch": (
            "diff --git a/tests/test_cli.py b/tests/test_cli.py\n+def test_quiet(): ...\n"
        ),
        "FAIL_TO_PASS": '["tests/test_cli.py::test_quiet"]',
        "PASS_TO_PASS": ["tests/test_cli.py::test_existing"],
        "version": "1.2",
        "image_digest": PINNED_IMAGE,
        "difficulty": 2,
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
        "patch": "SECRET_ORACLE_PATCH",
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


if __name__ == "__main__":
    unittest.main()
