import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.repository_synthetic import DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES
from easy_agentic_data.seed_corpus import build_seed_corpus, rehearse_registry_import
from easy_agentic_data.seed_library import SUPPORTED_TASK_FAMILIES, SeedLibraryPolicy

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


def _allowlist_record(
    *,
    source_uri: str = "https://github.com/example/tool.git",
) -> dict[str, object]:
    return {
        "repository": "example/tool",
        "source_uri": source_uri,
        "license": "MIT",
        "language": "Python",
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


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


if __name__ == "__main__":
    unittest.main()
