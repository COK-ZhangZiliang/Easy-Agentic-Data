import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.repository_synthetic import DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES
from easy_agentic_data.seed_corpus import build_seed_corpus
from easy_agentic_data.seed_library import SUPPORTED_TASK_FAMILIES

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


def _write_corpus_inputs(
    root: Path,
    *,
    coverage_budgets: dict[str, object] | None = None,
) -> Path:
    public_source = root / "public.jsonl"
    synthetic_source = root / "synthetic.json"
    holdout_source = root / "holdout.jsonl"
    public_source.write_text(json.dumps(_public_issue_record()) + "\n", encoding="utf-8")
    synthetic_source.write_text(json.dumps(_repository_synthesis_spec()) + "\n", encoding="utf-8")
    holdout_source.write_text(json.dumps(_swe_bench_record()) + "\n", encoding="utf-8")
    config = {
        "train_registry_root": "train",
        "holdout_registry_root": "holdout",
        "manifest_output": "manifest.json",
        "seed_audit_output": "seed-audit.json",
        "scenario_audit_output": "scenario-audit.json",
        "review_queue_output": "seed-review.jsonl",
        "overwrite_outputs": True,
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


if __name__ == "__main__":
    unittest.main()
