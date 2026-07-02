import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.repository_synthetic import (
    DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES,
    generate_repository_synthetic_scenarios,
    scenario_from_repository_synthetic_spec,
)
from easy_agentic_data.seed_library import SeedLibraryPolicy, audit_seed_library

PINNED_IMAGE = "python@sha256:" + ("f" * 64)


class RepositorySyntheticTests(unittest.TestCase):
    def test_generator_creates_default_multi_family_trainable_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            summary = generate_repository_synthetic_scenarios(
                registry,
                [_repository_synthesis_spec()],
                source_name="curated-repository-synthetic",
            )

            self.assertEqual(summary.generated, len(DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES))
            self.assertEqual(summary.skipped, 0)
            scenarios = [registry.get_scenario(scenario_id) for scenario_id in summary.scenario_ids]
            families = {scenario.query_seed.task_family for scenario in scenarios}

            self.assertEqual(families, set(DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES))
            self.assertTrue(all(scenario.query_seed.train_eligible for scenario in scenarios))
            self.assertTrue(
                all(
                    scenario.query_seed.source_method == "repository_grounded_synthetic"
                    for scenario in scenarios
                )
            )
            audit = audit_seed_library(
                [scenario.query_seed for scenario in scenarios],
                policy=SeedLibraryPolicy(
                    min_train_eligible=len(DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES),
                    required_task_families=list(DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES),
                ),
            )

            self.assertTrue(audit.valid)
            self.assertEqual(audit.train_eligible, len(DEFAULT_REPOSITORY_SYNTHETIC_TASK_FAMILIES))

    def test_generator_keeps_evaluator_commands_out_of_public_view(self) -> None:
        scenario = scenario_from_repository_synthetic_spec(
            _repository_synthesis_spec(),
            task_family="security-hardening",
            source_name="curated-repository-synthetic",
        )
        public = json.dumps(scenario.to_dict(include_hidden=False), sort_keys=True)

        self.assertEqual(scenario.query_seed.task_family, "security_hardening")
        self.assertIn("adversarial_test", scenario.query_seed.verifier_types)
        self.assertIn(
            "python -m pytest tests/security/test_parser.py",
            scenario.hidden_evaluator.hidden_tests,
        )
        self.assertNotIn("tests/security/test_parser.py", public)

    def test_generator_blocks_non_allowlisted_license_from_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            spec = {**_repository_synthesis_spec(), "license": "GPL-3.0"}
            summary = generate_repository_synthetic_scenarios(
                registry,
                [spec],
                source_name="curated-repository-synthetic",
                task_families=["test-authoring"],
            )

            scenario = registry.get_scenario(summary.scenario_ids[0])
            self.assertFalse(scenario.query_seed.train_eligible)
            self.assertIn("license_not_allowlisted", scenario.query_seed.contamination_tags)

    def test_generator_skips_family_without_minimum_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            spec = {
                **_repository_synthesis_spec(),
                "targets": [{"name": "empty", "paths": ["src/example.py"]}],
            }
            summary = generate_repository_synthetic_scenarios(
                registry,
                [spec],
                source_name="curated-repository-synthetic",
                task_families=["performance"],
            )

            self.assertEqual(summary.generated, 0)
            self.assertEqual(summary.skipped, 1)
            self.assertIn("missing verifier evidence for performance", summary.issues[0])

    def test_generator_rejects_mutable_revision(self) -> None:
        spec = {**_repository_synthesis_spec(), "source_revision": "main"}

        with self.assertRaisesRegex(ValueError, "40-character fixed source revision"):
            scenario_from_repository_synthetic_spec(
                spec,
                task_family="test_authoring",
            )

    def test_cli_generate_synthetic_and_seed_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "repo-synthesis.json"
            root = Path(directory) / "registry"
            source.write_text(json.dumps(_repository_synthesis_spec()) + "\n", encoding="utf-8")
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "generate-synthetic",
                        "--root",
                        str(root),
                        "--source",
                        str(source),
                        "--source-name",
                        "curated-repository-synthetic",
                        "--task-family",
                        "docs-examples",
                        "--task-family",
                        "ci-build",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["generated"], 2)

            audit_stdout = io.StringIO()
            with redirect_stdout(audit_stdout):
                audit_exit_code = main(
                    [
                        "registry",
                        "seed-audit",
                        "--root",
                        str(root),
                        "--require-task-family",
                        "docs-examples",
                        "--require-task-family",
                        "ci-build",
                    ]
                )

            audit = json.loads(audit_stdout.getvalue())
            self.assertEqual(audit_exit_code, 0)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["train_eligible"], 2)


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


if __name__ == "__main__":
    unittest.main()
