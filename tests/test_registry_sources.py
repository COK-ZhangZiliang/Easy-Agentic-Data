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
    import_swe_style_records,
    scenario_from_swe_style_record,
)

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
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["imported"], 1)
            self.assertEqual(len(ScenarioRegistry(root).list_scenarios()), 1)

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


if __name__ == "__main__":
    unittest.main()
