import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import _run_setup_commands, main
from easy_agentic_data.real_seed_sources import (
    prepare_real_seed_registry,
    records_from_huggingface_rows_payload,
)
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.sandbox import CommandResult

PINNED_IMAGE = "python@sha256:" + ("b" * 64)


class RealSeedSourceTests(unittest.TestCase):
    def test_huggingface_rows_payload_extracts_seed_records(self) -> None:
        records = records_from_huggingface_rows_payload(
            {
                "rows": [
                    {
                        "row": {
                            "instance_id": "django__django-1",
                            "repo": "django/django",
                            "base_commit": "a" * 40,
                            "problem_statement": "Fix the regression.",
                        }
                    }
                ]
            }
        )

        self.assertEqual(records[0]["repo"], "django/django")
        self.assertEqual(records[0]["problem_statement"], "Fix the regression.")

    def test_prepare_real_seed_registry_clones_repo_and_imports_file_backed_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_repo, commit = _make_git_repo(root / "upstream")
            summary = prepare_real_seed_registry(
                registry_root=root / "registry",
                cache_root=root / "cache",
                records=[_seed_record(source_repo, commit)],
                source_name="princeton-nlp/SWE-bench_Lite",
                image_digest=PINNED_IMAGE,
                setup_commands=["python -m pip install --no-deps -e ."],
                network_policy="disabled",
                strict=True,
            )

            registry = ScenarioRegistry(root / "registry")
            scenario = registry.get_scenario(summary.import_summary.scenario_ids[0])
            checkout = Path(scenario.environment.source_uri[7:])
            head = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            public = json.dumps(scenario.to_dict(include_hidden=False), sort_keys=True)

            self.assertEqual(summary.import_summary.imported, 1)
            self.assertTrue(checkout.exists())
            self.assertEqual(head, commit)
            self.assertEqual(scenario.environment.source_revision, commit)
            self.assertEqual(scenario.environment.image_digest, PINNED_IMAGE)
            self.assertEqual(
                scenario.environment.setup_commands,
                ["python -m pip install --no-deps -e ."],
            )
            self.assertEqual(scenario.environment.network_policy, "disabled")
            self.assertEqual(scenario.query_seed.split, "validation")
            self.assertEqual(
                scenario.hidden_evaluator.hidden_tests,
                ["python -m pytest tests/test_real.py::test_behavior"],
            )
            self.assertNotIn("gold fix", public)
            self.assertNotIn("test_behavior", public)

    def test_cli_real_seed_demo_prepares_registry_from_local_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_repo, commit = _make_git_repo(root / "upstream")
            source = root / "seeds.jsonl"
            source.write_text(
                json.dumps(_seed_record(source_repo, commit)) + "\n", encoding="utf-8"
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "synthesis",
                        "real-seed-demo",
                        "--output",
                        str(root / "run"),
                        "--source",
                        str(source),
                        "--image-digest",
                        PINNED_IMAGE,
                        "--setup-command",
                        "python -m pip install --no-deps -e .",
                    ]
                )
            payload = json.loads(stdout.getvalue())

            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["import_summary"]["imported"], 1)
            self.assertEqual(len(ScenarioRegistry(root / "run" / "registry").list_scenarios()), 1)
            self.assertEqual(payload["repositories"][0]["revision"], commit)
            scenario = ScenarioRegistry(root / "run" / "registry").get_scenario(
                payload["import_summary"]["scenario_ids"][0]
            )
            self.assertEqual(
                scenario.environment.setup_commands,
                ["python -m pip install --no-deps -e ."],
            )

    def test_setup_commands_run_before_agent_and_fail_fast(self) -> None:
        sandbox = _SetupSandbox()

        _run_setup_commands(sandbox, ["python -m pip install --no-deps -e ."])

        self.assertEqual(
            sandbox.commands, [["python", "-m", "pip", "install", "--no-deps", "-e", "."]]
        )

        with self.assertRaisesRegex(RuntimeError, "Environment setup command failed"):
            _run_setup_commands(_SetupSandbox(exit_code=2), ["false"])


def _make_git_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    (path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(["init", "-q"], path)
    _git(["config", "user.name", "Test"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["add", "."], path)
    _git(["commit", "-qm", "initial"], path)
    commit = _git(["rev-parse", "HEAD"], path).stdout.strip()
    return path, commit


def _seed_record(source_repo: Path, commit: str) -> dict[str, object]:
    return {
        "instance_id": "real__demo-1",
        "repo": "example/real-demo",
        "source_uri": source_repo.as_uri(),
        "base_commit": commit,
        "problem_statement": "Fix the behavior reported by the upstream issue.",
        "hints_text": "Start by inspecting module.py.",
        "patch": "diff --git a/module.py b/module.py\n+# gold fix\n",
        "test_patch": (
            "diff --git a/tests/test_real.py b/tests/test_real.py\n+def test_behavior(): ...\n"
        ),
        "FAIL_TO_PASS": ["tests/test_real.py::test_behavior"],
        "PASS_TO_PASS": ["tests/test_real.py::test_existing"],
    }


def _git(arguments: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


class _SetupSandbox:
    def __init__(self, *, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.commands: list[list[str]] = []

    def execute(self, command: list[str]) -> CommandResult:
        self.commands.append(command)
        return CommandResult(self.exit_code, "", "failed" if self.exit_code else "", 1.0, False)


if __name__ == "__main__":
    unittest.main()
