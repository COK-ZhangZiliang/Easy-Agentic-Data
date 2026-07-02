import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.scenario_decontamination import audit_scenario_decontamination
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


class ScenarioDecontaminationTests(unittest.TestCase):
    def test_scenario_audit_flags_held_out_oracle_overlap(self) -> None:
        train = _scenario(
            query="Improve parser tests.",
            split="train",
            train_eligible=True,
            scenario_suffix="train",
        )
        holdout = _scenario(
            query="Fix parser whitespace.",
            split="evaluation",
            train_eligible=False,
            scenario_suffix="holdout",
        )

        audit = audit_scenario_decontamination([train], holdout_scenarios=[holdout])
        codes = {issue.code for issue in audit.issues}

        self.assertFalse(audit.valid)
        self.assertIn("holdout_hidden_test_overlap", codes)
        self.assertIn("holdout_reference_artifact_overlap", codes)
        self.assertIn("holdout_oracle_hash_overlap", codes)
        self.assertIn("holdout_scenario_source_instance_overlap", codes)
        self.assertEqual(audit.overlap_counts["holdout_hidden_test_overlap"], 1)

    def test_scenario_audit_ignores_self_matches_inside_same_registry(self) -> None:
        scenario = _scenario(
            query="Improve parser tests.",
            split="train",
            train_eligible=True,
            scenario_suffix="train",
        )

        audit = audit_scenario_decontamination([scenario])

        self.assertTrue(audit.valid)
        self.assertEqual(audit.overlap_counts, {})

    def test_cli_scenario_audit_uses_holdout_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "registry"
            holdout_root = Path(directory) / "holdout"
            ScenarioRegistry(root).add_scenario(
                _scenario(
                    query="Improve parser tests.",
                    split="train",
                    train_eligible=True,
                    scenario_suffix="train",
                )
            )
            ScenarioRegistry(holdout_root).add_scenario(
                _scenario(
                    query="Fix parser whitespace.",
                    split="evaluation",
                    train_eligible=False,
                    scenario_suffix="holdout",
                )
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "scenario-audit",
                        "--root",
                        str(root),
                        "--holdout-root",
                        str(holdout_root),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            codes = {issue["code"] for issue in payload["issues"]}
            self.assertEqual(exit_code, 2)
            self.assertIn("holdout_hidden_test_overlap", codes)
            self.assertIn("holdout_reference_artifact_overlap", codes)


def _scenario(
    *,
    query: str,
    split: str,
    train_eligible: bool,
    scenario_suffix: str,
) -> Scenario:
    seed = QuerySeed(
        PublicTaskContext(
            query,
            context={"repository": "example/tool"},
        ),
        split=split,
        license="MIT",
        task_family="test_authoring",
        source_method="repository_grounded_synthetic",
        train_eligible=train_eligible,
        verifier_types=["hidden_command"],
        metadata={
            "source_name": "curated",
            "source_instance_id": "example__tool-1",
        },
    )
    return Scenario(
        query_seed=seed,
        environment=EnvironmentSpec(
            name=f"fixture-{scenario_suffix}",
            version="1",
            source_uri="https://github.com/example/tool.git",
            source_revision="a" * 40,
        ),
        hidden_evaluator=HiddenEvaluatorContext(
            reference_artifacts=["source://curated/example__tool-1/test_patch"],
            hidden_tests=["python -m pytest tests/test_parser.py::test_whitespace"],
            metadata={
                "patch_sha256": "b" * 64,
                "test_patch_sha256": "c" * 64,
            },
        ),
        metadata={
            "source_name": "curated",
            "source_instance_id": "example__tool-1",
        },
    )


if __name__ == "__main__":
    unittest.main()
