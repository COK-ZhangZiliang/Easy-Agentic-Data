import subprocess
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.registry import (
    ScenarioRegistry,
    import_repository_environment,
    issue_commit_scenario,
    materialize_environment_source,
    mutation_seed,
    semantic_duplicate_candidates,
)
from easy_agentic_data.sandbox import MemorySandbox
from easy_agentic_data.scenarios import Scenario
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


class RegistryTests(unittest.TestCase):
    def test_registry_round_trip_materialization_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            scenario = Scenario(
                mutation_seed(
                    query="Repair parser behavior.",
                    failing_test="test_parser",
                    provenance="fixture/parser",
                ),
                EnvironmentSpec(name="parser", version="1"),
            )
            registry.add_scenario(scenario)

            restored = registry.get_scenario(scenario.scenario_id)
            left = registry.materialize(scenario.scenario_id, random_seed=9)
            right = registry.materialize(scenario.scenario_id, random_seed=9)

            self.assertEqual(restored.scenario_id, scenario.scenario_id)
            self.assertEqual(left.instance_id, right.instance_id)
            self.assertEqual(len(registry.list_scenarios()), 1)
            self.assertTrue(registry.validate().valid)

    def test_validation_detects_train_evaluation_leakage_and_mutable_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = ScenarioRegistry(directory)
            train = QuerySeed(
                PublicTaskContext("Fix parser whitespace"),
                split="train",
                provenance="same-source",
            )
            evaluation = QuerySeed(
                PublicTaskContext("Fix parser whitespace"),
                split="evaluation",
                provenance="same-source",
            )
            registry.add_seed(train)
            registry.add_seed(evaluation)
            registry.add_environment(
                EnvironmentSpec(name="bad-image", version="1", image_digest="python:latest")
            )

            codes = {issue.code for issue in registry.validate().issues}

            self.assertIn("split_leakage", codes)
            self.assertIn("source_leakage", codes)
            self.assertIn("mutable_image", codes)

    def test_repository_and_issue_commit_adapters_hide_reference_patch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
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
            environment = import_repository_environment(repository, "HEAD", name="fixture")
            (repository / "app.py").write_text("value = 999\n", encoding="utf-8")
            materialized = Path(directory) / "materialized"
            materialize_environment_source(environment, materialized)
            scenario = issue_commit_scenario(
                issue_text="Correct the value.",
                environment=environment,
                reference_patch_artifact="artifact_secret_patch",
                hidden_tests=["hidden/test_value.py"],
                provenance="fixture-issue",
            )

            public = scenario.to_dict(include_hidden=False)

            self.assertEqual(len(environment.source_revision), 40)
            self.assertEqual((materialized / "app.py").read_text(encoding="utf-8"), "value = 1\n")
            self.assertNotIn("artifact_secret_patch", repr(public))
            self.assertNotIn("hidden/test_value.py", repr(public))

    def test_twenty_fixture_scenarios_reset_to_identical_health_state(self) -> None:
        seeds = []
        for index in range(20):
            sandbox = MemorySandbox({"health.txt": f"ok-{index}\n"})
            sandbox.create()
            initial = sandbox.state_hash()
            snapshot = sandbox.snapshot()
            sandbox.write("health.txt", "broken\n")
            sandbox.restore(snapshot)
            self.assertEqual(sandbox.state_hash(), initial)
            self.assertEqual(sandbox.read("health.txt"), f"ok-{index}\n")
            seeds.append(QuerySeed(PublicTaskContext(f"Repair fixture parser {index}")))
        self.assertTrue(semantic_duplicate_candidates(seeds))


if __name__ == "__main__":
    unittest.main()
