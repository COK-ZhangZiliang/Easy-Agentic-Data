import unittest

from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.scenarios import HiddenEvaluatorContext, Scenario, ScenarioInstance
from easy_agentic_data.seeds import HiddenUserContext, PublicTaskContext, QuerySeed


class ScenarioContractTests(unittest.TestCase):
    def test_scenario_round_trip_preserves_content_ids(self) -> None:
        scenario = _scenario()

        restored = Scenario.from_dict(scenario.to_dict())
        instance = ScenarioInstance.materialize(
            scenario,
            random_seed=17,
            parameters={"variant": "small"},
            initial_state_hash="state_initial",
        )
        restored_instance = ScenarioInstance.from_dict(instance.to_dict())

        self.assertEqual(restored.scenario_id, scenario.scenario_id)
        self.assertEqual(restored.environment.environment_id, scenario.environment.environment_id)
        self.assertEqual(restored.query_seed.seed_id, scenario.query_seed.seed_id)
        self.assertEqual(restored_instance.instance_id, instance.instance_id)

    def test_public_views_exclude_hidden_context(self) -> None:
        scenario = _scenario()
        instance = ScenarioInstance.materialize(
            scenario,
            random_seed=17,
            initial_state_hash="state_initial",
        )

        public_scenario = scenario.to_dict(include_hidden=False)
        public_instance = instance.public_view()
        encoded = repr({"scenario": public_scenario, "instance": public_instance})

        self.assertNotIn("hidden_user", public_scenario["query_seed"])
        self.assertNotIn("hidden_evaluator", public_scenario)
        self.assertNotIn("hidden_user", public_instance)
        self.assertNotIn("hidden_evaluator", public_instance)
        self.assertNotIn("USER_CANARY_12345", encoded)
        self.assertNotIn("EVALUATOR_CANARY_67890", encoded)
        self.assertNotIn("HIDDEN_TEST_PATCH_CANARY", encoded)
        self.assertIn("HIDDEN_TEST_PATCH_CANARY", instance.sensitive_strings())
        self.assertIn("REQUIRED_STATE_CANARY_24680", instance.sensitive_strings())
        self.assertIn("FORBIDDEN_STATE_CANARY_13579", instance.sensitive_strings())
        self.assertIn("NESTED_METADATA_CANARY_11223", instance.sensitive_strings())

    def test_environment_metadata_rejects_secret_like_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "secret-like"):
            EnvironmentSpec(
                name="unsafe",
                version="1",
                metadata={"api_key": "not-allowed"},
            )


def _scenario() -> Scenario:
    seed = QuerySeed(
        public=PublicTaskContext(
            query="Repair the failing parser test.",
            context={"repository": "fixture"},
            constraints=["Do not change the public API."],
        ),
        hidden_user=HiddenUserContext(
            goal="USER_CANARY_12345",
            persona="A maintainer who answers concise clarification questions.",
            known_facts={"affected_module": "parser"},
        ),
        category="software_engineering",
        provenance="test fixture",
        license="Apache-2.0",
    )
    environment = EnvironmentSpec(
        name="parser-fixture",
        version="1",
        image_digest="sha256:fixture",
        source_uri="fixture://parser",
        source_revision="abc123",
    )
    return Scenario(
        query_seed=seed,
        environment=environment,
        hidden_evaluator=HiddenEvaluatorContext(
            reference_answer="EVALUATOR_CANARY_67890",
            hidden_tests=["tests/hidden/test_parser.py"],
            required_state={
                "file_contains": {"src/private.py": "REQUIRED_STATE_CANARY_24680"}
            },
            forbidden_state={
                "nested": [{"value": "FORBIDDEN_STATE_CANARY_13579"}]
            },
            metadata={
                "test_patch": "HIDDEN_TEST_PATCH_CANARY",
                "private": {
                    "PRIVATE_METADATA_KEY_CANARY": ["NESTED_METADATA_CANARY_11223"]
                },
            },
        ),
    )


if __name__ == "__main__":
    unittest.main()
