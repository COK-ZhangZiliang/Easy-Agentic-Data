import json
import os
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from easy_agentic_data.config import LLMConfig
from easy_agentic_data.pilot_contract import (
    GOLD20_REQUIRED_VALIDATION_GATES,
    Gold20Binding,
    PilotBudgets,
    PilotQualityTargets,
    PilotRunContract,
    PilotVersionHashes,
    PricingSpec,
    ProviderConfigBinding,
    canonical_sha256,
)


class PilotContractTests(unittest.TestCase):
    def test_provider_binding_is_secret_free_and_normalizes_endpoint(self) -> None:
        first = LLMConfig(
            provider="openai_compatible",
            model="model-a",
            base_url="HTTPS://API.Example.COM:443/v1/",
            api_key_env="EAD_TEST_API_KEY",
            temperature=0.0,
            max_tokens=4096,
            request_body={"thinking": {"type": "disabled"}, "seed": 17},
        )
        second = LLMConfig(
            provider="openai_compatible",
            model="model-a",
            base_url="https://api.example.com/v1",
            api_key_env="EAD_TEST_API_KEY",
            temperature=0.0,
            max_tokens=4096,
            request_body={"seed": 17, "thinking": {"type": "disabled"}},
        )

        with patch.dict(os.environ, {"EAD_TEST_API_KEY": "top-secret-value"}):
            first_binding = ProviderConfigBinding.from_config(first)
            second_binding = ProviderConfigBinding.from_config(second)

        encoded = first_binding.to_json()
        self.assertEqual(first_binding.endpoint_sha256, second_binding.endpoint_sha256)
        self.assertEqual(first_binding.config_sha256, second_binding.config_sha256)
        self.assertNotIn("api.example.com", encoded)
        self.assertNotIn("top-secret-value", encoded)
        self.assertEqual(first_binding.api_key_env, "EAD_TEST_API_KEY")
        self.assertEqual(json.loads(encoded)["config_sha256"], first_binding.config_sha256)

    def test_provider_binding_rejects_embedded_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            ProviderConfigBinding.from_config(
                LLMConfig(
                    provider="openai_compatible",
                    model="model-a",
                    base_url="https://user:password@example.test/v1",
                )
            )

    def test_provider_binding_preserves_api_path_trailing_slash(self) -> None:
        without_slash = ProviderConfigBinding.from_config(
            LLMConfig(
                provider="local_openai_compatible",
                model="model-a",
                chat_completions_path="/chat/completions",
                temperature=0.0,
            )
        )
        with_slash = ProviderConfigBinding.from_config(
            LLMConfig(
                provider="local_openai_compatible",
                model="model-a",
                chat_completions_path="/chat/completions/",
                temperature=0.0,
            )
        )

        self.assertNotEqual(
            without_slash.chat_completions_path_sha256,
            with_slash.chat_completions_path_sha256,
        )
        self.assertNotEqual(without_slash.config_sha256, with_slash.config_sha256)
        with self.assertRaisesRegex(ValueError, "credential-like"):
            ProviderConfigBinding.from_config(
                LLMConfig(
                    provider="openai_compatible",
                    model="model-a",
                    temperature=0.0,
                    request_body={"authorization": "Bearer secret"},
                )
            )

    def test_pilot_provider_requires_seed_capability_or_greedy_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature=0"):
            ProviderConfigBinding.from_config(
                LLMConfig(
                    provider="openai_compatible",
                    model="model-a",
                    temperature=0.2,
                )
            )

        seeded = ProviderConfigBinding.from_config(
            LLMConfig(
                provider="openai_compatible",
                model="model-a",
                temperature=0.2,
                seed_request_field="seed",
            )
        )
        self.assertEqual(seeded.seed_request_field, "seed")

    def test_provider_response_aliases_are_frozen_and_content_bound(self) -> None:
        config = LLMConfig(
            provider="openai_compatible",
            model="deepseek-chat",
            temperature=0.0,
            response_model_aliases=["deepseek-v3", "deepseek-v3.1"],
        )
        binding = ProviderConfigBinding.from_config(config)

        self.assertEqual(
            binding.response_model_aliases,
            ("deepseek-v3", "deepseek-v3.1"),
        )
        self.assertEqual(
            ProviderConfigBinding.from_dict(binding.to_dict()),
            binding,
        )
        with self.assertRaisesRegex(ValueError, "repeat"):
            LLMConfig(
                provider="openai_compatible",
                model="deepseek-chat",
                response_model_aliases=["deepseek-chat"],
            )

    def test_gold20_binding_requires_exact_manifest_and_registry_scenarios(self) -> None:
        manifest, snapshots = _gold20_manifest()
        binding = Gold20Binding.from_manifest(manifest)

        self.assertEqual(binding.corpus_id, "gold20_test")
        self.assertEqual(binding.manifest_sha256, canonical_sha256(manifest))
        self.assertEqual(len(binding.scenarios), 20)
        binding.assert_exact_scenarios(
            {scenario_id: canonical_sha256(value) for scenario_id, value in snapshots.items()}
        )

        missing = dict(snapshots)
        missing.pop("scenario_19")
        with self.assertRaisesRegex(ValueError, "scenario set"):
            binding.assert_exact_scenarios(
                {scenario_id: canonical_sha256(value) for scenario_id, value in missing.items()}
            )

        mismatched = {
            scenario_id: canonical_sha256(value) for scenario_id, value in snapshots.items()
        }
        mismatched["scenario_00"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            binding.assert_exact_scenarios(mismatched)

        manifest["records"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            Gold20Binding.from_manifest(manifest)

        manifest, _ = _gold20_manifest()
        manifest["records"][0]["seed_id"] = "seed_tampered"
        with self.assertRaisesRegex(ValueError, "record_sha256"):
            Gold20Binding.from_manifest(manifest)

        manifest, _ = _gold20_manifest()
        serialized = Gold20Binding.from_manifest(manifest).to_dict()
        serialized["environment_bundle_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "environment_bundle_sha256"):
            Gold20Binding.from_dict(serialized)

    def test_gold20_binding_loads_json_file_without_binding_path(self) -> None:
        manifest, _ = _gold20_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold20.json"
            path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            from_path = Gold20Binding.from_manifest(path)
            from_value = Gold20Binding.from_manifest(manifest)

        self.assertEqual(from_path, from_value)
        self.assertNotIn(str(path), from_path.to_json())

    def test_pricing_calculates_real_cost_from_usage(self) -> None:
        pricing = PricingSpec(
            input_usd_per_million_tokens="2",
            cached_input_usd_per_million_tokens="0.5",
            output_usd_per_million_tokens="6",
        )

        cost = pricing.calculate_cost(
            {
                "prompt_tokens": 1_000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 200},
            }
        )

        self.assertEqual(cost.input_tokens, 1_000)
        self.assertEqual(cost.cached_input_tokens, 200)
        self.assertEqual(cost.output_tokens, 500)
        self.assertEqual(cost.total_tokens, 1_500)
        self.assertEqual(cost.cost_usd, Decimal("0.0047"))
        self.assertEqual(cost.to_dict()["cost_usd"], "0.0047")
        self.assertEqual(cost.pricing_sha256, pricing.pricing_sha256)

    def test_pricing_supports_cache_hit_and_miss_usage(self) -> None:
        pricing = PricingSpec(
            input_usd_per_million_tokens="1",
            cached_input_usd_per_million_tokens="0.1",
            output_usd_per_million_tokens="2",
        )
        cost = pricing.calculate_cost(
            {
                "prompt_cache_hit_tokens": 400,
                "prompt_cache_miss_tokens": 600,
                "completion_tokens": 100,
            }
        )

        self.assertEqual(cost.input_tokens, 1_000)
        self.assertEqual(cost.cached_input_tokens, 400)
        self.assertEqual(cost.cost_usd, Decimal("0.00084"))

    def test_pricing_rejects_inconsistent_total_tokens(self) -> None:
        pricing = PricingSpec(
            input_usd_per_million_tokens="1",
            cached_input_usd_per_million_tokens="0.1",
            output_usd_per_million_tokens="2",
        )

        with self.assertRaisesRegex(ValueError, "Total token count"):
            pricing.calculate_cost(
                {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 999}
            )

    def test_pricing_rejects_missing_or_conflicting_usage(self) -> None:
        pricing = PricingSpec(
            input_usd_per_million_tokens="1",
            cached_input_usd_per_million_tokens="0.1",
            output_usd_per_million_tokens="2",
        )

        with self.assertRaisesRegex(ValueError, "non-empty"):
            pricing.calculate_cost({})
        with self.assertRaisesRegex(ValueError, "input token"):
            pricing.calculate_cost({"completion_tokens": 10})
        with self.assertRaisesRegex(ValueError, "output token"):
            pricing.calculate_cost({"prompt_tokens": 10})
        with self.assertRaisesRegex(ValueError, "Conflicting input token"):
            pricing.calculate_cost(
                {
                    "input_tokens": 10,
                    "prompt_tokens": 11,
                    "output_tokens": 2,
                }
            )
        with self.assertRaisesRegex(ValueError, "Conflicting output token"):
            pricing.calculate_cost(
                {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "completion_tokens": 3,
                }
            )

        cost = pricing.calculate_cost(
            {
                "input_tokens": 10,
                "prompt_tokens": 10,
                "output_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 12,
            }
        )
        self.assertEqual(cost.total_tokens, 12)

    def test_contract_is_stable_and_plans_exactly_two_rollouts_per_scenario(self) -> None:
        manifest, _ = _gold20_manifest()
        contract = _contract(Gold20Binding.from_manifest(manifest))

        self.assertEqual(len(contract.rollouts), 40)
        self.assertEqual(
            Counter(rollout.scenario_id for rollout in contract.rollouts),
            {f"scenario_{index:02d}": 2 for index in range(20)},
        )
        for scenario_id in contract.corpus.scenario_ids:
            planned = [
                rollout for rollout in contract.rollouts if rollout.scenario_id == scenario_id
            ]
            self.assertEqual([item.rollout_index for item in planned], [0, 1])
            self.assertEqual([item.random_seed for item in planned], [101, 202])
            self.assertTrue(all(item.rollout_id.startswith("rollout_") for item in planned))
            self.assertTrue(all(item.contract_id == contract.contract_id for item in planned))

        encoded = contract.to_json()
        restored = PilotRunContract.from_dict(json.loads(encoded))
        self.assertEqual(restored, contract)
        self.assertEqual(restored.contract_id, contract.contract_id)
        self.assertEqual(restored.to_json(), encoded)

        tampered = json.loads(encoded)
        tampered["rollouts"][0]["random_seed"] = 999
        with self.assertRaisesRegex(ValueError, "rollout assignments"):
            PilotRunContract.from_dict(tampered)

    def test_contract_id_changes_with_a_budget_or_seed_change(self) -> None:
        manifest, _ = _gold20_manifest()
        corpus = Gold20Binding.from_manifest(manifest)
        original = _contract(corpus)
        changed_budget = _contract(
            corpus,
            budgets=PilotBudgets(
                max_agent_turns=21,
                max_agent_tool_calls=50,
                max_agent_tokens=100_000,
                max_agent_seconds=600,
                max_total_tokens=4_000_000,
                max_total_cost_usd="50",
                max_total_seconds=28_800,
            ),
        )
        changed_seed = _contract(corpus, rollout_seeds=(101, 303))
        changed_quality_target = _contract(
            corpus,
            quality_targets=PilotQualityTargets(minimum_successes=2),
        )

        self.assertNotEqual(original.contract_id, changed_budget.contract_id)
        self.assertNotEqual(original.contract_id, changed_seed.contract_id)
        self.assertNotEqual(original.contract_id, changed_quality_target.contract_id)
        self.assertEqual(
            original.quality_targets.to_dict(),
            {
                "minimum_successes": 1,
                "minimum_sft": 1,
                "minimum_rl": 1,
                "minimum_preference": 0,
            },
        )

    def test_contract_rejects_non_distinct_or_non_pair_seed_schedule(self) -> None:
        manifest, _ = _gold20_manifest()
        corpus = Gold20Binding.from_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            _contract(corpus, rollout_seeds=(1, 2, 3))
        with self.assertRaisesRegex(ValueError, "distinct"):
            _contract(corpus, rollout_seeds=(7, 7))


def _contract(
    corpus: Gold20Binding,
    *,
    budgets: PilotBudgets | None = None,
    rollout_seeds: tuple[int, ...] = (101, 202),
    quality_targets: PilotQualityTargets | None = None,
) -> PilotRunContract:
    provider = ProviderConfigBinding.from_config(
        LLMConfig(
            provider="local_openai_compatible",
            model="model-a",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env=None,
            temperature=0.0,
            max_tokens=4096,
        )
    )
    return PilotRunContract(
        corpus=corpus,
        provider=provider,
        budgets=budgets
        or PilotBudgets(
            max_agent_turns=20,
            max_agent_tool_calls=50,
            max_agent_tokens=100_000,
            max_agent_seconds=600,
            max_total_tokens=4_000_000,
            max_total_cost_usd="50",
            max_total_seconds=28_800,
        ),
        versions=PilotVersionHashes(
            prompt_sha256=canonical_sha256("prompt-v1"),
            tool_schema_sha256=canonical_sha256({"tools": ["read_file"]}),
            evaluator_sha256=canonical_sha256("evaluator-v1"),
            environment_sha256=canonical_sha256("environment-bundle-v1"),
            exporter_sha256=canonical_sha256("exporter-v1"),
        ),
        pricing=PricingSpec(
            input_usd_per_million_tokens="1",
            cached_input_usd_per_million_tokens="0.1",
            output_usd_per_million_tokens="2",
        ),
        quality_targets=quality_targets or PilotQualityTargets(),
        rollout_seeds=rollout_seeds,
    )


def _gold20_manifest() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    snapshots = {
        f"scenario_{index:02d}": {
            "scenario_id": f"scenario_{index:02d}",
            "query": f"Repair task {index}",
            "environment": f"env_{index:02d}",
        }
        for index in range(20)
    }
    records = []
    for index, (scenario_id, snapshot) in enumerate(sorted(snapshots.items())):
        record = {
            "scenario_id": scenario_id,
            "seed_id": f"seed_{index:02d}",
            "environment_id": f"env_{index:02d}",
            "valid": True,
            "hashes": {
                "scenario_sha256": canonical_sha256(snapshot),
                "environment_sha256": canonical_sha256({"environment": index}),
                "evaluator_sha256": canonical_sha256({"evaluator": index}),
            },
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
    return (
        {
            "schema_version": "easy_agentic_data.gold20_manifest.v1",
            "corpus_id": "gold20_test",
            "expected_seed_count": 20,
            "valid": True,
            "issues": [],
            "validation": dict.fromkeys(GOLD20_REQUIRED_VALIDATION_GATES, True),
            "evidence": {"registry_snapshot_sha256": canonical_sha256("registry")},
            "records": records,
        },
        snapshots,
    )


if __name__ == "__main__":
    unittest.main()
