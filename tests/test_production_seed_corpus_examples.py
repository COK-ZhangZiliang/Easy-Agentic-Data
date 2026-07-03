import json
import unittest
from pathlib import Path

from easy_agentic_data.repository_allowlist import (
    audit_repository_allowlist,
    load_repository_allowlist,
)
from easy_agentic_data.seed_library import SUPPORTED_TASK_FAMILIES
from easy_agentic_data.source_collection import build_source_collection_plan

ROOT = Path(__file__).resolve().parents[1]


class ProductionSeedCorpusExampleTests(unittest.TestCase):
    def test_production_allowlist_passes_audit_and_collection_plan(self) -> None:
        records = load_repository_allowlist(
            ROOT / "examples" / "production-repository-allowlist.json"
        )

        audit = audit_repository_allowlist(records)
        self.assertTrue(audit.valid, [issue.code for issue in audit.issues])
        self.assertEqual(audit.total, 5)
        self.assertEqual(audit.approved, 5)
        self.assertEqual(audit.blocked, 0)
        self.assertEqual(
            audit.license_counts,
            {"apache_2.0": 1, "bsd_3_clause": 2, "mit": 2},
        )
        self.assertEqual(audit.language_counts, {"python": 5})
        self.assertEqual(
            audit.collection_source_counts,
            {"ci": 5, "issues": 5, "pull_requests": 5},
        )

        plan = build_source_collection_plan(
            records,
            output_root="runs/source-exports",
            source_name="production-public-python-sources",
        )
        self.assertTrue(plan["valid"], plan["allowlist_audit"]["issues"])
        self.assertEqual(plan["total_tasks"], 15)
        self.assertEqual(
            {task["collection_source"] for task in plan["tasks"]},
            {"ci", "issues", "pull_requests"},
        )
        self.assertTrue(all(task["stable_commands"] for task in plan["tasks"]))
        self.assertTrue(
            all("source_revision" in task["required_record_fields"] for task in plan["tasks"])
        )

    def test_production_seed_policy_matches_supported_families_and_blocks_scale(self) -> None:
        policy = json.loads(
            (ROOT / "examples" / "production-seed-corpus-policy.json").read_text(
                encoding="utf-8"
            )
        )

        target = policy["target_train_eligible"]
        seed_policy = policy["seed_policy"]
        coverage = policy["coverage_budgets"]
        candidate_status = policy["current_candidate_status"]

        self.assertEqual(policy["schema_version"], "easy_agentic_data.seed_corpus_policy.v1")
        self.assertEqual(target, 1000)
        self.assertEqual(seed_policy["min_train_eligible"], target)
        self.assertEqual(set(seed_policy["required_task_families"]), SUPPORTED_TASK_FAMILIES)
        self.assertEqual(
            set(coverage["min_task_family_counts"]),
            SUPPORTED_TASK_FAMILIES,
        )
        self.assertLessEqual(sum(coverage["min_task_family_counts"].values()), target)
        self.assertGreaterEqual(coverage["min_language_counts"]["python"], 700)
        self.assertEqual(candidate_status["repository_candidates"], 5)
        self.assertGreater(
            candidate_status["minimum_repositories_required_by_share_cap"],
            candidate_status["repository_candidates"],
        )
        self.assertTrue(policy["review"]["required"])
        self.assertFalse(policy["scale_decision"]["approved"])
        self.assertFalse(candidate_status["approved_for_scale"])
        self.assertTrue(policy["rollout_plan"])


if __name__ == "__main__":
    unittest.main()
