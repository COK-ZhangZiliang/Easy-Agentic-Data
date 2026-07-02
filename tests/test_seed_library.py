import unittest

from easy_agentic_data.seed_library import SeedLibraryPolicy, audit_seed_library
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


class SeedLibraryTests(unittest.TestCase):
    def test_legacy_seed_dict_gets_safe_defaults(self) -> None:
        seed = QuerySeed.from_dict(
            {
                "public": {"query": "Repair the parser."},
                "category": "software_engineering",
                "difficulty": 2,
                "provenance": "fixture:1",
                "license": "MIT",
                "split": "train",
            }
        )

        self.assertEqual(seed.task_family, "general")
        self.assertEqual(seed.source_method, "unspecified")
        self.assertTrue(seed.train_eligible)
        self.assertEqual(seed.contamination_tags, [])

    def test_audit_flags_train_eligible_benchmark_source(self) -> None:
        seed = QuerySeed(
            PublicTaskContext("Fix a benchmark issue."),
            license="MIT",
            split="train",
            task_family="bug_repair",
            source_method="external_issue_workspace",
            train_eligible=True,
            verifier_types=["hidden_command"],
            metadata={"source_name": "princeton-nlp/SWE-bench_Lite"},
        )

        audit = audit_seed_library([seed])

        self.assertFalse(audit.valid)
        self.assertEqual(audit.train_eligible, 1)
        self.assertEqual(
            [issue.code for issue in audit.issues],
            ["benchmark_train_eligible"],
        )

    def test_audit_counts_blocked_benchmark_and_coverage(self) -> None:
        seed = QuerySeed(
            PublicTaskContext("Fix a held-out issue."),
            split="validation",
            task_family="bug_repair",
            source_method="external_issue_workspace",
            train_eligible=False,
            contamination_tags=["benchmark_source"],
            verifier_types=["hidden_test_patch"],
            coverage_tags=["language:python"],
            metadata={"source_name": "princeton-nlp/SWE-bench_Lite"},
        )

        audit = audit_seed_library([seed])

        self.assertTrue(audit.valid)
        self.assertEqual(audit.train_blocked, 1)
        self.assertEqual(audit.benchmark_blocked, 1)
        self.assertEqual(audit.task_family_counts, {"bug_repair": 1})
        self.assertEqual(audit.coverage_tag_counts, {"language:python": 1})

    def test_policy_gates_trainable_coverage_distribution(self) -> None:
        seeds = [
            QuerySeed(
                PublicTaskContext(
                    "Fix parser behavior.",
                    context={"repository": "example/tool"},
                ),
                license="MIT",
                task_family="bug_repair",
                source_method="curated_issue_workspace",
                verifier_types=["hidden_command"],
                coverage_tags=["language:python"],
            ),
            QuerySeed(
                PublicTaskContext(
                    "Fix CLI behavior.",
                    context={"repository": "example/tool"},
                ),
                license="MIT",
                task_family="bug_repair",
                source_method="curated_issue_workspace",
                verifier_types=["hidden_command"],
                coverage_tags=["language:python"],
            ),
        ]

        audit = audit_seed_library(
            seeds,
            policy=SeedLibraryPolicy(
                min_train_eligible=3,
                required_task_families=["test-authoring"],
                required_verifier_types=["hidden-test-patch"],
                max_task_family_share=0.75,
                max_repository_share=0.75,
            ),
        )
        codes = {issue.code for issue in audit.issues}

        self.assertFalse(audit.valid)
        self.assertIn("min_train_eligible_not_met", codes)
        self.assertIn("missing_required_task_family", codes)
        self.assertIn("missing_required_verifier", codes)
        self.assertIn("task_family_dominance", codes)
        self.assertIn("repository_dominance", codes)
        self.assertEqual(audit.train_task_family_counts, {"bug_repair": 2})

    def test_decontamination_flags_train_holdout_overlap(self) -> None:
        train = QuerySeed(
            PublicTaskContext(
                "Fix parser whitespace handling.",
                context={"repository": "example/tool"},
            ),
            license="MIT",
            provenance="curated:example__tool-1",
            task_family="bug_repair",
            source_method="curated_issue_workspace",
            verifier_types=["hidden_command"],
            metadata={
                "source_name": "curated",
                "source_instance_id": "example__tool-1",
            },
        )
        holdout = QuerySeed(
            PublicTaskContext(
                "Fix parser whitespace handling!",
                context={"repository": "example/tool"},
            ),
            split="evaluation",
            provenance="curated:example__tool-1",
            task_family="bug_repair",
            source_method="external_issue_workspace",
            verifier_types=["hidden_command"],
            metadata={
                "source_name": "curated",
                "source_instance_id": "example__tool-1",
            },
        )

        audit = audit_seed_library([train], holdout_seeds=[holdout])
        codes = {issue.code for issue in audit.issues}

        self.assertFalse(audit.valid)
        self.assertIn("holdout_query_overlap", codes)
        self.assertIn("holdout_provenance_overlap", codes)
        self.assertIn("holdout_source_instance_overlap", codes)
        self.assertIn("holdout_repository_overlap", codes)
        self.assertEqual(audit.decontamination_counts["holdout_query_overlap"], 1)

    def test_family_verifier_templates_require_matching_evidence(self) -> None:
        weak = QuerySeed(
            PublicTaskContext("Speed up report generation."),
            license="MIT",
            task_family="performance",
            source_method="synthetic_issue_workspace",
            verifier_types=["hidden_command"],
        )
        strong = QuerySeed(
            PublicTaskContext("Speed up report export."),
            license="MIT",
            task_family="performance",
            source_method="synthetic_issue_workspace",
            verifier_types=["benchmark_command", "performance_threshold"],
        )

        audit = audit_seed_library([weak, strong])
        issues_by_seed = {issue.seed_id: issue.code for issue in audit.issues}

        self.assertFalse(audit.valid)
        self.assertEqual(issues_by_seed[weak.seed_id], "family_verifier_gap")
        self.assertNotIn(strong.seed_id, issues_by_seed)


if __name__ == "__main__":
    unittest.main()
