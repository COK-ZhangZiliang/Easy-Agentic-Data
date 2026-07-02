import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.batch import load_human_reviews
from easy_agentic_data.cli import main
from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.registry import ScenarioRegistry
from easy_agentic_data.scenarios import Scenario
from easy_agentic_data.seed_review import build_seed_review_queue
from easy_agentic_data.seeds import PublicTaskContext, QuerySeed


class SeedReviewTests(unittest.TestCase):
    def test_review_queue_samples_by_family_difficulty_source_and_verifier(self) -> None:
        queue = build_seed_review_queue(
            [
                _scenario(
                    "Add parser tests.",
                    task_family="test_authoring",
                    difficulty=2,
                    source_method="repository_grounded_synthetic",
                    verifier_types=["hidden_command"],
                ),
                _scenario(
                    "Fix docs example.",
                    task_family="docs_examples",
                    difficulty=3,
                    source_method="repository_grounded_synthetic",
                    verifier_types=["doctest", "example_command"],
                ),
            ],
            sample_per_stratum=1,
        )

        strata = set(queue.stratum_counts)
        self.assertEqual(queue.total_scenarios, 2)
        self.assertEqual(queue.selected, 3)
        self.assertIn(
            "family=test_authoring|difficulty=2|source=repository_grounded_synthetic"
            "|verifier=hidden_command",
            strata,
        )
        self.assertIn(
            "family=docs_examples|difficulty=3|source=repository_grounded_synthetic"
            "|verifier=doctest",
            strata,
        )
        self.assertTrue(all(record["review_questions"] for record in queue.records))

    def test_review_queue_respects_max_records(self) -> None:
        queue = build_seed_review_queue(
            [
                _scenario(
                    "Fix docs example.",
                    task_family="docs_examples",
                    difficulty=3,
                    source_method="repository_grounded_synthetic",
                    verifier_types=["doctest", "example_command"],
                )
            ],
            max_records=1,
        )

        self.assertEqual(queue.selected, 1)
        self.assertEqual(len(queue.records), 1)

    def test_cli_review_queue_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "registry"
            output = Path(directory) / "review.jsonl"
            ScenarioRegistry(root).add_scenario(
                _scenario(
                    "Add parser tests.",
                    task_family="test_authoring",
                    difficulty=2,
                    source_method="repository_grounded_synthetic",
                    verifier_types=["hidden_command"],
                )
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "registry",
                        "review-queue",
                        "--root",
                        str(root),
                        "--output",
                        str(output),
                        "--overwrite",
                    ]
                )

            payload = json.loads(stdout.getvalue())
            records = load_human_reviews(output)
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["selected"], 1)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["task_family"], "test_authoring")


def _scenario(
    query: str,
    *,
    task_family: str,
    difficulty: int,
    source_method: str,
    verifier_types: list[str],
) -> Scenario:
    return Scenario(
        query_seed=QuerySeed(
            PublicTaskContext(
                query,
                context={"repository": "example/tool"},
            ),
            license="MIT",
            task_family=task_family,
            difficulty=difficulty,
            source_method=source_method,
            verifier_types=verifier_types,
            coverage_tags=["language:python"],
            metadata={
                "source_name": "curated",
                "source_instance_id": query.lower().replace(" ", "-"),
            },
        ),
        environment=EnvironmentSpec(
            name="fixture",
            version="1",
            source_uri="https://github.com/example/tool.git",
            source_revision="d" * 40,
        ),
    )


if __name__ == "__main__":
    unittest.main()
