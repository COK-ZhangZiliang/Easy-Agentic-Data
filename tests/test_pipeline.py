import json
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.cli import build_pipeline
from easy_agentic_data.config import GenerationConfig, OutputConfig, PipelineConfig
from easy_agentic_data.models import Message, Task, Trajectory
from easy_agentic_data.pipeline import SynthesisPipeline


class PipelineTests(unittest.TestCase):
    def test_pipeline_writes_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = PipelineConfig(
                run_name="test",
                generation=GenerationConfig(
                    num_tasks=2,
                    rollouts_per_task=1,
                    evolve_rounds=1,
                    max_turns=3,
                ),
                output=OutputConfig(directory=str(output)),
            )

            summary = build_pipeline(config).run()

            self.assertEqual(summary["tasks"], 2)
            self.assertEqual(summary["selected"], 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["random_seed"], 42)
            sft_rows = (output / "sft.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(sft_rows), 2)
            self.assertTrue((output / "llm_calls.jsonl").exists())

    def test_equal_reward_candidates_do_not_create_preference(self) -> None:
        task = Task(instruction="Answer")
        candidates = [
            Trajectory(
                task,
                [Message("user", "Answer"), Message("assistant", str(index))],
                reward=0.8,
                metadata={"rollout_index": index},
            )
            for index in range(2)
        ]
        pipeline = object.__new__(SynthesisPipeline)

        pairs = pipeline._build_preferences(candidates)

        self.assertEqual(pairs, [])
        self.assertNotEqual(candidates[0].trajectory_id, candidates[1].trajectory_id)


if __name__ == "__main__":
    unittest.main()
