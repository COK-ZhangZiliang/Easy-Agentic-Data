import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from easy_agentic_data.cli import main
from easy_agentic_data.synthesis_tiers import (
    default_synthesis_tiers,
    run_complex_synthetic_demo,
)
from easy_agentic_data.traces import load_trace


class SynthesisTierTests(unittest.TestCase):
    def test_default_tiers_describe_local_and_registry_paths(self) -> None:
        tiers = default_synthesis_tiers()

        self.assertEqual(
            [tier.tier_id for tier in tiers],
            ["complex_synthetic", "registry_backed"],
        )
        self.assertIn("HeadlessAgent", tiers[0].runtime)
        self.assertIn("Docker", tiers[1].runtime)

    def test_complex_synthetic_demo_writes_replayable_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = run_complex_synthetic_demo(directory)
            root = Path(directory)
            trace_text = (root / "trace.jsonl").read_text(encoding="utf-8")
            trace = load_trace(root / "trace.jsonl")
            sft = json.loads((root / "sft.json").read_text(encoding="utf-8"))
            episode = json.loads((root / "rl_episode.json").read_text(encoding="utf-8"))
            report = json.loads((root / "report.json").read_text(encoding="utf-8"))

        self.assertTrue(summary["success"])
        self.assertEqual(summary["reward"], 1)
        self.assertGreaterEqual(summary["tool_calls"], 8)
        self.assertGreaterEqual(summary["user_turns"], 1)
        self.assertGreaterEqual(len(trace.events), 20)
        self.assertIn("reasoning_content", trace_text)
        self.assertNotIn("tests/test_hidden_service.py", trace_text)
        self.assertEqual(sft["reward"], 1)
        self.assertTrue(any(message.get("reasoning_content") for message in sft["messages"]))
        self.assertTrue(episode["success"])
        self.assertEqual(report["reward"], 1)

    def test_cli_lists_tiers_and_runs_complex_demo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tiers_stdout = io.StringIO()
            with redirect_stdout(tiers_stdout):
                self.assertEqual(main(["synthesis", "tiers"]), 0)
            payload = json.loads(tiers_stdout.getvalue())

            run_stdout = io.StringIO()
            with redirect_stdout(run_stdout):
                self.assertEqual(main(["synthesis", "complex-demo", "--output", directory]), 0)
            summary = json.loads(run_stdout.getvalue())

        self.assertEqual(payload[0]["tier_id"], "complex_synthetic")
        self.assertEqual(summary["tier"], "complex_synthetic")
        self.assertTrue(summary["success"])


if __name__ == "__main__":
    unittest.main()
