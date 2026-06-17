from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

from easy_agentic_data.config import PipelineConfig
from easy_agentic_data.exporters import (
    preference_to_training,
    trajectory_to_sft,
    write_json,
    write_jsonl,
)
from easy_agentic_data.generation import EvolTaskGenerator, SelfInstructTaskGenerator
from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import PreferencePair, Task, Trajectory, utc_now
from easy_agentic_data.runner import AgentRunner
from easy_agentic_data.tools import ToolRegistry
from easy_agentic_data.verification import VerificationSuite


class SynthesisPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        client: LLMClient,
        tools: ToolRegistry,
        verification: VerificationSuite,
    ) -> None:
        self.config = config
        self.client = client
        self.tools = tools
        self.verification = verification

    def run(self) -> dict[str, int | str]:
        random.seed(self.config.random_seed)
        started_at = utc_now()
        generation = self.config.generation
        task_generator = SelfInstructTaskGenerator(self.client)
        tasks = task_generator.generate(generation.num_tasks, generation.seed_topics)
        if generation.evolve_rounds > 0:
            tasks = EvolTaskGenerator(self.client).evolve(tasks, generation.evolve_rounds)

        runner = AgentRunner(self.client, self.tools, generation.max_turns)
        trajectories: list[Trajectory] = []
        for task in tasks:
            for rollout_index in range(generation.rollouts_per_task):
                trajectory = runner.run(task, rollout_index)
                trajectories.append(self.verification.evaluate(trajectory))

        selected = self._select_best(trajectories)
        preferences = self._build_preferences(trajectories)
        summary = {
            "run_name": self.config.run_name,
            "started_at": started_at,
            "finished_at": utc_now(),
            "tasks": len(tasks),
            "trajectories": len(trajectories),
            "selected": len(selected),
            "preferences": len(preferences),
        }
        self._write_outputs(tasks, trajectories, selected, preferences, summary)
        return summary

    def _select_best(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        grouped = self._group_by_task(trajectories)
        selected: list[Trajectory] = []
        for candidates in grouped.values():
            best = max(candidates, key=lambda trajectory: trajectory.reward)
            if best.reward >= self.config.generation.min_reward:
                selected.append(best)
        return selected

    def _build_preferences(self, trajectories: list[Trajectory]) -> list[PreferencePair]:
        pairs: list[PreferencePair] = []
        for candidates in self._group_by_task(trajectories).values():
            ranked = sorted(candidates, key=lambda trajectory: trajectory.reward, reverse=True)
            if len(ranked) < 2:
                continue
            margin = ranked[0].reward - ranked[-1].reward
            if margin <= 0:
                continue
            pairs.append(
                PreferencePair(
                    task=ranked[0].task,
                    chosen=ranked[0],
                    rejected=ranked[-1],
                    margin=margin,
                )
            )
        return pairs

    @staticmethod
    def _group_by_task(trajectories: list[Trajectory]) -> dict[str, list[Trajectory]]:
        grouped: dict[str, list[Trajectory]] = {}
        for trajectory in trajectories:
            grouped.setdefault(trajectory.task.task_id, []).append(trajectory)
        return grouped

    def _write_outputs(
        self,
        tasks: list[Task],
        trajectories: list[Trajectory],
        selected: list[Trajectory],
        preferences: list[PreferencePair],
        summary: dict[str, int | str],
    ) -> None:
        output = Path(self.config.output.directory)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "manifest.json", {"config": asdict(self.config), "summary": summary})
        write_jsonl(output / "tasks.jsonl", (asdict(task) for task in tasks))
        write_jsonl(
            output / "trajectories.jsonl",
            (trajectory.to_dict() for trajectory in trajectories),
        )
        records = getattr(self.client, "records", None)
        if records is not None:
            write_jsonl(output / "llm_calls.jsonl", iter(records))
        if self.config.output.export_sft:
            write_jsonl(output / "sft.jsonl", (trajectory_to_sft(item) for item in selected))
        if self.config.output.export_preferences:
            write_jsonl(
                output / "preferences.jsonl",
                (preference_to_training(item) for item in preferences),
            )
