from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from easy_agentic_data.models import PreferencePair, Trajectory


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def trajectory_to_sft(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "id": trajectory.trajectory_id,
        "messages": [message.to_training_dict() for message in trajectory.messages],
        "metadata": {
            "task_id": trajectory.task.task_id,
            "category": trajectory.task.category,
            "difficulty": trajectory.task.difficulty,
            "reward": trajectory.reward,
            "tool_events": [
                {
                    "name": event.name,
                    "arguments": event.arguments,
                    "output": event.output,
                    "error": event.error,
                }
                for event in trajectory.tool_events
            ],
        },
    }


def preference_to_training(pair: PreferencePair) -> dict[str, Any]:
    prompt = [
        message.to_training_dict()
        for message in pair.chosen.messages
        if message.role in {"system", "user"}
    ]
    return {
        "id": pair.pair_id,
        "prompt": prompt,
        "chosen": [
            message.to_training_dict()
            for message in pair.chosen.messages
            if message.role not in {"system", "user"}
        ],
        "rejected": [
            message.to_training_dict()
            for message in pair.rejected.messages
            if message.role not in {"system", "user"}
        ],
        "metadata": {
            "task_id": pair.task.task_id,
            "chosen_reward": pair.chosen.reward,
            "rejected_reward": pair.rejected.reward,
            "margin": pair.margin,
        },
    }
