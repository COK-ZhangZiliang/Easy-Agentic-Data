from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Iterable, List

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import Message, Task


TASK_DESIGNER_PROMPT = """\
TASK_DESIGNER
Create diverse, realistic tasks for training an LLM agent. Tasks must be solvable with the
available tools, include explicit constraints, and avoid private or unsafe data. Return only a
JSON array. Each item must contain instruction, category, difficulty (1-5), constraints,
expected_tools, and may contain reference and metadata.
"""

TASK_EVOLVER_PROMPT = """\
TASK_EVOLVER
Increase the task's useful difficulty while preserving solvability. Add one meaningful constraint,
reasoning step, or tool dependency. Do not merely make the wording longer. Return only one JSON
object with the same schema.
"""


class SelfInstructTaskGenerator:
    """Generate task blueprints from topic seeds, inspired by Self-Instruct and AgentInstruct."""

    def __init__(self, client: LLMClient, batch_size: int = 8) -> None:
        self.client = client
        self.batch_size = batch_size

    def generate(self, count: int, topics: Iterable[str]) -> List[Task]:
        topics_text = ", ".join(topics)
        tasks: List[Task] = []
        while len(tasks) < count:
            requested = min(self.batch_size, count - len(tasks))
            response = self.client.complete(
                [
                    Message("system", TASK_DESIGNER_PROMPT),
                    Message(
                        "user",
                        f"COUNT={requested} Topics: {topics_text}. "
                        "Available tools: calculator.",
                    ),
                ],
                temperature=0.9,
            )
            payload = _parse_json(response.message.content)
            if not isinstance(payload, list):
                raise ValueError("Task generator must return a JSON array")
            tasks.extend(Task(**_normalize_task(item)) for item in payload)
        return tasks[:count]


class EvolTaskGenerator:
    """Apply controlled Evol-Instruct-style mutations to task blueprints."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def evolve(self, tasks: Iterable[Task], rounds: int) -> List[Task]:
        evolved = list(tasks)
        for _ in range(rounds):
            next_round: List[Task] = []
            for task in evolved:
                source = asdict(task)
                source.pop("task_id", None)
                response = self.client.complete(
                    [
                        Message("system", TASK_EVOLVER_PROMPT),
                        Message("user", json.dumps(source, ensure_ascii=True)),
                    ],
                    temperature=0.7,
                )
                payload = _parse_json(response.message.content)
                if not isinstance(payload, dict):
                    raise ValueError("Task evolver must return one JSON object")
                metadata = dict(payload.get("metadata", {}))
                metadata["parent_task_id"] = task.task_id
                payload["metadata"] = metadata
                next_round.append(Task(**_normalize_task(payload)))
            evolved = next_round
        return evolved


def _parse_json(content: str | None) -> Any:
    if not content:
        raise ValueError("Expected JSON content, received an empty response")
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(text)


def _normalize_task(item: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "instruction",
        "category",
        "difficulty",
        "constraints",
        "expected_tools",
        "reference",
        "metadata",
    }
    normalized = {key: value for key, value in item.items() if key in allowed}
    if not normalized.get("instruction"):
        raise ValueError("Generated task is missing instruction")
    normalized["difficulty"] = max(1, min(5, int(normalized.get("difficulty", 1))))
    normalized.setdefault("category", "general")
    normalized.setdefault("constraints", [])
    normalized.setdefault("expected_tools", [])
    normalized.setdefault("metadata", {})
    return normalized

