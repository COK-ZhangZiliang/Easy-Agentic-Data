from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import Message, Task

TASK_DESIGNER_PROMPT = """\
TASK_DESIGNER
Create realistic, self-contained tasks for training a tool-using LLM agent.

Rules:
- Produce exactly the requested number of tasks.
- Every task must be solvable using only the listed tools and information in its instruction.
- Make constraints objectively checkable and consistent with the instruction.
- Use only listed tool names in expected_tools. Do not invent capabilities.
- Keep tasks meaningfully different from one another.
- Do not request private data, credentials, network access, or unsafe actions.

Return one JSON object and no prose. The tasks array must follow this example:
{
  "tasks": [
    {
      "instruction": "Use the calculator to add 17 and 25, then report the result.",
      "category": "calculation",
      "difficulty": 1,
      "constraints": ["Use the calculator exactly once.", "State both operands."],
      "expected_tools": ["calculator"],
      "reference": "42",
      "metadata": {"skill": "arithmetic"}
    }
  ]
}
"""

TASK_EVOLVER_PROMPT = """\
TASK_EVOLVER
Increase the task's useful difficulty by exactly one level while preserving its intent,
solvability, listed tools, and reference correctness. Add one meaningful and objectively
checkable constraint or reasoning step. Do not add unavailable tools, hidden information,
subjective grading criteria, or decorative verbosity.

Return one JSON object and no prose, using the same fields as the input task:
{"instruction": "...", "category": "...", "difficulty": 2, "constraints": [],
 "expected_tools": [], "reference": null, "metadata": {}}
"""

JSON_OBJECT_FORMAT = {"type": "json_object"}


class SelfInstructTaskGenerator:
    """Generate task blueprints from topic seeds, inspired by Self-Instruct and AgentInstruct."""

    def __init__(self, client: LLMClient, batch_size: int = 8) -> None:
        self.client = client
        self.batch_size = batch_size

    def generate(self, count: int, topics: Iterable[str]) -> list[Task]:
        topics_text = ", ".join(topics)
        tasks: list[Task] = []
        while len(tasks) < count:
            requested = min(self.batch_size, count - len(tasks))
            response = self.client.complete(
                [
                    Message("system", TASK_DESIGNER_PROMPT),
                    Message(
                        "user",
                        f"Generate JSON now. COUNT={requested}. Topics: {topics_text}. "
                        "Available tools: calculator.",
                    ),
                ],
                temperature=0.9,
                response_format=JSON_OBJECT_FORMAT,
            )
            payload = _parse_json(response.message.content)
            if isinstance(payload, dict):
                payload = payload.get("tasks")
            if not isinstance(payload, list):
                raise ValueError("Task generator must return a JSON object containing tasks")
            if not payload:
                raise ValueError("Task generator returned an empty tasks array")
            tasks.extend(Task(**_normalize_task(item)) for item in payload)
        return tasks[:count]


class EvolTaskGenerator:
    """Apply controlled Evol-Instruct-style mutations to task blueprints."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def evolve(self, tasks: Iterable[Task], rounds: int) -> list[Task]:
        evolved = list(tasks)
        for _ in range(rounds):
            next_round: list[Task] = []
            for task in evolved:
                source = asdict(task)
                source.pop("task_id", None)
                response = self.client.complete(
                    [
                        Message("system", TASK_EVOLVER_PROMPT),
                        Message(
                            "user",
                            "Evolve this task and return JSON:\n"
                            + json.dumps(source, ensure_ascii=True),
                        ),
                    ],
                    temperature=0.7,
                    response_format=JSON_OBJECT_FORMAT,
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


def _normalize_task(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Generated task must be a JSON object")
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
