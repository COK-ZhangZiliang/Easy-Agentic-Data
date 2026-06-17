from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from easy_agentic_data.models import LLMResponse, Message


class MockLLMClient:
    """Deterministic local backend used by examples and contract tests."""

    model = "mock-agent"

    def __init__(self) -> None:
        self._task_counter = 0

    def complete(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens, response_format
        system = next(
            (message.content or "" for message in messages if message.role == "system"), ""
        )
        last = messages[-1]

        if "TASK_DESIGNER" in system:
            self._task_counter += 1
            count = _extract_requested_count(last.content or "")
            tasks = []
            for offset in range(count):
                number = self._task_counter + offset
                tasks.append(
                    {
                        "instruction": (
                            f"Use the calculator to add {number} and {number + 2}, "
                            "then explain the result."
                        ),
                        "category": "tool_use",
                        "difficulty": 1,
                        "constraints": ["Use the calculator tool exactly once."],
                        "expected_tools": ["calculator"],
                    }
                )
            self._task_counter += max(0, count - 1)
            content = json.dumps(tasks)
            return LLMResponse(Message("assistant", content), self.model)

        if "TASK_EVOLVER" in system:
            content = last.content or "{}"
            task = json.loads(content[content.find("{") :])
            task["instruction"] += " Also state which operands were used."
            task["difficulty"] = min(5, int(task.get("difficulty", 1)) + 1)
            task.setdefault("constraints", []).append("State the operands in the final answer.")
            return LLMResponse(Message("assistant", json.dumps(task)), self.model)

        if "SEMANTIC_JUDGE" in system:
            content = json.dumps(
                {"passed": True, "score": 0.9, "reason": "The answer follows the task."}
            )
            return LLMResponse(Message("assistant", content), self.model)

        tool_result = next(
            (message.content for message in reversed(messages) if message.role == "tool"),
            None,
        )
        if tools and tool_result is None:
            left, right = _numbers_from_text(last.content or "")
            tool_call = {
                "id": f"mock_call_{len(messages)}",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"operation": "add", "a": left, "b": right}),
                },
            }
            return LLMResponse(
                Message(role="assistant", content=None, tool_calls=[tool_call]),
                self.model,
            )

        if tool_result is not None:
            parsed = json.loads(tool_result)
            content = (
                f"The result is {parsed['result']}. "
                f"The operands were {parsed['a']} and {parsed['b']}."
            )
            return LLMResponse(Message("assistant", content), self.model)

        return LLMResponse(Message("assistant", "Completed."), self.model)


def _extract_requested_count(text: str) -> int:
    match = re.search(r"\bCOUNT=(\d+)\b", text)
    if match is None:
        return 1
    return max(1, int(match.group(1)))


def _numbers_from_text(text: str) -> tuple[float, float]:
    numbers: list[float] = []
    for token in text.replace(",", " ").split():
        cleaned = token.strip(".,;:!?")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    if len(numbers) < 2:
        return 1.0, 1.0
    return numbers[0], numbers[1]
