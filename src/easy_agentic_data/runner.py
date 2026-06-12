from __future__ import annotations

import json
from typing import List

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import Message, Task, ToolEvent, Trajectory
from easy_agentic_data.tools import ToolRegistry


AGENT_SYSTEM_PROMPT = """\
You are an agent solving a training task. Use tools when they improve correctness. Respect every
constraint, inspect tool errors, and finish with a concise answer grounded in observed results.
"""


class AgentRunner:
    def __init__(self, client: LLMClient, tools: ToolRegistry, max_turns: int = 8) -> None:
        self.client = client
        self.tools = tools
        self.max_turns = max_turns

    def run(self, task: Task, rollout_index: int = 0) -> Trajectory:
        messages: List[Message] = [
            Message("system", AGENT_SYSTEM_PROMPT),
            Message("user", task.instruction),
        ]
        events: List[ToolEvent] = []

        try:
            for turn in range(self.max_turns):
                response = self.client.complete(messages, tools=self.tools.schemas())
                assistant = response.message
                messages.append(assistant)
                if not assistant.tool_calls:
                    return Trajectory(
                        task=task,
                        messages=messages,
                        tool_events=events,
                        metadata={
                            "model": response.model,
                            "rollout_index": rollout_index,
                            "turns": turn + 1,
                            "usage": response.usage,
                        },
                    )
                for call in assistant.tool_calls:
                    event = self._execute_tool_call(call)
                    events.append(event)
                    result = {
                        "ok": event.error is None,
                        "output": event.output,
                        "error": event.error,
                    }
                    if isinstance(event.output, dict):
                        result.update(event.output)
                    messages.append(
                        Message(
                            role="tool",
                            content=json.dumps(result, ensure_ascii=True),
                            name=event.name,
                            tool_call_id=event.call_id,
                        )
                    )
            return Trajectory(
                task=task,
                messages=messages,
                tool_events=events,
                status="max_turns",
                error=f"Agent exceeded {self.max_turns} turns",
                metadata={"rollout_index": rollout_index},
            )
        except Exception as exc:
            return Trajectory(
                task=task,
                messages=messages,
                tool_events=events,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                metadata={"rollout_index": rollout_index},
            )

    def _execute_tool_call(self, call: dict) -> ToolEvent:
        function = call.get("function", {})
        name = function.get("name", "")
        raw_arguments = function.get("arguments", "{}")
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return ToolEvent(
                call_id=call.get("id", "unknown"),
                name=name,
                arguments={},
                error=f"Invalid tool arguments: {exc}",
            )
        return self.tools.execute(call.get("id", "unknown"), name, arguments)

