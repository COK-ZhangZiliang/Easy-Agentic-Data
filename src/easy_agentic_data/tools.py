from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from easy_agentic_data.models import ToolEvent


ToolHandler = Callable[..., Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler

    def api_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: List[Tool] | None = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> List[Dict[str, Any]]:
        return [tool.api_schema() for tool in self._tools.values()]

    def execute(self, call_id: str, name: str, arguments: Dict[str, Any]) -> ToolEvent:
        started = time.perf_counter()
        tool = self._tools.get(name)
        if tool is None:
            return ToolEvent(
                call_id=call_id,
                name=name,
                arguments=arguments,
                error=f"Unknown tool: {name}",
            )
        try:
            output = tool.handler(**arguments)
            error = None
        except Exception as exc:  # Tool failures belong in the trajectory, not the runner.
            output = None
            error = f"{type(exc).__name__}: {exc}"
        return ToolEvent(
            call_id=call_id,
            name=name,
            arguments=arguments,
            output=output,
            error=error,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def calculator(operation: str, a: float, b: float) -> Dict[str, float | str]:
    operations: Dict[str, Callable[[float, float], float]] = {
        "add": lambda left, right: left + right,
        "subtract": lambda left, right: left - right,
        "multiply": lambda left, right: left * right,
        "divide": lambda left, right: left / right,
    }
    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")
    result = operations[operation](a, b)
    return {"operation": operation, "a": a, "b": b, "result": result}


def default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="calculator",
                description="Perform one arithmetic operation on two numbers.",
                parameters={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                        },
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                    "additionalProperties": False,
                },
                handler=calculator,
            )
        ]
    )

