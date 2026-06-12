from __future__ import annotations

from typing import Dict

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import Message


def probe_tool_calling(client: LLMClient) -> Dict[str, object]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "capability_probe",
                "description": "Return the supplied value.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    response = client.complete(
        [
            Message("system", "Call capability_probe with value 'ok'."),
            Message("user", "Run the capability probe."),
        ],
        tools=tools,
        temperature=0.0,
        max_tokens=128,
    )
    supported = bool(response.message.tool_calls)
    return {
        "supported": supported,
        "model": response.model,
        "reason": "Tool call returned" if supported else "Model returned no tool call",
    }
