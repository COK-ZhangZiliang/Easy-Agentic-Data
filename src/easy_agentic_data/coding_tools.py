from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from easy_agentic_data.policy import PolicyDecision, ToolPolicy
from easy_agentic_data.sandbox import Sandbox


@dataclass
class CodingToolResult:
    output: Any = None
    error: str | None = None
    policy: PolicyDecision | None = None


SCHEMAS: dict[str, dict[str, Any]] = {
    "list_files": {"required": [], "properties": {"path": str}},
    "read_file": {"required": ["path"], "properties": {"path": str}},
    "search_files": {"required": ["query"], "properties": {"query": str}},
    "apply_patch": {
        "required": ["path", "old", "new"],
        "properties": {"path": str, "old": str, "new": str},
    },
    "run_command": {"required": ["command"], "properties": {"command": list}},
    "git_status": {"required": [], "properties": {}},
    "git_diff": {"required": [], "properties": {}},
    "ask_user": {"required": ["question"], "properties": {"question": str}},
}

DESCRIPTIONS = {
    "list_files": "List workspace files under an optional relative directory.",
    "read_file": "Read a UTF-8 text file at a workspace-relative path.",
    "search_files": "Search text files for an exact string and return matching lines.",
    "apply_patch": (
        "Replace the first exact occurrence of old with new in one workspace file. "
        "Read the file first so old contains precise context."
    ),
    "run_command": (
        "Run an allowed command as an argument array inside the sandbox. "
        "Use this for focused tests and validation."
    ),
    "git_status": "Show the short Git status for the sandboxed workspace.",
    "git_diff": "Show the current workspace diff for review after edits.",
    "ask_user": "Ask one concise question only when required information is unavailable.",
}


class CodingToolRuntime:
    def __init__(self, sandbox: Sandbox, policy: ToolPolicy) -> None:
        self.sandbox = sandbox
        self.policy = policy

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": DESCRIPTIONS[name],
                    "parameters": _json_schema(schema),
                },
            }
            for name, schema in SCHEMAS.items()
            if name in self.policy.allowed_tools
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> CodingToolResult:
        try:
            _validate(name, arguments)
        except (TypeError, ValueError) as exc:
            return CodingToolResult(error=str(exc))
        decision = self.policy.evaluate(name, arguments)
        if not decision.allowed:
            return CodingToolResult(error=decision.reason, policy=decision)
        try:
            output = self._execute_allowed(name, arguments)
            return CodingToolResult(output=output, policy=decision)
        except Exception as exc:
            return CodingToolResult(error=f"{type(exc).__name__}: {exc}", policy=decision)

    def _execute_allowed(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "list_files":
            return self.sandbox.list_files(arguments.get("path", "."))
        if name == "read_file":
            return self.sandbox.read(arguments["path"])
        if name == "search_files":
            query = arguments["query"]
            return [
                {"path": path, "line": number, "text": line}
                for path in self.sandbox.list_files()
                for number, line in enumerate(self.sandbox.read(path).splitlines(), 1)
                if query in line
            ]
        if name == "apply_patch":
            content = self.sandbox.read(arguments["path"])
            if arguments["old"] not in content:
                raise ValueError("Patch context was not found")
            self.sandbox.write(
                arguments["path"], content.replace(arguments["old"], arguments["new"], 1)
            )
            return {"state_hash": self.sandbox.state_hash()}
        if name == "run_command":
            return self.sandbox.execute(arguments["command"]).__dict__
        if name == "git_status":
            return self.sandbox.execute(["git", "status", "--short"]).__dict__
        if name == "git_diff":
            return self.sandbox.diff()
        if name == "ask_user":
            return {"question": arguments["question"], "awaiting_user": True}
        raise ValueError(f"Unknown coding tool: {name}")


def _validate(name: str, arguments: dict[str, Any]) -> None:
    if name not in SCHEMAS:
        raise ValueError(f"Unknown coding tool: {name}")
    schema = SCHEMAS[name]
    missing = set(schema["required"]) - arguments.keys()
    if missing:
        raise ValueError(f"Missing tool arguments: {sorted(missing)}")
    extra = arguments.keys() - schema["properties"].keys()
    if extra:
        raise ValueError(f"Unexpected tool arguments: {sorted(extra)}")
    for key, expected in schema["properties"].items():
        if key in arguments and not isinstance(arguments[key], expected):
            raise TypeError(f"Tool argument {key} must be {expected.__name__}")


def _json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    types = {str: "string", list: "array"}
    properties = {}
    for key, value in schema["properties"].items():
        property_schema = {"type": types[value]}
        if value is list:
            property_schema["items"] = {"type": "string"}
        properties[key] = property_schema
    return {
        "type": "object",
        "properties": properties,
        "required": schema["required"],
        "additionalProperties": False,
    }
