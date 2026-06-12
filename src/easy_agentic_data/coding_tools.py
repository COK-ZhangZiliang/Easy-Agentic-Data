from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from easy_agentic_data.policy import PolicyDecision, ToolPolicy
from easy_agentic_data.sandbox import Sandbox


@dataclass
class CodingToolResult:
    output: Any = None
    error: str | None = None
    policy: PolicyDecision | None = None


SCHEMAS: Dict[str, Dict[str, Any]] = {
    "list_files": {"required": [], "properties": {"path": str}},
    "read_file": {"required": ["path"], "properties": {"path": str}},
    "search_files": {"required": ["query"], "properties": {"query": str}},
    "apply_patch": {"required": ["path", "old", "new"], "properties": {"path": str, "old": str, "new": str}},
    "run_command": {"required": ["command"], "properties": {"command": list}},
    "git_status": {"required": [], "properties": {}},
    "git_diff": {"required": [], "properties": {}},
    "ask_user": {"required": ["question"], "properties": {"question": str}},
}


class CodingToolRuntime:
    def __init__(self, sandbox: Sandbox, policy: ToolPolicy) -> None:
        self.sandbox = sandbox
        self.policy = policy

    def schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Sandboxed coding operation: {name}.",
                    "parameters": _json_schema(schema),
                },
            }
            for name, schema in SCHEMAS.items()
        ]

    def execute(self, name: str, arguments: Dict[str, Any]) -> CodingToolResult:
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

    def _execute_allowed(self, name: str, arguments: Dict[str, Any]) -> Any:
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
            self.sandbox.write(arguments["path"], content.replace(arguments["old"], arguments["new"], 1))
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


def _validate(name: str, arguments: Dict[str, Any]) -> None:
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


def _json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    types = {str: "string", list: "array"}
    return {
        "type": "object",
        "properties": {key: {"type": types[value]} for key, value in schema["properties"].items()},
        "required": schema["required"],
        "additionalProperties": False,
    }
