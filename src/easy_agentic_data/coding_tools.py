from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from easy_agentic_data.policy import PolicyDecision, ToolPolicy
from easy_agentic_data.sandbox import Sandbox

MAX_LIST_FILES = 500
MAX_READ_CHARS = 40_000
MAX_SEARCH_MATCHES = 200
MAX_SEARCH_LINE_CHARS = 500
MAX_DIFF_CHARS = 40_000


@dataclass
class CodingToolResult:
    output: Any = None
    error: str | None = None
    policy: PolicyDecision | None = None


SCHEMAS: dict[str, dict[str, Any]] = {
    "list_files": {"required": [], "properties": {"path": str}},
    "read_file": {"required": ["path"], "properties": {"path": str, "offset": int, "limit": int}},
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
    "list_files": (
        "List workspace files under an optional relative directory, excluding Git control data, "
        "with bounded output."
    ),
    "read_file": (
        "Read a UTF-8 text file at a workspace-relative path with bounded output. Optional "
        "offset and limit select 1-based line ranges."
    ),
    "search_files": (
        "Search text files for an exact string and return matching lines. Binary or unreadable "
        "files are skipped."
    ),
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
            files = [
                path
                for path in self.sandbox.list_files(arguments.get("path", "."))
                if ".git" not in path.split("/")
            ]
            return {
                "files": files[:MAX_LIST_FILES],
                "file_count": len(files),
                "truncated": len(files) > MAX_LIST_FILES,
            }
        if name == "read_file":
            content = self.sandbox.read(arguments["path"])
            content = _slice_lines(
                content,
                offset=arguments.get("offset"),
                limit=arguments.get("limit"),
            )
            return {
                "path": arguments["path"],
                "content": content[:MAX_READ_CHARS],
                "chars": len(content),
                "truncated": len(content) > MAX_READ_CHARS,
            }
        if name == "search_files":
            return self._search_files(arguments["query"])
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
            value = self.sandbox.diff()
            return {
                "diff": value[:MAX_DIFF_CHARS],
                "chars": len(value),
                "truncated": len(value) > MAX_DIFF_CHARS,
            }
        if name == "ask_user":
            return {"question": arguments["question"], "awaiting_user": True}
        raise ValueError(f"Unknown coding tool: {name}")

    def _search_files(self, query: str) -> dict[str, Any]:
        grep = self.sandbox.execute(
            [
                "grep",
                "-R",
                "-n",
                "-I",
                "-m",
                str(MAX_SEARCH_MATCHES),
                "--",
                query,
                ".",
            ]
        )
        if grep.exit_code in {0, 1}:
            matches = _parse_grep_matches(grep.stdout)
            return {
                "matches": matches[:MAX_SEARCH_MATCHES],
                "match_count": len(matches),
                "skipped_count": 0,
                "truncated": len(matches) > MAX_SEARCH_MATCHES or grep.truncated,
            }
        if grep.exit_code == 127 and "Unsupported test command" in grep.stderr:
            return self._search_files_by_reading(query)
        return {
            "matches": [],
            "match_count": 0,
            "skipped_count": 0,
            "truncated": False,
            "grep_exit_code": grep.exit_code,
            "grep_error": grep.stderr[:1_000],
        }

    def _search_files_by_reading(self, query: str) -> dict[str, Any]:
        matches = []
        skipped = []
        for path in self.sandbox.list_files():
            try:
                content = self.sandbox.read(path)
            except (OSError, UnicodeDecodeError) as exc:
                skipped.append({"path": path, "reason": type(exc).__name__})
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if query in line:
                    matches.append(_search_match(path, number, line))
                if len(matches) >= MAX_SEARCH_MATCHES:
                    return {
                        "matches": matches,
                        "match_count": len(matches),
                        "skipped_count": len(skipped),
                        "truncated": True,
                    }
        return {
            "matches": matches,
            "match_count": len(matches),
            "skipped_count": len(skipped),
            "truncated": False,
        }


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
    if name == "read_file":
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        if offset is not None and offset < 1:
            raise ValueError("Tool argument offset must be >= 1")
        if limit is not None and limit < 1:
            raise ValueError("Tool argument limit must be >= 1")


def _json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    types = {str: "string", list: "array", int: "integer"}
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


def _slice_lines(content: str, *, offset: int | None, limit: int | None) -> str:
    if offset is None and limit is None:
        return content
    lines = content.splitlines(keepends=True)
    start = (offset - 1) if offset is not None else 0
    stop = start + limit if limit is not None else None
    return "".join(lines[start:stop])


def _parse_grep_matches(output: str) -> list[dict[str, Any]]:
    matches = []
    for line in output.splitlines():
        path, number, text = _split_grep_line(line)
        if path:
            matches.append(_search_match(path, number, text))
    return matches


def _split_grep_line(line: str) -> tuple[str, int, str]:
    parts = line.split(":", 2)
    if len(parts) != 3:
        return "", 0, ""
    path, number, text = parts
    try:
        line_number = int(number)
    except ValueError:
        return "", 0, ""
    return path.removeprefix("./"), line_number, text


def _search_match(path: str, number: int, line: str) -> dict[str, Any]:
    return {
        "path": path,
        "line": number,
        "text": line[:MAX_SEARCH_LINE_CHARS],
        "text_truncated": len(line) > MAX_SEARCH_LINE_CHARS,
    }
