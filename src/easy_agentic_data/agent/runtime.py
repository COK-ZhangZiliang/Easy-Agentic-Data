from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.llm.observability import prompt_token_upper_bound
from easy_agentic_data.models import Message
from easy_agentic_data.scenarios import ScenarioInstance
from easy_agentic_data.traces import EventType, TerminationReason, TraceRecorder

DEFAULT_SYSTEM_PROMPT = (
    "You are a headless coding agent operating inside a restricted workspace.\n\n"
    "Follow this protocol:\n"
    "1. Inspect relevant files and repository state before editing.\n"
    "2. Make the smallest change that fully satisfies the user request.\n"
    "3. Use only available tools and paths inside the workspace. Never invent results.\n"
    "4. After editing, run the narrowest relevant validation, then inspect the diff.\n"
    "5. If validation passes and the diff matches the requested fix, stop and summarize.\n"
    "6. If a tool fails, diagnose the error and retry with a corrected action.\n"
    "7. Do not call unavailable tools or broad exploratory commands after a focused "
    "test passes.\n"
    "8. Ask the user only when required information cannot be discovered safely.\n"
    "9. Finish with a concise summary of changes and validation actually performed."
)


@dataclass(frozen=True)
class AgentBudgets:
    max_turns: int = 20
    max_tool_calls: int = 50
    max_tokens: int = 100_000
    max_seconds: float = 600.0
    malformed_tool_retries: int = 2


@dataclass(frozen=True)
class AgentRunResult:
    termination_reason: TerminationReason
    final_answer: str
    turns: int
    tool_calls: int
    tokens: int
    final_state_hash: str
    elapsed_ms: float


class HeadlessAgent:
    def __init__(
        self,
        client: LLMClient,
        tools: CodingToolRuntime,
        *,
        budgets: AgentBudgets | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.tools = tools
        self.budgets = budgets or AgentBudgets()
        self.system_prompt = system_prompt

    def run(
        self,
        instance: ScenarioInstance,
        recorder: TraceRecorder,
        *,
        ask_user: Callable[[str], str | None] | None = None,
        finalize: bool = True,
    ) -> AgentRunResult:
        started = time.monotonic()
        messages = [
            Message("system", self.system_prompt),
            Message("user", instance.public_task.query),
        ]
        recorder.start(instance, system_prompt=self.system_prompt)
        recorder.record(
            EventType.USER_MESSAGE,
            {"message_id": "user_0", "content": instance.public_task.query},
        )
        tool_calls = 0
        tool_messages = 0
        user_messages = 1
        tokens = 0
        malformed = 0
        final_answer = ""
        reason = TerminationReason.AGENT_STOP
        tool_schemas = self.tools.schemas()

        for turn in range(self.budgets.max_turns):
            elapsed = time.monotonic() - started
            if elapsed >= self.budgets.max_seconds:
                reason = TerminationReason.TIMEOUT
                break
            remaining_tokens = self.budgets.max_tokens - tokens
            if remaining_tokens <= 0:
                reason = TerminationReason.TOKEN_BUDGET
                break
            prompt_upper_bound = prompt_token_upper_bound(messages, tool_schemas)
            remaining_output_tokens = remaining_tokens - prompt_upper_bound
            if remaining_output_tokens <= 0:
                reason = TerminationReason.TOKEN_BUDGET
                break
            client_max_tokens = getattr(
                self.client,
                "max_tokens",
                remaining_output_tokens,
            )
            if (
                isinstance(client_max_tokens, bool)
                or not isinstance(client_max_tokens, int)
                or client_max_tokens <= 0
            ):
                client_max_tokens = remaining_output_tokens
            requested_output_tokens = min(
                client_max_tokens,
                remaining_output_tokens,
            )
            response = self.client.complete(
                messages,
                tools=tool_schemas,
                max_tokens=requested_output_tokens,
            )
            response_tokens = _usage_total_tokens(response.usage)
            _validate_response_token_bound(
                response.usage,
                prompt_upper_bound=prompt_upper_bound,
                requested_output_tokens=requested_output_tokens,
                remaining_tokens=remaining_tokens,
            )
            tokens += response_tokens
            assistant = response.message
            recorder.record(
                EventType.MODEL_RESPONSE,
                {
                    "message_id": f"assistant_{turn}",
                    "content": assistant.content,
                    "reasoning_content": assistant.reasoning_content,
                    "tool_calls": assistant.tool_calls,
                    "model": response.model,
                    "usage": response.usage,
                },
            )
            messages.append(assistant)
            elapsed = time.monotonic() - started
            if elapsed >= self.budgets.max_seconds:
                tool_messages = _record_cancelled_tool_messages(
                    recorder,
                    messages,
                    assistant.tool_calls,
                    tool_message_index=tool_messages,
                    error="Agent time budget exhausted",
                )
                reason = TerminationReason.TIMEOUT
                break
            if tokens > self.budgets.max_tokens:
                tool_messages = _record_cancelled_tool_messages(
                    recorder,
                    messages,
                    assistant.tool_calls,
                    tool_message_index=tool_messages,
                    error="Agent token budget exhausted",
                )
                reason = TerminationReason.TOKEN_BUDGET
                break
            if not assistant.tool_calls:
                final_answer = assistant.content or ""
                reason = TerminationReason.AGENT_STOP
                break

            for call_index, raw_call in enumerate(assistant.tool_calls):
                if tool_calls >= self.budgets.max_tool_calls:
                    tool_messages = _record_cancelled_tool_messages(
                        recorder,
                        messages,
                        assistant.tool_calls[call_index:],
                        tool_message_index=tool_messages,
                        error="Agent tool budget exhausted",
                    )
                    reason = TerminationReason.TOOL_BUDGET
                    break
                call_id = raw_call["id"]
                function = raw_call["function"]
                name = function["name"]
                try:
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    malformed += 1
                    content = json.dumps(
                        {"ok": False, "error": f"Invalid tool arguments: {exc}"}
                    )
                    _record_tool_message(
                        recorder,
                        messages,
                        message_id=f"tool_{tool_messages}",
                        name=name,
                        tool_call_id=call_id,
                        content=content,
                    )
                    tool_messages += 1
                    if malformed > self.budgets.malformed_tool_retries:
                        tool_messages = _record_cancelled_tool_messages(
                            recorder,
                            messages,
                            assistant.tool_calls[call_index + 1 :],
                            tool_message_index=tool_messages,
                            error="Malformed tool-call retry budget exhausted",
                        )
                        reason = TerminationReason.MALFORMED_TOOL_CALLS
                        break
                    continue

                before_hash = self.tools.sandbox.state_hash()
                recorder.record(
                    EventType.TOOL_REQUESTED,
                    {"call_id": call_id, "name": name, "arguments": arguments},
                )
                decision = self.tools.policy.evaluate(name, arguments)
                recorder.record(
                    EventType.POLICY_DECISION,
                    {"call_id": call_id, "decision": decision.decision, "reason": decision.reason},
                )
                if not decision.allowed:
                    result = {"ok": False, "error": decision.reason}
                    _record_tool_message(
                        recorder,
                        messages,
                        message_id=f"tool_{tool_messages}",
                        name=name,
                        tool_call_id=call_id,
                        content=json.dumps(result),
                    )
                    tool_messages += 1
                    tool_messages = _record_cancelled_tool_messages(
                        recorder,
                        messages,
                        assistant.tool_calls[call_index + 1 :],
                        tool_message_index=tool_messages,
                        error="Session stopped after a policy violation",
                    )
                    reason = TerminationReason.POLICY_VIOLATION
                    break

                recorder.record(EventType.TOOL_STARTED, {"call_id": call_id, "name": name})
                tool_result = self.tools.execute(name, arguments)
                tool_calls += 1
                after_hash = self.tools.sandbox.state_hash()
                recorder.record(
                    EventType.TOOL_FINISHED,
                    {
                        "call_id": call_id,
                        "status": "failed" if tool_result.error else "completed",
                        "output": tool_result.output,
                        "error": tool_result.error,
                        "state_hash": after_hash,
                    },
                )
                if before_hash != after_hash:
                    recorder.record(
                        EventType.WORKSPACE_DIFF,
                        {
                            "before_state_hash": before_hash,
                            "after_state_hash": after_hash,
                            "diff": self.tools.sandbox.diff(),
                        },
                    )
                if name == "ask_user" and not tool_result.error:
                    question = arguments["question"]
                    answer = ask_user(question) if ask_user else None
                    if answer is None:
                        _record_tool_message(
                            recorder,
                            messages,
                            message_id=f"tool_{tool_messages}",
                            name=name,
                            tool_call_id=call_id,
                            content=json.dumps(
                                {"ok": False, "error": "User stopped without an answer"}
                            ),
                        )
                        tool_messages += 1
                        tool_messages = _record_cancelled_tool_messages(
                            recorder,
                            messages,
                            assistant.tool_calls[call_index + 1 :],
                            tool_message_index=tool_messages,
                            error="Session stopped by the user",
                        )
                        reason = TerminationReason.USER_STOP
                        break
                    _record_tool_message(
                        recorder,
                        messages,
                        message_id=f"tool_{tool_messages}",
                        name=name,
                        tool_call_id=call_id,
                        content=json.dumps({"answer": answer}),
                    )
                    tool_messages += 1
                    recorder.record(
                        EventType.USER_MESSAGE,
                        {"message_id": f"user_{user_messages}", "content": answer},
                    )
                    user_messages += 1
                    messages.append(Message("user", answer))
                else:
                    _record_tool_message(
                        recorder,
                        messages,
                        message_id=f"tool_{tool_messages}",
                        name=name,
                        tool_call_id=call_id,
                        content=json.dumps(
                            {
                                "ok": tool_result.error is None,
                                "output": tool_result.output,
                                "error": tool_result.error,
                            },
                            ensure_ascii=True,
                        )
                    )
                    tool_messages += 1
            else:
                continue
            break
        else:
            turn = self.budgets.max_turns
            reason = TerminationReason.TIMEOUT

        elapsed_ms = (time.monotonic() - started) * 1000
        final_hash = self.tools.sandbox.state_hash()
        if finalize:
            recorder.record(
                EventType.SESSION_FINISHED,
                {
                    "termination_reason": reason.value,
                    "final_state_hash": final_hash,
                    "success": False,
                },
            )
        return AgentRunResult(
            reason,
            final_answer,
            min(turn + 1, self.budgets.max_turns),
            tool_calls,
            tokens,
            final_hash,
            elapsed_ms,
        )


def _usage_total_tokens(usage: dict[str, int]) -> int:
    value = usage.get("total_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(value, 0)
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return sum(
        max(item, 0)
        for item in (prompt, completion)
        if isinstance(item, int) and not isinstance(item, bool)
    )


def _validate_response_token_bound(
    usage: dict[str, int],
    *,
    prompt_upper_bound: int,
    requested_output_tokens: int,
    remaining_tokens: int,
) -> None:
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if total is not None and (
        isinstance(total, bool) or not isinstance(total, int) or total < 0
    ):
        raise ValueError("Provider total token usage must be a non-negative integer")
    if prompt is not None and (
        isinstance(prompt, bool) or not isinstance(prompt, int) or prompt < 0
    ):
        raise ValueError("Provider input token usage must be a non-negative integer")
    if completion is not None and (
        isinstance(completion, bool)
        or not isinstance(completion, int)
        or completion < 0
    ):
        raise ValueError("Provider output token usage must be a non-negative integer")
    if isinstance(prompt, int) and prompt > prompt_upper_bound:
        raise ValueError("Provider input usage exceeded its pre-request token bound")
    if isinstance(completion, int) and completion > requested_output_tokens:
        raise ValueError("Provider output usage exceeded the requested token bound")
    if (
        isinstance(total, int)
        and not isinstance(total, bool)
        and isinstance(prompt, int)
        and not isinstance(prompt, bool)
        and isinstance(completion, int)
        and not isinstance(completion, bool)
        and total != prompt + completion
    ):
        raise ValueError("Provider total token usage is inconsistent with input and output")
    if _usage_total_tokens(usage) > remaining_tokens:
        raise ValueError("Provider usage exceeded the remaining agent token budget")


def _record_tool_message(
    recorder: TraceRecorder,
    messages: list[Message],
    *,
    message_id: str,
    name: str,
    tool_call_id: str,
    content: str,
) -> None:
    recorder.record(
        EventType.TOOL_MESSAGE,
        {
            "message_id": message_id,
            "name": name,
            "tool_call_id": tool_call_id,
            "content": content,
        },
    )
    messages.append(Message("tool", content, name=name, tool_call_id=tool_call_id))


def _record_cancelled_tool_messages(
    recorder: TraceRecorder,
    messages: list[Message],
    tool_calls: list[dict],
    *,
    tool_message_index: int,
    error: str,
) -> int:
    for call in tool_calls:
        function = call["function"]
        _record_tool_message(
            recorder,
            messages,
            message_id=f"tool_{tool_message_index}",
            name=function["name"],
            tool_call_id=call["id"],
            content=json.dumps({"ok": False, "error": error}),
        )
        tool_message_index += 1
    return tool_message_index
