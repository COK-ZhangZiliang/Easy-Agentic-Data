from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from easy_agentic_data.coding_tools import CodingToolRuntime
from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import Message
from easy_agentic_data.scenarios import ScenarioInstance
from easy_agentic_data.traces import EventType, TerminationReason, TraceRecorder


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


class HeadlessAgent:
    def __init__(
        self,
        client: LLMClient,
        tools: CodingToolRuntime,
        *,
        budgets: AgentBudgets | None = None,
        system_prompt: str = (
            "You are a headless coding agent. Inspect the workspace, use tools carefully, "
            "run relevant tests, and report the completed work."
        ),
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
        ask_user: Optional[Callable[[str], str | None]] = None,
        finalize: bool = True,
    ) -> AgentRunResult:
        started = time.monotonic()
        messages = [
            Message("system", self.system_prompt),
            Message("user", instance.public_task.query),
        ]
        recorder.start(instance)
        recorder.record(
            EventType.USER_MESSAGE,
            {"message_id": "user_0", "content": instance.public_task.query},
        )
        tool_calls = 0
        tokens = 0
        malformed = 0
        final_answer = ""
        reason = TerminationReason.AGENT_STOP

        for turn in range(self.budgets.max_turns):
            elapsed = time.monotonic() - started
            if elapsed >= self.budgets.max_seconds:
                reason = TerminationReason.TIMEOUT
                break
            response = self.client.complete(messages, tools=self.tools.schemas())
            tokens += int(response.usage.get("total_tokens", 0))
            if tokens > self.budgets.max_tokens:
                reason = TerminationReason.TOKEN_BUDGET
                break
            assistant = response.message
            messages.append(assistant)
            recorder.record(
                EventType.MODEL_RESPONSE,
                {
                    "message_id": f"assistant_{turn}",
                    "content": assistant.content,
                    "tool_calls": assistant.tool_calls,
                    "model": response.model,
                    "usage": response.usage,
                },
            )
            if not assistant.tool_calls:
                final_answer = assistant.content or ""
                reason = TerminationReason.AGENT_STOP
                break

            for raw_call in assistant.tool_calls:
                if tool_calls >= self.budgets.max_tool_calls:
                    reason = TerminationReason.TOOL_BUDGET
                    break
                call_id = raw_call.get("id", f"call_{tool_calls}")
                function = raw_call.get("function", {})
                name = str(function.get("name", ""))
                try:
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    malformed += 1
                    messages.append(
                        Message(
                            "tool",
                            json.dumps({"ok": False, "error": f"Invalid tool arguments: {exc}"}),
                            name=name,
                            tool_call_id=call_id,
                        )
                    )
                    if malformed > self.budgets.malformed_tool_retries:
                        reason = TerminationReason.INFRASTRUCTURE_FAILURE
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
                    messages.append(
                        Message("tool", json.dumps(result), name=name, tool_call_id=call_id)
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
                        reason = TerminationReason.USER_STOP
                        break
                    recorder.record(
                        EventType.USER_MESSAGE,
                        {"message_id": f"user_{turn + 1}", "content": answer},
                    )
                    messages.append(Message("tool", json.dumps({"answer": answer}), name=name, tool_call_id=call_id))
                    messages.append(Message("user", answer))
                else:
                    messages.append(
                        Message(
                            "tool",
                            json.dumps(
                                {
                                    "ok": tool_result.error is None,
                                    "output": tool_result.output,
                                    "error": tool_result.error,
                                },
                                ensure_ascii=True,
                            ),
                            name=name,
                            tool_call_id=call_id,
                        )
                    )
            else:
                continue
            break
        else:
            turn = self.budgets.max_turns
            reason = TerminationReason.TIMEOUT

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
        return AgentRunResult(reason, final_answer, min(turn + 1, self.budgets.max_turns), tool_calls, tokens, final_hash)
