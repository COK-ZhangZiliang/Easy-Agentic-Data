from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import LLMResponse, Message, utc_now
from easy_agentic_data.traces import EventType, Trace, replay_trace

PROMPT_TOKEN_SAFETY_MARGIN = 1024


class CallJournal(Protocol):
    def call_started(self, observed: Mapping[str, Any]) -> None: ...

    def call_completed(self, observed: Mapping[str, Any]) -> None: ...


class ObservedLLMClient:
    """Record a prompt-safe audit row for every delegated LLM call."""

    def __init__(self, inner: LLMClient, *, call_journal: CallJournal | None = None) -> None:
        self.inner = inner
        self.model = inner.model
        self.temperature = getattr(inner, "temperature", None)
        self.max_tokens = getattr(inner, "max_tokens", None)
        self.records: list[dict[str, Any]] = []
        self.call_journal = call_journal

    def complete(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "call_index": len(self.records),
            "started_at": utc_now(),
            "model": self.model,
            "message_count": len(messages),
            "tool_count": len(tools or []),
            "temperature": (
                temperature if temperature is not None else getattr(self.inner, "temperature", None)
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else getattr(self.inner, "max_tokens", None)
            ),
            "retry_count": 0,
            "prompt_hash": prompt_hash(messages, tools),
            "prompt_token_upper_bound": prompt_token_upper_bound(messages, tools),
            "response_format": response_format,
        }
        try:
            if self.call_journal is not None:
                self.call_journal.call_started(record)
            response = self.inner.complete(
                messages,
                tools,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            record.update(
                {
                    "status": "completed",
                    "response_model": response.model,
                    "usage": response.usage,
                    "retry_count": response.retry_count,
                    **provider_response_provenance(response),
                }
            )
            return response
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        finally:
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            try:
                if self.call_journal is not None and record.get("status") == "completed":
                    self.call_journal.call_completed(record)
            finally:
                self.records.append(record)


def provider_response_provenance(response: LLMResponse) -> dict[str, Any]:
    """Hash a provider response and retain only its non-secret receipt identity."""

    raw = response.raw
    if not isinstance(raw, Mapping):
        raise ValueError("Provider response raw payload must be an object")
    identity = {
        key: raw[key]
        for key in ("id", "created", "object", "model", "system_fingerprint")
        if key in raw
    }
    return {
        "provider_response_identity": identity,
        "provider_response_identity_sha256": _canonical_sha256(identity),
        "provider_response_sha256": _canonical_sha256(raw),
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_hash(
    messages: Sequence[Message],
    tools: list[dict[str, Any]] | None,
) -> str:
    """Hash the exact provider-visible messages and tool schemas for one request."""

    payload = {
        "messages": [message.to_api_dict(include_reasoning_content=True) for message in messages],
        "tools": tools or [],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_token_upper_bound(
    messages: Sequence[Message],
    tools: list[dict[str, Any]] | None,
) -> int:
    """Return a conservative pre-request bound for provider input tokens.

    OpenAI-compatible prompts are text payloads.  Their token count cannot exceed
    the ASCII-escaped JSON byte count for the exact messages and tools, apart from
    provider-added chat control tokens.  A fixed safety margin covers that framing
    while keeping the bound independent of a provider tokenizer or network call.
    """

    payload = {
        "messages": [message.to_api_dict(include_reasoning_content=True) for message in messages],
        "tools": tools or [],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return len(encoded) + PROMPT_TOKEN_SAFETY_MARGIN


def trace_prompt_fingerprints(
    trace: Trace,
    system_prompt: str,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Recompute requests from trace messages, asserting the expected prompt binding."""

    if not isinstance(system_prompt, str) or not system_prompt:
        raise ValueError("system_prompt must be a non-empty string")
    replay_trace(trace)
    messages: list[Message] = []
    traced_system_prompt: str | None = None
    fingerprints: list[dict[str, Any]] = []
    for event in trace.events:
        payload = event.payload
        if event.event_type is EventType.SYSTEM_MESSAGE:
            traced_system_prompt = payload["content"]
            messages.append(Message("system", traced_system_prompt))
        elif event.event_type is EventType.USER_MESSAGE:
            messages.append(Message("user", payload["content"]))
        elif event.event_type is EventType.MODEL_RESPONSE:
            if traced_system_prompt is None:
                raise ValueError("Trace model response has no canonical system_message")
            fingerprints.append(
                {
                    "prompt_hash": prompt_hash(messages, tools),
                    "prompt_token_upper_bound": prompt_token_upper_bound(
                        messages,
                        tools,
                    ),
                    "message_count": len(messages),
                    "tool_count": len(tools or []),
                }
            )
            messages.append(
                Message(
                    "assistant",
                    payload.get("content"),
                    tool_calls=payload.get("tool_calls", []),
                    reasoning_content=payload.get("reasoning_content"),
                )
            )
        elif event.event_type is EventType.TOOL_MESSAGE:
            messages.append(
                Message(
                    "tool",
                    payload["content"],
                    name=payload["name"],
                    tool_call_id=payload["tool_call_id"],
                )
            )
    if traced_system_prompt is None:
        raise ValueError("Trace has no canonical system_message")
    if traced_system_prompt != system_prompt:
        raise ValueError("Expected system prompt does not match canonical trace")
    return fingerprints


def validate_observed_prompt_lineage(
    records: Sequence[Mapping[str, Any]],
    trace: Trace,
    system_prompt: str,
    tools: list[dict[str, Any]] | None,
) -> None:
    """Require observed request hashes and counts to match trace-reconstructed prompts."""

    expected = trace_prompt_fingerprints(trace, system_prompt, tools)
    if len(records) != len(expected):
        raise ValueError("Observed model calls do not match trace-reconstructed prompts")
    for index, (record, fingerprint) in enumerate(zip(records, expected, strict=True)):
        mismatched = [
            key for key, value in fingerprint.items() if record.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                f"Observed prompt lineage mismatch at call {index}: {mismatched}"
            )
