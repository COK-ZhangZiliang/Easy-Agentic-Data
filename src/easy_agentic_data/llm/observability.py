from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Sequence

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import LLMResponse, Message, utc_now


class ObservedLLMClient:
    """Record a prompt-safe audit row for every delegated LLM call."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.model = inner.model
        self.records: List[Dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        record: Dict[str, Any] = {
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
            "prompt_hash": _prompt_hash(messages, tools),
        }
        try:
            response = self.inner.complete(
                messages,
                tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            record.update(
                {
                    "status": "completed",
                    "response_model": response.model,
                    "usage": response.usage,
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
            self.records.append(record)


def _prompt_hash(
    messages: Sequence[Message],
    tools: Optional[List[Dict[str, Any]]],
) -> str:
    payload = {
        "messages": [message.to_api_dict() for message in messages],
        "tools": tools or [],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
