from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from easy_agentic_data.models import LLMResponse, Message


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return one assistant message for the supplied conversation."""
