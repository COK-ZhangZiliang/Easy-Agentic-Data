from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence

from easy_agentic_data.models import LLMResponse, Message


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Return one assistant message for the supplied conversation."""

