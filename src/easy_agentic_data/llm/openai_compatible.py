from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from easy_agentic_data.config import LLMConfig
from easy_agentic_data.models import LLMResponse, Message


class OpenAICompatibleClient:
    """Small dependency-free client for OpenAI-compatible chat completion APIs."""

    def __init__(self, config: LLMConfig, *, require_api_key: bool = True) -> None:
        self.model = config.model
        self.base_url = config.base_url.rstrip("/")
        self.chat_completions_path = _normalize_path(config.chat_completions_path)
        self.api_key_env = config.api_key_env
        self.api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        self.timeout_seconds = config.timeout_seconds
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        if require_api_key and not self.api_key:
            variable = config.api_key_env or "<not configured>"
            raise ValueError(f"Missing API key in environment variable {variable}")

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_api_dict() for message in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.base_url}{self.chat_completions_path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc

        choice = body["choices"][0]["message"]
        message = Message(
            role=choice.get("role", "assistant"),
            content=choice.get("content"),
            tool_calls=choice.get("tool_calls", []),
        )
        return LLMResponse(
            message=message,
            model=body.get("model", self.model),
            usage=body.get("usage", {}),
            raw=body,
        )


class LocalOpenAICompatibleClient(OpenAICompatibleClient):
    """Client for local OpenAI-compatible servers where authentication is optional."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config, require_api_key=False)


def _normalize_path(path: str) -> str:
    normalized = path.strip()
    if not normalized:
        raise ValueError("chat_completions_path cannot be empty")
    return normalized if normalized.startswith("/") else f"/{normalized}"
