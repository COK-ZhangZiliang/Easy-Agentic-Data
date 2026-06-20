from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

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
        self.max_retries = config.max_retries
        self.retry_backoff_seconds = config.retry_backoff_seconds
        self.request_body = dict(config.request_body)
        self.ssl_context = _ssl_context(config.ca_bundle_env)
        if require_api_key and not self.api_key:
            variable = config.api_key_env or "<not configured>"
            raise ValueError(f"Missing API key in environment variable {variable}")

    def complete(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            **self.request_body,
            "model": self.model,
            "messages": [
                message.to_api_dict(include_reasoning_content=True) for message in messages
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body, retry_count = self._request(payload, headers)
        choice = _response_message(body)
        message = Message(
            role=choice.get("role", "assistant"),
            content=choice.get("content"),
            tool_calls=choice.get("tool_calls") or [],
            reasoning_content=choice.get("reasoning_content"),
        )
        return LLMResponse(
            message=message,
            model=body.get("model", self.model),
            usage=body.get("usage", {}),
            raw=body,
            retry_count=retry_count,
        )

    def _request(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], int]:
        request = urllib.request.Request(
            f"{self.base_url}{self.chat_completions_path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                    context=self.ssl_context,
                ) as response:
                    raw_body = response.read().decode("utf-8")
                body = json.loads(raw_body)
                if not isinstance(body, dict):
                    raise RuntimeError("LLM API returned a non-object JSON response")
                return body, attempt
            except urllib.error.HTTPError as exc:
                detail = _error_detail(exc.read())
                if attempt < self.max_retries and exc.code in _RETRYABLE_HTTP_CODES:
                    self._backoff(attempt)
                    continue
                raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    self._backoff(attempt)
                    continue
                raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("LLM API returned invalid JSON") from exc
        raise AssertionError("unreachable")

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_backoff_seconds * (2**attempt)
        if delay:
            time.sleep(delay)


class LocalOpenAICompatibleClient(OpenAICompatibleClient):
    """Client for local OpenAI-compatible servers where authentication is optional."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config, require_api_key=False)


def _normalize_path(path: str) -> str:
    normalized = path.strip()
    if not normalized:
        raise ValueError("chat_completions_path cannot be empty")
    return normalized if normalized.startswith("/") else f"/{normalized}"


_RETRYABLE_HTTP_CODES = {408, 409, 429, 500, 502, 503, 504}


def _ssl_context(ca_bundle_env: str | None) -> ssl.SSLContext:
    ca_bundle = os.environ.get(ca_bundle_env, "") if ca_bundle_env else ""
    if ca_bundle:
        if not os.path.isfile(ca_bundle):
            raise ValueError(f"CA bundle from {ca_bundle_env} does not exist: {ca_bundle}")
        return ssl.create_default_context(cafile=ca_bundle)
    return ssl.create_default_context()


def _response_message(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM API response is missing choices")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise RuntimeError("LLM API response is missing the first message")
    return choice["message"]


def _error_detail(raw: bytes, limit: int = 2000) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    return text[:limit] if text else "no response body"
