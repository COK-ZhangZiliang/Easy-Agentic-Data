from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = "OPENAI_API_KEY"
    chat_completions_path: str = "/chat/completions"
    timeout_seconds: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 2048
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    request_body: dict[str, Any] = field(default_factory=dict)
    ca_bundle_env: str | None = "SSL_CERT_FILE"

    def __post_init__(self) -> None:
        if self.provider not in {"openai_compatible", "local_openai_compatible"}:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        reserved = {"model", "messages", "tools", "tool_choice", "temperature", "max_tokens"}
        conflicts = reserved.intersection(self.request_body)
        if conflicts:
            raise ValueError(f"request_body cannot override reserved fields: {sorted(conflicts)}")


def load_llm_config(path: str | Path) -> LLMConfig:
    """Load the provider settings used by headless agent rollouts."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("llm"), dict):
        raise ValueError("Configuration must contain an 'llm' object")
    return LLMConfig(**data["llm"])
