from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REQUEST_FIELD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
    seed_request_field: str | None = None
    response_model_aliases: tuple[str, ...] = ()

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
        if self.seed_request_field is not None:
            if not _REQUEST_FIELD_PATTERN.fullmatch(self.seed_request_field):
                raise ValueError("seed_request_field must be a simple request field name")
            if self.seed_request_field in reserved:
                raise ValueError("seed_request_field cannot override a reserved field")
            if self.seed_request_field in self.request_body:
                raise ValueError(
                    "seed_request_field must not also be present in request_body"
                )
        aliases = self.response_model_aliases
        if isinstance(aliases, (str, bytes)):
            raise ValueError("response_model_aliases must be a sequence of model names")
        if not isinstance(aliases, (list, tuple)) or not all(
            isinstance(alias, str) and alias.strip() for alias in aliases
        ):
            raise ValueError("response_model_aliases must contain non-empty strings")
        normalized_aliases = tuple(sorted(alias.strip() for alias in aliases))
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("response_model_aliases must be unique")
        if self.model in normalized_aliases:
            raise ValueError("response_model_aliases must not repeat the requested model")
        self.response_model_aliases = normalized_aliases


def load_llm_config(path: str | Path) -> LLMConfig:
    """Load the provider settings used by headless agent rollouts."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("llm"), dict):
        raise ValueError("Configuration must contain an 'llm' object")
    return LLMConfig(**data["llm"])
