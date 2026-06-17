from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: str = "mock-agent"
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


@dataclass
class GenerationConfig:
    num_tasks: int = 4
    rollouts_per_task: int = 2
    max_turns: int = 6
    min_reward: float = 0.5
    seed_topics: list[str] = field(
        default_factory=lambda: ["calculation", "information lookup", "planning"]
    )
    evolve_rounds: int = 1


@dataclass
class OutputConfig:
    directory: str = "runs/demo"
    export_sft: bool = True
    export_preferences: bool = True


@dataclass
class PipelineConfig:
    run_name: str = "demo"
    random_seed: int = 42
    llm: LLMConfig = field(default_factory=LLMConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return cls(
            run_name=data.get("run_name", "demo"),
            random_seed=int(data.get("random_seed", 42)),
            llm=LLMConfig(**data.get("llm", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            output=OutputConfig(**data.get("output", {})),
        )


def load_config(path: str | Path) -> PipelineConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PipelineConfig.from_dict(json.load(handle))
