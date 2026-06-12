from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: str = "mock-agent"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: Optional[str] = "OPENAI_API_KEY"
    chat_completions_path: str = "/chat/completions"
    timeout_seconds: float = 60.0
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class GenerationConfig:
    num_tasks: int = 4
    rollouts_per_task: int = 2
    max_turns: int = 6
    min_reward: float = 0.5
    seed_topics: List[str] = field(
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
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
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
