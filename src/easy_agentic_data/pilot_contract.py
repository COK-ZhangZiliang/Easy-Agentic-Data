from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from easy_agentic_data.config import LLMConfig

PILOT_SCHEMA_VERSION = "easy_agentic_data.pilot_run_contract.v1"
GOLD20_MANIFEST_SCHEMA_VERSION = "easy_agentic_data.gold20_manifest.v1"
EXPECTED_GOLD20_SCENARIOS = 20
ROLLOUTS_PER_SCENARIO = 2
GOLD20_REQUIRED_VALIDATION_GATES = frozenset(
    {
        "container_replay_valid",
        "decontamination_valid",
        "exact_count",
        "exact_source_set",
        "hidden_patch_rehearsal_valid",
        "materialization_reset_valid",
        "registry_valid",
        "repair_validation_valid",
    }
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible content deterministically and reject non-finite numbers."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON content."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderConfigBinding:
    """Secret-free, content-addressed provider settings used by one pilot."""

    provider: str
    model: str
    endpoint_sha256: str
    chat_completions_path_sha256: str
    api_key_env: str | None
    ca_bundle_env: str | None
    timeout_seconds: float
    temperature: float
    max_tokens: int
    max_retries: int
    retry_backoff_seconds: float
    request_body_sha256: str
    seed_request_field: str | None = None
    response_model_aliases: tuple[str, ...] = ()
    config_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("Provider and model must be non-empty")
        _require_sha256(self.endpoint_sha256, "endpoint_sha256")
        _require_sha256(self.request_body_sha256, "request_body_sha256")
        _require_sha256(
            self.chat_completions_path_sha256,
            "chat_completions_path_sha256",
        )
        _require_positive_float(self.timeout_seconds, "timeout_seconds")
        _require_finite_float(self.temperature, "temperature")
        _require_positive_int(self.max_tokens, "max_tokens")
        _require_nonnegative_int(self.max_retries, "max_retries")
        _require_nonnegative_float(self.retry_backoff_seconds, "retry_backoff_seconds")
        if self.seed_request_field is None and self.temperature != 0.0:
            raise ValueError(
                "Pilot providers without seed_request_field must use temperature=0"
            )
        aliases = tuple(self.response_model_aliases)
        if not all(isinstance(alias, str) and alias for alias in aliases):
            raise ValueError("response_model_aliases must contain non-empty strings")
        if aliases != tuple(sorted(set(aliases))):
            raise ValueError("response_model_aliases must be sorted and unique")
        if self.model in aliases:
            raise ValueError("response_model_aliases must not repeat the requested model")
        object.__setattr__(self, "response_model_aliases", aliases)
        expected = canonical_sha256(self._identity_payload())
        if self.config_sha256 and self.config_sha256 != expected:
            raise ValueError("config_sha256 does not match provider settings")
        object.__setattr__(self, "config_sha256", expected)

    @classmethod
    def from_config(cls, config: LLMConfig) -> ProviderConfigBinding:
        _reject_credential_fields(config.request_body)
        normalized_endpoint = _normalize_endpoint(config.base_url)
        endpoint_sha256 = hashlib.sha256(normalized_endpoint.encode("utf-8")).hexdigest()
        return cls(
            provider=config.provider,
            model=config.model,
            endpoint_sha256=endpoint_sha256,
            chat_completions_path_sha256=hashlib.sha256(
                _normalize_api_path(config.chat_completions_path).encode("utf-8")
            ).hexdigest(),
            api_key_env=config.api_key_env,
            ca_bundle_env=config.ca_bundle_env,
            timeout_seconds=float(config.timeout_seconds),
            temperature=float(config.temperature),
            max_tokens=config.max_tokens,
            max_retries=config.max_retries,
            retry_backoff_seconds=float(config.retry_backoff_seconds),
            request_body_sha256=canonical_sha256(config.request_body),
            seed_request_field=config.seed_request_field,
            response_model_aliases=config.response_model_aliases,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProviderConfigBinding:
        raw_aliases = value.get("response_model_aliases")
        if not isinstance(raw_aliases, list):
            raise ValueError("response_model_aliases must be a list")
        return cls(
            provider=_required_string(value, "provider"),
            model=_required_string(value, "model"),
            endpoint_sha256=_required_string(value, "endpoint_sha256"),
            chat_completions_path_sha256=_required_string(
                value,
                "chat_completions_path_sha256",
            ),
            api_key_env=_optional_string(value.get("api_key_env"), "api_key_env"),
            ca_bundle_env=_optional_string(value.get("ca_bundle_env"), "ca_bundle_env"),
            timeout_seconds=_number(value, "timeout_seconds"),
            temperature=_number(value, "temperature"),
            max_tokens=_integer(value, "max_tokens"),
            max_retries=_integer(value, "max_retries"),
            retry_backoff_seconds=_number(value, "retry_backoff_seconds"),
            request_body_sha256=_required_string(value, "request_body_sha256"),
            seed_request_field=_optional_string(
                value.get("seed_request_field"), "seed_request_field"
            ),
            response_model_aliases=tuple(
                _plain_required_string(item, "response_model_aliases")
                for item in raw_aliases
            ),
            config_sha256=_required_string(value, "config_sha256"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "api_key_env": self.api_key_env,
            "ca_bundle_env": self.ca_bundle_env,
            "chat_completions_path_sha256": self.chat_completions_path_sha256,
            "endpoint_sha256": self.endpoint_sha256,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "model": self.model,
            "provider": self.provider,
            "request_body_sha256": self.request_body_sha256,
            "response_model_aliases": list(self.response_model_aliases),
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "seed_request_field": self.seed_request_field,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "config_sha256": self.config_sha256}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class Gold20ScenarioBinding:
    scenario_id: str
    seed_id: str
    environment_id: str
    record_sha256: str
    scenario_sha256: str
    environment_sha256: str
    evaluator_sha256: str

    def __post_init__(self) -> None:
        for name in ("scenario_id", "seed_id", "environment_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in (
            "record_sha256",
            "scenario_sha256",
            "environment_sha256",
            "evaluator_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Gold20ScenarioBinding:
        return cls(
            scenario_id=_required_string(value, "scenario_id"),
            seed_id=_required_string(value, "seed_id"),
            environment_id=_required_string(value, "environment_id"),
            record_sha256=_required_string(value, "record_sha256"),
            scenario_sha256=_required_string(value, "scenario_sha256"),
            environment_sha256=_required_string(value, "environment_sha256"),
            evaluator_sha256=_required_string(value, "evaluator_sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "environment_id": self.environment_id,
            "environment_sha256": self.environment_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "record_sha256": self.record_sha256,
            "scenario_id": self.scenario_id,
            "scenario_sha256": self.scenario_sha256,
            "seed_id": self.seed_id,
        }


@dataclass(frozen=True)
class Gold20Binding:
    """The exact frozen corpus and scenario set admitted to the M2 pilot."""

    corpus_id: str
    manifest_schema_version: str
    manifest_sha256: str
    registry_snapshot_sha256: str
    scenarios: tuple[Gold20ScenarioBinding, ...]

    def __post_init__(self) -> None:
        if not self.corpus_id.startswith("gold20_"):
            raise ValueError("Gold-20 corpus_id must start with gold20_")
        if self.manifest_schema_version != GOLD20_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported Gold-20 manifest schema")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        _require_sha256(self.registry_snapshot_sha256, "registry_snapshot_sha256")
        if len(self.scenarios) != EXPECTED_GOLD20_SCENARIOS:
            raise ValueError("Gold-20 binding must contain exactly 20 scenarios")
        ordered = tuple(sorted(self.scenarios, key=lambda item: item.scenario_id))
        if len({item.scenario_id for item in ordered}) != EXPECTED_GOLD20_SCENARIOS:
            raise ValueError("Gold-20 scenario IDs must be unique")
        object.__setattr__(self, "scenarios", ordered)

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any] | str | Path,
    ) -> Gold20Binding:
        payload = _load_json_object(manifest)
        if payload.get("valid") is not True or payload.get("issues") != []:
            raise ValueError("Gold-20 manifest must be valid and issue-free")
        if payload.get("expected_seed_count") != EXPECTED_GOLD20_SCENARIOS:
            raise ValueError("Gold-20 manifest must declare exactly 20 seeds")
        validation = payload.get("validation")
        if (
            not isinstance(validation, dict)
            or set(validation) != GOLD20_REQUIRED_VALIDATION_GATES
            or not all(value is True for value in validation.values())
        ):
            raise ValueError("Gold-20 manifest validation gates must all pass")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != EXPECTED_GOLD20_SCENARIOS:
            raise ValueError("Gold-20 manifest must contain exactly 20 records")
        scenarios = []
        for record in records:
            if not isinstance(record, dict) or record.get("valid") is not True:
                raise ValueError("Every Gold-20 record must be a valid object")
            record_sha256 = _required_string(record, "record_sha256")
            record_payload = dict(record)
            record_payload.pop("record_sha256")
            if record_sha256 != canonical_sha256(record_payload):
                raise ValueError("Gold-20 record_sha256 does not match record content")
            hashes = record.get("hashes")
            if not isinstance(hashes, dict):
                raise ValueError("Every Gold-20 record must contain hashes")
            scenarios.append(
                Gold20ScenarioBinding(
                    scenario_id=_required_string(record, "scenario_id"),
                    seed_id=_required_string(record, "seed_id"),
                    environment_id=_required_string(record, "environment_id"),
                    record_sha256=record_sha256,
                    scenario_sha256=_required_string(hashes, "scenario_sha256"),
                    environment_sha256=_required_string(hashes, "environment_sha256"),
                    evaluator_sha256=_required_string(hashes, "evaluator_sha256"),
                )
            )
        evidence = payload.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Gold-20 manifest evidence must be an object")
        return cls(
            corpus_id=_required_string(payload, "corpus_id"),
            manifest_schema_version=_required_string(payload, "schema_version"),
            manifest_sha256=canonical_sha256(payload),
            registry_snapshot_sha256=_required_string(evidence, "registry_snapshot_sha256"),
            scenarios=tuple(scenarios),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Gold20Binding:
        raw_scenarios = value.get("scenarios")
        if not isinstance(raw_scenarios, list):
            raise ValueError("Gold-20 scenarios must be a list")
        binding = cls(
            corpus_id=_required_string(value, "corpus_id"),
            manifest_schema_version=_required_string(value, "manifest_schema_version"),
            manifest_sha256=_required_string(value, "manifest_sha256"),
            registry_snapshot_sha256=_required_string(value, "registry_snapshot_sha256"),
            scenarios=tuple(
                Gold20ScenarioBinding.from_dict(_mapping(item, "scenario binding"))
                for item in raw_scenarios
            ),
        )
        supplied_environment_hash = _required_string(
            value, "environment_bundle_sha256"
        )
        if supplied_environment_hash != binding.environment_bundle_sha256:
            raise ValueError(
                "environment_bundle_sha256 does not match Gold-20 scenario bindings"
            )
        supplied_evaluator_hash = _required_string(value, "evaluator_bundle_sha256")
        if supplied_evaluator_hash != binding.evaluator_bundle_sha256:
            raise ValueError(
                "evaluator_bundle_sha256 does not match Gold-20 scenario bindings"
            )
        return binding

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(item.scenario_id for item in self.scenarios)

    @property
    def environment_bundle_sha256(self) -> str:
        return canonical_sha256(
            {item.scenario_id: item.environment_sha256 for item in self.scenarios}
        )

    @property
    def evaluator_bundle_sha256(self) -> str:
        return canonical_sha256(
            {item.scenario_id: item.evaluator_sha256 for item in self.scenarios}
        )

    def assert_exact_scenarios(self, scenario_sha256s: Mapping[str, str]) -> None:
        expected = {item.scenario_id: item.scenario_sha256 for item in self.scenarios}
        actual = dict(scenario_sha256s)
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(
                f"Registry scenario set does not match Gold-20: missing={missing} extra={extra}"
            )
        invalid = sorted(
            scenario_id
            for scenario_id, digest in actual.items()
            if digest != expected[scenario_id]
        )
        if invalid:
            raise ValueError(f"Registry scenario hash mismatch: {invalid}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "environment_bundle_sha256": self.environment_bundle_sha256,
            "evaluator_bundle_sha256": self.evaluator_bundle_sha256,
            "manifest_schema_version": self.manifest_schema_version,
            "manifest_sha256": self.manifest_sha256,
            "registry_snapshot_sha256": self.registry_snapshot_sha256,
            "scenarios": [item.to_dict() for item in self.scenarios],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class PilotBudgets:
    max_agent_turns: int
    max_agent_tool_calls: int
    max_agent_tokens: int
    max_agent_seconds: float
    max_total_tokens: int
    max_total_cost_usd: Decimal
    max_total_seconds: float
    malformed_tool_retries: int = 2
    max_infrastructure_retries: int = 2
    max_workers: int = 1

    def __post_init__(self) -> None:
        for name in (
            "max_agent_turns",
            "max_agent_tool_calls",
            "max_agent_tokens",
            "max_total_tokens",
        ):
            _require_positive_int(getattr(self, name), name)
        _require_nonnegative_int(self.malformed_tool_retries, "malformed_tool_retries")
        _require_nonnegative_int(
            self.max_infrastructure_retries,
            "max_infrastructure_retries",
        )
        _require_positive_int(self.max_workers, "max_workers")
        for name in ("max_agent_seconds", "max_total_seconds"):
            value = float(getattr(self, name))
            _require_positive_float(value, name)
            object.__setattr__(self, name, value)
        cost = _decimal(self.max_total_cost_usd, "max_total_cost_usd")
        if cost <= 0:
            raise ValueError("max_total_cost_usd must be positive")
        object.__setattr__(self, "max_total_cost_usd", cost)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PilotBudgets:
        return cls(
            max_agent_turns=_integer(value, "max_agent_turns"),
            max_agent_tool_calls=_integer(value, "max_agent_tool_calls"),
            max_agent_tokens=_integer(value, "max_agent_tokens"),
            max_agent_seconds=_number(value, "max_agent_seconds"),
            max_total_tokens=_integer(value, "max_total_tokens"),
            max_total_cost_usd=_required_string(value, "max_total_cost_usd"),
            max_total_seconds=_number(value, "max_total_seconds"),
            malformed_tool_retries=_integer(value, "malformed_tool_retries"),
            max_infrastructure_retries=_integer(value, "max_infrastructure_retries"),
            max_workers=_integer(value, "max_workers"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_agent_seconds": self.max_agent_seconds,
            "max_agent_tokens": self.max_agent_tokens,
            "max_agent_tool_calls": self.max_agent_tool_calls,
            "max_agent_turns": self.max_agent_turns,
            "max_total_cost_usd": _decimal_text(self.max_total_cost_usd),
            "max_total_seconds": self.max_total_seconds,
            "max_total_tokens": self.max_total_tokens,
            "malformed_tool_retries": self.malformed_tool_retries,
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "max_workers": self.max_workers,
        }


@dataclass(frozen=True)
class PilotVersionHashes:
    prompt_sha256: str
    tool_schema_sha256: str
    evaluator_sha256: str
    environment_sha256: str
    exporter_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "prompt_sha256",
            "tool_schema_sha256",
            "evaluator_sha256",
            "environment_sha256",
            "exporter_sha256",
        ):
            _require_sha256(getattr(self, name), name)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PilotVersionHashes:
        return cls(
            prompt_sha256=_required_string(value, "prompt_sha256"),
            tool_schema_sha256=_required_string(value, "tool_schema_sha256"),
            evaluator_sha256=_required_string(value, "evaluator_sha256"),
            environment_sha256=_required_string(value, "environment_sha256"),
            exporter_sha256=_required_string(value, "exporter_sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "environment_sha256": self.environment_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "exporter_sha256": self.exporter_sha256,
            "prompt_sha256": self.prompt_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
        }


@dataclass(frozen=True)
class PilotQualityTargets:
    """Contract-bound minimum output counts for a valid pilot."""

    minimum_successes: int = 1
    minimum_sft: int = 1
    minimum_rl: int = 1
    minimum_preference: int = 0

    def __post_init__(self) -> None:
        for name in ("minimum_successes", "minimum_sft", "minimum_rl"):
            _require_positive_int(getattr(self, name), name)
        _require_nonnegative_int(self.minimum_preference, "minimum_preference")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PilotQualityTargets:
        expected = {
            "minimum_successes",
            "minimum_sft",
            "minimum_rl",
            "minimum_preference",
        }
        if set(value) != expected:
            raise ValueError("Pilot quality target fields do not match the schema")
        return cls(
            minimum_successes=_integer(value, "minimum_successes"),
            minimum_sft=_integer(value, "minimum_sft"),
            minimum_rl=_integer(value, "minimum_rl"),
            minimum_preference=_integer(value, "minimum_preference"),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "minimum_successes": self.minimum_successes,
            "minimum_sft": self.minimum_sft,
            "minimum_rl": self.minimum_rl,
            "minimum_preference": self.minimum_preference,
        }


@dataclass(frozen=True)
class UsageCost:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: Decimal
    pricing_sha256: str
    usage_cost_id: str = ""

    def __post_init__(self) -> None:
        for name in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
            _require_nonnegative_int(getattr(self, name), name)
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens plus output_tokens")
        cost = _decimal(self.cost_usd, "cost_usd")
        _require_sha256(self.pricing_sha256, "pricing_sha256")
        object.__setattr__(self, "cost_usd", cost)
        expected = _stable_id("usage_cost", self._identity_payload())
        if self.usage_cost_id and self.usage_cost_id != expected:
            raise ValueError("usage_cost_id does not match usage cost content")
        object.__setattr__(self, "usage_cost_id", expected)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> UsageCost:
        return cls(
            input_tokens=_integer(value, "input_tokens"),
            cached_input_tokens=_integer(value, "cached_input_tokens"),
            output_tokens=_integer(value, "output_tokens"),
            total_tokens=_integer(value, "total_tokens"),
            cost_usd=_required_string(value, "cost_usd"),
            pricing_sha256=_required_string(value, "pricing_sha256"),
            usage_cost_id=_required_string(value, "usage_cost_id"),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "cached_input_tokens": self.cached_input_tokens,
            "cost_usd": _decimal_text(self.cost_usd),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "pricing_sha256": self.pricing_sha256,
            "total_tokens": self.total_tokens,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "usage_cost_id": self.usage_cost_id}


@dataclass(frozen=True)
class PricingSpec:
    input_usd_per_million_tokens: Decimal
    cached_input_usd_per_million_tokens: Decimal
    output_usd_per_million_tokens: Decimal
    currency: str = "USD"
    pricing_sha256: str = ""

    def __post_init__(self) -> None:
        if self.currency != "USD":
            raise ValueError("Pilot pricing currency must be USD")
        for name in (
            "input_usd_per_million_tokens",
            "cached_input_usd_per_million_tokens",
            "output_usd_per_million_tokens",
        ):
            value = _decimal(getattr(self, name), name)
            object.__setattr__(self, name, value)
        expected = canonical_sha256(self._identity_payload())
        if self.pricing_sha256 and self.pricing_sha256 != expected:
            raise ValueError("pricing_sha256 does not match pricing content")
        object.__setattr__(self, "pricing_sha256", expected)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PricingSpec:
        return cls(
            input_usd_per_million_tokens=_required_string(
                value, "input_usd_per_million_tokens"
            ),
            cached_input_usd_per_million_tokens=_required_string(
                value, "cached_input_usd_per_million_tokens"
            ),
            output_usd_per_million_tokens=_required_string(
                value, "output_usd_per_million_tokens"
            ),
            currency=_required_string(value, "currency"),
            pricing_sha256=_required_string(value, "pricing_sha256"),
        )

    def _identity_payload(self) -> dict[str, str]:
        return {
            "cached_input_usd_per_million_tokens": _decimal_text(
                self.cached_input_usd_per_million_tokens
            ),
            "currency": self.currency,
            "input_usd_per_million_tokens": _decimal_text(
                self.input_usd_per_million_tokens
            ),
            "output_usd_per_million_tokens": _decimal_text(
                self.output_usd_per_million_tokens
            ),
        }

    def to_dict(self) -> dict[str, str]:
        return {**self._identity_payload(), "pricing_sha256": self.pricing_sha256}

    def calculate_cost(self, usage: Mapping[str, Any]) -> UsageCost:
        if not usage:
            raise ValueError("Token usage must be a non-empty object")
        hit_tokens = _usage_int(usage, "prompt_cache_hit_tokens", default=0)
        miss_tokens = _usage_int(usage, "prompt_cache_miss_tokens", default=0)
        input_tokens = _coalesced_usage_int(
            usage,
            ("input_tokens", "prompt_tokens"),
            "input token count",
        )
        if input_tokens is None:
            if not any(
                key in usage
                for key in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
            ):
                raise ValueError("Token usage is missing an input token count")
            input_tokens = hit_tokens + miss_tokens
        details = _coalesced_usage_details(
            usage,
            ("input_tokens_details", "prompt_tokens_details"),
        )
        if details is None:
            details = {}
        detail_cached = _usage_int(details, "cached_tokens", default=0)
        cached_tokens = hit_tokens if hit_tokens else detail_cached
        if hit_tokens and detail_cached and hit_tokens != detail_cached:
            raise ValueError("Conflicting cached token counts")
        if miss_tokens and hit_tokens + miss_tokens != input_tokens:
            raise ValueError("Cache hit and miss token counts do not match input tokens")
        if cached_tokens > input_tokens:
            raise ValueError("Cached tokens cannot exceed input tokens")
        output_tokens = _coalesced_usage_int(
            usage,
            ("output_tokens", "completion_tokens"),
            "output token count",
        )
        if output_tokens is None:
            raise ValueError("Token usage is missing an output token count")
        declared_total = _coalesced_usage_int(
            usage,
            ("total_tokens",),
            "total token count",
        )
        if declared_total is not None and declared_total != input_tokens + output_tokens:
            raise ValueError("Total token count does not match input plus output tokens")
        uncached_tokens = input_tokens - cached_tokens
        cost = (
            Decimal(uncached_tokens) * self.input_usd_per_million_tokens
            + Decimal(cached_tokens) * self.cached_input_usd_per_million_tokens
            + Decimal(output_tokens) * self.output_usd_per_million_tokens
        ) / Decimal(1_000_000)
        return UsageCost(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            pricing_sha256=self.pricing_sha256,
        )


@dataclass(frozen=True)
class PilotRolloutAssignment:
    contract_id: str
    scenario_id: str
    rollout_index: int
    random_seed: int
    job_id: str = ""

    def __post_init__(self) -> None:
        if not self.contract_id.startswith("pilot_"):
            raise ValueError("Rollout contract_id must be a pilot ID")
        if not self.scenario_id:
            raise ValueError("Rollout scenario_id must be non-empty")
        if self.rollout_index not in {0, 1}:
            raise ValueError("Rollout index must be 0 or 1")
        _require_nonnegative_int(self.random_seed, "random_seed")
        expected = _stable_id(
            "rollout",
            {
                "contract_id": self.contract_id,
                "random_seed": self.random_seed,
                "rollout_index": self.rollout_index,
                "scenario_id": self.scenario_id,
            },
        )
        if self.job_id and self.job_id != expected:
            raise ValueError("Pilot rollout assignments contain a mismatched job_id")
        object.__setattr__(self, "job_id", expected)

    @property
    def rollout_id(self) -> str:
        return self.job_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "job_id": self.job_id,
            "random_seed": self.random_seed,
            "rollout_index": self.rollout_index,
            "scenario_id": self.scenario_id,
        }


@dataclass(frozen=True)
class PilotRunContract:
    """Immutable M2 run identity, budgets, versions, and exact 20x2 schedule."""

    corpus: Gold20Binding
    provider: ProviderConfigBinding
    budgets: PilotBudgets
    versions: PilotVersionHashes
    pricing: PricingSpec
    quality_targets: PilotQualityTargets = field(default_factory=PilotQualityTargets)
    rollout_seeds: tuple[int, ...] = (0, 1)
    schema_version: str = PILOT_SCHEMA_VERSION
    contract_id: str = ""
    _rollouts: tuple[PilotRolloutAssignment, ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.schema_version != PILOT_SCHEMA_VERSION:
            raise ValueError("Unsupported pilot contract schema")
        seeds = tuple(self.rollout_seeds)
        if len(seeds) != ROLLOUTS_PER_SCENARIO:
            raise ValueError("Pilot seed schedule must contain exactly two seeds")
        for seed in seeds:
            _require_nonnegative_int(seed, "rollout seed")
        if len(set(seeds)) != ROLLOUTS_PER_SCENARIO:
            raise ValueError("Pilot rollout seeds must be distinct")
        object.__setattr__(self, "rollout_seeds", seeds)
        expected = _stable_id("pilot", self._identity_payload())
        if self.contract_id and self.contract_id != expected:
            raise ValueError("contract_id does not match pilot contract content")
        object.__setattr__(self, "contract_id", expected)
        expected_rollouts = self._build_rollouts()
        if self._rollouts and self._rollouts != expected_rollouts:
            raise ValueError("Pilot rollout assignments do not match contract")
        object.__setattr__(self, "_rollouts", expected_rollouts)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PilotRunContract:
        raw_seeds = value.get("rollout_seeds")
        if not isinstance(raw_seeds, list):
            raise ValueError("rollout_seeds must be a list")
        raw_rollouts = value.get("rollouts")
        if not isinstance(raw_rollouts, list):
            raise ValueError("rollout assignments must be a list")
        contract = cls(
            corpus=Gold20Binding.from_dict(_mapping(value.get("corpus"), "corpus")),
            provider=ProviderConfigBinding.from_dict(
                _mapping(value.get("provider"), "provider")
            ),
            budgets=PilotBudgets.from_dict(_mapping(value.get("budgets"), "budgets")),
            versions=PilotVersionHashes.from_dict(
                _mapping(value.get("versions"), "versions")
            ),
            pricing=PricingSpec.from_dict(_mapping(value.get("pricing"), "pricing")),
            quality_targets=PilotQualityTargets.from_dict(
                _mapping(value.get("quality_targets"), "quality_targets")
            ),
            rollout_seeds=tuple(_plain_integer(item, "rollout seed") for item in raw_seeds),
            schema_version=_required_string(value, "schema_version"),
            contract_id=_required_string(value, "contract_id"),
        )
        supplied_rollouts = tuple(
            PilotRolloutAssignment(
                contract_id=_required_string(_mapping(item, "rollout"), "contract_id"),
                scenario_id=_required_string(_mapping(item, "rollout"), "scenario_id"),
                rollout_index=_integer(_mapping(item, "rollout"), "rollout_index"),
                random_seed=_integer(_mapping(item, "rollout"), "random_seed"),
                job_id=_required_string(_mapping(item, "rollout"), "job_id"),
            )
            for item in raw_rollouts
        )
        if supplied_rollouts != contract.rollouts:
            raise ValueError("Pilot rollout assignments do not match contract")
        return contract

    @property
    def rollouts(self) -> tuple[PilotRolloutAssignment, ...]:
        return self._rollouts

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "budgets": self.budgets.to_dict(),
            "corpus": self.corpus.to_dict(),
            "pricing": self.pricing.to_dict(),
            "provider": self.provider.to_dict(),
            "quality_targets": self.quality_targets.to_dict(),
            "rollout_seeds": list(self.rollout_seeds),
            "schema_version": self.schema_version,
            "versions": self.versions.to_dict(),
        }

    def _build_rollouts(self) -> tuple[PilotRolloutAssignment, ...]:
        rollouts = []
        for scenario_id in self.corpus.scenario_ids:
            for rollout_index, random_seed in enumerate(self.rollout_seeds):
                rollouts.append(
                    PilotRolloutAssignment(
                        contract_id=self.contract_id,
                        scenario_id=scenario_id,
                        rollout_index=rollout_index,
                        random_seed=random_seed,
                    )
                )
        expected = EXPECTED_GOLD20_SCENARIOS * ROLLOUTS_PER_SCENARIO
        if len(rollouts) != expected:
            raise ValueError(f"Pilot must contain exactly {expected} rollout assignments")
        return tuple(rollouts)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_payload(),
            "contract_id": self.contract_id,
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def _normalize_endpoint(value: str) -> str:
    endpoint = value.strip()
    parsed = urlsplit(endpoint)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Provider endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Provider endpoint must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Provider endpoint must not contain a query or fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Provider endpoint port is invalid") from exc
    hostname = parsed.hostname.encode("idna").decode("ascii").lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = 443 if scheme == "https" else 80
    netloc = hostname if port in {None, default_port} else f"{hostname}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def _normalize_api_path(value: str) -> str:
    path = value.strip()
    if not path:
        raise ValueError("chat_completions_path cannot be empty")
    return path if path.startswith("/") else f"/{path}"


def _reject_credential_fields(value: Any, path: str = "request_body") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key in _CREDENTIAL_KEYS or key.endswith(("_api_key", "_password", "_secret")):
                raise ValueError(
                    "Provider request_body contains credential-like field: "
                    f"{path}.{key}"
                )
            _reject_credential_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_credential_fields(item, f"{path}[{index}]")


def _load_json_object(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gold-20 manifest must be a JSON object")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _plain_required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must contain non-empty strings")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string or null")
    return value


def _integer(value: Mapping[str, Any], key: str) -> int:
    return _plain_integer(value.get(key), key)


def _plain_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: Mapping[str, Any], key: str) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(item)


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_finite_float(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_positive_float(value: float, name: str) -> None:
    _require_finite_float(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_float(value: float, name: str) -> None:
    _require_finite_float(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _decimal(value: Any, name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    return "0" if normalized == 0 else text


def _usage_int(value: Mapping[str, Any], key: str, *, default: int) -> int:
    item = value.get(key, default)
    _require_nonnegative_int(item, key)
    return item


def _coalesced_usage_int(
    value: Mapping[str, Any],
    keys: Sequence[str],
    name: str,
) -> int | None:
    present = [(key, _usage_int(value, key, default=0)) for key in keys if key in value]
    if not present:
        return None
    amounts = {amount for _, amount in present}
    if len(amounts) != 1:
        aliases = ", ".join(key for key, _ in present)
        raise ValueError(f"Conflicting {name} aliases: {aliases}")
    return present[0][1]


def _coalesced_usage_details(
    value: Mapping[str, Any],
    keys: Sequence[str],
) -> Mapping[str, Any] | None:
    present = [(key, value[key]) for key in keys if key in value]
    for key, details in present:
        if not isinstance(details, Mapping):
            raise ValueError(f"{key} must be an object")
    if len(present) > 1 and any(details != present[0][1] for _, details in present[1:]):
        aliases = ", ".join(key for key, _ in present)
        raise ValueError(f"Conflicting token usage detail aliases: {aliases}")
    return present[0][1] if present else None


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{canonical_sha256(value)[:20]}"
