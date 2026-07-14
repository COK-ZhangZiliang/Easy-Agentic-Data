from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.models import stable_id
from easy_agentic_data.seeds import HiddenUserContext, PublicTaskContext, QuerySeed


def json_payload_contains_string(payload: Any, value: str) -> bool:
    """Match a decoded string against its canonical JSON-escaped payload representation."""

    if not value:
        return False
    encoded_value = json.dumps(value, ensure_ascii=True)[1:-1]
    encoded_payload = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return bool(encoded_value and encoded_value in encoded_payload)


@dataclass
class HiddenEvaluatorContext:
    """Private evaluator state that must never enter agent or user observations."""

    reference_answer: str = ""
    reference_artifacts: list[str] = field(default_factory=list)
    hidden_tests: list[str] = field(default_factory=list)
    required_state: dict[str, Any] = field(default_factory=dict)
    forbidden_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HiddenEvaluatorContext:
        return cls(**value)


@dataclass
class Scenario:
    """A query seed bound to a reproducible environment and hidden evaluator state."""

    query_seed: QuerySeed
    environment: EnvironmentSpec
    hidden_evaluator: HiddenEvaluatorContext = field(default_factory=HiddenEvaluatorContext)
    metadata: dict[str, Any] = field(default_factory=dict)
    scenario_id: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id:
            content = asdict(self)
            content.pop("scenario_id", None)
            self.scenario_id = stable_id("scenario", content)

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        value = {
            "scenario_id": self.scenario_id,
            "query_seed": self.query_seed.to_dict(include_hidden=include_hidden),
            "environment": self.environment.to_dict(),
            "metadata": self.metadata,
        }
        if include_hidden:
            value["hidden_evaluator"] = self.hidden_evaluator.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Scenario:
        data = dict(value)
        data["query_seed"] = QuerySeed.from_dict(data["query_seed"])
        data["environment"] = EnvironmentSpec.from_dict(data["environment"])
        data["hidden_evaluator"] = HiddenEvaluatorContext.from_dict(
            data.get("hidden_evaluator", {})
        )
        return cls(**data)


@dataclass
class ScenarioInstance:
    """Materialized scenario with explicit public and hidden context boundaries."""

    scenario_id: str
    environment_id: str
    public_task: PublicTaskContext
    hidden_user: HiddenUserContext
    hidden_evaluator: HiddenEvaluatorContext
    random_seed: int
    parameters: dict[str, Any] = field(default_factory=dict)
    initial_state_hash: str = ""
    instance_id: str = ""

    def __post_init__(self) -> None:
        if not self.instance_id:
            self.instance_id = stable_id(
                "instance",
                {
                    "scenario_id": self.scenario_id,
                    "environment_id": self.environment_id,
                    "public_task": self.public_task.to_dict(),
                    "hidden_user": self.hidden_user.to_dict(),
                    "hidden_evaluator": self.hidden_evaluator.to_dict(),
                    "random_seed": self.random_seed,
                    "parameters": self.parameters,
                    "initial_state_hash": self.initial_state_hash,
                },
            )

    @classmethod
    def materialize(
        cls,
        scenario: Scenario,
        *,
        random_seed: int,
        parameters: dict[str, Any] | None = None,
        initial_state_hash: str = "",
    ) -> ScenarioInstance:
        return cls(
            scenario_id=scenario.scenario_id,
            environment_id=scenario.environment.environment_id,
            public_task=scenario.query_seed.public,
            hidden_user=scenario.query_seed.hidden_user,
            hidden_evaluator=scenario.hidden_evaluator,
            random_seed=random_seed,
            parameters=parameters or {},
            initial_state_hash=initial_state_hash,
        )

    def public_view(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "scenario_id": self.scenario_id,
            "environment_id": self.environment_id,
            "public_task": self.public_task.to_dict(),
            "random_seed": self.random_seed,
            "parameters": self.parameters,
            "initial_state_hash": self.initial_state_hash,
        }

    def sensitive_strings(self) -> list[str]:
        """Return every distinct private string for post-run contamination auditing."""

        hidden_values: list[str] = []
        _collect_strings(self.hidden_user.goal, hidden_values)
        _collect_strings(self.hidden_user.unavailable_facts, hidden_values)
        _collect_strings(self.hidden_evaluator.reference_answer, hidden_values)
        _collect_strings(self.hidden_evaluator.reference_artifacts, hidden_values)
        _collect_strings(self.hidden_evaluator.hidden_tests, hidden_values)
        _collect_strings(self.hidden_evaluator.required_state, hidden_values)
        _collect_strings(self.hidden_evaluator.forbidden_state, hidden_values)
        _collect_strings(self.hidden_evaluator.metadata, hidden_values)
        return _distinct_private_strings(self, hidden_values)

    def trace_forbidden_strings(self) -> list[str]:
        """Return high-confidence private strings safe for literal live-trace rejection.

        Required-state fragments can legitimately be rediscovered as candidate code. They remain
        part of the complete post-run contamination audit, while the live recorder rejects only
        explicit private markers from those free-form fields to avoid blocking correct repairs.
        """

        hidden_values: list[str] = []
        _collect_strings(self.hidden_user.goal, hidden_values)
        _collect_strings(self.hidden_user.unavailable_facts, hidden_values)
        _collect_strings(self.hidden_evaluator.reference_answer, hidden_values)
        _collect_strings(self.hidden_evaluator.reference_artifacts, hidden_values)
        _collect_strings(self.hidden_evaluator.hidden_tests, hidden_values)
        _collect_strings(self.hidden_evaluator.metadata.get("test_patch"), hidden_values)
        nested_values: list[str] = []
        _collect_strings(self.hidden_evaluator.required_state, nested_values)
        _collect_strings(self.hidden_evaluator.forbidden_state, nested_values)
        _collect_strings(self.hidden_evaluator.metadata, nested_values)
        hidden_values.extend(value for value in nested_values if _is_private_marker(value))
        return _distinct_private_strings(self, hidden_values)

    def _observable_strings(self) -> list[str]:
        observable_values: list[str] = []
        _collect_strings(self.public_task.to_dict(), observable_values)
        _collect_strings(self.hidden_user.known_facts, observable_values)
        _collect_strings(
            self.hidden_user.interaction_policy.get("corrections", {}),
            observable_values,
        )
        return observable_values

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        value = self.public_view()
        if include_hidden:
            value["hidden_user"] = self.hidden_user.to_dict()
            value["hidden_evaluator"] = self.hidden_evaluator.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScenarioInstance:
        data = dict(value)
        data["public_task"] = PublicTaskContext.from_dict(data["public_task"])
        data["hidden_user"] = HiddenUserContext.from_dict(data.get("hidden_user", {}))
        data["hidden_evaluator"] = HiddenEvaluatorContext.from_dict(
            data.get("hidden_evaluator", {})
        )
        return cls(**data)


def _collect_strings(value: Any, output: list[str]) -> None:
    if isinstance(value, str):
        if value:
            output.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, output)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_strings(item, output)


def _distinct_private_strings(instance: ScenarioInstance, values: list[str]) -> list[str]:
    observable_values = instance._observable_strings()
    return sorted(
        {
            value
            for value in values
            if len(value) >= 8
            and not any(value in observable for observable in observable_values)
        }
    )


def _is_private_marker(value: str) -> bool:
    lowered = value.lower()
    markers = (
        "canary",
        "credential",
        "password",
        "private key",
        "secret",
        "api_key",
        "api-key",
        "authorization",
    )
    return len(value) >= 128 or any(marker in lowered for marker in markers)
