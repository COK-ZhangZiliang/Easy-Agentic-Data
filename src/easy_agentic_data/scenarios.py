from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from easy_agentic_data.environments import EnvironmentSpec
from easy_agentic_data.models import stable_id
from easy_agentic_data.seeds import HiddenUserContext, PublicTaskContext, QuerySeed


@dataclass
class HiddenEvaluatorContext:
    """Private evaluator state that must never enter agent or user observations."""

    reference_answer: str = ""
    reference_artifacts: List[str] = field(default_factory=list)
    hidden_tests: List[str] = field(default_factory=list)
    required_state: Dict[str, Any] = field(default_factory=dict)
    forbidden_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "HiddenEvaluatorContext":
        return cls(**value)


@dataclass
class Scenario:
    """A query seed bound to a reproducible environment and hidden evaluator state."""

    query_seed: QuerySeed
    environment: EnvironmentSpec
    hidden_evaluator: HiddenEvaluatorContext = field(default_factory=HiddenEvaluatorContext)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scenario_id: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id:
            content = asdict(self)
            content.pop("scenario_id", None)
            self.scenario_id = stable_id("scenario", content)

    def to_dict(self, *, include_hidden: bool = True) -> Dict[str, Any]:
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
    def from_dict(cls, value: Dict[str, Any]) -> "Scenario":
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
    parameters: Dict[str, Any] = field(default_factory=dict)
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
        parameters: Dict[str, Any] | None = None,
        initial_state_hash: str = "",
    ) -> "ScenarioInstance":
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

    def public_view(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "scenario_id": self.scenario_id,
            "environment_id": self.environment_id,
            "public_task": self.public_task.to_dict(),
            "random_seed": self.random_seed,
            "parameters": self.parameters,
            "initial_state_hash": self.initial_state_hash,
        }

    def sensitive_strings(self) -> List[str]:
        hidden_values: List[str] = []
        public_values: List[str] = []
        _collect_strings(self.hidden_user.goal, hidden_values)
        _collect_strings(self.hidden_user.unavailable_facts, hidden_values)
        _collect_strings(self.hidden_evaluator.reference_answer, hidden_values)
        _collect_strings(self.hidden_evaluator.reference_artifacts, hidden_values)
        _collect_strings(self.hidden_evaluator.hidden_tests, hidden_values)
        _collect_strings(self.public_task.to_dict(), public_values)
        public = set(public_values)
        return sorted(
            {
                value
                for value in hidden_values
                if len(value) >= 8 and value not in public
            }
        )

    def to_dict(self, *, include_hidden: bool = True) -> Dict[str, Any]:
        value = self.public_view()
        if include_hidden:
            value["hidden_user"] = self.hidden_user.to_dict()
            value["hidden_evaluator"] = self.hidden_evaluator.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ScenarioInstance":
        data = dict(value)
        data["public_task"] = PublicTaskContext.from_dict(data["public_task"])
        data["hidden_user"] = HiddenUserContext.from_dict(data.get("hidden_user", {}))
        data["hidden_evaluator"] = HiddenEvaluatorContext.from_dict(
            data.get("hidden_evaluator", {})
        )
        return cls(**data)


def _collect_strings(value: Any, output: List[str]) -> None:
    if isinstance(value, str):
        if value:
            output.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, output)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_strings(item, output)
