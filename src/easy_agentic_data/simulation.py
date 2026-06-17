from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import Message
from easy_agentic_data.scenarios import ScenarioInstance
from easy_agentic_data.seeds import HiddenUserContext


@dataclass(frozen=True)
class UserObservation:
    question: str
    public_query: str
    turn: int


@dataclass(frozen=True)
class UserResponse:
    content: str | None
    action: str
    reason: str


@dataclass
class InteractionMetrics:
    turns: int = 0
    clarifications: int = 0
    corrections: int = 0
    refusals: int = 0
    confirmations: int = 0
    contradictions: int = 0
    early_stops: int = 0
    goal_components_total: int = 0
    goal_components_satisfied: int = 0
    disclosure_violations: int = 0
    unavailable_fact_requests: int = 0
    unavailable_fact_leaks: int = 0
    critical_simulator_errors: int = 0
    responses: list[str] = field(default_factory=list)

    @property
    def diversity(self) -> float:
        return len(set(self.responses)) / len(self.responses) if self.responses else 0.0

    @property
    def goal_alignment(self) -> float:
        if self.goal_components_total == 0:
            return 1.0
        return self.goal_components_satisfied / self.goal_components_total

    @property
    def simulator_error_rate(self) -> float:
        if self.turns == 0:
            return 0.0
        return self.critical_simulator_errors / self.turns

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "clarifications": self.clarifications,
            "corrections": self.corrections,
            "refusals": self.refusals,
            "confirmations": self.confirmations,
            "contradictions": self.contradictions,
            "early_stops": self.early_stops,
            "diversity": self.diversity,
            "goal_components_total": self.goal_components_total,
            "goal_components_satisfied": self.goal_components_satisfied,
            "goal_alignment": self.goal_alignment,
            "disclosure_violations": self.disclosure_violations,
            "unavailable_fact_requests": self.unavailable_fact_requests,
            "unavailable_fact_leaks": self.unavailable_fact_leaks,
            "critical_simulator_errors": self.critical_simulator_errors,
            "simulator_error_rate": self.simulator_error_rate,
        }


class UserGoalTracker:
    """Track hidden user-goal coverage without exposing hidden values."""

    def __init__(self, state: HiddenUserContext) -> None:
        components = state.goal_components or {key: key for key in state.known_facts}
        self.components = {str(key): str(value) for key, value in components.items()}
        self.status = {key: "pending" for key in self.components}

    @property
    def total(self) -> int:
        return len(self.status)

    @property
    def satisfied(self) -> int:
        return sum(1 for status in self.status.values() if status == "satisfied")

    def update_from_known_fact(self, key: str) -> None:
        if key in self.status:
            self.status[key] = "satisfied"

    def update_from_text(self, text: str) -> None:
        lowered = text.lower()
        for key, description in self.components.items():
            aliases = {key.lower().replace("_", " "), key.lower(), description.lower()}
            if any(alias and alias in lowered for alias in aliases):
                self.status[key] = "satisfied"

    @property
    def complete(self) -> bool:
        return self.total == 0 or self.satisfied == self.total


class UserSimulator(Protocol):
    metrics: InteractionMetrics

    def respond(self, observation: UserObservation) -> UserResponse: ...


class RuleBasedUserSimulator:
    """Deterministic simulator that can reveal only explicitly known facts."""

    def __init__(self, instance: ScenarioInstance) -> None:
        self.state = instance.hidden_user
        self.metrics = InteractionMetrics()
        self.goal_tracker = UserGoalTracker(self.state)
        self._sync_goal_metrics()

    def respond(self, observation: UserObservation) -> UserResponse:
        self.metrics.turns += 1
        if self.metrics.turns > self.state.patience_turns:
            self.metrics.early_stops += 1
            if not self.goal_tracker.complete:
                self.metrics.critical_simulator_errors += 1
            self._sync_goal_metrics()
            return UserResponse(None, "stop", "Patience budget exhausted")
        question = observation.question.lower()
        corrections = self.state.interaction_policy.get("corrections", {})
        for topic, correction in corrections.items():
            if topic.lower() in question:
                content = str(correction)
                self.metrics.corrections += 1
                self.metrics.responses.append(content)
                self._observe_response(observation.question, content)
                return UserResponse(content, "correct", f"Corrected topic: {topic}")
        for key, value in self.state.known_facts.items():
            if key.lower().replace("_", " ") in question or key.lower() in question:
                content = str(value)
                self.metrics.clarifications += 1
                self.metrics.responses.append(content)
                self.goal_tracker.update_from_known_fact(key)
                self._observe_response(observation.question, content)
                return UserResponse(content, "clarify", f"Answered known fact: {key}")
        if any(fact.lower() in question for fact in self.state.unavailable_facts):
            self.metrics.refusals += 1
            self.metrics.unavailable_fact_requests += 1
            content = "I do not have that information."
            self.metrics.responses.append(content)
            self._observe_response(observation.question, content)
            return UserResponse(content, "refuse", "Requested fact is unavailable")
        if "confirm" in question or "proceed" in question:
            self.metrics.confirmations += 1
            content = "Yes, proceed within the stated constraints."
            self.metrics.responses.append(content)
            self._observe_response(observation.question, content)
            return UserResponse(content, "confirm", "Confirmed requested action")
        self.metrics.refusals += 1
        content = "Please use the information already provided."
        self.metrics.responses.append(content)
        if _mentions_known_fact(observation.question, self.state.known_facts):
            self.metrics.critical_simulator_errors += 1
        self._observe_response(observation.question, content)
        return UserResponse(content, "refuse", "Question does not match a known fact")

    def _observe_response(self, question: str, content: str) -> None:
        self.goal_tracker.update_from_text(question)
        self.goal_tracker.update_from_text(content)
        if _leaks_unavailable_fact(content, self.state.unavailable_facts):
            self.metrics.unavailable_fact_leaks += 1
            self.metrics.critical_simulator_errors += 1
        if _discloses_unasked_known_fact(question, content, self.state):
            self.metrics.disclosure_violations += 1
        self._sync_goal_metrics()

    def _sync_goal_metrics(self) -> None:
        self.metrics.goal_components_total = self.goal_tracker.total
        self.metrics.goal_components_satisfied = self.goal_tracker.satisfied


class LLMUserSimulator:
    """LLM simulator constrained to user state and a narrow observation boundary."""

    def __init__(self, client: LLMClient, instance: ScenarioInstance) -> None:
        self.client = client
        self.instance = instance
        self.metrics = InteractionMetrics()
        self._answers: dict[str, str] = {}
        self.goal_tracker = UserGoalTracker(instance.hidden_user)
        self._sync_goal_metrics()

    def respond(self, observation: UserObservation) -> UserResponse:
        self.metrics.turns += 1
        if self.metrics.turns > self.instance.hidden_user.patience_turns:
            self.metrics.early_stops += 1
            if not self.goal_tracker.complete:
                self.metrics.critical_simulator_errors += 1
            self._sync_goal_metrics()
            return UserResponse(None, "stop", "Patience budget exhausted")
        payload = {
            "public_query": observation.public_query,
            "question": observation.question,
            "persona": self.instance.hidden_user.persona,
            "goal_components": list(self.instance.hidden_user.goal_components.keys()),
            "known_facts": self.instance.hidden_user.known_facts,
            "unavailable_facts": self.instance.hidden_user.unavailable_facts,
            "constraints": self.instance.hidden_user.constraints,
            "disclosure_policy": self.instance.hidden_user.disclosure_policy,
            "stop_conditions": self.instance.hidden_user.stop_conditions,
        }
        response = self.client.complete(
            [
                Message(
                    "system",
                    "Act only as the user described by the supplied state. Use known_facts "
                    "verbatim when they answer the question. Never infer unavailable facts, "
                    "evaluation answers, or workspace state. Keep the response concise and "
                    "consistent with earlier answers. Return one JSON object and no prose: "
                    '{"content": "answer or null", '
                    '"action": "clarify|correct|refuse|confirm|stop", '
                    '"reason": "brief explanation"}.',
                ),
                Message(
                    "user",
                    "Respond to this observation as JSON:\n"
                    + json.dumps(payload, ensure_ascii=True),
                ),
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        value = json.loads(response.message.content or "{}")
        result = UserResponse(
            value.get("content"),
            str(value.get("action", "clarify")),
            str(value.get("reason", "")),
        )
        if result.content:
            previous = self._answers.get(observation.question)
            if previous is not None and previous != result.content:
                self.metrics.contradictions += 1
                self.metrics.critical_simulator_errors += 1
            self._answers[observation.question] = result.content
            self.metrics.responses.append(result.content)
            self._observe_response(observation.question, result.content, result.action)
        if result.action == "clarify":
            self.metrics.clarifications += 1
        elif result.action == "correct":
            self.metrics.corrections += 1
        elif result.action == "refuse":
            self.metrics.refusals += 1
            if _mentions_known_fact(observation.question, self.instance.hidden_user.known_facts):
                self.metrics.critical_simulator_errors += 1
        elif result.action == "confirm":
            self.metrics.confirmations += 1
        elif result.action == "stop":
            self.metrics.early_stops += 1
            if not self.goal_tracker.complete:
                self.metrics.critical_simulator_errors += 1
        self._sync_goal_metrics()
        return result

    def _observe_response(self, question: str, content: str, action: str) -> None:
        state = self.instance.hidden_user
        self.goal_tracker.update_from_text(question)
        self.goal_tracker.update_from_text(content)
        if _leaks_unavailable_fact(content, state.unavailable_facts):
            self.metrics.unavailable_fact_leaks += 1
            self.metrics.critical_simulator_errors += 1
        if _discloses_unasked_known_fact(question, content, state):
            self.metrics.disclosure_violations += 1
        if action == "clarify":
            for key in state.known_facts:
                if _matches_fact_name(question, key):
                    self.goal_tracker.update_from_known_fact(key)
        self._sync_goal_metrics()

    def _sync_goal_metrics(self) -> None:
        self.metrics.goal_components_total = self.goal_tracker.total
        self.metrics.goal_components_satisfied = self.goal_tracker.satisfied


def user_callback(simulator: UserSimulator, instance: ScenarioInstance):
    turn = 0

    def answer(question: str) -> str | None:
        nonlocal turn
        turn += 1
        return simulator.respond(
            UserObservation(question, instance.public_task.query, turn)
        ).content

    return answer


def _mentions_known_fact(question: str, facts: dict[str, Any]) -> bool:
    return any(_matches_fact_name(question, key) for key in facts)


def _matches_fact_name(text: str, key: str) -> bool:
    lowered = text.lower()
    return key.lower() in lowered or key.lower().replace("_", " ") in lowered


def _leaks_unavailable_fact(content: str, unavailable_facts: list[str]) -> bool:
    lowered = content.lower()
    return any(fact.lower() in lowered for fact in unavailable_facts)


def _discloses_unasked_known_fact(
    question: str,
    content: str,
    state: HiddenUserContext,
) -> bool:
    only_when_asked = set(state.disclosure_policy.get("only_when_asked", []))
    if not only_when_asked:
        return False
    lowered_content = content.lower()
    for key in only_when_asked:
        if _matches_fact_name(question, key):
            continue
        if key in state.known_facts and str(state.known_facts[key]).lower() in lowered_content:
            return True
    return False
