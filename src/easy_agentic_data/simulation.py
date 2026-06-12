from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import Message
from easy_agentic_data.scenarios import ScenarioInstance


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
    responses: List[str] = field(default_factory=list)

    @property
    def diversity(self) -> float:
        return len(set(self.responses)) / len(self.responses) if self.responses else 0.0


class UserSimulator(Protocol):
    metrics: InteractionMetrics

    def respond(self, observation: UserObservation) -> UserResponse: ...


class RuleBasedUserSimulator:
    """Deterministic simulator that can reveal only explicitly known facts."""

    def __init__(self, instance: ScenarioInstance) -> None:
        self.state = instance.hidden_user
        self.metrics = InteractionMetrics()

    def respond(self, observation: UserObservation) -> UserResponse:
        self.metrics.turns += 1
        if self.metrics.turns > self.state.patience_turns:
            self.metrics.early_stops += 1
            return UserResponse(None, "stop", "Patience budget exhausted")
        question = observation.question.lower()
        corrections = self.state.interaction_policy.get("corrections", {})
        for topic, correction in corrections.items():
            if topic.lower() in question:
                content = str(correction)
                self.metrics.corrections += 1
                self.metrics.responses.append(content)
                return UserResponse(content, "correct", f"Corrected topic: {topic}")
        for key, value in self.state.known_facts.items():
            if key.lower().replace("_", " ") in question or key.lower() in question:
                content = str(value)
                self.metrics.clarifications += 1
                self.metrics.responses.append(content)
                return UserResponse(content, "clarify", f"Answered known fact: {key}")
        if any(fact.lower() in question for fact in self.state.unavailable_facts):
            self.metrics.refusals += 1
            content = "I do not have that information."
            self.metrics.responses.append(content)
            return UserResponse(content, "refuse", "Requested fact is unavailable")
        if "confirm" in question or "proceed" in question:
            self.metrics.confirmations += 1
            content = "Yes, proceed within the stated constraints."
            self.metrics.responses.append(content)
            return UserResponse(content, "confirm", "Confirmed requested action")
        self.metrics.refusals += 1
        content = "Please use the information already provided."
        self.metrics.responses.append(content)
        return UserResponse(content, "refuse", "Question does not match a known fact")


class LLMUserSimulator:
    """LLM simulator constrained to user state and a narrow observation boundary."""

    def __init__(self, client: LLMClient, instance: ScenarioInstance) -> None:
        self.client = client
        self.instance = instance
        self.metrics = InteractionMetrics()
        self._answers: Dict[str, str] = {}

    def respond(self, observation: UserObservation) -> UserResponse:
        self.metrics.turns += 1
        if self.metrics.turns > self.instance.hidden_user.patience_turns:
            self.metrics.early_stops += 1
            return UserResponse(None, "stop", "Patience budget exhausted")
        payload = {
            "public_query": observation.public_query,
            "question": observation.question,
            "persona": self.instance.hidden_user.persona,
            "known_facts": self.instance.hidden_user.known_facts,
            "unavailable_facts": self.instance.hidden_user.unavailable_facts,
            "constraints": self.instance.hidden_user.constraints,
        }
        response = self.client.complete(
            [
                Message(
                    "system",
                    "Act as the user. Use only supplied known_facts. Never infer hidden answers. "
                    "Return JSON with content, action, and reason.",
                ),
                Message("user", json.dumps(payload, ensure_ascii=True)),
            ],
            temperature=0.7,
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
            self._answers[observation.question] = result.content
            self.metrics.responses.append(result.content)
        if result.action == "clarify":
            self.metrics.clarifications += 1
        elif result.action == "correct":
            self.metrics.corrections += 1
        elif result.action == "refuse":
            self.metrics.refusals += 1
        elif result.action == "confirm":
            self.metrics.confirmations += 1
        elif result.action == "stop":
            self.metrics.early_stops += 1
        return result


def user_callback(simulator: UserSimulator, instance: ScenarioInstance):
    turn = 0

    def answer(question: str) -> str | None:
        nonlocal turn
        turn += 1
        return simulator.respond(
            UserObservation(question, instance.public_task.query, turn)
        ).content

    return answer
