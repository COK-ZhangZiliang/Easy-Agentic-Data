from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Protocol

from easy_agentic_data.llm.base import LLMClient
from easy_agentic_data.models import Message, Trajectory, Verification


class Verifier(Protocol):
    name: str

    def verify(self, trajectory: Trajectory) -> Verification:
        """Score a trajectory in the inclusive range [0, 1]."""


class StructuralVerifier:
    name = "structural"

    def verify(self, trajectory: Trajectory) -> Verification:
        reasons: list[str] = []
        if trajectory.status != "completed":
            reasons.append(f"status={trajectory.status}")
        if not trajectory.messages or trajectory.messages[-1].role != "assistant":
            reasons.append("missing final assistant message")
        if not (trajectory.messages[-1].content if trajectory.messages else None):
            reasons.append("empty final answer")
        score = 1.0 if not reasons else 0.0
        return Verification(self.name, not reasons, score, "; ".join(reasons) or "Valid structure")


class ToolExecutionVerifier:
    name = "tool_execution"

    def verify(self, trajectory: Trajectory) -> Verification:
        expected = set(trajectory.task.expected_tools)
        observed = {event.name for event in trajectory.tool_events if event.error is None}
        failed = [event for event in trajectory.tool_events if event.error is not None]
        missing = expected - observed
        passed = not failed and not missing
        if passed:
            reason = "All expected tool calls executed successfully"
            score = 1.0
        else:
            details = []
            if missing:
                details.append(f"missing tools: {sorted(missing)}")
            if failed:
                details.append(f"failed calls: {len(failed)}")
            reason = "; ".join(details)
            score = max(0.0, 1.0 - 0.5 * len(missing) - 0.25 * len(failed))
        return Verification(self.name, passed, score, reason)


class SemanticLLMVerifier:
    name = "semantic_llm"
    system_prompt = """\
SEMANTIC_JUDGE
Evaluate only the supplied task, messages, and tool events.

Rules:
- Treat successful tool outputs as ground truth.
- Fail if the final answer contradicts tool output, misses any explicit constraint, or claims work
  that was not observed.
- Do not reward style, confidence, or unsupported reasoning.
- Set passed=true only when the task is fully solved.
- Use score=1 for a complete correct solution, a value below 1 for minor quality issues, and 0 for
  an incorrect or incomplete solution.

Return one JSON object and no prose:
{"passed": true, "score": 1.0, "reason": "Concise evidence-based explanation."}
"""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def verify(self, trajectory: Trajectory) -> Verification:
        payload = {
            "task": {
                "instruction": trajectory.task.instruction,
                "constraints": trajectory.task.constraints,
                "reference": trajectory.task.reference,
            },
            "messages": [message.to_api_dict() for message in trajectory.messages],
            "tool_events": [
                {
                    "name": event.name,
                    "arguments": event.arguments,
                    "output": event.output,
                    "error": event.error,
                }
                for event in trajectory.tool_events
            ],
        }
        response = self.client.complete(
            [
                Message("system", self.system_prompt),
                Message(
                    "user",
                    "Judge this trajectory and return JSON:\n"
                    + json.dumps(payload, ensure_ascii=True),
                ),
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.message.content or "{}")
        score = max(0.0, min(1.0, float(result.get("score", 0.0))))
        return Verification(
            self.name,
            bool(result.get("passed", False)),
            score,
            str(result.get("reason", "No reason supplied")),
        )


class VerificationSuite:
    def __init__(self, verifiers: Iterable[Verifier]) -> None:
        self.verifiers = list(verifiers)

    def evaluate(self, trajectory: Trajectory) -> Trajectory:
        results: list[Verification] = []
        for verifier in self.verifiers:
            try:
                results.append(verifier.verify(trajectory))
            except Exception as exc:
                results.append(
                    Verification(
                        verifier=verifier.name,
                        passed=False,
                        score=0.0,
                        reason=f"Verifier error: {type(exc).__name__}: {exc}",
                    )
                )
        trajectory.verifications = results
        if not trajectory.verifications:
            trajectory.reward = 0.0
        else:
            # Any failed verifier is conservative by default; policy can become configurable later.
            all_pass = all(result.passed for result in trajectory.verifications)
            mean_score = sum(result.score for result in trajectory.verifications) / len(
                trajectory.verifications
            )
            trajectory.reward = mean_score if all_pass else 0.0
        return trajectory
