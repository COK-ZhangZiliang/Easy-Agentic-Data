from __future__ import annotations

import copy
import hashlib
import json
import re
import unittest

from easy_agentic_data.trajectory_review import (
    ReviewDecision,
    build_quarantine_set,
    build_trajectory_review_queue,
    summarize_review_gate,
    validate_review_decisions,
    validate_review_gate,
)

_CONTRACT_ID = "pilot_fixture_contract"


def _trace_summaries() -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    failure_terminations = ("max_turns", "tool_budget", "provider_error")
    for scenario_index in range(20):
        for rollout_index in range(2):
            success = rollout_index == 0
            job_id = f"pilotjob_{scenario_index:02d}_{rollout_index}"
            summary: dict[str, object] = {
                "contract_id": _CONTRACT_ID,
                "job_id": job_id,
                "trace_path": f"{job_id}.jsonl",
                "trace_id": f"trace_{scenario_index:02d}_{rollout_index}",
                "scenario_id": f"scenario_{scenario_index:02d}",
                "repository": f"org/repo-{scenario_index % 8}",
                "language": "python" if scenario_index < 16 else "javascript",
                "success": success,
                "termination_reason": (
                    "completed"
                    if success
                    else failure_terminations[scenario_index % len(failure_terminations)]
                ),
                "risk_score": 0 if success else 10 + scenario_index,
                "schema_valid": True,
                "replay_valid": True,
                "infrastructure_failure": False,
            }
            if scenario_index == 0 and rollout_index == 1:
                summary["hidden_content_leak"] = True
            summaries.append(summary)
    return summaries


def _decisions(
    queue: dict[str, object],
    verdicts: list[str],
    *,
    quarantine_indexes: set[int] | None = None,
) -> list[ReviewDecision]:
    quarantine_indexes = quarantine_indexes or set()
    items = queue["items"]
    assert isinstance(items, list)
    decisions = []
    for index, (item, verdict) in enumerate(zip(items, verdicts, strict=True)):
        assert isinstance(item, dict)
        decisions.append(
            ReviewDecision(
                trace_id=str(item["trace_id"]),
                reviewer_alias="reviewer-a",
                timestamp="2026-07-14T08:00:00+08:00",
                verdict=verdict,
                issue_codes=() if verdict == "acceptable" else (f"quality.{verdict}",),
                notes="Human-entered review note.",
                quarantine=index in quarantine_indexes,
            )
        )
    return decisions


def _rehash_gate(gate: dict[str, object]) -> dict[str, object]:
    material = {
        key: value for key, value in gate.items() if key != "review_gate_sha256"
    }
    encoded = json.dumps(
        material,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {**material, "review_gate_sha256": hashlib.sha256(encoded).hexdigest()}


class TrajectoryReviewTests(unittest.TestCase):
    def test_queue_is_exact_deterministic_risk_first_and_stratified(self) -> None:
        summaries = _trace_summaries()

        first = build_trajectory_review_queue(summaries)
        second = build_trajectory_review_queue(list(reversed(summaries)))

        self.assertEqual(first, second)
        self.assertEqual(first["source_trace_count"], 40)
        self.assertEqual(first["contract_id"], _CONTRACT_ID)
        self.assertEqual(first["sample_size"], 20)
        self.assertRegex(str(first["sample_sha256"]), r"^[0-9a-f]{64}$")
        self.assertRegex(str(first["queue_sha256"]), r"^[0-9a-f]{64}$")
        items = first["items"]
        assert isinstance(items, list)
        self.assertEqual(len(items), 20)
        self.assertEqual(len({item["trace_id"] for item in items}), 20)
        self.assertIn("trace_00_1", {item["trace_id"] for item in items})
        self.assertEqual(len({item["scenario_id"] for item in items}), 20)
        self.assertEqual({item["language"] for item in items}, {"python", "javascript"})
        self.assertEqual(len({item["repository"] for item in items}), 8)
        self.assertEqual({item["success"] for item in items}, {True, False})
        self.assertEqual(
            {item["termination_reason"] for item in items},
            {"completed", "max_turns", "tool_budget", "provider_error"},
        )
        self.assertEqual(items[0]["trace_id"], "trace_00_1")
        self.assertEqual(items[0]["job_id"], "pilotjob_00_1")
        self.assertEqual(items[0]["trace_path"], "pilotjob_00_1.jsonl")
        self.assertIn("hidden_content_leak", items[0]["risk_reasons"])
        self.assertNotIn("notes", items[0])

    def test_queue_rejects_noncanonical_source_counts_and_duplicate_trace_ids(self) -> None:
        summaries = _trace_summaries()
        with self.assertRaisesRegex(ValueError, "exactly 40"):
            build_trajectory_review_queue(summaries[:-1])

        summaries[-1]["trace_id"] = summaries[0]["trace_id"]
        with self.assertRaisesRegex(ValueError, "unique trace_id"):
            build_trajectory_review_queue(summaries)

    def test_queue_rejects_cross_contract_and_unsafe_or_mismatched_trace_paths(self) -> None:
        summaries = _trace_summaries()
        summaries[-1]["contract_id"] = "pilot_other_contract"
        with self.assertRaisesRegex(ValueError, "one contract_id"):
            build_trajectory_review_queue(summaries)

        for trace_path in (
            "../pilotjob_00_0.jsonl",
            "/tmp/pilotjob_00_0.jsonl",
            "nested/pilotjob_00_0.jsonl",
            "another-job.jsonl",
        ):
            with self.subTest(trace_path=trace_path):
                summaries = _trace_summaries()
                summaries[0]["trace_path"] = trace_path
                with self.assertRaisesRegex(ValueError, "safe canonical relative path"):
                    build_trajectory_review_queue(summaries)

    def test_queue_rejects_duplicate_job_ids(self) -> None:
        summaries = _trace_summaries()
        summaries[-1]["job_id"] = summaries[0]["job_id"]
        summaries[-1]["trace_path"] = summaries[0]["trace_path"]

        with self.assertRaisesRegex(ValueError, "unique job_id"):
            build_trajectory_review_queue(summaries)

    def test_review_decision_is_structured_and_canonically_serialized(self) -> None:
        decision = ReviewDecision(
            trace_id="trace_a",
            reviewer_alias="reviewer-a",
            timestamp="2026-07-14T08:00:00+08:00",
            verdict="minor",
            issue_codes=("quality.z", "quality.a"),
            notes="  Human note.\r\n",
            quarantine=False,
        )

        self.assertEqual(
            decision.to_dict(),
            {
                "schema_version": "easy_agentic_data.trajectory_review_decision.v1",
                "trace_id": "trace_a",
                "reviewer_alias": "reviewer-a",
                "timestamp": "2026-07-14T00:00:00Z",
                "verdict": "minor",
                "issue_codes": ["quality.a", "quality.z"],
                "notes": "Human note.",
                "quarantine": False,
            },
        )
        self.assertEqual(ReviewDecision.from_dict(decision.to_dict()), decision)
        with self.assertRaisesRegex(ValueError, "verdict"):
            ReviewDecision(
                trace_id="trace_a",
                reviewer_alias="reviewer-a",
                timestamp="2026-07-14T00:00:00Z",
                verdict="pending",
            )
        with self.assertRaisesRegex(ValueError, "issue code"):
            ReviewDecision(
                trace_id="trace_a",
                reviewer_alias="reviewer-a",
                timestamp="2026-07-14T00:00:00Z",
                verdict="critical",
            )

    def test_gate_passes_at_90_percent_and_quarantine_hash_is_stable(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())
        decisions = _decisions(queue, ["acceptable"] * 18 + ["minor"] * 2, quarantine_indexes={19})

        validated = validate_review_decisions(queue, list(reversed(decisions)))
        first_quarantine = build_quarantine_set(queue, decisions)
        second_quarantine = build_quarantine_set(queue, list(reversed(decisions)))
        summary = summarize_review_gate(queue, decisions)

        self.assertEqual(len(validated), 20)
        self.assertEqual(first_quarantine, second_quarantine)
        self.assertEqual(len(first_quarantine["trace_ids"]), 1)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", first_quarantine["quarantine_sha256"]))
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["acceptable_rate"], 0.9)
        self.assertEqual(summary["unresolved_trace_ids"], [])
        self.assertEqual(summary["critical_unquarantined_trace_ids"], [])

    def test_gate_is_self_contained_content_addressed_and_strictly_validated(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())
        decisions = _decisions(
            queue,
            ["acceptable"] * 18 + ["minor"] * 2,
            quarantine_indexes={19},
        )

        gate = summarize_review_gate(queue, list(reversed(decisions)))
        validated = validate_review_gate(queue, gate)

        self.assertEqual(validated, gate)
        self.assertEqual(
            gate["schema_version"],
            "easy_agentic_data.trajectory_review_gate.v3",
        )
        self.assertEqual(gate["contract_id"], _CONTRACT_ID)
        self.assertRegex(str(gate["review_gate_sha256"]), r"^[0-9a-f]{64}$")
        embedded = gate["decisions"]
        assert isinstance(embedded, list)
        self.assertEqual(len(embedded), 20)
        self.assertEqual(
            [decision["trace_id"] for decision in embedded],
            sorted(decision.trace_id for decision in decisions),
        )
        self.assertEqual(
            gate["quarantined_trace_ids"],
            sorted(decision.trace_id for decision in decisions if decision.quarantine),
        )

    def test_review_gate_rejects_tampering_even_when_outer_hash_is_recomputed(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())
        decisions = _decisions(
            queue,
            ["acceptable"] * 18 + ["minor"] * 2,
            quarantine_indexes={19},
        )
        gate = summarize_review_gate(queue, decisions)

        stale_hash = copy.deepcopy(gate)
        stale_hash["acceptable_count"] = 17
        with self.assertRaisesRegex(ValueError, "gate hash mismatch"):
            validate_review_gate(queue, stale_hash)

        tampered_rate = copy.deepcopy(gate)
        tampered_rate["acceptable_rate"] = 0.95
        with self.assertRaisesRegex(ValueError, "derived fields mismatch"):
            validate_review_gate(queue, _rehash_gate(tampered_rate))

        tampered_quarantine = copy.deepcopy(gate)
        tampered_quarantine["quarantined_trace_ids"] = []
        with self.assertRaisesRegex(ValueError, "derived fields mismatch"):
            validate_review_gate(queue, _rehash_gate(tampered_quarantine))

        bool_like_pass = copy.deepcopy(gate)
        bool_like_pass["passed"] = "true"
        with self.assertRaisesRegex(ValueError, "derived fields mismatch"):
            validate_review_gate(queue, _rehash_gate(bool_like_pass))

        extra_field = {**gate, "approval": True}
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_review_gate(queue, extra_field)

    def test_review_gate_rejects_tampered_decisions_and_cross_queue_binding(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())
        decisions = _decisions(queue, ["acceptable"] * 20)
        gate = summarize_review_gate(queue, decisions)

        tampered_decision = copy.deepcopy(gate)
        embedded = tampered_decision["decisions"]
        assert isinstance(embedded, list)
        embedded[0]["reviewer_alias"] = "reviewer-b"
        with self.assertRaisesRegex(ValueError, "derived fields mismatch"):
            validate_review_gate(queue, _rehash_gate(tampered_decision))

        malformed_decision = copy.deepcopy(gate)
        embedded = malformed_decision["decisions"]
        assert isinstance(embedded, list)
        embedded[0]["quarantine"] = "false"
        with self.assertRaisesRegex(ValueError, "quarantine must be a boolean"):
            validate_review_gate(queue, _rehash_gate(malformed_decision))

        summaries = _trace_summaries()
        summaries[0]["repository"] = "org/different-repo"
        other_queue = build_trajectory_review_queue(summaries)
        with self.assertRaisesRegex(ValueError, "queue binding mismatch"):
            validate_review_gate(other_queue, gate)

        other_contract_summaries = _trace_summaries()
        for summary in other_contract_summaries:
            summary["contract_id"] = "pilot_other_contract"
        other_contract_queue = build_trajectory_review_queue(other_contract_summaries)
        with self.assertRaisesRegex(ValueError, "contract binding mismatch"):
            validate_review_gate(other_contract_queue, gate)

        cross_contract = copy.deepcopy(queue)
        cross_contract["contract_id"] = "pilot_other_contract"
        cross_contract_material = {
            key: value for key, value in cross_contract.items() if key != "queue_sha256"
        }
        cross_contract["queue_sha256"] = hashlib.sha256(
            json.dumps(
                cross_contract_material,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "item contract binding mismatch"):
            validate_review_gate(cross_contract, gate)

    def test_validate_review_gate_enforces_the_human_quality_gate(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())

        low_acceptance = summarize_review_gate(
            queue,
            _decisions(queue, ["acceptable"] * 17 + ["minor"] * 3),
        )
        with self.assertRaisesRegex(ValueError, "below 90 percent"):
            validate_review_gate(queue, low_acceptance)

        critical = summarize_review_gate(
            queue,
            _decisions(queue, ["acceptable"] * 19 + ["critical"]),
        )
        with self.assertRaisesRegex(ValueError, "must be quarantined"):
            validate_review_gate(queue, critical)

        unresolved_decisions = _decisions(queue, ["acceptable"] * 19 + ["minor"])
        unresolved_decisions[-1] = ReviewDecision.from_dict(
            {
                **unresolved_decisions[-1].to_dict(),
                "issue_codes": ["unresolved"],
            }
        )
        unresolved = summarize_review_gate(queue, unresolved_decisions)
        with self.assertRaisesRegex(ValueError, "unresolved decisions"):
            validate_review_gate(queue, unresolved)

        incomplete = summarize_review_gate(
            queue,
            _decisions(queue, ["acceptable"] * 20)[:-1],
        )
        with self.assertRaisesRegex(ValueError, "exactly 20 decisions"):
            validate_review_gate(queue, incomplete)

    def test_gate_requires_every_critical_decision_to_be_quarantined(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())
        decisions = _decisions(queue, ["acceptable"] * 19 + ["critical"])

        held = summarize_review_gate(queue, decisions)
        quarantined_decisions = list(decisions)
        quarantined_decisions[-1] = ReviewDecision.from_dict(
            {**decisions[-1].to_dict(), "quarantine": True}
        )
        passed = summarize_review_gate(queue, quarantined_decisions)

        self.assertFalse(held["passed"])
        self.assertEqual(
            held["critical_unquarantined_trace_ids"],
            [decisions[-1].trace_id],
        )
        self.assertTrue(passed["passed"])

    def test_gate_does_not_invent_missing_human_decisions(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())
        decisions = _decisions(queue, ["acceptable"] * 20)

        incomplete = summarize_review_gate(queue, decisions[:-1])

        self.assertFalse(incomplete["passed"])
        self.assertEqual(incomplete["decision_count"], 19)
        self.assertEqual(incomplete["unresolved_trace_ids"], [decisions[-1].trace_id])
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            validate_review_decisions(queue, decisions[:-1])

    def test_gate_rejects_low_acceptance_and_duplicate_decisions(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())
        decisions = _decisions(queue, ["acceptable"] * 17 + ["minor"] * 3)

        summary = summarize_review_gate(queue, decisions)

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["acceptable_rate"], 0.85)
        with self.assertRaisesRegex(ValueError, "unique trace_id"):
            validate_review_decisions(queue, decisions[:-1] + [decisions[0]])

    def test_serialized_decision_rejects_missing_extra_and_bool_like_fields(self) -> None:
        queue = build_trajectory_review_queue(_trace_summaries())
        value = _decisions(queue, ["acceptable"] * 20)[0].to_dict()

        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            ReviewDecision.from_dict({**value, "inferred": True})
        missing_schema = dict(value)
        del missing_schema["schema_version"]
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            ReviewDecision.from_dict(missing_schema)
        with self.assertRaisesRegex(ValueError, "quarantine must be a boolean"):
            ReviewDecision.from_dict({**value, "quarantine": "false"})


if __name__ == "__main__":
    unittest.main()
