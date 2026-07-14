from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from easy_agentic_data.batch import RolloutOutcome
from easy_agentic_data.pilot_contract import Gold20Binding, canonical_sha256
from easy_agentic_data.pilot_usage_ledger import (
    LEDGER_DIRECTORY_NAME,
    PilotUsageAttempt,
    UnknownProviderUsageError,
    audit_pilot_usage_ledger,
)
from tests.test_pilot_contract import _contract, _gold20_manifest


class PilotUsageLedgerTests(unittest.TestCase):
    def test_completed_attempt_recomputes_usage_without_prompt_content(self) -> None:
        contract = _pilot_contract()
        assignment = contract.rollouts[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = PilotUsageAttempt(
                root,
                contract_id=contract.contract_id,
                job_id=assignment.job_id,
                attempt_id="attempt_1",
            )
            _record_call(attempt, response_id="completion_1")
            attempt.finalize(
                RolloutOutcome(
                    infrastructure_failure=True,
                    tokens=15,
                    cost=0.00002,
                    error="retryable verifier host failure",
                ),
                elapsed_ms=12.5,
            )
            rows = _rows(contract)
            row = next(item for item in rows if item["job_id"] == assignment.job_id)
            row["attempts"] = 1
            audit = audit_pilot_usage_ledger(contract, rows, root)
            ledger_text = "".join(
                path.read_text(encoding="utf-8")
                for path in (root / LEDGER_DIRECTORY_NAME).rglob("*.json")
            )
            partials = list((root / LEDGER_DIRECTORY_NAME).rglob("*.tmp"))

        state = audit.jobs[assignment.job_id]
        self.assertEqual(state.totals.tokens, 15)
        self.assertAlmostEqual(state.totals.cost, 0.00002)
        self.assertEqual(state.totals.elapsed_ms, 12.5)
        self.assertEqual(state.attempt_count, 1)
        self.assertEqual(state.call_count, 1)
        self.assertNotIn("private prompt content", ledger_text)
        self.assertEqual(partials, [])

    def test_started_only_and_missing_terminal_fail_closed(self) -> None:
        contract = _pilot_contract()
        assignment = contract.rollouts[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = PilotUsageAttempt(
                root,
                contract_id=contract.contract_id,
                job_id=assignment.job_id,
                attempt_id="attempt_1",
            )
            _record_started(attempt)
            rows = _rows(contract)
            next(item for item in rows if item["job_id"] == assignment.job_id)["attempts"] = 1

            with self.assertRaisesRegex(UnknownProviderUsageError, "terminal"):
                audit_pilot_usage_ledger(contract, rows, root)

    def test_content_hash_tampering_is_rejected(self) -> None:
        contract = _pilot_contract()
        assignment = contract.rollouts[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = PilotUsageAttempt(
                root,
                contract_id=contract.contract_id,
                job_id=assignment.job_id,
                attempt_id="attempt_1",
            )
            _record_call(attempt, response_id="completion_1")
            attempt.finalize(
                RolloutOutcome(
                    infrastructure_failure=True,
                    tokens=15,
                    cost=0.00002,
                ),
                elapsed_ms=1.0,
            )
            completed = next(attempt.directory.glob("call-*.completed.*.json"))
            value = json.loads(completed.read_text(encoding="utf-8"))
            value["usage"]["prompt_tokens"] = 999
            completed.write_text(json.dumps(value), encoding="utf-8")
            rows = _rows(contract)
            next(item for item in rows if item["job_id"] == assignment.job_id)["attempts"] = 1

            with self.assertRaisesRegex(ValueError, "content hash"):
                audit_pilot_usage_ledger(contract, rows, root)

    def test_provider_receipt_id_cannot_be_counted_twice_across_attempts(self) -> None:
        contract = _pilot_contract()
        assignment = contract.rollouts[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (1, 2):
                attempt = PilotUsageAttempt(
                    root,
                    contract_id=contract.contract_id,
                    job_id=assignment.job_id,
                    attempt_id=f"attempt_{index}",
                )
                _record_call(attempt, response_id="completion_reused")
                attempt.finalize(
                    RolloutOutcome(
                        infrastructure_failure=True,
                        tokens=15,
                        cost=0.00002,
                    ),
                    elapsed_ms=1.0,
                )
            rows = _rows(contract)
            next(item for item in rows if item["job_id"] == assignment.job_id)["attempts"] = 2

            with self.assertRaisesRegex(ValueError, "globally unique"):
                audit_pilot_usage_ledger(contract, rows, root)

    def test_scheduler_attempt_count_must_match_terminal_attempts(self) -> None:
        contract = _pilot_contract()
        assignment = contract.rollouts[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = PilotUsageAttempt(
                root,
                contract_id=contract.contract_id,
                job_id=assignment.job_id,
                attempt_id="attempt_1",
            )
            attempt.finalize(
                RolloutOutcome(infrastructure_failure=True),
                elapsed_ms=1.0,
            )

            with self.assertRaisesRegex(UnknownProviderUsageError, "attempt count"):
                audit_pilot_usage_ledger(contract, _rows(contract), root)

    def test_ledger_and_attempt_symlinks_are_rejected(self) -> None:
        contract = _pilot_contract()
        assignment = contract.rollouts[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_root = root / "traces"
            outside = root / "outside"
            trace_root.mkdir()
            outside.mkdir()
            (trace_root / LEDGER_DIRECTORY_NAME).symlink_to(
                outside,
                target_is_directory=True,
            )

            with self.assertRaisesRegex(ValueError, "non-symlink directory"):
                PilotUsageAttempt(
                    trace_root,
                    contract_id=contract.contract_id,
                    job_id=assignment.job_id,
                    attempt_id="attempt_1",
                )
            self.assertEqual(list(outside.iterdir()), [])

            (trace_root / LEDGER_DIRECTORY_NAME).unlink()
            job_root = trace_root / LEDGER_DIRECTORY_NAME / assignment.job_id
            job_root.mkdir(parents=True)
            (job_root / "attempt_1").symlink_to(
                outside,
                target_is_directory=True,
            )
            rows = _rows(contract)
            next(item for item in rows if item["job_id"] == assignment.job_id)["attempts"] = 1
            with self.assertRaisesRegex(ValueError, "unsafe attempt paths"):
                audit_pilot_usage_ledger(contract, rows, trace_root)

    def test_attempt_id_cannot_escape_ledger_directory(self) -> None:
        contract = _pilot_contract()
        assignment = contract.rollouts[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "safe path component"):
                PilotUsageAttempt(
                    root,
                    contract_id=contract.contract_id,
                    job_id=assignment.job_id,
                    attempt_id="attempt_../escape",
                )

            self.assertFalse((root / LEDGER_DIRECTORY_NAME).exists())


def _pilot_contract():
    manifest, _ = _gold20_manifest()
    return _contract(Gold20Binding.from_manifest(manifest))


def _rows(contract) -> list[dict]:
    return [
        {
            "job_id": assignment.job_id,
            "status": "pending",
            "attempts": 0,
            "tokens": 0,
            "cost": 0.0,
            "consumed_tokens": 0,
            "consumed_cost": 0.0,
            "consumed_elapsed_ms": 0.0,
        }
        for assignment in contract.rollouts
    ]


def _record_started(attempt: PilotUsageAttempt) -> None:
    attempt.call_started(
        {
            "call_index": 0,
            "started_at": "2026-07-14T00:00:00Z",
            "model": "model-a",
            "prompt_hash": canonical_sha256("private prompt content"),
            "message_count": 2,
            "tool_count": 1,
            "temperature": 0.0,
            "max_tokens": 4096,
            "response_format": None,
        }
    )


def _record_call(attempt: PilotUsageAttempt, *, response_id: str) -> None:
    _record_started(attempt)
    identity = {
        "id": response_id,
        "created": 1,
        "object": "chat.completion",
        "model": "model-a",
    }
    attempt.call_completed(
        {
            "call_index": 0,
            "response_model": "model-a",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "retry_count": 0,
            "latency_ms": 1.0,
            "provider_response_identity": identity,
            "provider_response_identity_sha256": canonical_sha256(identity),
            "provider_response_sha256": canonical_sha256(identity),
        }
    )


if __name__ == "__main__":
    unittest.main()
