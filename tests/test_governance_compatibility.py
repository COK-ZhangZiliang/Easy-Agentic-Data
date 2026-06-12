import os
import tempfile
import time
import unittest
from pathlib import Path

from easy_agentic_data.compatibility import probe_tool_calling
from easy_agentic_data.governance import purge_expired_artifacts, sensitive_findings
from easy_agentic_data.models import LLMResponse, Message


class GovernanceCompatibilityTests(unittest.TestCase):
    def test_sensitive_scanner_and_retention(self) -> None:
        findings = sensitive_findings(
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz and password=long-secret"
        )
        self.assertEqual({item["kind"] for item in findings}, {"bearer_token", "password_assignment"})
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "old.txt"
            fresh = Path(directory) / "fresh.txt"
            old.write_text("old", encoding="utf-8")
            fresh.write_text("fresh", encoding="utf-8")
            os.utime(old, (1, 1))
            removed = purge_expired_artifacts(
                directory, retention_seconds=60, now=time.time()
            )
            self.assertIn("old.txt", removed)
            self.assertTrue(fresh.exists())

    def test_tool_calling_probe_detects_support(self) -> None:
        class Client:
            model = "tool-model"

            def complete(self, messages, tools=None, **kwargs):
                del messages, tools, kwargs
                return LLMResponse(
                    Message(
                        "assistant",
                        tool_calls=[
                            {
                                "id": "probe",
                                "type": "function",
                                "function": {
                                    "name": "capability_probe",
                                    "arguments": '{"value":"ok"}',
                                },
                            }
                        ],
                    ),
                    self.model,
                )

        self.assertTrue(probe_tool_calling(Client())["supported"])


if __name__ == "__main__":
    unittest.main()
