import json
import os
import unittest
from unittest.mock import patch

from easy_agentic_data.config import LLMConfig
from easy_agentic_data.llm.openai_compatible import (
    LocalOpenAICompatibleClient,
    OpenAICompatibleClient,
)
from easy_agentic_data.models import Message


class _FakeHTTPResponse:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self) -> bytes:
        return self.body


class LLMClientTests(unittest.TestCase):
    def test_local_client_calls_server_without_api_key(self) -> None:
        config = LLMConfig(
            provider="local_openai_compatible",
            model="local-test-model",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env=None,
            chat_completions_path="chat/completions",
        )
        client = LocalOpenAICompatibleClient(config)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Add numbers.",
                    "parameters": {"type": "object"},
                },
            }
        ]
        fake_response = _FakeHTTPResponse(
            {
                "model": "local-test-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Local response",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 2,
                    "total_tokens": 6,
                },
            }
        )

        with patch(
            "easy_agentic_data.llm.openai_compatible.urllib.request.urlopen",
            return_value=fake_response,
        ) as urlopen:
            response = client.complete([Message("user", "Hello")], tools=tools)

        request = urlopen.call_args.args[0]
        request_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(response.message.content, "Local response")
        self.assertEqual(response.usage["total_tokens"], 6)
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(request_body["model"], "local-test-model")
        self.assertEqual(request_body["tools"], tools)
        self.assertNotIn("Authorization", request.headers)

    def test_local_client_uses_optional_api_key(self) -> None:
        config = LLMConfig(
            provider="local_openai_compatible",
            api_key_env="EAD_LOCAL_API_KEY",
        )
        with patch.dict(os.environ, {"EAD_LOCAL_API_KEY": "local-secret"}, clear=True):
            client = LocalOpenAICompatibleClient(config)
        self.assertEqual(client.api_key, "local-secret")

    def test_hosted_client_still_requires_api_key(self) -> None:
        config = LLMConfig(api_key_env="EAD_TEST_MISSING_API_KEY")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "EAD_TEST_MISSING_API_KEY"):
                OpenAICompatibleClient(config)


if __name__ == "__main__":
    unittest.main()
