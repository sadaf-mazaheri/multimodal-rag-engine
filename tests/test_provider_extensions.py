"""Optional seed / structured output on providers, and secret redaction.

No network: the OpenAI client is replaced with a recorder, so what is asserted is
the exact request the provider would send.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mmrag.generation.providers.base import Message, ProviderError, redact_secrets
from mmrag.generation.providers.echo import EchoProvider
from mmrag.generation.providers.openai_provider import OpenAIProvider

FAKE_KEY = "sk-proj-THISISNOTAREALKEY1234567890abcdef"


class _RecordingCompletions:
    def __init__(self, *, fail_with: Exception | None = None):
        self.requests: list[dict] = []
        self.fail_with = fail_with

    def create(self, **request):
        self.requests.append(request)
        if self.fail_with is not None:
            raise self.fail_with
        return SimpleNamespace(
            model="gpt-4o-mini-2024-07-18",
            system_fingerprint="fp_test123",
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=" an answer [1] "),
                finish_reason="stop",
            )],
        )


def _provider(**kwargs) -> tuple[OpenAIProvider, _RecordingCompletions]:
    provider = OpenAIProvider(api_key=FAKE_KEY)
    completions = _RecordingCompletions(**kwargs)
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


MESSAGES = [Message(role="system", content="s"), Message(role="user", content="u")]


class TestOpenAIRequestShape:
    def test_existing_callers_send_the_request_they_always_did(self):
        """No seed and no response_format keys at all, not even as None."""
        provider, rec = _provider()
        provider.complete(MESSAGES, model="gpt-4o-mini", temperature=0.0, max_output_tokens=50)
        assert set(rec.requests[0]) == {"model", "messages", "temperature", "max_tokens"}

    def test_seed_is_sent_when_given(self):
        provider, rec = _provider()
        provider.complete(MESSAGES, model="m", seed=42)
        assert rec.requests[0]["seed"] == 42

    def test_response_format_is_sent_when_given(self):
        provider, rec = _provider()
        schema = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
        provider.complete(MESSAGES, model="m", response_format=schema)
        assert rec.requests[0]["response_format"] == schema

    def test_system_fingerprint_is_surfaced(self):
        provider, _ = _provider()
        completion = provider.complete(MESSAGES, model="m")
        assert completion.metadata["system_fingerprint"] == "fp_test123"
        assert completion.text == "an answer [1]"
        assert completion.usage.total_tokens == 18


class TestSecretSafety:
    def test_redact_removes_a_full_key(self):
        assert FAKE_KEY not in redact_secrets(f"bad key {FAKE_KEY} used")
        assert "[REDACTED]" in redact_secrets(f"bad key {FAKE_KEY} used")

    def test_redact_removes_a_partially_masked_key(self):
        """SDK auth errors echo keys like sk-proj-****abcd."""
        assert "abcd" not in redact_secrets("Incorrect API key provided: sk-proj-****abcd.")

    def test_redact_leaves_ordinary_text_alone(self):
        text = "Figure SPM.8 shows risk-based scenarios"
        assert redact_secrets(text) == text

    def test_a_provider_error_never_carries_the_key(self):
        provider, _ = _provider(fail_with=RuntimeError(f"401 Incorrect API key: {FAKE_KEY}"))
        with pytest.raises(ProviderError) as info:
            provider.complete(MESSAGES, model="m")
        assert FAKE_KEY not in str(info.value)
        # The original exception is not chained, so a traceback cannot print it.
        assert info.value.__cause__ is None
        assert info.value.__suppress_context__ is True


class TestEchoAcceptsTheNewParameters:
    def test_seed_and_response_format_are_accepted_and_ignored(self):
        plain = EchoProvider().complete(MESSAGES, model="echo")
        extended = EchoProvider().complete(
            MESSAGES, model="echo", seed=7, response_format={"type": "json_object"}
        )
        assert plain.text == extended.text


class TestRegistryTransportSettings:
    def test_timeout_and_retries_reach_the_client(self, monkeypatch):
        from mmrag.config import Settings
        from mmrag.generation.providers.registry import get_provider

        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        provider = get_provider(Settings(), name="openai", timeout=12.5, max_retries=1)
        assert provider.client.timeout == 12.5
        assert provider.client.max_retries == 1
