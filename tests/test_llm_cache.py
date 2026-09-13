"""The evaluation LLM cache. No network: the wrapped provider is a recorder."""

from __future__ import annotations

import json

import pytest

from mmrag.evaluation.llm_cache import CachingProvider, cache_key
from mmrag.generation.providers.base import Completion, Message, ProviderError, Usage

FAKE_KEY = "sk-proj-NOTAREALKEY0000000000000000"


class RecordingProvider:
    """Counts calls and returns a deterministic completion."""

    name = "openai"

    def __init__(self, *, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail
        # Present to prove a credential on the wrapped object never leaks into
        # a cache file.
        self.api_key = FAKE_KEY

    def supports_images(self) -> bool:
        return False

    def complete(self, messages, *, model, temperature=0.0, max_output_tokens=1024,
                 seed=None, response_format=None):
        self.calls.append({"model": model, "seed": seed, "response_format": response_format})
        if self.fail:
            raise ProviderError("boom")
        return Completion(
            text=f"answer {len(self.calls)} [1]",
            model="gpt-4o-mini-2024-07-18",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            latency_ms=12.0,
            metadata={"finish_reason": "stop", "system_fingerprint": "fp_1"},
        )


MSGS = [Message(role="system", content="rules"), Message(role="user", content="q + sources")]
BASE = dict(kind="generation", provider="openai", model="gpt-4o-mini", temperature=0.0,
            max_output_tokens=512, seed=42, response_format=None, prompt_version="v1",
            messages=MSGS)


class TestCacheKey:
    def test_is_deterministic(self):
        assert cache_key(**BASE) == cache_key(**BASE)

    @pytest.mark.parametrize("field,value", [
        ("kind", "judge"),
        ("provider", "local"),
        ("model", "gpt-4o"),
        ("temperature", 0.2),
        ("max_output_tokens", 513),
        ("seed", 7),
        ("seed", None),
        ("response_format", {"type": "json_object"}),
        ("prompt_version", "v2"),
        ("messages", [Message(role="system", content="rules"), Message(role="user", content="other")]),
        ("messages", [Message(role="user", content="rules"), Message(role="user", content="q + sources")]),
    ])
    def test_every_request_field_changes_the_key(self, field, value):
        assert cache_key(**{**BASE, field: value}) != cache_key(**BASE)

    def test_is_a_hash_not_a_transcript(self):
        key = cache_key(**BASE)
        assert len(key) == 64 and all(c in "0123456789abcdef" for c in key)


class TestCachingProvider:
    def _cp(self, tmp_path, inner=None, **kwargs):
        inner = inner or RecordingProvider()
        return CachingProvider(inner, tmp_path, kind="generation", prompt_version="v1",
                               seed=42, **kwargs), inner

    def _call(self, cp):
        return cp.complete(MSGS, model="gpt-4o-mini", temperature=0.0, max_output_tokens=512)

    def test_a_miss_calls_once_and_writes(self, tmp_path):
        cp, inner = self._cp(tmp_path)
        first = self._call(cp)
        assert len(inner.calls) == 1
        assert first.metadata["cache_hit"] is False
        assert cp.stats.writes == 1

    def test_a_hit_makes_no_provider_call(self, tmp_path):
        cp, inner = self._cp(tmp_path)
        first = self._call(cp)
        second = self._call(cp)
        assert len(inner.calls) == 1
        assert second.metadata["cache_hit"] is True
        assert second.text == first.text
        assert second.usage.as_dict() == first.usage.as_dict()
        assert second.metadata["system_fingerprint"] == "fp_1"

    def test_the_cache_survives_a_new_wrapper(self, tmp_path):
        cp, _ = self._cp(tmp_path)
        self._call(cp)
        again, inner = self._cp(tmp_path)
        assert self._call(again).metadata["cache_hit"] is True
        assert inner.calls == []

    def test_seed_is_injected_when_the_caller_gives_none(self, tmp_path):
        cp, inner = self._cp(tmp_path)
        self._call(cp)
        assert inner.calls[0]["seed"] == 42

    def test_disabled_always_calls_and_never_writes(self, tmp_path):
        cp, inner = self._cp(tmp_path, enabled=False)
        self._call(cp)
        self._call(cp)
        assert len(inner.calls) == 2
        assert not any(tmp_path.rglob("*.json"))

    def test_refresh_calls_again_and_overwrites(self, tmp_path):
        cp, _ = self._cp(tmp_path)
        self._call(cp)
        refreshing, inner = self._cp(tmp_path, refresh=True)
        result = self._call(refreshing)
        assert len(inner.calls) == 1 and result.metadata["cache_hit"] is False

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        cp, inner = self._cp(tmp_path)
        self._call(cp)
        path = next(tmp_path.rglob("*.json"))
        path.write_text("{not json", encoding="utf-8")
        result = self._call(cp)
        assert result.metadata["cache_hit"] is False
        assert cp.stats.corrupt == 1 and len(inner.calls) == 2

    def test_a_failure_is_not_cached(self, tmp_path):
        cp, _ = self._cp(tmp_path, inner=RecordingProvider(fail=True))
        with pytest.raises(ProviderError):
            self._call(cp)
        assert not any(tmp_path.rglob("*.json"))

    def test_lookup_never_calls_the_provider(self, tmp_path):
        cp, inner = self._cp(tmp_path)
        request = dict(model="gpt-4o-mini", temperature=0.0, max_output_tokens=512)
        assert cp.lookup(MSGS, **request) is None
        self._call(cp)
        assert cp.lookup(MSGS, **request) is not None
        assert len(inner.calls) == 1

    def test_no_secret_reaches_disk(self, tmp_path):
        cp, _ = self._cp(tmp_path)
        self._call(cp)
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert FAKE_KEY not in path.read_text(encoding="utf-8")
                assert "api_key" not in path.read_text(encoding="utf-8")

    def test_entries_record_the_request_but_not_the_messages(self, tmp_path):
        cp, _ = self._cp(tmp_path)
        self._call(cp)
        entry = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
        assert entry["request"]["model"] == "gpt-4o-mini"
        assert entry["request"]["seed"] == 42
        assert "q + sources" not in json.dumps(entry)
