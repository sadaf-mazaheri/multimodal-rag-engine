"""Content-addressed cache for LLM calls made during evaluation.

Generation and judging both cost money, and both are re-run far more often than
their inputs change: after a report tweak, when re-judging, or when a second arm
retrieves exactly what the first did (31 of 42 queries for Method 2 with and
without metadata). Keying on *what was asked* rather than *when* makes every
identical request free after the first, across runs and across arms.

``CachingProvider`` wraps any ``LLMProvider``, so the code under evaluation never
knows a cache exists.

What the key covers: request kind, provider, model, temperature, output-token
cap, seed, response-format schema, a prompt-version string, and every message's
role, content and attached-image bytes. Change any of those and the key changes.
What it never covers: the API key, or anything else about the machine. A cache
entry therefore records a request and its response and nothing that would be
unsafe to keep.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmrag.generation.providers.base import Completion, LLMProvider, Message, Usage
from mmrag.logging_utils import get_logger

log = get_logger(__name__)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.images:
        # Image bytes, not paths: a re-rendered page at the same path is a
        # different request.
        payload["images"] = [
            {
                "detail": image.detail,
                "sha256": hashlib.sha256(Path(image.path).read_bytes()).hexdigest(),
            }
            for image in message.images
        ]
    return payload


def cache_key(
    *,
    kind: str,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    seed: int | None,
    response_format: dict[str, Any] | None,
    prompt_version: str,
    messages: list[Message],
) -> str:
    """Deterministic key for one LLM request."""
    request = {
        "kind": kind,
        "provider": provider,
        "model": model,
        "temperature": round(float(temperature), 6),
        "max_output_tokens": int(max_output_tokens),
        "seed": seed,
        "response_format": response_format,
        "prompt_version": prompt_version,
        "messages": [_message_payload(m) for m in messages],
    }
    return _sha256(_canonical(request))


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    corrupt: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def as_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes, "corrupt": self.corrupt}


class CachingProvider:
    """An ``LLMProvider`` that answers repeated requests from disk.

    ``seed`` is applied to every call that does not supply its own, so the
    evaluation can fix one seed without the answering code knowing about it.
    ``enabled=False`` bypasses the cache entirely; ``refresh=True`` always calls
    the provider and overwrites what was stored.
    """

    def __init__(
        self,
        inner: LLMProvider,
        cache_dir: str | Path,
        *,
        kind: str,
        prompt_version: str,
        seed: int | None = None,
        enabled: bool = True,
        refresh: bool = False,
    ):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.kind = kind
        self.prompt_version = prompt_version
        self.seed = seed
        self.enabled = enabled
        self.refresh = refresh
        self.stats = CacheStats()

    @property
    def name(self) -> str:
        return getattr(self.inner, "name", "unknown")

    def supports_images(self) -> bool:
        return self.inner.supports_images()

    # -- keys and paths --------------------------------------------------------

    def key_for(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return cache_key(
            kind=self.kind,
            provider=self.name,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=seed if seed is not None else self.seed,
            response_format=response_format,
            prompt_version=self.prompt_version,
            messages=messages,
        )

    def _path(self, key: str) -> Path:
        return self.cache_dir / self.kind / key[:2] / f"{key}.json"

    # -- reads -------------------------------------------------------------------

    def _read(self, key: str) -> Completion | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            stored = entry["completion"]
            usage = stored.get("usage", {})
            return Completion(
                text=stored["text"],
                model=stored["model"],
                usage=Usage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                ),
                latency_ms=float(stored.get("latency_ms", 0.0)),
                metadata=dict(stored.get("metadata", {})),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # A torn or hand-edited entry is a miss, not a crash; it is
            # overwritten by the next successful call.
            self.stats.bump("corrupt")
            log.warning("ignoring unreadable cache entry %s (%s)", path.name, type(exc).__name__)
            return None

    def lookup(self, messages: list[Message], **request: Any) -> Completion | None:
        """What the cache holds for a request, without ever calling the provider.

        Used by ``--dry-run`` to count how many calls a run would really make.
        """
        if not self.enabled or self.refresh:
            return None
        return self._read(self.key_for(messages, **request))

    # -- the provider interface ------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        effective_seed = seed if seed is not None else self.seed
        key = self.key_for(
            messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=effective_seed,
            response_format=response_format,
        )

        if self.enabled and not self.refresh:
            cached = self._read(key)
            if cached is not None:
                self.stats.bump("hits")
                cached.metadata = {**cached.metadata, "cache_hit": True, "cache_key": key}
                return cached

        self.stats.bump("misses")
        # Errors propagate and nothing is written: a failure must be retried,
        # never replayed.
        completion = self.inner.complete(
            messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=effective_seed,
            response_format=response_format,
        )
        if self.enabled:
            self._write(key, completion, model=model, temperature=temperature,
                        max_output_tokens=max_output_tokens, seed=effective_seed,
                        response_format=response_format)
        completion.metadata = {**completion.metadata, "cache_hit": False, "cache_key": key}
        return completion

    def _write(self, key: str, completion: Completion, **request: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        response_format = request.pop("response_format", None)
        entry = {
            "key": key,
            "kind": self.kind,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "request": {
                **request,
                "provider": self.name,
                "prompt_version": self.prompt_version,
                "response_format_sha256": _sha256(_canonical(response_format))
                if response_format is not None
                else None,
            },
            "completion": {
                "text": completion.text,
                "model": completion.model,
                "usage": completion.usage.as_dict(),
                "latency_ms": completion.latency_ms,
                "metadata": {k: v for k, v in completion.metadata.items()
                             if k not in ("cache_hit", "cache_key")},
            },
        }
        # Write-then-rename, so a concurrent reader or a killed process never
        # sees half an entry.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(entry, handle, ensure_ascii=False)
            os.replace(tmp, path)
            self.stats.bump("writes")
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise
