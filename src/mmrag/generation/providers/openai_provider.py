"""OpenAI-compatible generation provider.

Covers both the OpenAI API and any server speaking the same protocol (Ollama,
llama.cpp, vLLM), which is why the ``local`` provider is this class with a
different base URL rather than a separate implementation.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

from mmrag.generation.providers.base import (
    Completion,
    ImageInput,
    Message,
    ProviderError,
    Usage,
    redact_secrets,
)
from mmrag.logging_utils import get_logger

log = get_logger(__name__)

# Images are sent inline as data URLs rather than by URL: the corpus renders are
# local files, and uploading them somewhere first would be both slower and a
# needless disclosure of document content to a third host.
_DEFAULT_IMAGE_MIME = "image/png"


def encode_image(path: Path) -> str:
    """Read an image file into a base64 data URL."""
    if not path.exists():
        raise ProviderError(f"image not found: {path}")
    mime = mimetypes.guess_type(path.name)[0] or _DEFAULT_IMAGE_MIME
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class OpenAIProvider:
    """Chat completions via the OpenAI SDK."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "the openai package is not installed; pip install -e '.[openai]'"
            ) from exc

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            # The SDK retries connection errors and 429/5xx with backoff.
            max_retries=max_retries,
        )

    def supports_images(self) -> bool:
        return True

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
        payload = [self._to_openai(m) for m in messages]
        request: dict[str, Any] = {
            "model": model,
            # The SDK's message types are a large union of TypedDicts; the dicts
            # built above are structurally correct but not statically
            # recognisable as any single member of it.
            "messages": payload,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        # Only sent when asked for, so existing callers produce byte-identical
        # requests to before.
        if seed is not None:
            request["seed"] = seed
        if response_format is not None:
            request["response_format"] = response_format

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            # SDK errors can echo a partially masked key; never let one reach a
            # log line or a run artefact.
            raise ProviderError(redact_secrets(f"{type(exc).__name__}: {exc}")) from None
        elapsed = (time.perf_counter() - started) * 1000

        choice = response.choices[0]
        usage = response.usage
        return Completion(
            text=(choice.message.content or "").strip(),
            model=response.model,
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            latency_ms=elapsed,
            metadata={
                "finish_reason": choice.finish_reason,
                # Seeded sampling is best-effort; the fingerprint is what explains
                # a replay that differs.
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            },
        )

    @staticmethod
    def _to_openai(message: Message) -> dict[str, Any]:
        if not message.images:
            return {"role": message.role, "content": message.content}

        parts: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
        parts.extend(_image_part(image) for image in message.images)
        return {"role": message.role, "content": parts}


def _image_part(image: ImageInput) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": encode_image(image.path), "detail": image.detail},
    }
