"""Provider-agnostic generation interface.

The whole retrieval stack is local and open-source; only this last step talks to
a vendor. Keeping that behind a narrow interface is what stops retrieval quality
from being silently confounded with a model vendor's behaviour -- and it is what
lets the same experiment be re-run against a local model to check that a result
is a property of the architecture rather than of GPT-4o-mini.

The interface is deliberately small: messages in, text out, with optional
images. Anything richer would leak provider-specific concepts into the methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class ImageInput:
    """An image to attach to a message.

    Method 3 sends rendered page images; Method 1 never does. It lives in the
    shared interface so the two methods differ only in what they pass, not in
    which provider API they call.
    """

    path: Path
    detail: str = "high"


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str
    images: list[ImageInput] = field(default_factory=list)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Completion:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A provider failed in a way the caller should surface, not swallow."""


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal contract every generation backend implements."""

    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> Completion: ...

    def supports_images(self) -> bool: ...
