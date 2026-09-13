"""A provider that calls nothing.

Exists so the entire pipeline -- retrieval, prompt assembly, citation parsing,
evaluation plumbing -- can be exercised end to end with no API key and no cost.
Every test that is not specifically about generation quality uses this, which is
what keeps the suite free to run and independent of a vendor being reachable.

It returns a deterministic, obviously-synthetic answer that still carries valid
citation markers, so citation parsing is genuinely tested rather than bypassed.
"""

from __future__ import annotations

import re
from typing import Any

from mmrag.generation.providers.base import Completion, Message, Usage

_CITE_SOURCE = re.compile(r"^\[(\d+)\]", re.MULTILINE)


class EchoProvider:
    """Deterministic stub. Never contacts a network."""

    name = "echo"

    def supports_images(self) -> bool:
        # Reports True so Method 3's image path is exercised in tests; the
        # images are counted and described rather than looked at.
        return True

    def complete(
        self,
        messages: list[Message],
        *,
        model: str = "echo",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        seed: int | None = None,  # noqa: ARG002 - accepted for interface parity
        response_format: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> Completion:
        user = next((m for m in reversed(messages) if m.role == "user"), None)
        content = user.content if user else ""
        n_images = sum(len(m.images) for m in messages)

        # Echo back the source numbers the prompt offered, so the answer parses
        # as cited and the citation machinery is actually under test.
        sources = _CITE_SOURCE.findall(content)
        citations = " ".join(f"[{s}]" for s in sources[:3]) or "[1]"

        text = (
            f"ECHO: generated without a model from {len(sources)} source(s) "
            f"and {n_images} image(s). {citations}"
        )
        return Completion(
            text=text,
            model=model,
            usage=Usage(prompt_tokens=len(content) // 4, completion_tokens=len(text) // 4),
            latency_ms=0.0,
            metadata={"stub": True, "n_sources": len(sources), "n_images": n_images},
        )
