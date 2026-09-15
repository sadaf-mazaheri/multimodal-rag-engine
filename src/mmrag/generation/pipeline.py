"""Choosing a generation pipeline.

``generation.pipeline`` selects how retrieved chunks become an answer. ``v1`` is
the original ``Answerer`` and the default, so every existing configuration and
run keeps its behaviour; ``v2`` is ``AnswererV2``. Both take the same config and
provider and expose ``build_prompt`` and ``answer``, so callers never branch.
"""

from __future__ import annotations

import hashlib
from typing import Any

from mmrag.config import GenerationConfig
from mmrag.generation.answerer import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_REFUSAL, Answerer
from mmrag.generation.providers.base import LLMProvider
from mmrag.textify.tokens import TokenCounter

PIPELINES = ("v1", "v2", "v2.1")
# V2 variants share AnswererV2 and the evidence pack; only the prompt differs.
_V2_VARIANTS = ("v2", "v2.1")

# V1's prompt version, computed exactly as evaluation always has. It is pinned by
# tests/test_generation_v1_golden.py.
PROMPT_VERSION_V1 = hashlib.sha256(
    (SYSTEM_PROMPT + "\x00" + SYSTEM_PROMPT_NO_REFUSAL).encode("utf-8")
).hexdigest()[:16]


def build_answerer(
    config: GenerationConfig,
    provider: LLMProvider,
    *,
    token_counter: TokenCounter | None = None,
) -> Any:
    """The answerer ``config.pipeline`` selects."""
    if config.pipeline == "v1":
        return Answerer(config, provider, token_counter=token_counter)
    if config.pipeline in _V2_VARIANTS:
        from mmrag.generation.answerer_v2 import AnswererV2

        return AnswererV2(config, provider, token_counter=token_counter, variant=config.pipeline)
    raise ValueError(
        f"unknown generation pipeline {config.pipeline!r}; expected one of {PIPELINES}"
    )


def prompt_version_for(pipeline: str) -> str:
    """The prompt version that keys a pipeline's cached answers."""
    if pipeline == "v1":
        return PROMPT_VERSION_V1
    if pipeline in _V2_VARIANTS:
        from mmrag.generation.answerer_v2 import VARIANTS

        return VARIANTS[pipeline][2]
    raise ValueError(f"unknown generation pipeline {pipeline!r}; expected one of {PIPELINES}")
