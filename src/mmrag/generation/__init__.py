"""Answer generation: prompt assembly, providers, citation resolution.

Two pipelines, selected by ``generation.pipeline``: ``v1`` (``Answerer``, the
default) and ``v2`` (``AnswererV2``: evidence pack, one call, validation).
"""

from mmrag.generation.answerer import Answerer, is_refusal, resolve_citations
from mmrag.generation.pipeline import PIPELINES, build_answerer, prompt_version_for
from mmrag.generation.providers import get_provider

__all__ = [
    "PIPELINES",
    "Answerer",
    "build_answerer",
    "get_provider",
    "is_refusal",
    "prompt_version_for",
    "resolve_citations",
]
