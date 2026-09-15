"""Measure prompt construction and answer post-processing offline.

Neither step was ever timed during evaluation, and both are deterministic and
local, so they are re-executed here with the frozen answerer the generation run
used. Two guarantees make the timing trustworthy:

* each rebuilt prompt's sha256 is compared with the one the generation run
  recorded; a timing is used only when they match, so what was timed is byte for
  byte the prompt that was sent;
* post-processing re-runs exactly the deterministic steps ``generate_one``
  applies to a recorded answer -- citation parsing and resolution, refusal
  checks and ``AnswerValidator`` -- over that same answer text.

No provider is ever called: the answerer is handed a provider that raises.

Only queries present in the retrieval run are timed. Unanswerable queries were
retrieved live during generation and their stored sources carry no retrieval
scores, so their prompts cannot be rebuilt exactly and are skipped.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mmrag.evaluation.generation_eval import (
    DryRunProvider,
    GenerationRecord,
    _prompt_sha256,
    rebuild_retrieved,
)
from mmrag.evaluation.generation_metrics import is_mixed_refusal, is_refusal
from mmrag.evaluation.retrieval_eval import RetrievalRun
from mmrag.generation.answerer import parse_citation_numbers, resolve_citations
from mmrag.generation.validation import AnswerValidator, answer_sentences
from mmrag.schemas import Chunk, ScoredChunk


@dataclass
class PromptTiming:
    query_id: str
    sha_match: bool
    prompt_construction_ms: float | None
    postprocess_ms: float | None


def _median_ms(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    samples: list[float] = []
    result: Any = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples), result


def postprocess(text: str, sources: list[ScoredChunk]) -> dict[str, Any]:
    """The deterministic steps ``generate_one`` applies to a provider's answer."""
    citations, unresolved = resolve_citations(text, sources)
    numbers = [n for n in parse_citation_numbers(text) if 1 <= n <= len(sources)]
    return {
        "citations": citations,
        "unresolved": unresolved,
        "numbers": numbers,
        "refused": is_refusal(text),
        "mixed_refusal": is_mixed_refusal(text, len(citations)),
        "sentences": len(answer_sentences(text)),
        "validation": AnswerValidator().validate(text, sources).as_dict(),
    }


def time_prompts(
    records: list[GenerationRecord],
    retrieval_run: RetrievalRun,
    chunks: dict[str, Chunk],
    answerer: Any,
    *,
    repeats: int = 5,
    warmup: bool = True,
) -> dict[str, PromptTiming]:
    """Per-query median prompt-construction and post-processing time, sha-checked."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    by_id = {q.query_id: q for q in retrieval_run.per_query}
    timeable = [r for r in records if r.query_id in by_id]
    timings: dict[str, PromptTiming] = {}

    if warmup and timeable:
        # First use can load a tokenizer or compile regexes; not a per-query cost.
        first = timeable[0]
        retrieved = rebuild_retrieved(by_id[first.query_id], chunks)
        prompt = answerer.build_prompt(first.query, retrieved)
        if first.answer is not None:
            postprocess(first.answer, prompt.sources)

    for record in timeable:
        retrieved = rebuild_retrieved(by_id[record.query_id], chunks)
        build_ms, prompt = _median_ms(
            lambda r=record, ret=retrieved: answerer.build_prompt(r.query, ret), repeats
        )
        match = _prompt_sha256(prompt.messages) == record.prompt_sha256
        post_ms: float | None = None
        if match and record.status == "ok" and record.answer is not None:
            post_ms, _ = _median_ms(
                lambda text=record.answer, src=prompt.sources: postprocess(text, src), repeats
            )
        timings[record.query_id] = PromptTiming(
            query_id=record.query_id,
            sha_match=match,
            prompt_construction_ms=build_ms if match else None,
            postprocess_ms=post_ms,
        )
    return timings


def offline_answerer(generation_config: Any, provider_name: str = "offline") -> Any:
    """The frozen answerer a pipeline selects, over a provider that refuses calls."""
    from mmrag.generation.pipeline import build_answerer

    return build_answerer(generation_config, DryRunProvider(provider_name))
