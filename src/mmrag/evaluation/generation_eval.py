"""Generate answers from a saved retrieval run, and record them for judging.

Answers are produced from **the exact chunks a retrieval run scored**, rebuilt by
id, rather than by calling ``method.answer()``. Three consequences:

* the link from retrieval quality to generation quality is exact, since both
  numbers describe the same retrieved list;
* retrieval is not paid for twice -- on CPU the reranker is ~18 s a query;
* every retrieval arm, including ``--no-metadata``, can be generated for.

It is safe because a chunk id hashes its body text: if every id in the run still
exists in the index, every body is the one that was scored. A run whose ids have
gone missing is refused outright rather than half-generated.

The one exception is the unanswerable queries, which are not in any retrieval
run. They are retrieved live with the retrieval run's own configuration and
overrides, and their retrieved lists are stored in the generation run so the
result stays reproducible.

Prompt assembly and citation resolution are ``Answerer``'s own
``build_prompt`` and ``resolve_citations``, and the provider is called with the
same model, temperature and output cap ``Answerer.answer`` uses; a test pins that
parity. Prompts are built sequentially and only provider calls run concurrently,
because the tokenizer used for the context budget is not guaranteed thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from mmrag.config import GenerationConfig
from mmrag.evaluation.generation_gold import GenerationGold
from mmrag.evaluation.generation_metrics import (
    contains_gold_evidence,
    fact_lexical_coverage,
    is_mixed_refusal,
    is_refusal,
    mean,
    percentile,
    rate,
)
from mmrag.evaluation.gold import GoldQuery, GoldSet
from mmrag.evaluation.llm_cache import CachingProvider
from mmrag.evaluation.retrieval_eval import RetrievalRun
from mmrag.generation.answerer import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_REFUSAL,
    Answerer,
    PromptBuild,
    parse_citation_numbers,
    resolve_citations,
)
from mmrag.generation.providers.base import Message, ProviderError, redact_secrets
from mmrag.logging_utils import get_logger
from mmrag.schemas import Chunk, ChunkType, Modality, ScoredChunk

log = get_logger(__name__)

# Identifies the answering prompt. Changing Answerer's system prompt changes this,
# which changes every cache key, so a stale cached answer can never be replayed
# under a new prompt.
PROMPT_VERSION = hashlib.sha256(
    (SYSTEM_PROMPT + "\x00" + SYSTEM_PROMPT_NO_REFUSAL).encode("utf-8")
).hexdigest()[:16]

_SOURCES_PREFIX = "Sources:\n\n"


class RetrievalRunMismatch(RuntimeError):
    """A retrieval run cannot be generated from as-is."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class SourceRef(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    page: int
    chunk_type: str
    rank: int
    is_gold: bool | None = None


class CitationRef(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    page: int
    is_gold: bool | None = None


class GenerationRecord(BaseModel):
    """One query's generated answer and everything needed to debug it."""

    query_id: str
    query: str
    answerable: bool
    stratum: str | None = None
    requires: str | None = None
    required_facts: list[str] = Field(default_factory=list)

    status: Literal["ok", "error", "dry_run"]
    error: str | None = None
    attempts: int = 0

    retrieved: list[SourceRef] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    dropped_for_budget: int = 0
    evidence_retrieved: bool | None = None
    evidence_in_context: bool | None = None

    # Exactly what the generator saw between "Sources:" and "Question:", so the
    # judge can be given the same text rather than a reconstruction of it.
    sources_block: str = ""
    prompt_sha256: str = ""
    prompt_tokens_estimated: int = 0

    answer: str | None = None
    refused: bool | None = None
    mixed_refusal: bool | None = None
    citations: list[CitationRef] = Field(default_factory=list)
    unresolved_citations: list[int] = Field(default_factory=list)
    gold_page_cited: bool | None = None
    fact_lexical: dict[str, int] | None = None

    model: str | None = None
    system_fingerprint: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float | None = None
    cache_hit: bool | None = None


class GenerationRun(BaseModel):
    kind: Literal["generation"] = "generation"
    schema_version: int = 1
    method: str
    label: str
    created_at: str
    elapsed_s: float
    dry_run: bool = False
    retrieval_run: dict[str, Any] = Field(default_factory=dict)
    gold: dict[str, Any] = Field(default_factory=dict)
    generation_gold: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    cache: dict[str, int] = Field(default_factory=dict)
    totals: dict[str, Any] = Field(default_factory=dict)
    records: list[GenerationRecord] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> GenerationRun:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------


@dataclass
class WorkItem:
    query_id: str
    query: str
    answerable: bool
    stratum: str | None
    requires: str | None
    facts: list[str]
    gold: GoldQuery | None
    retrieved: list[ScoredChunk]


def _modality_of(chunk: Chunk) -> Modality:
    return {ChunkType.TABLE: Modality.TABLE, ChunkType.FIGURE: Modality.IMAGE}.get(
        chunk.chunk_type, Modality.TEXT
    )


def rebuild_retrieved(query_result, chunks: dict[str, Chunk]) -> list[ScoredChunk]:
    """The ranked chunks a retrieval run returned for one query, by id."""
    missing = [hit.chunk_id for hit in query_result.retrieved if hit.chunk_id not in chunks]
    if missing:
        raise RetrievalRunMismatch(
            f"{query_result.query_id}: {len(missing)} retrieved chunk id(s) are not in the "
            f"current index (e.g. {missing[0]}). The index was rebuilt after this retrieval "
            "run; re-run retrieval before generating from it."
        )
    return [
        ScoredChunk(
            chunk=chunks[hit.chunk_id],
            score=hit.score,
            rank=hit.rank,
            retriever=hit.retriever,
            modality=_modality_of(chunks[hit.chunk_id]),
        )
        for hit in sorted(query_result.retrieved, key=lambda h: h.rank)
    ]


def select_work(
    retrieval_run: RetrievalRun,
    gold: GoldSet,
    generation_gold: GenerationGold,
    chunks: dict[str, Chunk],
    *,
    include_unanswerable: bool = True,
    query_ids: Sequence[str] | None = None,
    limit: int | None = None,
    live_retrieve: Callable[[str], list[ScoredChunk]] | None = None,
) -> list[WorkItem]:
    """Answerable queries in gold order, then unanswerable ones, then filtered.

    Filtering happens before any live retrieval, so ``--queries q001`` never
    loads a model it does not need.
    """
    by_id = {q.query_id: q for q in retrieval_run.per_query}
    candidates: list[tuple[str, bool]] = [(q.id, True) for q in gold.queries]
    if include_unanswerable or query_ids:
        candidates += [(u.id, False) for u in generation_gold.unanswerable]

    if query_ids:
        wanted = list(dict.fromkeys(query_ids))
        known = {cid for cid, _ in candidates}
        unknown = [q for q in wanted if q not in known]
        if unknown:
            raise ValueError(f"unknown query id(s): {unknown}")
        keep = set(wanted)
        candidates = [c for c in candidates if c[0] in keep]
    if limit is not None:
        candidates = candidates[:limit]

    gold_by_id = {q.id: q for q in gold.queries}
    unanswerable_by_id = {u.id: u for u in generation_gold.unanswerable}
    items: list[WorkItem] = []
    for query_id, answerable in candidates:
        if answerable:
            if query_id not in by_id:
                raise RetrievalRunMismatch(f"retrieval run has no result for {query_id}")
            g = gold_by_id[query_id]
            items.append(WorkItem(
                query_id=query_id, query=g.query, answerable=True,
                stratum=g.stratum, requires=g.requires,
                facts=generation_gold.facts_for(query_id), gold=g,
                retrieved=rebuild_retrieved(by_id[query_id], chunks),
            ))
        else:
            if live_retrieve is None:
                raise RetrievalRunMismatch(
                    f"{query_id} is unanswerable and needs live retrieval, but no retriever "
                    "was supplied"
                )
            u = unanswerable_by_id[query_id]
            items.append(WorkItem(
                query_id=query_id, query=u.query, answerable=False,
                stratum=None, requires=None, facts=[], gold=None,
                retrieved=live_retrieve(u.query),
            ))
    return items


# ---------------------------------------------------------------------------
# One query
# ---------------------------------------------------------------------------


def sources_block_of(prompt: PromptBuild, query: str) -> str:
    """The source text between the template's fixed prefix and suffix.

    Exact rather than heuristic: the user message is Answerer's template filled
    in, so stripping its known ends recovers the body byte for byte. If the
    template changes, this fails loudly instead of handing the judge the wrong
    text.
    """
    content = prompt.messages[-1].content
    suffix = f"\n\nQuestion: {query}\n\nAnswer:"
    if not (content.startswith(_SOURCES_PREFIX) and content.endswith(suffix)):
        raise ValueError(
            "Answerer's user-message template has changed; update sources_block_of"
        )
    return content[len(_SOURCES_PREFIX) : -len(suffix)]


def _prompt_sha256(messages: list[Message]) -> str:
    payload = json.dumps([[m.role, m.content] for m in messages], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_refs(chunks: Sequence[ScoredChunk], gold: GoldQuery | None) -> list[SourceRef]:
    return [
        SourceRef(
            number=i,
            chunk_id=s.chunk.chunk_id,
            doc_id=s.chunk.doc_id,
            page=s.chunk.page_number,
            chunk_type=s.chunk.chunk_type.value,
            rank=s.rank,
            is_gold=(gold.match_index(s.chunk) is not None) if gold else None,
        )
        for i, s in enumerate(chunks, start=1)
    ]


def _complete_with_retries(
    provider: CachingProvider,
    messages: list[Message],
    config: GenerationConfig,
    *,
    attempts: int,
    backoff_s: float,
):
    """Call the provider, retrying ProviderError a bounded number of times.

    Transport-level retries (429, 5xx, connection) already happen inside the
    OpenAI SDK; this covers failures that surface past it. Returns the
    completion and how many attempts it took, or raises the last error.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            completion = provider.complete(
                messages,
                model=config.text_model,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
            return completion, attempt
        except ProviderError as exc:
            last = exc
            if attempt < attempts:
                time.sleep(backoff_s * attempt)
    assert last is not None
    raise last


def generate_one(
    item: WorkItem,
    prompt: PromptBuild,
    provider: CachingProvider,
    config: GenerationConfig,
    *,
    dry_run: bool,
    attempts: int = 2,
    backoff_s: float = 2.0,
) -> GenerationRecord:
    sources = prompt.sources
    record = GenerationRecord(
        query_id=item.query_id,
        query=item.query,
        answerable=item.answerable,
        stratum=item.stratum,
        requires=item.requires,
        required_facts=item.facts,
        status="dry_run" if dry_run else "ok",
        retrieved=_source_refs(item.retrieved, item.gold),
        sources=_source_refs(sources, item.gold),
        dropped_for_budget=prompt.dropped_for_budget,
        evidence_retrieved=contains_gold_evidence((s.chunk for s in item.retrieved), item.gold),
        evidence_in_context=contains_gold_evidence((s.chunk for s in sources), item.gold),
        sources_block=sources_block_of(prompt, item.query),
        prompt_sha256=_prompt_sha256(prompt.messages),
        prompt_tokens_estimated=prompt.prompt_tokens,
    )

    if dry_run:
        cached = provider.lookup(
            prompt.messages,
            model=config.text_model,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )
        record.cache_hit = cached is not None
        return record

    try:
        completion, record.attempts = _complete_with_retries(
            provider, prompt.messages, config, attempts=attempts, backoff_s=backoff_s
        )
    except Exception as exc:  # recorded, never raised: one failure must not sink a run
        record.status = "error"
        record.attempts = attempts
        record.error = redact_secrets(f"{type(exc).__name__}: {exc}")
        log.warning("%s: generation failed: %s", item.query_id, record.error)
        return record

    text = completion.text
    citations, unresolved = resolve_citations(text, sources)
    numbers = [n for n in parse_citation_numbers(text) if 1 <= n <= len(sources)]

    record.answer = text
    record.refused = is_refusal(text)
    record.mixed_refusal = is_mixed_refusal(text, len(citations))
    record.citations = [
        CitationRef(
            number=number,
            chunk_id=citation.chunk_id or "",
            doc_id=citation.doc_id,
            page=citation.page_number,
            is_gold=(
                item.gold.match_index(sources[number - 1].chunk) is not None
                if item.gold else None
            ),
        )
        for number, citation in zip(numbers, citations, strict=True)
    ]
    record.unresolved_citations = unresolved
    if item.gold is not None and not record.refused:
        record.gold_page_cited = any(c.is_gold for c in record.citations)
    record.fact_lexical = fact_lexical_coverage(item.facts, text) if item.facts else None
    record.model = completion.model
    record.system_fingerprint = completion.metadata.get("system_fingerprint")
    record.finish_reason = completion.metadata.get("finish_reason")
    record.usage = completion.usage.as_dict()
    record.latency_ms = round(completion.latency_ms, 1)
    record.cache_hit = bool(completion.metadata.get("cache_hit"))
    return record


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------


def run_generation(
    items: Sequence[WorkItem],
    answerer: Answerer,
    provider: CachingProvider,
    *,
    dry_run: bool = False,
    concurrency: int = 4,
    attempts: int = 2,
    backoff_s: float = 2.0,
    on_record: Callable[[GenerationRecord], None] | None = None,
) -> list[GenerationRecord]:
    """Build prompts sequentially, call the provider concurrently, keep input order."""
    config = answerer.config
    prompts = [answerer.build_prompt(item.query, item.retrieved) for item in items]

    def work(pair: tuple[WorkItem, PromptBuild]) -> GenerationRecord:
        record = generate_one(pair[0], pair[1], provider, config, dry_run=dry_run,
                              attempts=attempts, backoff_s=backoff_s)
        if on_record is not None:
            on_record(record)
        return record

    pairs = list(zip(items, prompts, strict=True))
    if dry_run or concurrency <= 1:
        return [work(p) for p in pairs]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # map() preserves input order regardless of completion order.
        return list(pool.map(work, pairs))


def aggregate(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    """Deterministic metrics, with counts beside every rate."""
    ok = [r for r in records if r.status == "ok"]
    answerable = [r for r in ok if r.answerable]
    unanswerable = [r for r in ok if not r.answerable]
    answered = [r for r in answerable if not r.refused]

    def slice_by(field: str) -> dict[str, Any]:
        groups: dict[str, list[GenerationRecord]] = {}
        for r in answerable:
            groups.setdefault(getattr(r, field) or "-", []).append(r)
        return {
            key: {
                "evidence_in_context": rate(r.evidence_in_context for r in group),
                "refused": rate(r.refused for r in group),
            }
            for key, group in sorted(groups.items())
        }

    found = sum((r.fact_lexical or {}).get("found", 0) for r in answerable)
    checkable = sum((r.fact_lexical or {}).get("checkable", 0) for r in answerable)
    latencies = [r.latency_ms for r in ok if r.latency_ms is not None and not r.cache_hit]

    return {
        "n_records": len(records),
        "n_ok": len(ok),
        "n_errors": sum(1 for r in records if r.status == "error"),
        "answerable": {
            "n": len(answerable),
            "evidence_retrieved": rate(r.evidence_retrieved for r in answerable),
            "evidence_in_context": rate(r.evidence_in_context for r in answerable),
            "lost_to_budget": rate(
                bool(r.evidence_retrieved and not r.evidence_in_context) for r in answerable
            ),
            "refused": rate(r.refused for r in answerable),
            "mixed_refusal": rate(r.mixed_refusal for r in answerable),
            "citation_presence": rate(bool(r.citations) for r in answered),
            "any_unresolved_citation": rate(bool(r.unresolved_citations) for r in answered),
            "gold_page_cited": rate(r.gold_page_cited for r in answered),
            "fact_lexical": {"found": found, "checkable": checkable,
                             "rate": round(found / checkable, 4) if checkable else None},
            "by_requires": slice_by("requires"),
            "by_stratum": slice_by("stratum"),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "refused": rate(r.refused for r in unanswerable),
        },
        "latency_ms": {
            "n_uncached": len(latencies),
            "median": percentile(latencies, 0.5),
            "p90": percentile(latencies, 0.9),
        },
        "sources_per_prompt": mean(len(r.sources) for r in records),
    }


def totals(records: Sequence[GenerationRecord], cache: CachingProvider) -> dict[str, Any]:
    return {
        "calls_made": cache.stats.misses,
        "cache_hits": cache.stats.hits,
        "prompt_tokens": sum(r.usage.get("prompt_tokens", 0) for r in records),
        "completion_tokens": sum(r.usage.get("completion_tokens", 0) for r in records),
        "prompt_tokens_estimated": sum(r.prompt_tokens_estimated for r in records),
    }


def estimate(
    records: Sequence[GenerationRecord],
    *,
    output_tokens_per_answer: int = 250,
    judge_prompt_overhead_tokens: int = 900,
    judge_output_tokens: int = 700,
    price_in_per_m: float | None = None,
    price_out_per_m: float | None = None,
) -> dict[str, Any]:
    """Projected calls, tokens and cost for generating and then judging these records.

    Token counts use the context-budget tokenizer, not OpenAI's, so they are an
    estimate to within roughly ten to twenty per cent. The judge re-reads every
    source block, which is why judging costs about as much as generating.
    Prices are supplied by the caller and never hard-coded.
    """
    to_generate = [r for r in records if not r.cache_hit]
    gen_in = sum(r.prompt_tokens_estimated for r in to_generate)
    gen_out = output_tokens_per_answer * len(to_generate)
    judge_in = sum(
        r.prompt_tokens_estimated + output_tokens_per_answer + judge_prompt_overhead_tokens
        for r in records
    )
    judge_out = judge_output_tokens * len(records)

    def cost(tokens_in: int, tokens_out: int) -> float | None:
        if price_in_per_m is None or price_out_per_m is None:
            return None
        return round(tokens_in / 1e6 * price_in_per_m + tokens_out / 1e6 * price_out_per_m, 4)

    return {
        "generation": {"calls": len(to_generate), "cache_hits": len(records) - len(to_generate),
                       "input_tokens": gen_in, "output_tokens": gen_out,
                       "cost_usd": cost(gen_in, gen_out)},
        "judge": {"calls": len(records), "input_tokens": judge_in,
                  "output_tokens": judge_out, "cost_usd": cost(judge_in, judge_out)},
        "total": {"calls": len(to_generate) + len(records),
                  "input_tokens": gen_in + judge_in, "output_tokens": gen_out + judge_out,
                  "cost_usd": cost(gen_in + judge_in, gen_out + judge_out)},
        "assumptions": {"output_tokens_per_answer": output_tokens_per_answer,
                        "judge_prompt_overhead_tokens": judge_prompt_overhead_tokens,
                        "judge_output_tokens": judge_output_tokens,
                        "price_in_per_m": price_in_per_m, "price_out_per_m": price_out_per_m},
    }


class DryRunProvider:
    """Stands in for the real provider during --dry-run and refuses to be called.

    Carries the real provider's name so cache keys match the ones a live run
    would write; any attempt to complete is a bug and fails the run.
    """

    def __init__(self, name: str):
        self.name = name

    def supports_images(self) -> bool:
        return False

    def complete(self, *args: Any, **kwargs: Any):
        raise AssertionError("a dry run attempted a provider call")


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return {"path": str(p).replace("\\", "/"),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}


def environment() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "git_commit": commit}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
