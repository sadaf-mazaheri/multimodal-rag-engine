"""Method 2's retrieval orchestrator: route, fan out, fuse, rerank.

**Ownership: unique to Method 2.** The counterpart to Method 1's
``HybridRetriever``, which is left untouched.

The shape of a query's journey:

    query -> router -> {text, table, image} retrievers -> weighted RRF -> rerank

Everything after the fan-out is shared machinery -- the same
``reciprocal_rank_fusion`` and ``CrossEncoderReranker`` Method 1 uses (or could
use). What is new is the routing and the per-modality fan-out.

The result reports which retrievers fired, which contributed, and which found
something nothing else did. Without that, a modality-aware system that has
quietly degenerated into "text retrieval plus noise" is indistinguishable from
one that is working.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mmrag.config import RetrievalConfig
from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import MetadataFilter, Retriever, RetrieverOutput
from mmrag.retrieval.fusion import RankedList, fusion_diagnostics, reciprocal_rank_fusion
from mmrag.retrieval.modality_retrievers import summarize
from mmrag.retrieval.router import HeuristicRouter, RoutingDecision
from mmrag.schemas import Chunk, Modality, ScoredChunk

log = get_logger(__name__)


@dataclass
class ModalityRetrievalResult:
    """Retrieved chunks, plus the routing and fusion story behind them."""

    query: str
    results: list[ScoredChunk] = field(default_factory=list)
    routing: RoutingDecision | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)


class ModalityAwareRetriever:
    """Routes a query to modality retrievers and fuses their ranked lists."""

    def __init__(
        self,
        config: RetrievalConfig,
        *,
        router: HeuristicRouter,
        retrievers: dict[str, Retriever],
        chunks: dict[str, Chunk],
        reranker: Any | None = None,
    ):
        self.config = config
        self.router = router
        self.retrievers = retrievers
        self.chunks = chunks
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        force_modalities: list[Modality] | None = None,
    ) -> ModalityRetrievalResult:
        top_k = top_k or self.config.top_k
        pool = self.config.candidates_per_retriever
        latency: dict[str, float] = {}

        started = time.perf_counter()
        base = MetadataFilter(doc_ids=list(doc_ids)) if doc_ids else None
        decision = self.router.route(query, base_filters=base)
        if force_modalities is not None:
            # Used by the ablation: fire an exact retriever set regardless of
            # what the router would have chosen.
            decision.modalities = list(force_modalities)
            decision.strategy = "forced"
        latency["routing_ms"] = (time.perf_counter() - started) * 1000

        selected = self._select(decision.modalities)
        if not selected:
            log.warning("router selected no available retriever for %r", query)

        outputs: list[RetrieverOutput] = []
        for name, retriever in selected.items():
            output = retriever.retrieve(query, pool, decision.filters)
            latency[f"{name}_ms"] = output.latency_ms
            outputs.append(output)

        started = time.perf_counter()
        fuse_to = self.config.rerank_top_n if self.reranker else top_k
        fused = reciprocal_rank_fusion(
            [RankedList(o.retriever, o.chunk_ids) for o in outputs],
            k=self.config.rrf_k,
            weights=self.config.fusion_weights,
            top_k=max(fuse_to, top_k),
        )
        latency["fusion_ms"] = (time.perf_counter() - started) * 1000

        scored: list[ScoredChunk] = []
        for result in fused:
            chunk = self.chunks.get(result.chunk_id)
            if chunk is None:
                continue
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=result.score,
                    rank=result.rank,
                    retriever="+".join(result.retrievers),
                    modality=_modality_of(chunk),
                    component_ranks=result.component_ranks,
                )
            )

        missing = len(fused) - len(scored)
        if missing:
            log.warning("%d fused hits had no chunk record; index may be stale", missing)

        if self.reranker is not None and scored:
            started = time.perf_counter()
            scored = self.reranker.rerank(query, scored, top_k=top_k)
            latency["rerank_ms"] = (time.perf_counter() - started) * 1000
        else:
            scored = scored[:top_k]

        # Excludes one-off model loads, which retrievers report separately.
        latency["total_ms"] = sum(
            v for key, v in latency.items() if key.endswith("_ms") and key != "total_ms"
        )

        return ModalityRetrievalResult(
            query=query,
            results=scored,
            routing=decision,
            latency_ms=latency,
            diagnostics={
                **fusion_diagnostics(
                    fused[:top_k], [RankedList(o.retriever, o.chunk_ids) for o in outputs]
                ),
                "routed_to": [m.value for m in decision.modalities],
                "retrievers_fired": list(selected),
                "per_retriever": summarize(outputs),
                "reranked": self.reranker is not None,
                "missing_chunk_records": missing,
                "returned_by_modality": _count_by_modality(scored),
            },
        )

    def _select(self, modalities: list[Modality]) -> dict[str, Retriever]:
        """The retrievers whose modality the router asked for.

        Order is the registration order, so fusion input order is deterministic
        and two runs of one config produce identical results.
        """
        wanted = set(modalities)
        return {
            name: retriever
            for name, retriever in self.retrievers.items()
            if retriever.modality in wanted
        }


def _modality_of(chunk: Chunk) -> Modality:
    """The modality a chunk *is*, independent of how it was retrieved."""
    return {
        "table": Modality.TABLE,
        "figure": Modality.IMAGE,
    }.get(chunk.chunk_type.value, Modality.TEXT)


def _count_by_modality(scored: list[ScoredChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in scored:
        key = hit.chunk.chunk_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts
