"""Method 1's retriever: BM25 + dense, fused with RRF, optionally reranked.

The two retrievers are kept as separate ranked lists all the way to fusion, and
their per-retriever ranks are preserved on every result. That is what makes the
ablation ("was the hybrid actually better than either alone?") answerable after
the fact rather than requiring three separate runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from mmrag.config import RetrievalConfig
from mmrag.embeddings.text import TextEmbedder
from mmrag.logging_utils import get_logger
from mmrag.retrieval.fusion import RankedList, fusion_diagnostics, reciprocal_rank_fusion
from mmrag.schemas import Chunk, Modality, ScoredChunk
from mmrag.stores.bm25 import BM25Index
from mmrag.stores.qdrant import QdrantStore

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Retrieved chunks plus how they were found."""

    query: str
    results: list[ScoredChunk] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)


class HybridRetriever:
    """BM25 + dense hybrid over one chunk variant."""

    def __init__(
        self,
        config: RetrievalConfig,
        *,
        bm25: BM25Index,
        qdrant: QdrantStore,
        embedder: TextEmbedder,
        chunks: dict[str, Chunk],
        reranker: Any | None = None,
    ):
        self.config = config
        self.bm25 = bm25
        self.qdrant = qdrant
        self.embedder = embedder
        self.chunks = chunks
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.config.top_k
        pool = self.config.candidates_per_retriever
        latency: dict[str, float] = {}

        started = time.perf_counter()
        bm25_hits = self.bm25.search(query, k=pool)
        latency["bm25_ms"] = (time.perf_counter() - started) * 1000

        # Model loading is a one-off startup cost. Timing it as part of the
        # query would report ~20s of "embedding" on the first call of every run.
        load_ms = self.embedder.ensure_loaded()
        if load_ms:
            latency["model_load_ms"] = load_ms

        started = time.perf_counter()
        vector = self.embedder.embed_query(query)
        latency["embed_ms"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dense_hits = self.qdrant.search(vector, k=pool, doc_ids=doc_ids)
        latency["dense_ms"] = (time.perf_counter() - started) * 1000

        # Restrict BM25 to the same document filter as the dense side, so both
        # retrievers see the same candidate universe and fusion stays fair.
        if doc_ids:
            allowed = set(doc_ids)
            bm25_hits = [
                h
                for h in bm25_hits
                if (c := self.chunks.get(h.chunk_id)) is not None and c.doc_id in allowed
            ]

        ranked_lists = [
            RankedList(
                retriever="bm25",
                chunk_ids=[h.chunk_id for h in bm25_hits],
                scores={h.chunk_id: h.score for h in bm25_hits},
            ),
            RankedList(
                retriever="dense",
                chunk_ids=[h.chunk_id for h in dense_hits],
                scores={h.chunk_id: h.score for h in dense_hits},
            ),
        ]

        started = time.perf_counter()
        # Fuse a wider slice than requested when reranking, so the reranker has
        # something to reorder rather than merely confirming the fused order.
        fuse_to = self.config.rerank_top_n if self.reranker else top_k
        fused = reciprocal_rank_fusion(
            ranked_lists,
            k=self.config.rrf_k,
            weights=self.config.fusion_weights,
            top_k=max(fuse_to, top_k),
        )
        latency["fusion_ms"] = (time.perf_counter() - started) * 1000

        scored = [
            ScoredChunk(
                chunk=chunk,
                score=result.score,
                rank=result.rank,
                retriever="+".join(result.retrievers),
                modality=_modality_for(chunk),
                component_ranks=result.component_ranks,
            )
            for result in fused
            if (chunk := self.chunks.get(result.chunk_id)) is not None
        ]

        missing = len(fused) - len(scored)
        if missing:
            # An index that outlived its chunk table. Surfaced rather than
            # silently returning fewer results than requested.
            log.warning("%d fused hits had no chunk record; index may be stale", missing)

        if self.reranker is not None and scored:
            started = time.perf_counter()
            scored = self.reranker.rerank(query, scored, top_k=top_k)
            latency["rerank_ms"] = (time.perf_counter() - started) * 1000
        else:
            scored = scored[:top_k]

        # Excludes model_load_ms: total_ms is per-query cost, and a one-off
        # startup charge in it would make the first query incomparable to the rest.
        latency["total_ms"] = sum(
            v
            for key, v in latency.items()
            if key.endswith("_ms") and key not in {"total_ms", "model_load_ms"}
        )

        return RetrievalResult(
            query=query,
            results=scored,
            latency_ms=latency,
            diagnostics={
                **fusion_diagnostics(fused[:top_k], ranked_lists),
                "reranked": self.reranker is not None,
                "missing_chunk_records": missing,
            },
        )


def _modality_for(chunk: Chunk) -> Modality:
    """Method 1 retrieves everything as text -- that is the whole point.

    The chunk's own type is still reported, so evaluation can slice results by
    what the evidence originally *was* even though it was retrieved as text.
    """
    return {
        "table": Modality.TABLE,
        "figure": Modality.IMAGE,
    }.get(chunk.chunk_type.value, Modality.TEXT)
