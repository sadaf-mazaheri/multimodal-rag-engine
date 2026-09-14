"""Visual page retrieval: late-interaction search over rendered pages.

A retriever like any other: it implements ``Retriever``, is registered with the
engine, and is fused by the same machinery. Nothing downstream of fusion knows
it exists.

It works in two steps, kept apart so either can change on its own:

1. :meth:`VisualPageRetriever.rank_pages` scores pages by ColQwen2 MaxSim. This
   is the visual signal itself, and the unit is a whole page.
2. :meth:`VisualPageRetriever.retrieve` expands the ranked pages into **the
   chunks already indexed on each page**, in document order, each carrying its
   page's score. That is what lets the engine fuse, rerank, cite and generate
   from the page signal with no special case.

Expansion has three consequences worth knowing:

* provenance is exact by construction -- every hit is a real chunk with its own
  document, page and bounding box, and nothing spans a page boundary;
* evidence recorded as ``(doc_id, page, modality)`` matches exactly as it does
  for any other retriever, so evaluation needs no special case;
* the generator reads the same chunk text it reads for every other retriever.

What it cannot do is return a page that produced no chunks; those are counted,
not hidden. The candidate budget is also spent in chunks, not pages, so a page
with many chunks uses more of it -- see the README's Method 3 notes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mmrag.embeddings.visual import VisualEncoder, VisualModelUnavailableError, compatible_identity
from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import Hit, MetadataFilter, RetrieverOutput
from mmrag.retrieval.modality_retrievers import _apply_chunk_filters
from mmrag.schemas import Chunk, Modality
from mmrag.stores.multivector import IndexIntegrityError, PageEmbeddingIndex, QueryEmbeddingCache

log = get_logger(__name__)


def chunks_by_page(chunks: dict[str, Chunk]) -> dict[tuple[str, int], list[str]]:
    """Chunk ids per (document, page), in the chunk set's own order."""
    out: dict[tuple[str, int], list[str]] = {}
    for chunk_id, chunk in chunks.items():
        out.setdefault((chunk.doc_id, chunk.page_number), []).append(chunk_id)
    return out


@dataclass(frozen=True)
class PageHit:
    """One scored page."""

    page_id: str
    doc_id: str
    page_number: int
    score: float
    rank: int


@dataclass
class PageRanking:
    pages: list[PageHit]
    query_embedding: str
    query_tokens: int
    pages_scored: int
    model_load_ms: float = 0.0


class VisualPageRetriever:
    """ColQwen2 MaxSim over pages, expanded to same-page chunks."""

    name = "visual_page"
    modality = Modality.VISUAL_PAGE

    def __init__(
        self,
        index: PageEmbeddingIndex,
        chunks: dict[str, Chunk],
        *,
        cache: QueryEmbeddingCache | None = None,
        encoder_factory: Callable[[], VisualEncoder] | None = None,
    ):
        self.index = index
        self.chunks = chunks
        self.cache = cache
        self.encoder_factory = encoder_factory
        self._encoder: VisualEncoder | None = None
        self._by_page = chunks_by_page(chunks)
        if cache is not None:
            cache.check_identity(index.identity)

    # -- query embedding -----------------------------------------------------

    def _embed(self, query: str) -> tuple[np.ndarray, str, float]:
        """Query vectors, where they came from, and any one-off model load time."""
        if self.cache is not None:
            cached = self.cache.get(query)
            if cached is not None:
                return cached, "cache", 0.0

        if self.encoder_factory is None:
            raise VisualModelUnavailableError(
                f"no cached embedding for query {query[:60]!r} and no visual model configured. "
                "Precompute it with 'mmrag index embed-queries --config method3' on the machine "
                "that has the model, or install the visual extra to encode queries here."
            )
        load_ms = 0.0
        if self._encoder is None:
            started = time.perf_counter()
            encoder = self.encoder_factory()
            identity = encoder.identity()
            load_ms = (time.perf_counter() - started) * 1000
            if not compatible_identity(identity, self.index.identity):
                raise IndexIntegrityError(
                    f"query encoder {identity} is not the model that built the page index "
                    f"({self.index.identity})"
                )
            self._encoder = encoder
        return self._encoder.encode_queries([query])[0], "encoded", load_ms

    # -- page ranking --------------------------------------------------------

    def rank_pages(self, query: str, filters: MetadataFilter | None = None) -> PageRanking:
        """Every candidate page, best first. Page-level filters apply here."""
        vectors, source, load_ms = self._embed(query)
        candidates = self.index.candidate_pages(
            doc_ids=filters.doc_ids if filters else None,
            page_numbers=filters.page_numbers if filters else None,
        )
        pages = [
            PageHit(
                page_id=(record := self.index.pages[page_index]).page_id,
                doc_id=record.doc_id,
                page_number=record.page_number,
                score=score,
                rank=rank,
            )
            for rank, (page_index, score) in enumerate(self.index.rank(vectors, candidates), 1)
        ]
        return PageRanking(
            pages=pages,
            query_embedding=source,
            query_tokens=int(vectors.shape[0]),
            pages_scored=len(candidates),
            model_load_ms=load_ms,
        )

    # -- retrieval -----------------------------------------------------------

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> RetrieverOutput:
        # Query encoding is part of per-query cost; a one-off model load is not,
        # matching how the other retrievers report latency.
        started = time.perf_counter()
        ranking = self.rank_pages(query, filters)
        started += ranking.model_load_ms / 1000

        hits: list[Hit] = []
        pages_used = 0
        pages_without_chunks = 0
        top_pages: list[dict[str, Any]] = []
        for page in ranking.pages:
            chunk_ids = self._by_page.get((page.doc_id, page.page_number), [])
            if len(top_pages) < 5:
                top_pages.append({"page": page.page_id, "score": round(page.score, 4)})
            if not chunk_ids:
                pages_without_chunks += 1
                continue
            page_hits = [Hit(chunk_id=c, score=page.score, rank=0) for c in chunk_ids]
            # Chunk-level filters (type, doc type, domain) apply per chunk and
            # renumber ranks; the running total is renumbered again below.
            page_hits = _apply_chunk_filters(page_hits, self.chunks, filters)
            if page_hits:
                pages_used += 1
                hits.extend(page_hits)
            if len(hits) >= k:
                break

        hits = [Hit(chunk_id=h.chunk_id, score=h.score, rank=i) for i, h in enumerate(hits[:k], 1)]
        return RetrieverOutput(
            retriever=self.name,
            modality=self.modality,
            hits=hits,
            latency_ms=(time.perf_counter() - started) * 1000,
            diagnostics={
                "query_embedding": ranking.query_embedding,
                "query_tokens": ranking.query_tokens,
                "pages_scored": ranking.pages_scored,
                "pages_contributing": pages_used,
                "pages_without_chunks_skipped": pages_without_chunks,
                "top_pages": top_pages,
                **(
                    {"model_load_ms": round(ranking.model_load_ms, 1)}
                    if ranking.model_load_ms
                    else {}
                ),
            },
        )
