"""The four modality retrievers Method 2 routes between.

**Ownership: unique to Method 2.** Method 1 has a single retriever pair over a
single flattened index; these are per-modality and independently addressable.

Each retriever emits one :class:`RetrieverOutput`, and the names line up exactly
with the ``fusion_weights`` keys in ``configs/method2.yaml`` -- ``bm25``,
``dense``, ``table``, ``image`` -- so a weight in the config maps to a component
you can point at.

``TableRetriever`` and ``ImageRetriever`` are internally hybrid: each combines
two signals of its own and fuses them before returning. That keeps the top-level
fusion interpretable (four weights, four modalities) while letting each modality
use the retrieval strategy that actually suits it.
"""

from __future__ import annotations

import time
from typing import Any

from mmrag.embeddings.image import ImageEmbedder
from mmrag.embeddings.text import TextEmbedder
from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import Hit, MetadataFilter, RetrieverOutput
from mmrag.retrieval.fusion import RankedList, reciprocal_rank_fusion
from mmrag.schemas import Chunk, Modality
from mmrag.stores.bm25 import BM25Index
from mmrag.stores.qdrant import QdrantStore

log = get_logger(__name__)

# RRF constant for the *internal* fusion inside a composite retriever. Kept
# separate from the top-level rrf_k so tuning one does not silently move the
# other.
INTERNAL_RRF_K = 60


def _apply_chunk_filters(
    hits: list[Hit], chunks: dict[str, Chunk], filters: MetadataFilter | None
) -> list[Hit]:
    """Apply metadata constraints a backend could not enforce itself.

    Qdrant filters server-side; BM25 has no notion of payload, so its hits are
    filtered here. Ranks are renumbered afterwards -- leaving gaps would corrupt
    RRF, which reads rank position directly.
    """
    if filters is None or filters.is_empty:
        return hits

    kept: list[Hit] = []
    for hit in hits:
        chunk = chunks.get(hit.chunk_id)
        if chunk is None:
            continue
        if filters.doc_ids and chunk.doc_id not in filters.doc_ids:
            continue
        if filters.page_numbers and chunk.page_number not in filters.page_numbers:
            continue
        if filters.chunk_types and chunk.chunk_type.value not in filters.chunk_types:
            continue
        if filters.doc_types and chunk.metadata.get("doc_type") not in filters.doc_types:
            continue
        if filters.domains and chunk.metadata.get("domain") not in filters.domains:
            continue
        kept.append(hit)

    return [Hit(chunk_id=h.chunk_id, score=h.score, rank=i) for i, h in enumerate(kept, start=1)]


def _search_or_empty(store: QdrantStore, vector, k: int, doc_ids=None, chunk_types=None):
    """Search a collection, tolerating one that was never built.

    A corpus slice with no figures leaves the image collection absent. Firing
    the image retriever at it must degrade to "found nothing", not crash the
    whole query -- the router fans out by design, so it will routinely ask a
    modality that this corpus happens not to contain.
    """
    try:
        if not store.exists():
            return [], "collection missing"
        return store.search(vector, k=k, doc_ids=doc_ids, chunk_types=chunk_types), None
    except Exception as exc:  # pragma: no cover - depends on the service
        log.warning("%s: search failed (%s)", store.collection, exc)
        return [], f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class BM25Retriever:
    """Lexical retrieval over one modality's chunks."""

    modality = Modality.TEXT

    def __init__(
        self,
        index: BM25Index,
        chunks: dict[str, Chunk],
        *,
        name: str = "bm25",
        modality: Modality = Modality.TEXT,
    ):
        self.index = index
        self.chunks = chunks
        self.name = name
        self.modality = modality

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> RetrieverOutput:
        started = time.perf_counter()
        # Over-fetch when filtering, or a restrictive filter leaves almost
        # nothing for fusion to work with.
        fetch = k * 4 if (filters and not filters.is_empty) else k
        raw = self.index.search(query, k=fetch)
        hits = [Hit(chunk_id=h.chunk_id, score=h.score, rank=h.rank) for h in raw]
        hits = _apply_chunk_filters(hits, self.chunks, filters)[:k]
        return RetrieverOutput(
            retriever=self.name,
            modality=self.modality,
            hits=hits,
            latency_ms=(time.perf_counter() - started) * 1000,
            diagnostics={"index_size": len(self.index), "fetched": len(raw)},
        )


class DenseRetriever:
    """Semantic retrieval over one modality's chunks."""

    modality = Modality.TEXT

    def __init__(
        self,
        store: QdrantStore,
        embedder: TextEmbedder,
        *,
        name: str = "dense",
        modality: Modality = Modality.TEXT,
    ):
        self.store = store
        self.embedder = embedder
        self.name = name
        self.modality = modality

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> RetrieverOutput:
        # One-off model load is not part of query cost; see HybridRetriever for
        # the same reasoning applied to Method 1.
        load_ms = self.embedder.ensure_loaded()
        started = time.perf_counter()
        vector = self.embedder.embed_query(query)
        raw, problem = _search_or_empty(
            self.store,
            vector,
            k,
            doc_ids=filters.doc_ids if filters else None,
            chunk_types=filters.chunk_types if filters else None,
        )
        hits = [Hit(chunk_id=h.chunk_id, score=h.score, rank=h.rank) for h in raw]
        diagnostics: dict[str, Any] = {}
        if load_ms:
            diagnostics["model_load_ms"] = round(load_ms, 1)
        if problem:
            diagnostics["unavailable"] = problem
        return RetrieverOutput(
            retriever=self.name,
            modality=self.modality,
            hits=hits,
            latency_ms=(time.perf_counter() - started) * 1000,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TableRetriever:
    """Structure-aware table retrieval.

    Two signals over the same chunks, fused internally:

    * **lexical over the content view** finds the table containing a literal
      value -- ``22,360``, ``GPIO_OE`` -- which is what BM25 is for.
    * **semantic over the schema view** finds the table *about* something, from
      its caption and column headers, even when no cell matches the query.

    Method 1 can only do the first, and only incidentally: with schema and cells
    flattened into one string, the headers are a handful of tokens competing
    against hundreds of digits.
    """

    name = "table"
    modality = Modality.TABLE

    def __init__(
        self,
        *,
        content_index: BM25Index,
        schema_store: QdrantStore,
        embedder: TextEmbedder,
        chunks: dict[str, Chunk],
    ):
        self.content_index = content_index
        self.schema_store = schema_store
        self.embedder = embedder
        self.chunks = chunks

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> RetrieverOutput:
        load_ms = self.embedder.ensure_loaded()
        started = time.perf_counter()

        lexical = self.content_index.search(query, k=k * 2)
        lexical_hits = _apply_chunk_filters(
            [Hit(h.chunk_id, h.score, h.rank) for h in lexical], self.chunks, filters
        )

        vector = self.embedder.embed_query(query)
        semantic, schema_problem = _search_or_empty(
            self.schema_store, vector, k * 2, doc_ids=filters.doc_ids if filters else None
        )

        fused = reciprocal_rank_fusion(
            [
                RankedList("table_content", [h.chunk_id for h in lexical_hits]),
                RankedList("table_schema", [h.chunk_id for h in semantic]),
            ],
            k=INTERNAL_RRF_K,
            top_k=k,
        )
        hits = [Hit(chunk_id=r.chunk_id, score=r.score, rank=r.rank) for r in fused]

        return RetrieverOutput(
            retriever=self.name,
            modality=self.modality,
            hits=hits,
            latency_ms=(time.perf_counter() - started) * 1000,
            diagnostics={
                "content_hits": len(lexical_hits),
                "schema_hits": len(semantic),
                # How often the schema view found a table the cells did not.
                # This is the number that justifies the two-view design.
                "schema_only": sum(1 for r in fused if list(r.component_ranks) == ["table_schema"]),
                "content_only": sum(
                    1 for r in fused if list(r.component_ranks) == ["table_content"]
                ),
                **({"model_load_ms": round(load_ms, 1)} if load_ms else {}),
                **({"schema_unavailable": schema_problem} if schema_problem else {}),
            },
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


class ImageRetriever:
    """Figure retrieval by CLIP image similarity, plus any figure text.

    The CLIP side is the component that reaches figures Method 1 cannot see at
    all. The text side is kept because a captioned figure is usually easier to
    find by its caption than by its pixels -- dropping it would hand Method 2 a
    handicap that has nothing to do with the modality-aware hypothesis.
    """

    name = "image"
    modality = Modality.IMAGE

    def __init__(
        self,
        *,
        image_store: QdrantStore,
        image_embedder: ImageEmbedder,
        text_index: BM25Index | None,
        chunks: dict[str, Chunk],
    ):
        self.image_store = image_store
        self.image_embedder = image_embedder
        self.text_index = text_index
        self.chunks = chunks

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> RetrieverOutput:
        load_ms = self.image_embedder.ensure_loaded()
        started = time.perf_counter()

        vector = self.image_embedder.embed_query(query)
        visual, image_problem = _search_or_empty(
            self.image_store, vector, k * 2, doc_ids=filters.doc_ids if filters else None
        )

        ranked = [RankedList("image_clip", [h.chunk_id for h in visual])]
        textual_hits: list[Hit] = []
        if self.text_index is not None and len(self.text_index):
            raw = self.text_index.search(query, k=k * 2)
            textual_hits = _apply_chunk_filters(
                [Hit(h.chunk_id, h.score, h.rank) for h in raw], self.chunks, filters
            )
            ranked.append(RankedList("figure_text", [h.chunk_id for h in textual_hits]))

        fused = reciprocal_rank_fusion(ranked, k=INTERNAL_RRF_K, top_k=k)
        hits = [Hit(chunk_id=r.chunk_id, score=r.score, rank=r.rank) for r in fused]

        # The headline diagnostic for Method 2: hits that exist only because of
        # the image vector -- figures with no retrievable text at all, which
        # Method 1 could not have returned under any query.
        clip_only = [r.chunk_id for r in fused if list(r.component_ranks) == ["image_clip"]]
        text_invisible = sum(
            1
            for chunk_id in clip_only
            if (c := self.chunks.get(chunk_id)) is not None and not _has_text_beyond_header(c)
        )

        return RetrieverOutput(
            retriever=self.name,
            modality=self.modality,
            hits=hits,
            latency_ms=(time.perf_counter() - started) * 1000,
            diagnostics={
                "clip_hits": len(visual),
                "figure_text_hits": len(textual_hits),
                "clip_only": len(clip_only),
                "text_invisible_recovered": text_invisible,
                **({"model_load_ms": round(load_ms, 1)} if load_ms else {}),
                **({"unavailable": image_problem} if image_problem else {}),
            },
        )


def _has_text_beyond_header(chunk: Chunk) -> bool:
    from mmrag.retrieval.views import has_figure_text

    return has_figure_text(chunk)


def summarize(outputs: list[RetrieverOutput]) -> dict[str, Any]:
    """Per-retriever result counts and latencies, for the run record."""
    return {
        output.retriever: {
            "modality": output.modality.value,
            "n_hits": len(output),
            "latency_ms": round(output.latency_ms, 1),
            **output.diagnostics,
        }
        for output in outputs
    }
