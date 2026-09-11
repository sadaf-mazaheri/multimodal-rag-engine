"""Method 2's retrieval orchestrator: route, fan out, fuse, rerank.

**Ownership: unique to Method 2.** The counterpart to Method 1's
``HybridRetriever``, which is left untouched.

The shape of a query's journey:

    query -> router -> {bm25, dense, table, image} retrievers
          -> RRF within each modality        (text: bm25 + dense)
          -> weighted RRF across modalities  (text / table / image)
          -> modality-floored candidate pool
          -> cross-encoder rerank

Everything after the fan-out is shared machinery -- the same
``reciprocal_rank_fusion`` and ``CrossEncoderReranker`` Method 1 uses. What is
new is the routing and the per-modality fan-out.

Fusion runs in two stages for a reason. RRF is additive across ranked lists, so
a modality that supplies more lists gets more votes regardless of relevance.
``TableRetriever`` and ``ImageRetriever`` already fused their own two signals
internally, but text's ``bm25`` and ``dense`` arrived separately -- giving text
a 2.86x ceiling advantage over image before a single document was scored. On a
corpus-grounded query the best figure landed at fused rank 44 with a score of
exactly 0.7/(60+1), its structural ceiling, behind 43 text chunks.

Collapsing text first equalises the votes, but it is not sufficient on its own:
RRF cannot *abstain*, so with equal votes an irrelevant modality still injects
its best candidate at full strength. The modality floor plus the cross-encoder
resolves that -- see ``_balanced_pool``.

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
        per_modality = _fuse_within_modalities(outputs, k=self.config.rrf_k)
        # When a reranker will arbitrate, keep the whole fused ordering rather
        # than a window of it: the modality floor exists precisely to reach
        # candidates that cross-modality fusion buried, and truncating here
        # would hide them from it. The pool is bounded later instead.
        fuse_to = None if self.reranker else top_k
        fused = reciprocal_rank_fusion(
            [RankedList(name, ids) for name, ids in per_modality.items()],
            k=self.config.rrf_k,
            weights=_modality_weights(outputs, self.config.fusion_weights),
            top_k=fuse_to,
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

        pool_composition: dict[str, int] = {}
        if self.reranker is not None and scored:
            # Guarantee every fired modality a place in the pool before the
            # cross-encoder sees it. Without this the pool arrives pre-filtered
            # by RRF and the reranker can only reorder what already got through.
            scored = _balanced_pool(
                scored,
                per_modality,
                per_modality_floor=self.config.rerank_pool_per_modality,
                limit=max(self.config.rerank_top_n, top_k),
            )
            pool_composition = _count_by_modality(scored)
            started = time.perf_counter()
            scored = self.reranker.rerank(query, scored, top_k=top_k)
            latency["rerank_ms"] = (time.perf_counter() - started) * 1000
        else:
            # No arbiter, so no quotas: a modality floor without a reranker to
            # judge relevance would inject evidence nothing has vouched for.
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
                    fused[:top_k],
                    [RankedList(name, ids) for name, ids in per_modality.items()],
                ),
                "routed_to": [m.value for m in decision.modalities],
                "retrievers_fired": list(selected),
                "per_retriever": summarize(outputs),
                "fusion_stages": {
                    "within_modality": {n: len(ids) for n, ids in per_modality.items()},
                    "modality_weights": _modality_weights(outputs, self.config.fusion_weights),
                },
                "rerank_pool_by_modality": pool_composition,
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


def _fuse_within_modalities(
    outputs: list[RetrieverOutput], *, k: int
) -> dict[str, list[str]]:
    """Collapse each modality's retrievers into a single ranked list.

    Stage one of two. ``TableRetriever`` and ``ImageRetriever`` already fuse
    their own pair of signals internally; text's ``bm25`` and ``dense`` did not,
    so text reached the cross-modality stage as *two* lists and every other
    modality as one. Since RRF is additive, that handed text twice the
    achievable score before relevance was considered at all -- a figure ranked
    first by its own retriever could not outscore a text chunk that merely
    appeared in both text lists.

    Ordering follows the outputs, which follow registration order, so fusion
    input order stays deterministic.
    """
    grouped: dict[str, list[RankedList]] = {}
    for output in outputs:
        grouped.setdefault(output.modality.value, []).append(
            RankedList(output.retriever, output.chunk_ids)
        )

    fused: dict[str, list[str]] = {}
    for modality, lists in grouped.items():
        if len(lists) == 1:
            fused[modality] = list(lists[0].chunk_ids)
        else:
            fused[modality] = [r.chunk_id for r in reciprocal_rank_fusion(lists, k=k)]
    return fused


def _modality_weights(
    outputs: list[RetrieverOutput], weights: dict[str, float]
) -> dict[str, float]:
    """Resolve a weight per modality from weights configured per retriever.

    An explicit modality key wins. Otherwise a modality inherits the strongest
    weight among its own retrievers, which keeps a config written against
    retriever names meaning the same thing after the two stages were split
    apart: ``bm25: 1.0, dense: 1.0`` becomes ``text: 1.0`` rather than 2.0.
    """
    resolved: dict[str, float] = {}
    for output in outputs:
        modality = output.modality.value
        if modality in weights:
            resolved[modality] = weights[modality]
            continue
        candidate = weights.get(output.retriever, 1.0)
        resolved[modality] = max(resolved.get(modality, candidate), candidate)
    return resolved


def _balanced_pool(
    scored: list[ScoredChunk],
    per_modality: dict[str, list[str]],
    *,
    per_modality_floor: int,
    limit: int,
) -> list[ScoredChunk]:
    """Give every fired modality a floor in the rerank pool, then fill by rank.

    Stage two. RRF is scale-free: a retriever contributes its rank-1 candidate
    at full strength whether or not it holds anything relevant, and it has no
    way to abstain. No weighting fixes both directions of that -- raising the
    image weight surfaces figures on figure questions *and* on prose ones.

    The cross-encoder can abstain, because it reads the pair. So the floor
    decides only what gets *considered*; the reranker still decides the order,
    and scores an irrelevant table below the prose it was competing with.

    Candidates keep their fused order within the pool, and the pool is capped
    at ``limit`` so reranking cost stays bounded.
    """
    if per_modality_floor <= 0:
        return scored[:limit]

    by_id = {hit.chunk.chunk_id: hit for hit in scored}
    picked: dict[str, ScoredChunk] = {}

    for ids in per_modality.values():
        taken = 0
        for chunk_id in ids:
            if taken >= per_modality_floor or len(picked) >= limit:
                break
            hit = by_id.get(chunk_id)
            # Absent only if the chunk had no record at all; the caller passes
            # the untruncated fused list so a buried candidate is still here.
            if hit is None or chunk_id in picked:
                continue
            picked[chunk_id] = hit
            taken += 1

    # Top up in fused order, so a modality that earned more than its floor
    # keeps the surplus.
    for hit in scored:
        if len(picked) >= limit:
            break
        picked.setdefault(hit.chunk.chunk_id, hit)

    # Restore fused order: the floor decides membership, not position.
    return sorted(picked.values(), key=lambda h: h.rank)[:limit]


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
