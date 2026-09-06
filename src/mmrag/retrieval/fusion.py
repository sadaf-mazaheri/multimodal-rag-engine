"""Reciprocal Rank Fusion.

RRF combines ranked lists by rank rather than by score:

    score(d) = sum over retrievers r of  weight_r / (k + rank_r(d))

Rank-based fusion is the right default here because BM25 scores are unbounded
and corpus-dependent while cosine similarities sit in [-1, 1]. Any score-based
combination has to normalise first, and every normalisation choice (min-max over
the returned window, z-score, softmax) is itself a tunable that quietly changes
the results. RRF sidesteps that entirely: only the ordering matters.

``k`` damps the influence of the head of each list. At the conventional k=60 the
gap between rank 1 and rank 2 is small, so a document must rank well across
*several* retrievers to beat one that a single retriever loves. Lowering k makes
fusion more winner-take-all.

Weights are applied on top so a retriever known to be noisier -- CLIP text-image
similarity in Method 2, say -- can contribute less without being excluded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankedList:
    """One retriever's output: chunk ids in rank order."""

    retriever: str
    chunk_ids: list[str]
    scores: dict[str, float] = field(default_factory=dict)

    def ranks(self) -> dict[str, int]:
        """chunk_id -> 1-based rank, keeping the best rank on duplicates."""
        out: dict[str, int] = {}
        for rank, chunk_id in enumerate(self.chunk_ids, start=1):
            out.setdefault(chunk_id, rank)
        return out


@dataclass
class FusedResult:
    chunk_id: str
    score: float
    rank: int
    component_ranks: dict[str, int]

    @property
    def retrievers(self) -> list[str]:
        return sorted(self.component_ranks)


def reciprocal_rank_fusion(
    ranked_lists: list[RankedList],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedResult]:
    """Fuse ranked lists into one ordering.

    Ties are broken by the best single rank a chunk achieved, then by chunk id,
    so the output is deterministic. Non-determinism here would make two runs of
    the same config produce different metrics, which is exactly what this
    benchmark exists to rule out.
    """
    if k < 1:
        raise ValueError(f"rrf k must be >= 1, got {k}")

    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    components: dict[str, dict[str, int]] = defaultdict(dict)

    for ranked in ranked_lists:
        weight = weights.get(ranked.retriever, 1.0)
        if weight == 0:
            continue
        for chunk_id, rank in ranked.ranks().items():
            scores[chunk_id] += weight / (k + rank)
            components[chunk_id][ranked.retriever] = rank

    order = sorted(
        scores,
        key=lambda cid: (-scores[cid], min(components[cid].values()), cid),
    )
    if top_k is not None:
        order = order[:top_k]

    return [
        FusedResult(
            chunk_id=chunk_id,
            score=scores[chunk_id],
            rank=position,
            component_ranks=dict(components[chunk_id]),
        )
        for position, chunk_id in enumerate(order, start=1)
    ]


def fusion_diagnostics(
    results: list[FusedResult], ranked_lists: list[RankedList]
) -> dict[str, Any]:
    """Where the fused results came from.

    Answers the question the comparison actually needs: is the hybrid better
    than its parts, or is one retriever doing all the work? Without this, a
    fusion that silently degenerates to "BM25 only" looks like a working hybrid.
    """
    names = [r.retriever for r in ranked_lists]
    contributed = {name: sum(1 for r in results if name in r.component_ranks) for name in names}
    unique = {name: sum(1 for r in results if list(r.component_ranks) == [name]) for name in names}
    return {
        "n_results": len(results),
        "candidates_per_retriever": {r.retriever: len(r.chunk_ids) for r in ranked_lists},
        "contributed": contributed,
        "found_only_by": unique,
        "found_by_all": sum(1 for r in results if len(r.component_ranks) == len(names)),
    }
