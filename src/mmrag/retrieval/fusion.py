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
from typing import Any, Literal


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
    combine: Literal["sum", "max"] = "sum",
) -> list[FusedResult]:
    """Fuse ranked lists into one ordering.

    ``combine`` decides what several lists agreeing is worth.

    ``sum`` is classic RRF and the right default *across* modalities, where each
    list is an independent vote: two retrievers agreeing is real evidence.

    ``max`` suits fusion *within* one modality, where the lists are alternative
    routes to the same evidence rather than independent judges -- a figure found
    by its pixels or by its caption is the same figure, and the better route
    should decide. Summing there produced a defect: a candidate present in one
    list is capped at ``1/(k+1)`` however perfect its match, so it loses to any
    candidate sitting mid-table in both. On this corpus that demoted figures
    ranked *first* by CLIP or by figure-text to 24th, 28th and 39th, below the
    pool floor, so the cross-encoder never saw them.

    That asymmetry is not always recoverable by ranking better, because the
    sub-signals do not cover the same universe: 37 figures carry no text at all
    and are structurally absent from the figure-text index. Under ``sum`` their
    absence reads as "ranked worst" rather than "not applicable" -- penalising
    exactly the figures the image index exists to reach.

    Ties are broken by the best single rank a chunk achieved, then by chunk id,
    so the output is deterministic. Non-determinism here would make two runs of
    the same config produce different metrics, which is exactly what this
    benchmark exists to rule out.
    """
    if k < 1:
        raise ValueError(f"rrf k must be >= 1, got {k}")
    if combine not in ("sum", "max"):
        raise ValueError(f"combine must be 'sum' or 'max', got {combine!r}")

    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    components: dict[str, dict[str, int]] = defaultdict(dict)

    for ranked in ranked_lists:
        weight = weights.get(ranked.retriever, 1.0)
        if weight == 0:
            continue
        for chunk_id, rank in ranked.ranks().items():
            contribution = weight / (k + rank)
            if combine == "max":
                scores[chunk_id] = max(scores[chunk_id], contribution)
            else:
                scores[chunk_id] += contribution
            components[chunk_id][ranked.retriever] = rank

    if combine == "max":
        # Under max, the top of every list scores exactly w/(k+1), so rank-one
        # candidates tie by construction. Corroboration breaks that tie: among
        # equally good best-routes, the candidate more signals found wins. It
        # cannot distort the primary ordering, which is the property summing
        # failed to have. Applied only here, so `sum` -- and Method 1 with it --
        # keeps the tie-break it was measured under.
        def sort_key(cid: str) -> tuple[float, int, int, str]:
            return (-scores[cid], -len(components[cid]), min(components[cid].values()), cid)
    else:

        def sort_key(cid: str) -> tuple[float, int, int, str]:
            return (-scores[cid], 0, min(components[cid].values()), cid)

    order = sorted(scores, key=sort_key)
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
