"""Retrieval metrics.

Pure arithmetic over one query's ranked results. No schemas, no I/O, no config
-- so every metric can be unit-tested against hand-computed values rather than
against whatever the pipeline happened to return.

**Input contract.** Every function takes ``matched``: one entry per retrieved
result, in rank order, holding the *index of the gold evidence entry* that
result satisfies, or ``None`` for a miss.

That indirection matters. Gold evidence is ``(doc_id, page, modality)``, and a
page usually yields more than one chunk -- 2.09 on average, up to 17 -- so two
retrieved chunks routinely satisfy the *same* gold entry. Counting relevant
*results* would then report recall above 1.0 for a single-evidence query.
Recall, hit rate and nDCG therefore count **distinct gold entries covered**.

Precision is the deliberate exception: it asks what fraction of what the user
was shown was useful, so there a second chunk from the right page is a second
useful result and is counted as such.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Matched = Sequence[int | None]


def _covered(matched: Matched, k: int) -> set[int]:
    """Distinct gold entries satisfied within the top ``k``."""
    return {m for m in matched[:k] if m is not None}


def recall_at_k(matched: Matched, n_gold: int, k: int) -> float:
    """Fraction of the gold evidence reached within the top ``k``.

    The headline metric: it answers "did retrieval find the evidence", which is
    the question a retrieval architecture is responsible for.
    """
    if n_gold <= 0:
        raise ValueError("a gold query must carry at least one evidence entry")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return len(_covered(matched, k)) / n_gold


def precision_at_k(matched: Matched, k: int) -> float:
    """Fraction of the top ``k`` results that satisfy some gold entry.

    Counts results rather than distinct gold entries, and is reported but not
    leaned on: with one or two gold entries and k=10 its ceiling is set by the
    size of the gold set, not by retrieval quality.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    window = matched[:k]
    if not window:
        return 0.0
    return sum(1 for m in window if m is not None) / len(window)


def hit_rate_at_k(matched: Matched, k: int) -> float:
    """1.0 if anything relevant appears in the top ``k``, else 0.0.

    Averaged over queries this is "how often did we find *something*", which
    separates a method that misses narrowly from one that misses entirely.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return 1.0 if _covered(matched, k) else 0.0


def reciprocal_rank(matched: Matched) -> float:
    """1 / rank of the first relevant result, or 0.0 if there is none.

    Averaged over queries this is MRR. Unlike recall it is sensitive to *where*
    the evidence landed, which is the axis the fusion work moved.
    """
    for index, entry in enumerate(matched, start=1):
        if entry is not None:
            return 1.0 / index
    return 0.0


def ndcg_at_k(matched: Matched, n_gold: int, k: int) -> float:
    """Normalised discounted cumulative gain with binary gains.

    Relevance here is binary, so this mostly restates recall with a positional
    discount. It is included because ``EvaluationConfig.retrieval_metrics``
    lists it and because the discount makes it comparable with published
    retrieval numbers.

    A repeated gold entry scores zero gain: the second chunk from an
    already-covered page adds nothing the answerer did not already have.
    """
    if n_gold <= 0:
        raise ValueError("a gold query must carry at least one evidence entry")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    seen: set[int] = set()
    dcg = 0.0
    for position, entry in enumerate(matched[:k], start=1):
        if entry is None or entry in seen:
            continue
        seen.add(entry)
        dcg += 1.0 / math.log2(position + 1)

    # Ideal: every gold entry packed into the highest positions available.
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_gold, k) + 1))
    return dcg / ideal if ideal else 0.0


def summarize(matched: Matched, n_gold: int, k_values: Sequence[int]) -> dict[str, float]:
    """Every metric for one query, keyed as ``name@k`` (or bare for MRR)."""
    out: dict[str, float] = {"mrr": reciprocal_rank(matched)}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(matched, n_gold, k)
        out[f"precision@{k}"] = precision_at_k(matched, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(matched, k)
        out[f"ndcg@{k}"] = ndcg_at_k(matched, n_gold, k)
    return out


def mean(values: Sequence[float]) -> float:
    """Mean that returns 0.0 rather than raising on an empty slice.

    Slices are sliced by stratum and modality, and an empty one is a normal
    state -- a gold set with no table questions is a legitimate gold set.
    """
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average per-query metrics.

    Macro rather than micro: every query counts once regardless of how much
    gold evidence it carries, so a single query with five evidence pages cannot
    dominate a slice of ten.
    """
    if not per_query:
        return {}
    keys = sorted({key for row in per_query for key in row})
    return {key: mean([row.get(key, 0.0) for row in per_query]) for key in keys}
