"""Evaluation: gold set, retrieval metrics, and the comparison report.

Retrieval and generation are evaluated separately and on purpose. Retrieval is
what the three methods actually differ in, and it can be scored offline, for
free, and deterministically. Generation needs a provider, costs money, and
mixes a vendor's behaviour into a retrieval comparison -- so it lives behind its
own module and its own command, and is not required to produce the headline
numbers.
"""

from mmrag.evaluation.gold import (
    Evidence,
    GoldProblem,
    GoldQuery,
    GoldSet,
    describe,
    validate_against_chunks,
)
from mmrag.evaluation.metrics import (
    aggregate,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)

__all__ = [
    "Evidence",
    "GoldProblem",
    "GoldQuery",
    "GoldSet",
    "aggregate",
    "describe",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "summarize",
    "validate_against_chunks",
]
