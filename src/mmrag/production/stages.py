"""Canonical pipeline stages from the per-method latency keys retrieval runs record.

Method 1's ``HybridRetriever`` and the Method 2/3 ``ModalityAwareRetriever``
name their timings differently: Method 1 reports ``embed_ms`` separately, while a
Method 2 ``dense_ms`` already includes query embedding, and Method 2/3 report one
``<retriever>_ms`` per fired retriever plus ``routing_ms``. This module maps both
onto the same stages:

* ``retrieval``  -- first-stage candidate retrieval: routing, query embedding and
  every retriever (bm25, dense, table, image, visual_page, ...);
* ``fusion``     -- RRF fusion;
* ``reranking``  -- cross-encoder reranking;
* ``retrieval_total`` -- the recorded ``total_ms``, which equals the three above.

``model_load_ms`` is a one-off cold-start cost and is never part of a stage.

Known gaps, reported rather than papered over:

* Method 2/3 metadata resolution runs before the retriever's timer starts and is
  not in any recorded key.
* Method 3's ``visual_page_ms`` was recorded with precomputed ColQwen2 query
  embeddings, so it covers page scoring only, not query encoding.
"""

from __future__ import annotations

from typing import Any

STAGES = ("retrieval", "fusion", "reranking")
_FUSION = "fusion_ms"
_RERANK = "rerank_ms"
_TOTAL = "total_ms"
_MODEL_LOAD = "model_load_ms"
_NON_RETRIEVAL = {_FUSION, _RERANK, _TOTAL, _MODEL_LOAD}

# A recorded total that differs from the sum of its parts by more than this is
# flagged: it would mean a stage was timed but not mapped.
TOTAL_TOLERANCE_MS = 1.0


def canonical_stages(latency_ms: dict[str, float]) -> dict[str, Any]:
    """Map one query's recorded ``latency_ms`` onto canonical stages.

    Returns the three stages, ``retrieval_total`` (the recorded total), the
    one-off ``model_load_ms`` if present, which raw keys fed ``retrieval``, and
    whether the recorded total matches the sum of the stages.
    """
    retrieval_keys = sorted(
        k for k in latency_ms if k.endswith("_ms") and k not in _NON_RETRIEVAL
    )
    retrieval = sum(latency_ms[k] for k in retrieval_keys)
    fusion = latency_ms.get(_FUSION, 0.0)
    reranking = latency_ms.get(_RERANK)
    parts = retrieval + fusion + (reranking or 0.0)
    total = latency_ms.get(_TOTAL, parts)
    return {
        "retrieval": retrieval,
        "fusion": fusion,
        "reranking": reranking,
        "retrieval_total": total,
        "model_load_ms": latency_ms.get(_MODEL_LOAD),
        "retrieval_keys": retrieval_keys,
        "total_consistent": abs(total - parts) <= TOTAL_TOLERANCE_MS,
    }
