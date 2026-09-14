"""Retrievers, rank fusion, routing, and reranking.

Shared by every method: ``reciprocal_rank_fusion`` and ``CrossEncoderReranker``.
The engine (:mod:`mmrag.engine`) uses the router, the ``Retriever``
implementations -- ``BM25Retriever``, ``DenseRetriever``, ``TableRetriever``,
``ImageRetriever`` and ``visual_page.VisualPageRetriever`` -- and
``ModalityAwareRetriever``. Method 1's frozen pipeline owns ``HybridRetriever``.

``visual_page`` is not imported here, so importing this package stays light.
"""

from mmrag.retrieval.base import Hit, MetadataFilter, Retriever, RetrieverOutput
from mmrag.retrieval.fusion import (
    FusedResult,
    RankedList,
    fusion_diagnostics,
    reciprocal_rank_fusion,
)
from mmrag.retrieval.hybrid import HybridRetriever, RetrievalResult
from mmrag.retrieval.metadata import MetadataResolver
from mmrag.retrieval.modality import ModalityAwareRetriever, ModalityRetrievalResult
from mmrag.retrieval.modality_retrievers import (
    BM25Retriever,
    DenseRetriever,
    ImageRetriever,
    TableRetriever,
)
from mmrag.retrieval.router import HeuristicRouter, RoutingDecision

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "FusedResult",
    "HeuristicRouter",
    "Hit",
    "HybridRetriever",
    "ImageRetriever",
    "MetadataFilter",
    "MetadataResolver",
    "ModalityAwareRetriever",
    "ModalityRetrievalResult",
    "RankedList",
    "RetrievalResult",
    "Retriever",
    "RetrieverOutput",
    "RoutingDecision",
    "TableRetriever",
    "fusion_diagnostics",
    "reciprocal_rank_fusion",
]
