"""Retrievers, rank fusion, routing, and reranking.

Shared by every method: ``reciprocal_rank_fusion`` and ``CrossEncoderReranker``.
Method 1 owns ``HybridRetriever``; Method 2 owns the router, the per-modality
retrievers, and ``ModalityAwareRetriever``.
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
