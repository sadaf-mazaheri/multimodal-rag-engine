"""The benchmark methods: named configurations of the RAG system.

Methods 2 and 3 are configurations of :class:`mmrag.engine.RAGEngine` over
index components from :mod:`mmrag.indexing`; neither inherits from the other.
Method 1 is frozen: it established the baseline numbers, so its single-index
pipeline is kept as it was measured. All three share the ingestion layer, the
chunker, the stores, rank fusion and the generation path, and expose the same
``RAGMethod`` surface, which is all the CLI and evaluation depend on.
"""

from mmrag.methods.base import EngineMethod, RAGMethod
from mmrag.methods.method1_textified import IndexReport, Method1Textified
from mmrag.methods.method2_modality import Method2IndexReport, Method2ModalityAware
from mmrag.methods.method3_visual import Method3HybridVisual, Method3IndexReport
from mmrag.methods.registry import METHODS, build_method

__all__ = [
    "METHODS",
    "EngineMethod",
    "IndexReport",
    "Method1Textified",
    "Method2IndexReport",
    "Method2ModalityAware",
    "Method3HybridVisual",
    "Method3IndexReport",
    "RAGMethod",
    "build_method",
]
