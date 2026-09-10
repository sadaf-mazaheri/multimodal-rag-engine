"""The three end-to-end RAG pipelines.

Method 1 is frozen: it established the baseline numbers, so it is not modified
while later methods are built. Methods share the ingestion layer, the chunker,
the stores, rank fusion, and the generation path -- they differ only in how
chunks are indexed and searched, which is what makes the comparison attributable.
"""

from mmrag.methods.method1_textified import IndexReport, Method1Textified
from mmrag.methods.method2_modality import Method2IndexReport, Method2ModalityAware

__all__ = [
    "IndexReport",
    "Method1Textified",
    "Method2IndexReport",
    "Method2ModalityAware",
]
