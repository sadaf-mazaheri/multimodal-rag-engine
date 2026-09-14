"""Index components: build retrieval indexes from the parsed corpus, and open
retrievers over them.

Each component owns one family of on-disk and Qdrant artefacts and knows nothing
about routing, fusion or generation. Pipelines compose them:

* :class:`ModalityIndex` -- a modality-aware chunk set with text, table and
  figure sub-indexes (BM25, dense, CLIP).
* :class:`VisualPageIndexer` -- ColQwen2 multi-vector embeddings of every
  rendered page, independent of any chunk set.

The visual module is imported lazily by callers that need it, so importing this
package never pulls in anything heavier than numpy.
"""

from mmrag.indexing.modality import ModalityIndex, ModalityIndexReport

__all__ = ["ModalityIndex", "ModalityIndexReport"]
