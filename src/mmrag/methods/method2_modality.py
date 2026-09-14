"""Method 2: Modality-aware RAG.

Each modality keeps its native representation and gets a retriever suited to it,
a router picks which to fire, and their ranked lists are fused and reranked.

What this measures against Method 1 is the value of *not* flattening. The two
methods read the same parsed elements, chunk them with the same shared chunker,
and answer with the same prompt and provider. The only differences are how those
chunks are indexed and how they are searched -- which is what makes the
comparison attributable.

Composition
-----------
This module is a configuration, not an implementation:

* :class:`mmrag.indexing.ModalityIndex` builds and opens the ``method2`` chunk
  set and its text, table and figure sub-indexes;
* :class:`mmrag.engine.RAGEngine` resolves, routes, fuses, reranks and answers
  over the retrievers that index opens, with the Postgres ``MetadataResolver``.

Method 1 is not modified by any of it: Method 2 writes to its own chunk variant
and its own Qdrant collections, so both indexes coexist and either can be
rebuilt without disturbing the other.
"""

from __future__ import annotations

from pathlib import Path

from mmrag.config import ExperimentConfig
from mmrag.engine import RAGEngine, build_reranker
from mmrag.indexing.modality import (  # noqa: F401 - re-exported for callers
    CHUNKS_FILE,
    FIGURE_TEXT_INDEX,
    IMAGE_INDEX,
    MANIFEST_FILE,
    TABLE_CONTENT_INDEX,
    TABLE_SCHEMA_INDEX,
    TEXT_INDEX,
    ModalityIndex,
    ModalityIndexReport,
)
from mmrag.methods.base import EngineMethod
from mmrag.retrieval.router import HeuristicRouter
from mmrag.schemas import Chunk

METHOD_NAME = "method2"

# The report type kept its old name for existing callers.
Method2IndexReport = ModalityIndexReport



class Method2ModalityAware(EngineMethod):
    """Modality-aware RAG: the engine over the text, table and figure retrievers."""

    name = METHOD_NAME

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        index_base: Path | None = None,
        variant: str | None = None,
    ):
        super().__init__(config)
        self.index = ModalityIndex(
            config,
            variant=variant or config.chunk_variant,
            index_base=index_base,
            owner=METHOD_NAME,
        )

    @property
    def variant(self) -> str:
        return self.index.variant

    @property
    def index_dir(self) -> Path:
        return self.index.index_dir

    def collection(self, name: str) -> str:
        return self.index.collection(name)

    @property
    def chunks(self) -> dict[str, Chunk]:
        return self.index.chunks

    def build_index(self, doc_ids: list[str] | None = None) -> ModalityIndexReport:
        report = self.index.build(doc_ids)
        self.reset()
        return report

    def _build_engine(self) -> RAGEngine:
        return RAGEngine(
            self.config,
            name=self.name,
            retrievers=self.index.retrievers(),
            chunks=self.index.chunks,
            router=HeuristicRouter(self.config.router),
            reranker=build_reranker(self.config),
            resolver_factory=self.index.metadata_resolver,
        )
