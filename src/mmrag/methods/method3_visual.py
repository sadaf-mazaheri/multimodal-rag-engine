"""Method 3: Hybrid visual RAG.

Method 2's retrievers plus ColQwen2 late-interaction retrieval over the page
images ingestion already rendered. What it measures against Method 2 is the
value of *seeing the page*: layout, charts, and text that extraction and OCR
missed all reach the retriever as pixels.

Composition
-----------
A configuration of the engine, not a subclass of Method 2:

* :class:`mmrag.indexing.ModalityIndex` over the ``method2`` chunk set, opened
  **read-only** -- Method 3 never builds or writes it. With the page signal
  removed, what remains is exactly Method 2's retrieval, which is the control;
* :class:`mmrag.indexing.VisualPageIndexer` for the page index and query cache;
* :class:`mmrag.engine.RAGEngine` with the page retriever registered after
  Method 2's four and switched on for every query.

Held constant with Methods 1 and 2: the corpus lock, ingestion and page renders,
the gold sets, the metrics, the generator (text only -- no images are attached,
so answer quality differences remain attributable to retrieval) and the judge.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mmrag.config import ExperimentConfig
from mmrag.engine import RAGEngine, build_reranker
from mmrag.indexing.modality import CHUNKS_FILE, ModalityIndex
from mmrag.indexing.visual_pages import (  # noqa: F401 - re-exported for callers
    CpuIndexingRefusedError,
    EncoderFactory,
    VisualIndexReport,
    VisualPageIndexer,
    prepare_pages,
    resolve_page_image,
)
from mmrag.methods.base import EngineMethod
from mmrag.retrieval.router import HeuristicRouter
from mmrag.schemas import Chunk, Modality
from mmrag.stores.multivector import PageEmbeddingIndex

METHOD_NAME = "method3"
# Method 3 reads Method 2's chunk set and indexes rather than building its own.
BASE_VARIANT = "method2"

# The report type kept its old name for existing callers.
Method3IndexReport = VisualIndexReport


class Method3HybridVisual(EngineMethod):
    """Method 2's retrievers plus the visual page retriever, over the same chunks."""

    name = METHOD_NAME

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        index_base: Path | None = None,
        processed_dir: Path | None = None,
        lock_path: Path | None = None,
        encoder_factory: EncoderFactory | None = None,
    ):
        if config.method != METHOD_NAME:
            raise ValueError(f"Method3HybridVisual needs a method3 config, got {config.method}")
        super().__init__(config)
        self.index = ModalityIndex(
            config, variant=BASE_VARIANT, index_base=index_base, processed_dir=processed_dir,
            owner=BASE_VARIANT,
        )
        self.visual = VisualPageIndexer(
            config, index_base=index_base, processed_dir=processed_dir, lock_path=lock_path,
            encoder_factory=encoder_factory,
        )

    # -- indexes -------------------------------------------------------------

    @property
    def index_dir(self) -> Path:
        return self.index.index_dir

    @property
    def visual_dir(self) -> Path:
        return self.visual.index_dir

    @property
    def query_cache_dir(self) -> Path:
        return self.visual.query_cache_dir

    def collection(self, name: str) -> str:
        return self.index.collection(name)

    @property
    def chunks(self) -> dict[str, Chunk]:
        return self.index.chunks

    def build_index(
        self,
        doc_ids: list[str] | None = None,
        *,
        device: str | None = None,
        allow_cpu: bool | None = None,
        batch_size: int | None = None,
        max_pages: int | None = None,
    ) -> VisualIndexReport:
        """Build the page index only. Method 2's indexes are never touched."""
        report = self.visual.build(
            doc_ids, device=device, allow_cpu=allow_cpu, batch_size=batch_size,
            max_pages=max_pages, chunks_path=self.index.index_dir / CHUNKS_FILE,
        )
        self.reset()
        return report

    def embed_queries(
        self, queries: Sequence[str], *, device: str | None = None, batch_size: int | None = None
    ) -> dict[str, int]:
        return self.visual.embed_queries(queries, device=device, batch_size=batch_size)

    @property
    def visual_index(self) -> PageEmbeddingIndex:
        """The verified page index, checked to cover every page carrying a chunk."""
        index = self.visual.index
        if (self.index.index_dir / CHUNKS_FILE).exists():
            self.visual.check_covers(self.chunks)
        return index

    @property
    def visual_index_is_complete(self) -> bool:
        return self.visual_index.is_complete

    def describe_visual_index(self) -> dict[str, Any]:
        _ = self.visual_index
        return self.visual.describe()

    # -- engine --------------------------------------------------------------

    def _build_engine(self) -> RAGEngine:
        chunks = self.index.chunks
        return RAGEngine(
            self.config,
            name=self.name,
            retrievers={
                **self.index.retrievers(),
                "visual_page": self.visual.retriever(chunks),
            },
            chunks=chunks,
            router=HeuristicRouter(self.config.router, always=[Modality.VISUAL_PAGE]),
            reranker=build_reranker(self.config),
            resolver_factory=self.index.metadata_resolver,
        )
