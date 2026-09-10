"""Method 2: Modality-aware RAG.

Each modality keeps its native representation and gets a retriever suited to it,
a router picks which to fire, and their ranked lists are fused and reranked.

What this measures against Method 1 is the value of *not* flattening. The two
methods read the same parsed elements, chunk them with the same shared chunker,
and answer with the same prompt and provider. The only differences are how those
chunks are indexed and how they are searched -- which is what makes the
comparison attributable.

Component ownership
-------------------
**Shared with Method 1 (unchanged):** ingestion and the ``Element`` model, the
``Chunker``, ``BM25Index``, ``QdrantStore``, ``TextEmbedder``,
``reciprocal_rank_fusion``, ``CrossEncoderReranker``, ``Answerer`` and the
provider interface.

**Unique to Method 2:** the query router, the CLIP ``ImageEmbedder``, the
per-modality views, the four modality retrievers, ``ModalityAwareRetriever``,
the Postgres ``MetadataResolver``, and this pipeline.

Method 1 is not modified by any of it: Method 2 writes to its own chunk variant
and its own Qdrant collections, so both indexes coexist and either can be
rebuilt without disturbing the other.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmrag.config import INDEX_DIR, ExperimentConfig
from mmrag.embeddings.image import ImageEmbedder
from mmrag.embeddings.text import TextEmbedder
from mmrag.generation.answerer import Answerer
from mmrag.generation.providers.base import LLMProvider
from mmrag.ingestion.pipeline import read_sidecar, sidecar_path_for
from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import Retriever
from mmrag.retrieval.metadata import MetadataResolver
from mmrag.retrieval.modality import ModalityAwareRetriever, ModalityRetrievalResult
from mmrag.retrieval.modality_retrievers import (
    BM25Retriever,
    DenseRetriever,
    ImageRetriever,
    TableRetriever,
)
from mmrag.retrieval.router import HeuristicRouter
from mmrag.retrieval.views import (
    figure_text_view,
    has_figure_text,
    table_content_view,
    table_schema_view,
)
from mmrag.schemas import Answer, Chunk, ChunkType, Document, Element, Modality
from mmrag.stores.bm25 import BM25Index
from mmrag.stores.qdrant import QdrantStore
from mmrag.textify import Chunker

log = get_logger(__name__)

METHOD_NAME = "method2"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "index.json"

# Sub-index names, used for both on-disk BM25 directories and Qdrant collections.
TEXT_INDEX = "text"
TABLE_CONTENT_INDEX = "table_content"
TABLE_SCHEMA_INDEX = "table_schema"
FIGURE_TEXT_INDEX = "figure_text"
IMAGE_INDEX = "image"


@dataclass
class Method2IndexReport:
    """What a Method 2 index build produced, per modality."""

    variant: str
    n_documents: int = 0
    n_chunks: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    n_text_indexed: int = 0
    n_tables_indexed: int = 0
    n_figures_indexed: int = 0
    n_figure_images_embedded: int = 0
    n_figures_without_text: int = 0
    n_figures_without_image: int = 0
    elapsed_s: float = 0.0
    embedders: dict[str, Any] = field(default_factory=dict)
    per_document: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text_invisible_recoverable(self) -> int:
        """Figures with no text that nonetheless have an image vector.

        The headline Method 2 number: content Method 1 cannot retrieve under any
        query, which Method 2 can.
        """
        return self.n_figures_without_text - self.n_figures_without_image

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "n_documents": self.n_documents,
            "n_chunks": self.n_chunks,
            "by_type": self.by_type,
            "n_text_indexed": self.n_text_indexed,
            "n_tables_indexed": self.n_tables_indexed,
            "n_figures_indexed": self.n_figures_indexed,
            "n_figure_images_embedded": self.n_figure_images_embedded,
            "n_figures_without_text": self.n_figures_without_text,
            "n_figures_without_image": self.n_figures_without_image,
            "text_invisible_recoverable": self.text_invisible_recoverable,
            "elapsed_s": round(self.elapsed_s, 1),
            "embedders": self.embedders,
            "per_document": self.per_document,
        }


class Method2ModalityAware:
    """End-to-end modality-aware RAG: chunk, index per modality, route, answer."""

    name = METHOD_NAME

    def __init__(self, config: ExperimentConfig, *, index_base: Path | None = None):
        self.config = config
        self.variant = config.chunk_variant
        self.index_dir = (index_base or INDEX_DIR) / self.variant
        self.text_embedder = TextEmbedder(config.embedding)
        self.image_embedder = ImageEmbedder(config.embedding)
        self._chunks: dict[str, Chunk] | None = None
        self._retriever: ModalityAwareRetriever | None = None
        self._documents: list[Document] = []

    def collection(self, name: str) -> str:
        """Qdrant collection for one sub-index, namespaced by variant."""
        return f"{self.variant}_{name}"

    # -- indexing ------------------------------------------------------------

    def build_index(self, doc_ids: list[str] | None = None) -> Method2IndexReport:
        started = time.perf_counter()
        report = Method2IndexReport(variant=self.variant)

        chunker = Chunker(
            self.config.chunking,
            variant=self.variant,
            embedding_model=self.config.embedding.text_model,
            # Method 2 has a CLIP index for these; without this they would never
            # become chunks and the image retriever could not reach them at all.
            keep_textless_visuals=True,
        )

        all_chunks: list[Chunk] = []
        elements_by_id: dict[str, Element] = {}
        documents: list[Document] = []

        for sidecar in self._sidecars(doc_ids):
            parsed = read_sidecar(sidecar)
            chunks, chunk_report = chunker.chunk_document(parsed.document, parsed.elements)
            all_chunks.extend(chunks)
            documents.append(parsed.document)
            for element in parsed.elements:
                elements_by_id[element.element_id] = element

            report.n_documents += 1
            for key, value in (chunk_report.by_type or {}).items():
                report.by_type[key] = report.by_type.get(key, 0) + value
            report.per_document.append({"doc_id": parsed.document.doc_id, **chunk_report.as_dict()})

        if not all_chunks:
            raise RuntimeError("no chunks produced; run 'mmrag ingest run' first")

        duplicates = _duplicate_ids(all_chunks)
        if duplicates:
            raise RuntimeError(
                f"{len(duplicates)} duplicate chunk ids, e.g. {duplicates[:3]}. "
                "A collision overwrites a vector and misattributes citations."
            )

        report.n_chunks = len(all_chunks)
        self._documents = documents
        self._write_chunks(all_chunks)

        by_type = _partition(all_chunks)
        self._preflight(by_type)

        self._build_text_indexes(by_type[ChunkType.TEXT], report)
        self._build_table_indexes(by_type[ChunkType.TABLE], elements_by_id, report)
        self._build_figure_indexes(by_type[ChunkType.FIGURE], elements_by_id, report)

        report.embedders = {
            "text": self.text_embedder.describe(),
            "image": self.image_embedder.describe()
            if report.n_figure_images_embedded
            else {"model": self.config.embedding.image_model, "loaded": False},
        }
        report.elapsed_s = time.perf_counter() - started
        self._write_manifest(report)
        return report

    # -- preflight -----------------------------------------------------------

    def _preflight(self, by_type: dict[ChunkType, list[Chunk]]) -> None:
        """Prove every embedder this build needs is usable, before using it.

        A multi-modality build does expensive work in stages, so a problem with
        the *last* embedder is discovered only after the earlier stages have
        already run. That happened: CLIP does not report its embedding
        dimension, and the build failed on the image index twelve minutes after
        the text and table indexes were written -- leaving collections behind
        with no manifest.

        Resolving each dimension up front costs one model load and turns that
        into a failure in seconds. Only the embedders that will actually be
        used are touched, so a corpus with no figures does not pay to load CLIP.
        """
        checks: list[tuple[str, Any]] = [("text", self.text_embedder)]
        if by_type[ChunkType.FIGURE]:
            checks.append(("image", self.image_embedder))

        for label, embedder in checks:
            try:
                dimension = embedder.dimension
            except Exception as exc:
                raise RuntimeError(
                    f"{label} embedder is unusable before indexing began: {exc}. "
                    "Fix this rather than letting a later stage fail mid-build."
                ) from exc
            if dimension <= 0:
                raise RuntimeError(f"{label} embedder reports a non-positive dimension")
            log.info("preflight ok: %s embedder is %d-dimensional", label, dimension)

    # -- per-modality index construction -------------------------------------

    def _build_text_indexes(self, chunks: list[Chunk], report: Method2IndexReport) -> None:
        if not chunks:
            log.warning("no text chunks to index")
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]

        bm25 = BM25Index(k1=self.config.retrieval.bm25_k1, b=self.config.retrieval.bm25_b)
        bm25.build(ids, texts)
        bm25.save(self.index_dir / TEXT_INDEX)

        log.info("embedding %d text chunks", len(texts))
        vectors = self.text_embedder.embed_passages(texts, show_progress=True)
        store = QdrantStore(self.collection(TEXT_INDEX))
        store.recreate(self.text_embedder.dimension)
        store.upsert(chunks, vectors)
        _verify(store, len(chunks), TEXT_INDEX)
        report.n_text_indexed = len(chunks)

    def _build_table_indexes(
        self,
        chunks: list[Chunk],
        elements: dict[str, Element],
        report: Method2IndexReport,
    ) -> None:
        """Two indexes over the same table chunks: cells, and schema."""
        if not chunks:
            log.warning("no table chunks to index")
            return

        ids = [c.chunk_id for c in chunks]

        content = [table_content_view(c) for c in chunks]
        bm25 = BM25Index(k1=self.config.retrieval.bm25_k1, b=self.config.retrieval.bm25_b)
        bm25.build(ids, content)
        bm25.save(self.index_dir / TABLE_CONTENT_INDEX)

        schema = [table_schema_view(c, _source_element(c, elements)) for c in chunks]
        log.info("embedding %d table schema views", len(schema))
        vectors = self.text_embedder.embed_passages(schema, show_progress=True)
        store = QdrantStore(self.collection(TABLE_SCHEMA_INDEX))
        store.recreate(self.text_embedder.dimension)
        store.upsert(chunks, vectors)
        _verify(store, len(chunks), TABLE_SCHEMA_INDEX)
        report.n_tables_indexed = len(chunks)

    def _build_figure_indexes(
        self,
        chunks: list[Chunk],
        elements: dict[str, Element],
        report: Method2IndexReport,
    ) -> None:
        """A CLIP index over figure crops, plus BM25 over whatever text they have."""
        if not chunks:
            log.warning("no figure chunks to index")
            return

        report.n_figures_indexed = len(chunks)
        report.n_figures_without_text = sum(1 for c in chunks if not has_figure_text(c))

        # --- text side: only figures that actually carry text ---------------
        textual = [(c.chunk_id, figure_text_view(c)) for c in chunks if has_figure_text(c)]
        if textual:
            bm25 = BM25Index(k1=self.config.retrieval.bm25_k1, b=self.config.retrieval.bm25_b)
            bm25.build([i for i, _ in textual], [t for _, t in textual])
            bm25.save(self.index_dir / FIGURE_TEXT_INDEX)

        # --- image side: every figure with a crop on disk -------------------
        paths: list[Path] = []
        with_image: list[Chunk] = []
        for chunk in chunks:
            element = _source_element(chunk, elements)
            image_path = element.figure.image_path if element and element.figure else None
            if image_path and Path(image_path).exists():
                paths.append(Path(image_path))
                with_image.append(chunk)

        report.n_figures_without_image = len(chunks) - len(with_image)
        if not paths:
            log.warning("no figure images found on disk; the image retriever will be empty")
            return

        log.info("embedding %d figure images with CLIP", len(paths))
        vectors, kept = self.image_embedder.embed_images(paths, show_progress=True)
        embedded = [with_image[i] for i in kept]
        if not embedded:
            return

        store = QdrantStore(self.collection(IMAGE_INDEX))
        store.recreate(self.image_embedder.dimension)
        store.upsert(embedded, vectors)
        _verify(store, len(embedded), IMAGE_INDEX)
        report.n_figure_images_embedded = len(embedded)

    # -- persistence ---------------------------------------------------------

    def _sidecars(self, doc_ids: list[str] | None) -> list[Path]:
        from mmrag.config import PROCESSED_DIR

        if doc_ids:
            paths = [sidecar_path_for(d) for d in doc_ids]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"not parsed yet: {[p.parent.name for p in missing]}; run 'mmrag ingest run'"
                )
            return paths
        found = sorted(PROCESSED_DIR.glob("*/parsed.json"))
        if not found:
            raise FileNotFoundError("no parsed documents; run 'mmrag ingest run' first")
        return found

    def _write_chunks(self, chunks: list[Chunk]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with (self.index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def _write_manifest(self, report: Method2IndexReport) -> None:
        (self.index_dir / MANIFEST_FILE).write_text(
            json.dumps(
                {
                    "method": self.name,
                    "config": self.config.model_dump(mode="json"),
                    "report": report.as_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- loading -------------------------------------------------------------

    @property
    def chunks(self) -> dict[str, Chunk]:
        if self._chunks is None:
            path = self.index_dir / CHUNKS_FILE
            if not path.exists():
                raise FileNotFoundError(
                    f"no index at {self.index_dir}; run 'mmrag index build --config method2'"
                )
            loaded: dict[str, Chunk] = {}
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    chunk = Chunk.model_validate_json(line)
                    loaded[chunk.chunk_id] = chunk
            self._chunks = loaded
        return self._chunks

    @property
    def retriever(self) -> ModalityAwareRetriever:
        if self._retriever is None:
            self._retriever = self._build_retriever()
        return self._retriever

    def _build_retriever(self) -> ModalityAwareRetriever:
        chunks = self.chunks
        retrievers: dict[str, Retriever] = {}

        # Registration order fixes fusion input order, so runs are reproducible.
        text_bm25 = self._load_bm25(TEXT_INDEX)
        if text_bm25 is not None:
            retrievers["bm25"] = BM25Retriever(
                text_bm25, chunks, name="bm25", modality=Modality.TEXT
            )
        retrievers["dense"] = DenseRetriever(
            QdrantStore(self.collection(TEXT_INDEX)),
            self.text_embedder,
            name="dense",
            modality=Modality.TEXT,
        )

        table_content = self._load_bm25(TABLE_CONTENT_INDEX)
        if table_content is not None:
            retrievers["table"] = TableRetriever(
                content_index=table_content,
                schema_store=QdrantStore(self.collection(TABLE_SCHEMA_INDEX)),
                embedder=self.text_embedder,
                chunks=chunks,
            )

        retrievers["image"] = ImageRetriever(
            image_store=QdrantStore(self.collection(IMAGE_INDEX)),
            image_embedder=self.image_embedder,
            text_index=self._load_bm25(FIGURE_TEXT_INDEX),
            chunks=chunks,
        )

        reranker = None
        if self.config.retrieval.rerank_enabled:
            from mmrag.retrieval.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(
                self.config.retrieval, device=self.config.embedding.device
            )

        return ModalityAwareRetriever(
            self.config.retrieval,
            router=HeuristicRouter(self.config.router),
            retrievers=retrievers,
            chunks=chunks,
            reranker=reranker,
        )

    def _load_bm25(self, name: str) -> BM25Index | None:
        directory = self.index_dir / name
        if not (directory / "meta.json").exists():
            return None
        return BM25Index.load(directory)

    @property
    def metadata_resolver(self) -> MetadataResolver:
        """Postgres-backed document resolution, falling back to the sidecars."""
        if not hasattr(self, "_resolver"):
            self._resolver = MetadataResolver.load(
                fallback=self._documents or self._load_documents()
            )
        return self._resolver

    def _load_documents(self) -> list[Document]:
        from mmrag.config import PROCESSED_DIR

        documents: list[Document] = []
        for sidecar in sorted(PROCESSED_DIR.glob("*/parsed.json")):
            try:
                documents.append(read_sidecar(sidecar).document)
            except Exception as exc:  # pragma: no cover - corrupt sidecar
                log.warning("could not read %s: %s", sidecar, exc)
        return documents

    # -- querying ------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        force_modalities: list[Modality] | None = None,
        use_metadata: bool = True,
    ) -> ModalityRetrievalResult:
        """Route and retrieve.

        When ``doc_ids`` is not given, the Postgres metadata resolver gets a
        chance to infer one from the query -- "what does the IPCC report say"
        becomes a document filter rather than a lexical hope.
        """
        resolved = list(doc_ids) if doc_ids else None
        if resolved is None and use_metadata:
            inferred = self.metadata_resolver.resolve(query)
            if inferred.doc_ids:
                log.info("metadata resolver narrowed to %s", inferred.doc_ids)
                resolved = inferred.doc_ids

        return self.retriever.retrieve(
            query, top_k=top_k, doc_ids=resolved, force_modalities=force_modalities
        )

    def answer(
        self,
        query: str,
        provider: LLMProvider,
        *,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> Answer:
        """Retrieve, then generate a cited answer.

        Uses the same ``Answerer``, prompt and provider as Method 1, so any
        difference in answer quality is attributable to retrieval rather than to
        prompting. Like Method 1, no images are attached -- passing page renders
        to the model is Method 3's defining move, not Method 2's.
        """
        retrieval = self.retrieve(query, top_k=top_k, doc_ids=doc_ids)
        answerer = Answerer(self.config.generation, provider)
        answer = answerer.answer(query, retrieval.results, method=self.name)

        answer.latency_ms.update(retrieval.latency_ms)
        answer.metadata["retrieval"] = retrieval.diagnostics
        if retrieval.routing is not None:
            answer.metadata["routing"] = retrieval.routing.as_dict()
        return answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _partition(chunks: list[Chunk]) -> dict[ChunkType, list[Chunk]]:
    out: dict[ChunkType, list[Chunk]] = {t: [] for t in ChunkType}
    for chunk in chunks:
        out[chunk.chunk_type].append(chunk)
    return out


def _source_element(chunk: Chunk, elements: dict[str, Element]) -> Element | None:
    """The element a single-element chunk came from.

    Table and figure chunks always have exactly one source element, which is
    what makes the structural views buildable at all.
    """
    for element_id in chunk.element_ids:
        element = elements.get(element_id)
        if element is not None:
            return element
    return None


def _duplicate_ids(chunks: list[Chunk]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            duplicates.append(chunk.chunk_id)
        seen.add(chunk.chunk_id)
    return duplicates


def _verify(store: QdrantStore, expected: int, label: str) -> None:
    """A sub-index must hold exactly what was written to it."""
    stored = store.count()
    if stored != expected:
        raise RuntimeError(
            f"{label}: indexed {expected} chunks but the store holds {stored}; "
            "points were overwritten and retrieval would misattribute evidence"
        )
