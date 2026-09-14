"""The modality-aware index: one chunk set, one sub-index per modality.

Text chunks get BM25 and a dense collection; table chunks get BM25 over their
cells and a dense collection over their schema; figure chunks get BM25 over
their text and a CLIP collection over their crops. :meth:`ModalityIndex.retrievers`
opens the matching retrievers, in a fixed registration order.

This used to live inside the Method 2 pipeline class, which meant anything else
that wanted these indexes -- Method 3 -- had to inherit the whole pipeline to
get them. As a component it is shared by composition, and the pipeline that
routes, fuses and answers is :class:`mmrag.engine.RAGEngine`.

Nothing about the build changed in the move: same chunker settings, same
sub-index names, same Qdrant collections, same manifest.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmrag.config import INDEX_DIR, PROCESSED_DIR, ExperimentConfig
from mmrag.embeddings.image import ImageEmbedder
from mmrag.embeddings.text import TextEmbedder
from mmrag.ingestion.pipeline import read_sidecar, sidecar_path_for
from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import Retriever
from mmrag.retrieval.metadata import MetadataResolver
from mmrag.retrieval.modality_retrievers import (
    BM25Retriever,
    DenseRetriever,
    ImageRetriever,
    TableRetriever,
)
from mmrag.retrieval.views import (
    figure_text_view,
    has_figure_text,
    table_content_view,
    table_schema_view,
)
from mmrag.schemas import Chunk, ChunkType, Document, Element, Modality
from mmrag.stores.bm25 import BM25Index
from mmrag.stores.qdrant import QdrantStore
from mmrag.textify import Chunker

log = get_logger(__name__)

CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "index.json"

# Sub-index names, used for both on-disk BM25 directories and Qdrant collections.
TEXT_INDEX = "text"
TABLE_CONTENT_INDEX = "table_content"
TABLE_SCHEMA_INDEX = "table_schema"
FIGURE_TEXT_INDEX = "figure_text"
IMAGE_INDEX = "image"


@dataclass
class ModalityIndexReport:
    """What a modality-aware index build produced, per modality."""

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


class ModalityIndex:
    """A modality-aware chunk set and its per-modality sub-indexes."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        variant: str,
        index_base: Path | None = None,
        processed_dir: Path | None = None,
        text_embedder: TextEmbedder | None = None,
        image_embedder: ImageEmbedder | None = None,
        owner: str | None = None,
    ):
        self.config = config
        self.variant = variant
        self.index_dir = (index_base or INDEX_DIR) / variant
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.text_embedder = text_embedder or TextEmbedder(config.embedding)
        self.image_embedder = image_embedder or ImageEmbedder(config.embedding)
        # Recorded as "method" in the manifest; unchanged from when the build
        # lived in the Method 2 pipeline.
        self.owner = owner or variant
        self._chunks: dict[str, Chunk] | None = None
        self._documents: list[Document] = []

    def collection(self, name: str) -> str:
        """Qdrant collection for one sub-index, namespaced by variant."""
        return f"{self.variant}_{name}"

    @property
    def exists(self) -> bool:
        return (self.index_dir / MANIFEST_FILE).exists()

    # -- indexing ------------------------------------------------------------

    def build(self, doc_ids: list[str] | None = None) -> ModalityIndexReport:
        started = time.perf_counter()
        report = ModalityIndexReport(variant=self.variant)

        chunker = Chunker(
            self.config.chunking,
            variant=self.variant,
            embedding_model=self.config.embedding.text_model,
            # There is a CLIP index for these; without this they would never
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

        by_type = _partition(all_chunks)
        self._preflight(by_type)

        # The manifest is the "this index is usable" marker, so it is removed
        # before anything is rewritten and written again only once every
        # sub-index has been built and verified. An interrupted build therefore
        # leaves an index that reports itself missing, never one that is half
        # old and half new.
        (self.index_dir / MANIFEST_FILE).unlink(missing_ok=True)
        self._write_chunks(all_chunks)
        self._chunks = None

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

    def _bm25(self) -> BM25Index:
        return BM25Index(k1=self.config.retrieval.bm25_k1, b=self.config.retrieval.bm25_b)

    def _build_text_indexes(self, chunks: list[Chunk], report: ModalityIndexReport) -> None:
        if not chunks:
            log.warning("no text chunks to index")
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]

        bm25 = self._bm25()
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
        report: ModalityIndexReport,
    ) -> None:
        """Two indexes over the same table chunks: cells, and schema."""
        if not chunks:
            log.warning("no table chunks to index")
            return

        ids = [c.chunk_id for c in chunks]

        content = [table_content_view(c) for c in chunks]
        bm25 = self._bm25()
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
        report: ModalityIndexReport,
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
            bm25 = self._bm25()
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
        if doc_ids:
            paths = [sidecar_path_for(d) for d in doc_ids]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"not parsed yet: {[p.parent.name for p in missing]}; run 'mmrag ingest run'"
                )
            return paths
        found = sorted(self.processed_dir.glob("*/parsed.json"))
        if not found:
            raise FileNotFoundError("no parsed documents; run 'mmrag ingest run' first")
        return found

    def _write_chunks(self, chunks: list[Chunk]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_lines(
            self.index_dir / CHUNKS_FILE,
            (json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) for chunk in chunks),
        )

    def _write_manifest(self, report: ModalityIndexReport) -> None:
        _atomic_write_lines(
            self.index_dir / MANIFEST_FILE,
            [
                json.dumps(
                    {
                        "method": self.owner,
                        "config": self.config.model_dump(mode="json"),
                        "report": report.as_dict(),
                    },
                    indent=2,
                )
            ],
            trailing_newline=False,
        )

    # -- loading -------------------------------------------------------------

    @property
    def chunks(self) -> dict[str, Chunk]:
        if self._chunks is None:
            path = self.index_dir / CHUNKS_FILE
            if not path.exists():
                raise FileNotFoundError(
                    f"no index at {self.index_dir}; run 'mmrag index build --config "
                    f"{self.owner}'"
                )
            loaded: dict[str, Chunk] = {}
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    chunk = Chunk.model_validate_json(line)
                    loaded[chunk.chunk_id] = chunk
            self._chunks = loaded
        return self._chunks

    def retrievers(self) -> dict[str, Retriever]:
        """Open a retriever over every sub-index that exists.

        Registration order fixes fusion input order, so runs are reproducible.
        """
        chunks = self.chunks
        retrievers: dict[str, Retriever] = {}

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
        return retrievers

    def _load_bm25(self, name: str) -> BM25Index | None:
        directory = self.index_dir / name
        if not (directory / "meta.json").exists():
            return None
        return BM25Index.load(directory)

    def documents(self) -> list[Document]:
        """The corpus documents, from this build or else from the sidecars."""
        if self._documents:
            return self._documents
        documents: list[Document] = []
        for sidecar in sorted(self.processed_dir.glob("*/parsed.json")):
            try:
                documents.append(read_sidecar(sidecar).document)
            except Exception as exc:  # pragma: no cover - corrupt sidecar
                log.warning("could not read %s: %s", sidecar, exc)
        return documents

    def metadata_resolver(self) -> MetadataResolver:
        """Postgres-backed document resolution, falling back to the sidecars.

        The indexed chunks are handed over so the resolver can tell a
        document's name from one of its topic words. Titles alone cannot: with
        fourteen documents, "architecture" and "table" each appear in exactly
        one title and so looked like perfect identifiers, while appearing in
        nine and fourteen document bodies respectively.
        """
        return MetadataResolver.load(fallback=self.documents(), corpus=self.chunks.values())

    def describe(self) -> dict[str, Any]:
        """The build manifest, as written."""
        path = self.index_dir / MANIFEST_FILE
        if not path.exists():
            raise FileNotFoundError(f"no index at {self.index_dir}; run 'mmrag index build'")
        return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atomic_write_lines(path: Path, lines: Any, *, trailing_newline: bool = True) -> None:
    """Write text to a sibling temp file and move it into place.

    A reader -- or a crash -- can then only ever see the old file or the new
    one, never a truncated chunk set that still parses.
    """
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + ("\n" if trailing_newline else ""))
    os.replace(tmp, path)


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
