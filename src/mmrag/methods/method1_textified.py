"""Method 1: Textified multimodal RAG.

Every modality is flattened into text -- tables to Markdown, figures to caption
plus OCR plus VLM description -- and then retrieved by a single hybrid BM25 +
dense pipeline. It is the simplest thing that could work, and that is the point:
it is the baseline against which Methods 2 and 3 are measured, so the *cost* of
the flattening is the number this method exists to produce.

The pipeline records what it lost. ``FlattenReport.invisible_figures`` counts
figures with no retrievable text at all -- content that is present in the corpus
and structurally unreachable for this architecture. That is the mechanism behind
any deficit Method 3 later makes up.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmrag.config import INDEX_DIR, ExperimentConfig
from mmrag.embeddings.text import TextEmbedder
from mmrag.generation.answerer import Answerer
from mmrag.generation.providers.base import LLMProvider
from mmrag.ingestion.pipeline import read_sidecar, sidecar_path_for
from mmrag.logging_utils import get_logger
from mmrag.retrieval.hybrid import HybridRetriever, RetrievalResult
from mmrag.schemas import Answer, Chunk
from mmrag.stores.bm25 import BM25Index
from mmrag.stores.qdrant import QdrantStore
from mmrag.textify import Chunker

log = get_logger(__name__)

METHOD_NAME = "method1"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "index.json"


def index_dir_for(variant: str, base: Path | None = None) -> Path:
    return (base or INDEX_DIR) / variant


def _duplicate_ids(chunks: list[Chunk]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            duplicates.append(chunk.chunk_id)
        seen.add(chunk.chunk_id)
    return duplicates


@dataclass
class IndexReport:
    """What an index build produced."""

    variant: str
    n_documents: int = 0
    n_chunks: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    invisible_figures: int = 0
    elapsed_s: float = 0.0
    per_document: list[dict[str, Any]] = field(default_factory=list)
    embedder: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "n_documents": self.n_documents,
            "n_chunks": self.n_chunks,
            "by_type": self.by_type,
            "invisible_figures": self.invisible_figures,
            "elapsed_s": round(self.elapsed_s, 1),
            "embedder": self.embedder,
            "per_document": self.per_document,
        }


class Method1Textified:
    """End-to-end textified RAG: chunk, index, retrieve, answer."""

    name = METHOD_NAME

    def __init__(self, config: ExperimentConfig, *, index_base: Path | None = None):
        self.config = config
        self.variant = config.chunk_variant
        self.index_dir = index_dir_for(self.variant, index_base)
        self.embedder = TextEmbedder(config.embedding)
        self._chunks: dict[str, Chunk] | None = None
        self._retriever: HybridRetriever | None = None

    # -- indexing ------------------------------------------------------------

    def build_index(self, doc_ids: list[str] | None = None) -> IndexReport:
        """Chunk the parsed corpus and build both indexes.

        Reads the JSON sidecars rather than Postgres so an index can be built on
        a machine with no database -- the same reason the sidecars exist for the
        Colab visual-indexing step.
        """
        started = time.perf_counter()
        report = IndexReport(variant=self.variant)

        chunker = Chunker(
            self.config.chunking,
            variant=self.variant,
            embedding_model=self.config.embedding.text_model,
        )

        all_chunks: list[Chunk] = []
        for sidecar in self._sidecars(doc_ids):
            parsed = read_sidecar(sidecar)
            chunks, chunk_report = chunker.chunk_document(parsed.document, parsed.elements)
            all_chunks.extend(chunks)

            report.n_documents += 1
            report.invisible_figures += chunk_report.flatten.invisible_figures
            for key, value in (chunk_report.by_type or {}).items():
                report.by_type[key] = report.by_type.get(key, 0) + value
            report.per_document.append({"doc_id": parsed.document.doc_id, **chunk_report.as_dict()})
            log.info(
                "%s: %d chunks (%d figures with no text)",
                parsed.document.doc_id,
                len(chunks),
                chunk_report.flatten.invisible_figures,
            )

        if not all_chunks:
            raise RuntimeError(
                "no chunks produced; run 'mmrag ingest run' before building an index"
            )

        report.n_chunks = len(all_chunks)

        # Chunk ids address points in the vector store, so a duplicate silently
        # overwrites a point and maps a future retrieval hit onto the wrong
        # evidence. Cheap to check, and it fails the build rather than producing
        # an index that is quietly wrong about its own provenance.
        duplicates = _duplicate_ids(all_chunks)
        if duplicates:
            raise RuntimeError(
                f"{len(duplicates)} duplicate chunk ids, e.g. {duplicates[:3]}. "
                "Chunk ids must be unique: a collision overwrites a vector and "
                "misattributes citations."
            )

        self._write_chunks(all_chunks)

        texts = [c.text for c in all_chunks]
        ids = [c.chunk_id for c in all_chunks]

        log.info("building BM25 index over %d chunks", len(ids))
        bm25 = BM25Index(k1=self.config.retrieval.bm25_k1, b=self.config.retrieval.bm25_b)
        bm25.build(ids, texts)
        bm25.save(self.index_dir / "bm25")

        log.info("embedding %d chunks on %s", len(texts), self.embedder.device)
        vectors = self.embedder.embed_passages(texts, show_progress=True)
        report.embedder = self.embedder.describe()

        qdrant = QdrantStore(self.variant)
        qdrant.recreate(self.embedder.dimension)
        qdrant.upsert(all_chunks, vectors)

        # The vector store must hold exactly what was indexed. A shortfall means
        # points were overwritten, which is how the duplicate-id bug first
        # showed itself: 2,818 chunks in, 2,814 points out.
        stored = qdrant.count()
        if stored != len(all_chunks):
            raise RuntimeError(
                f"indexed {len(all_chunks)} chunks but the vector store holds {stored}; "
                "points were overwritten and retrieval would misattribute evidence"
            )

        report.elapsed_s = time.perf_counter() - started
        self._write_manifest(report)
        return report

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
        """Persist chunks as JSONL beside the indexes.

        The retriever needs full chunk records to return, and keeping them here
        means retrieval works without Postgres -- important because the index and
        the chunks must always be from the same build, and a database that has
        moved on would silently mismatch.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        path = self.index_dir / CHUNKS_FILE
        with path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def _write_manifest(self, report: IndexReport) -> None:
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
                    f"no index at {self.index_dir}; run 'mmrag index build --config "
                    f"{self.config.name}'"
                )
            loaded: dict[str, Chunk] = {}
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    chunk = Chunk.model_validate_json(line)
                    loaded[chunk.chunk_id] = chunk
            self._chunks = loaded
        return self._chunks

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            reranker = None
            if self.config.retrieval.rerank_enabled:
                from mmrag.retrieval.rerank import CrossEncoderReranker

                reranker = CrossEncoderReranker(
                    self.config.retrieval, device=self.config.embedding.device
                )
            self._retriever = HybridRetriever(
                self.config.retrieval,
                bm25=BM25Index.load(self.index_dir / "bm25"),
                qdrant=QdrantStore(self.variant),
                embedder=self.embedder,
                chunks=self.chunks,
                reranker=reranker,
            )
        return self._retriever

    # -- querying ------------------------------------------------------------

    def retrieve(
        self, query: str, *, top_k: int | None = None, doc_ids: list[str] | None = None
    ) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k=top_k, doc_ids=doc_ids)

    def answer(
        self,
        query: str,
        provider: LLMProvider,
        *,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> Answer:
        """Retrieve, then generate a cited answer.

        Method 1 never attaches images: flattening every modality to text is
        precisely its defining constraint, and passing a page render here would
        quietly turn the baseline into Method 3.
        """
        retrieval = self.retrieve(query, top_k=top_k, doc_ids=doc_ids)
        answerer = Answerer(self.config.generation, provider)
        answer = answerer.answer(query, retrieval.results, method=self.name)

        answer.latency_ms.update(retrieval.latency_ms)
        answer.metadata["retrieval"] = retrieval.diagnostics
        return answer
