"""The visual page index: ColQwen2 embeddings of every rendered page.

A corpus-level component. It embeds the page renders ingestion produced, and it
depends on the corpus lock and the model -- not on any chunk set, router or
method. Anything that wants a page signal opens a retriever over it with
:meth:`VisualPageIndexer.retriever`, handing over the chunk set the pages should
expand into.

On disk::

    data/indexes/visual_pages/
      index/         embeddings, offsets, page records, manifest (atomic)
      query_cache/   precomputed query vectors, bound to one model identity

Where it runs
-------------
Encoding ~950 pages needs a GPU. Scoring does not: the index is a set of numpy
files, and query vectors can be precomputed on the GPU machine with
``mmrag index embed-queries``, so retrieval runs on a CPU machine.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from mmrag.config import CONFIG_DIR, INDEX_DIR, PROCESSED_DIR, ExperimentConfig
from mmrag.embeddings.visual import (
    ColQwen2Encoder,
    VisualEncoder,
    compatible_identity,
    resolve_visual_device,
)
from mmrag.ingestion.pipeline import read_sidecar
from mmrag.logging_utils import get_logger
from mmrag.retrieval.visual_page import VisualPageRetriever
from mmrag.schemas import Chunk, Page
from mmrag.stores.multivector import (
    IndexIntegrityError,
    PageEmbeddingIndex,
    PageIndexWriter,
    PageRecord,
    QueryEmbeddingCache,
    sha256_file,
)

log = get_logger(__name__)

VISUAL_INDEX_DIR = "visual_pages"
INDEX_SUBDIR = "index"
QUERY_CACHE_SUBDIR = "query_cache"
# Recorded in the manifest; a directory holding anything else is refused.
INDEX_KIND = "visual_page_index"

EncoderFactory = Callable[[str], VisualEncoder]


class CpuIndexingRefusedError(RuntimeError):
    """A page index build resolved to CPU without being explicitly allowed to."""


# ---------------------------------------------------------------------------
# Page preparation
# ---------------------------------------------------------------------------


@dataclass
class PreparedPage:
    record: PageRecord
    path: Path


def resolve_page_image(page: Page, processed_dir: Path) -> Path | None:
    """The page render on this machine.

    Sidecars record the absolute path ingestion wrote to, which on another
    machine -- a Linux GPU box reading sidecars written on Windows -- does not
    exist. Renders always live at ``<processed>/<doc_id>/pages/<file>``, so the
    file name is enough to find them wherever the corpus directory was copied.
    """
    if not page.image_path:
        return None
    recorded = Path(page.image_path)
    if recorded.exists():
        return recorded
    windows = "\\" in page.image_path
    name = (PureWindowsPath if windows else PurePosixPath)(page.image_path).name
    candidate = processed_dir / page.doc_id / "pages" / name
    return candidate if candidate.exists() else None


def _relative_image(path: Path, processed_dir: Path, page: Page) -> str:
    try:
        return path.resolve().relative_to(processed_dir.resolve()).as_posix()
    except ValueError:
        return f"{page.doc_id}/pages/{path.name}"


def prepare_pages(
    processed_dir: Path,
    *,
    doc_ids: Sequence[str] | None = None,
    max_pages: int | None = None,
) -> tuple[list[PreparedPage], dict[str, str], list[str]]:
    """Every ingested page, with its render, checksum and provenance.

    Returns the pages in (document, page) order, each document's source sha256,
    and a list of problems -- pages with no render -- which the caller must not
    ignore: a page that cannot be embedded is a page that cannot be retrieved.
    """
    sidecars = sorted(processed_dir.glob("*/parsed.json"))
    if doc_ids:
        wanted = set(doc_ids)
        missing = wanted - {p.parent.name for p in sidecars}
        if missing:
            raise FileNotFoundError(f"not parsed: {sorted(missing)}; run 'mmrag ingest run'")
        sidecars = [p for p in sidecars if p.parent.name in wanted]
    if not sidecars:
        raise FileNotFoundError(f"no parsed documents under {processed_dir}")

    pages: list[PreparedPage] = []
    documents: dict[str, str] = {}
    problems: list[str] = []
    for sidecar in sidecars:
        parsed = read_sidecar(sidecar)
        documents[parsed.document.doc_id] = parsed.document.sha256
        for page in sorted(parsed.pages, key=lambda p: p.page_number):
            path = resolve_page_image(page, processed_dir)
            if path is None:
                problems.append(f"{page.page_id}: no page render at {page.image_path}")
                continue
            data = path.read_bytes()
            pages.append(PreparedPage(
                record=PageRecord(
                    page_id=page.page_id, doc_id=page.doc_id, page_number=page.page_number,
                    image=_relative_image(path, processed_dir, page),
                    image_sha256=hashlib.sha256(data).hexdigest(),
                    image_width=page.image_width or 0, image_height=page.image_height or 0,
                    image_dpi=page.image_dpi, n_tokens=0,
                ),
                path=path,
            ))
    if max_pages is not None:
        pages = pages[:max_pages]
    return pages, documents, problems


def corpus_lock(lock_path: Path) -> tuple[str, dict[str, str]]:
    """The lockfile's checksum and the source sha256 it pins per document."""
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    return sha256_file(lock_path), {doc: e["sha256"] for doc, e in payload["entries"].items()}


def check_documents_against_lock(documents: dict[str, str], pinned: dict[str, str]) -> None:
    drifted = sorted(d for d, sha in documents.items() if pinned.get(d) != sha)
    if drifted:
        raise IndexIntegrityError(
            f"parsed documents do not match configs/corpus.lock.yaml: {drifted}. The page "
            "index would describe a different corpus from the one the lockfile pins."
        )


def pages_missing_from(index: PageEmbeddingIndex, pages: Iterable[tuple[str, int]]) -> list[str]:
    """Pages (as ``doc#pN``) that are not in the index."""
    return sorted(f"{doc}#p{page}" for doc, page in pages if index.page_index(doc, page) is None)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class VisualIndexReport:
    variant: str = VISUAL_INDEX_DIR
    n_documents: int = 0
    n_pages: int = 0
    n_tokens: int = 0
    tokens_per_page: dict[str, float] = field(default_factory=dict)
    pages_without_render: list[str] = field(default_factory=list)
    pages_without_chunks: int | None = None
    complete: bool = False
    device: str = ""
    dtype: str = ""
    index_dir: str = ""
    index_bytes: int = 0
    elapsed_s: float = 0.0
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chunks(self) -> int:
        return self.n_pages

    @property
    def by_type(self) -> dict[str, int]:
        return {"page": self.n_pages}

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant, "n_documents": self.n_documents, "n_pages": self.n_pages,
            "n_tokens": self.n_tokens, "tokens_per_page": self.tokens_per_page,
            "pages_without_render": self.pages_without_render,
            "pages_without_chunks": self.pages_without_chunks, "complete": self.complete,
            "device": self.device, "dtype": self.dtype, "index_dir": self.index_dir,
            "index_bytes": self.index_bytes, "elapsed_s": round(self.elapsed_s, 1),
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------


def _is_out_of_memory(exc: BaseException) -> bool:
    return "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()


class VisualPageIndexer:
    """Build, validate and open the ColQwen2 page index."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        index_base: Path | None = None,
        processed_dir: Path | None = None,
        lock_path: Path | None = None,
        encoder_factory: EncoderFactory | None = None,
    ):
        self.config = config
        self.root = (index_base or INDEX_DIR) / VISUAL_INDEX_DIR
        self.index_dir = self.root / INDEX_SUBDIR
        self.query_cache_dir = self.root / QUERY_CACHE_SUBDIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.lock_path = lock_path or CONFIG_DIR / "corpus.lock.yaml"
        self._encoder_factory = encoder_factory or self._colqwen2
        self._index: PageEmbeddingIndex | None = None

    def _colqwen2(self, device: str) -> VisualEncoder:
        return ColQwen2Encoder(
            self.config.embedding.visual_model, self.config.visual,
            expected_dim=self.config.embedding.visual_dim, device=device,
        )

    @property
    def exists(self) -> bool:
        return PageEmbeddingIndex.exists(self.index_dir)

    # -- building ------------------------------------------------------------

    def build(
        self,
        doc_ids: list[str] | None = None,
        *,
        device: str | None = None,
        allow_cpu: bool | None = None,
        batch_size: int | None = None,
        max_pages: int | None = None,
        chunks_path: Path | None = None,
    ) -> VisualIndexReport:
        """Embed every page render and write the page index atomically.

        ``chunks_path`` names the chunk set retrieval will expand pages into. It
        does not affect the embeddings; it is hashed into the manifest, and
        pages carrying no chunk are counted, so the build reports up front how
        much of the index can reach the generator.
        """
        started = time.perf_counter()
        resolved = resolve_visual_device(device or self.config.visual.device)
        if resolved == "cpu" and not (allow_cpu or self.config.visual.allow_cpu_indexing):
            raise CpuIndexingRefusedError(
                "the page index would be built on CPU. ColQwen2 takes seconds per page there, "
                "so the full corpus takes hours and ~9 GB of RAM. Run this on a GPU machine "
                "(--device cuda), or pass --allow-cpu for a small --doc-id/--max-pages test."
            )

        pages, documents, problems = prepare_pages(
            self.processed_dir, doc_ids=doc_ids, max_pages=max_pages
        )
        if problems:
            raise FileNotFoundError(
                f"{len(problems)} page(s) have no render, e.g. {problems[:3]}. Copy "
                "data/processed/ from the ingestion machine, or re-run 'mmrag ingest run'."
            )
        lock_sha, pinned = corpus_lock(self.lock_path)
        check_documents_against_lock(documents, pinned)

        encoder = self._encoder_factory(resolved)
        identity = encoder.identity()
        description = encoder.describe()
        writer = PageIndexWriter(
            self.index_dir, dim=self.config.embedding.visual_dim,
            storage_dtype=self.config.visual.storage_dtype,
        )
        self._embed_pages(encoder, pages, writer, batch_size or self.config.visual.batch_size)

        complete = doc_ids is None and max_pages is None
        chunk_pages, chunks_sha = _chunk_pages(chunks_path)
        indexed = {(p.doc_id, p.page_number) for p in writer.records}
        without_chunks = (
            None if chunk_pages is None else sum(1 for key in indexed if key not in chunk_pages)
        )

        index = writer.finalize({
            "kind": INDEX_KIND,
            "model": {"identity": identity, "description": description},
            "preprocessing": {
                "image_source": "ingestion page renders (data/processed/<doc>/pages)",
                "image_dpi": sorted({p.record.image_dpi for p in pages if p.record.image_dpi}),
                "color_mode": "RGB",
                "resize": "model processor defaults",
                "processor": description.get("processor"),
            },
            "corpus": {
                "lock_sha256": lock_sha,
                "documents": documents,
                "complete": complete,
                "subset": None if complete else {"doc_ids": doc_ids, "max_pages": max_pages},
            },
            "chunk_set": {
                "path": chunks_path.as_posix() if chunks_path else None,
                "chunks_sha256": chunks_sha,
                "indexed_pages_without_chunks": without_chunks,
            },
            "config": {
                "visual": self.config.visual.model_dump(mode="json"),
                "visual_model": self.config.embedding.visual_model,
                "visual_dim": self.config.embedding.visual_dim,
            },
        })
        self._index = None

        tokens = [p.n_tokens for p in index.pages]
        return VisualIndexReport(
            n_documents=len(documents), n_pages=len(index.pages), n_tokens=sum(tokens),
            tokens_per_page={"min": min(tokens), "median": statistics.median(tokens),
                             "max": max(tokens)},
            pages_without_chunks=without_chunks, complete=complete,
            device=description.get("device", resolved), dtype=str(description.get("dtype", "")),
            index_dir=str(self.index_dir),
            index_bytes=sum(f.stat().st_size for f in self.index_dir.iterdir()),
            elapsed_s=time.perf_counter() - started, model=identity,
        )

    def _embed_pages(
        self, encoder: VisualEncoder, pages: list[PreparedPage], writer: PageIndexWriter,
        batch_size: int,
    ) -> None:
        from PIL import Image

        position = 0
        while position < len(pages):
            batch = pages[position : position + batch_size]
            images = []
            for page in batch:
                with Image.open(page.path) as handle:
                    images.append(handle.convert("RGB"))
            try:
                vectors = encoder.encode_images(images)
            except Exception as exc:
                if _is_out_of_memory(exc) and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    log.warning("out of memory; retrying with batch size %d", batch_size)
                    _empty_cuda_cache()
                    continue
                raise
            if len(vectors) != len(batch):
                raise IndexIntegrityError(
                    f"encoder returned {len(vectors)} results for {len(batch)} pages"
                )
            for page, page_vectors in zip(batch, vectors, strict=True):
                writer.add(replace(page.record, n_tokens=int(page_vectors.shape[0])), page_vectors)
            position += len(batch)
            log.info("embedded %d/%d pages", position, len(pages))

    # -- query embeddings ----------------------------------------------------

    def embed_queries(
        self, queries: Sequence[str], *, device: str | None = None, batch_size: int | None = None
    ) -> dict[str, int]:
        """Encode queries once and cache them for CPU-only retrieval."""
        texts = list(dict.fromkeys(q for q in queries if q.strip()))
        encoder = self._encoder_factory(resolve_visual_device(device or self.config.visual.device))
        identity = encoder.identity()
        if self.exists and not compatible_identity(identity, self.index.identity):
            raise IndexIntegrityError(
                f"encoder {identity} is not the model that built {self.index_dir} "
                f"({self.index.identity})"
            )
        cache = QueryEmbeddingCache(self.query_cache_dir)
        cache.check_identity(identity)
        todo = [t for t in texts if cache.get(t) is None]
        size = batch_size or self.config.visual.query_batch_size
        description = encoder.describe()
        written = 0
        for start in range(0, len(todo), size):
            batch = todo[start : start + size]
            written += cache.put_many(
                list(zip(batch, encoder.encode_queries(batch), strict=True)),
                identity=identity, encoder=description,
            )
        return {"requested": len(texts), "already_cached": len(texts) - len(todo),
                "encoded": written}

    # -- loading -------------------------------------------------------------

    @property
    def index(self) -> PageEmbeddingIndex:
        """The page index, verified against its files, the config and the corpus."""
        if self._index is None:
            if not self.exists:
                raise FileNotFoundError(
                    f"no visual page index at {self.index_dir}. Build it on a GPU machine "
                    "with 'mmrag index build --config method3 --device cuda' and copy "
                    f"{self.root} here."
                )
            index = PageEmbeddingIndex.load(self.index_dir)
            self._validate(index)
            self._index = index
        return self._index

    def _validate(self, index: PageEmbeddingIndex) -> None:
        if index.manifest.get("kind") != INDEX_KIND:
            raise IndexIntegrityError(f"{index.directory} is not a visual page index")
        expected = (self.config.embedding.visual_model, self.config.embedding.visual_dim)
        if (index.identity.get("model"), index.identity.get("dim")) != expected:
            raise IndexIntegrityError(
                f"page index was built with {index.identity}, but the config selects "
                f"model={expected[0]} dim={expected[1]}"
            )
        revision = self.config.visual.model_revision
        if revision and index.identity.get("revision") not in (None, revision):
            raise IndexIntegrityError(
                f"page index revision {index.identity.get('revision')} != config {revision}"
            )
        _, pinned = corpus_lock(self.lock_path)
        check_documents_against_lock(index.manifest["corpus"]["documents"], pinned)
        if not index.is_complete:
            log.warning("page index at %s is a partial test build: %s",
                        index.directory, index.manifest["corpus"].get("subset"))

    def check_covers(self, chunks: dict[str, Chunk]) -> None:
        """A complete index must hold every page the chunk set can expand into."""
        index = self.index
        if not index.is_complete:
            return
        missing = pages_missing_from(index, {(c.doc_id, c.page_number) for c in chunks.values()})
        if missing:
            raise IndexIntegrityError(
                f"page index claims to be complete but lacks {len(missing)} page(s) that "
                f"carry chunks, e.g. {missing[:3]}"
            )

    def retriever(self, chunks: dict[str, Chunk]) -> VisualPageRetriever:
        """A page retriever that expands into ``chunks``.

        Uses precomputed query vectors when a cache exists, and loads the model
        only for a query the cache does not hold.
        """
        self.check_covers(chunks)
        cache = None
        if self.config.visual.use_query_cache and (
            self.query_cache_dir / QueryEmbeddingCache.MANIFEST
        ).exists():
            cache = QueryEmbeddingCache(self.query_cache_dir)

        def query_encoder() -> VisualEncoder:
            return self._encoder_factory(resolve_visual_device(self.config.visual.device))

        return VisualPageRetriever(self.index, chunks, cache=cache, encoder_factory=query_encoder)

    def describe(self) -> dict[str, Any]:
        index = self.index
        cache = QueryEmbeddingCache(self.query_cache_dir)
        vectors = self.query_cache_dir / "vectors"
        cached = sum(1 for _ in vectors.glob("*.npy")) if vectors.exists() else 0
        return {
            "directory": str(index.directory),
            "index_version": index.manifest["index_version"],
            "created_at": index.manifest["created_at"],
            "complete": index.is_complete,
            "layout": index.manifest["layout"],
            "model": index.manifest["model"]["identity"],
            "device": index.manifest["model"]["description"].get("device"),
            "dtype": index.manifest["model"]["description"].get("dtype"),
            "corpus_lock_sha256": index.manifest["corpus"]["lock_sha256"],
            "embeddings_sha256": index.manifest["files"]["embeddings.npy"],
            "cached_queries": cached,
            "query_cache_identity": (cache.manifest or {}).get("identity"),
        }


def _chunk_pages(path: Path | None) -> tuple[set[tuple[str, int]] | None, str | None]:
    if path is None or not path.exists():
        if path is not None:
            log.warning("no chunk set at %s; page coverage is not checked", path)
        return None, None
    pages: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunk = Chunk.model_validate_json(line)
                pages.add((chunk.doc_id, chunk.page_number))
    return pages, sha256_file(path)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
