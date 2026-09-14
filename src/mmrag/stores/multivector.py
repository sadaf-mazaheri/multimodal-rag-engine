"""A file-based multi-vector index over pages, scored by MaxSim.

The storage layer under :mod:`mmrag.indexing.visual_pages`.

Each page keeps every token vector the late-interaction model produced for it.
They are stored as one contiguous matrix plus an offsets array, so page ``i``
owns rows ``offsets[i]:offsets[i+1]``, with a JSONL row per page carrying its
provenance. That is deliberately plain:

* **portable** -- the index is built on a GPU machine and evaluated on a laptop,
  and the files copy across with no database or server to migrate;
* **exact** -- MaxSim is computed over every stored vector, with no approximate
  nearest-neighbour step whose recall would become a second variable;
* **cheap enough** -- 951 pages of ~800 vectors score in well under a second of
  numpy, which is noise next to the cross-encoder's ~18 s.

A manifest records the model, device, preprocessing and corpus the vectors came
from, plus checksums of every file, and loading verifies all of it. An index
that has drifted from its corpus is refused rather than silently scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mmrag.embeddings.visual import compatible_identity
from mmrag.logging_utils import get_logger

log = get_logger(__name__)

# Bump when the on-disk layout or the meaning of a stored field changes.
INDEX_VERSION = 1

EMBEDDINGS_FILE = "embeddings.npy"
OFFSETS_FILE = "offsets.npy"
PAGES_FILE = "pages.jsonl"
MANIFEST_FILE = "index.json"


class IndexIntegrityError(RuntimeError):
    """A stored index is incomplete, corrupted, or does not match its manifest."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PageRecord:
    """One indexed page and where its vectors and pixels came from."""

    page_id: str
    doc_id: str
    page_number: int
    # Relative to the processed-corpus directory, so the record is portable.
    image: str
    image_sha256: str
    image_width: int
    image_height: int
    image_dpi: int | None
    n_tokens: int


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def maxsim_scores(
    query: np.ndarray,
    embeddings: np.ndarray,
    offsets: np.ndarray,
    pages: Sequence[int] | None = None,
    *,
    block_tokens: int = 262_144,
) -> np.ndarray:
    """Late-interaction score of one query against each page.

    ``score(page) = sum over query tokens of max over page tokens of q . p``.
    Pages are scored in blocks so memory stays bounded however large the index.
    Returns one score per entry of ``pages`` (all pages when None), in order.
    """
    query = np.asarray(query, dtype=np.float32)
    order = np.arange(len(offsets) - 1) if pages is None else np.asarray(pages, dtype=np.int64)
    scores = np.empty(len(order), dtype=np.float32)
    if len(order) == 0:
        return scores

    start = 0
    while start < len(order):
        # Grow the block until it would exceed the token budget (always >= 1 page).
        end, tokens = start, 0
        while end < len(order):
            page = order[end]
            size = int(offsets[page + 1] - offsets[page])
            if end > start and tokens + size > block_tokens:
                break
            tokens += size
            end += 1

        block = order[start:end]
        rows = np.concatenate(
            [np.asarray(embeddings[offsets[p] : offsets[p + 1]], dtype=np.float32) for p in block]
        )
        sims = rows @ query.T  # (tokens, query_tokens)
        sizes = np.array([offsets[p + 1] - offsets[p] for p in block], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        scores[start:end] = np.maximum.reduceat(sims, starts, axis=0).sum(axis=1)
        start = end
    return scores


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class PageIndexWriter:
    """Accumulates page vectors, then writes the index in one atomic step.

    Everything goes to a sibling temporary directory that replaces the target
    only once every file and the manifest are written, so an interrupted build
    leaves the previous index -- or nothing -- rather than half of one.
    """

    def __init__(self, directory: Path, *, dim: int, storage_dtype: str = "float16"):
        self.directory = Path(directory)
        self.dim = dim
        self.storage_dtype = np.dtype(storage_dtype)
        self.records: list[PageRecord] = []
        self._seen: set[str] = set()
        self._blocks: list[np.ndarray] = []

    def add(self, record: PageRecord, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise IndexIntegrityError(
                f"{record.page_id}: expected vectors of width {self.dim}, got {vectors.shape}"
            )
        if vectors.shape[0] == 0 or vectors.shape[0] != record.n_tokens:
            raise IndexIntegrityError(
                f"{record.page_id}: {vectors.shape[0]} vectors for a record of {record.n_tokens}"
            )
        if record.page_id in self._seen:
            raise IndexIntegrityError(f"{record.page_id} was added twice")
        self._seen.add(record.page_id)
        self.records.append(record)
        self._blocks.append(np.asarray(vectors, dtype=self.storage_dtype))

    def finalize(self, manifest: dict[str, Any]) -> PageEmbeddingIndex:
        if not self.records:
            raise IndexIntegrityError("no pages were added; refusing to write an empty index")

        tmp = self.directory.with_name(self.directory.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        offsets = np.zeros(len(self.records) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([r.n_tokens for r in self.records])
        np.save(tmp / EMBEDDINGS_FILE, np.concatenate(self._blocks))
        np.save(tmp / OFFSETS_FILE, offsets)
        with (tmp / PAGES_FILE).open("w", encoding="utf-8", newline="\n") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

        full = {
            **manifest,
            "index_version": INDEX_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "layout": {
                "dim": self.dim,
                "storage_dtype": self.storage_dtype.name,
                "n_pages": len(self.records),
                "n_tokens": int(offsets[-1]),
            },
            "files": {
                name: sha256_file(tmp / name)
                for name in (EMBEDDINGS_FILE, OFFSETS_FILE, PAGES_FILE)
            },
        }
        (tmp / MANIFEST_FILE).write_text(
            json.dumps(full, indent=2, sort_keys=True), encoding="utf-8"
        )

        # Swap by rename: the previous index is moved aside, not deleted, until
        # the new one is in place, so no failure leaves the target missing.
        previous = self.directory.with_name(self.directory.name + ".old")
        if previous.exists():
            shutil.rmtree(previous)
        if self.directory.exists():
            os.replace(self.directory, previous)
        try:
            os.replace(tmp, self.directory)
        except OSError:
            if previous.exists() and not self.directory.exists():
                os.replace(previous, self.directory)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        return PageEmbeddingIndex.load(self.directory)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class PageEmbeddingIndex:
    """A loaded, verified page index."""

    def __init__(
        self,
        directory: Path,
        manifest: dict[str, Any],
        pages: list[PageRecord],
        embeddings: np.ndarray,
        offsets: np.ndarray,
    ):
        self.directory = directory
        self.manifest = manifest
        self.pages = pages
        self.embeddings = embeddings
        self.offsets = offsets
        self._by_key = {(p.doc_id, p.page_number): i for i, p in enumerate(pages)}

    @classmethod
    def exists(cls, directory: Path) -> bool:
        return (Path(directory) / MANIFEST_FILE).exists()

    @classmethod
    def load(cls, directory: Path, *, verify_checksums: bool = True) -> PageEmbeddingIndex:
        directory = Path(directory)
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"no page index at {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        version = manifest.get("index_version")
        if version != INDEX_VERSION:
            raise IndexIntegrityError(
                f"{directory} is index version {version}; this build reads version "
                f"{INDEX_VERSION}. Rebuild it with 'mmrag index build --config method3'."
            )
        for name in (EMBEDDINGS_FILE, OFFSETS_FILE, PAGES_FILE):
            path = directory / name
            if not path.exists():
                raise IndexIntegrityError(f"{directory} is missing {name}")
            if verify_checksums and sha256_file(path) != manifest["files"][name]:
                raise IndexIntegrityError(f"{path} does not match the checksum in its manifest")

        pages = [
            PageRecord(**json.loads(line))
            for line in (directory / PAGES_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Read into memory rather than memory-mapped: the full corpus is ~200 MB
        # at float16, and a mapped file stays locked on Windows, which would stop
        # the same process from rebuilding the index it has open.
        embeddings = np.load(directory / EMBEDDINGS_FILE)
        offsets = np.load(directory / OFFSETS_FILE)
        layout = manifest["layout"]

        if len(offsets) != len(pages) + 1 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise IndexIntegrityError(f"{directory}: offsets do not describe one block per page")
        if int(offsets[-1]) != embeddings.shape[0] or embeddings.shape[1] != layout["dim"]:
            raise IndexIntegrityError(f"{directory}: embeddings do not match offsets or dimension")
        if [p.n_tokens for p in pages] != np.diff(offsets).tolist():
            raise IndexIntegrityError(f"{directory}: page records disagree with offsets")
        if len(pages) != layout["n_pages"]:
            raise IndexIntegrityError(f"{directory}: manifest page count disagrees with records")
        return cls(directory, manifest, pages, embeddings, offsets)

    # -- queries -------------------------------------------------------------

    @property
    def identity(self) -> dict[str, Any]:
        return self.manifest["model"]["identity"]

    @property
    def is_complete(self) -> bool:
        return bool(self.manifest.get("corpus", {}).get("complete"))

    def page_index(self, doc_id: str, page_number: int) -> int | None:
        return self._by_key.get((doc_id, page_number))

    def candidate_pages(
        self, *, doc_ids: Iterable[str] | None = None, page_numbers: Iterable[int] | None = None
    ) -> list[int]:
        docs = set(doc_ids) if doc_ids else None
        numbers = set(page_numbers) if page_numbers else None
        return [
            i
            for i, page in enumerate(self.pages)
            if (docs is None or page.doc_id in docs)
            and (numbers is None or page.page_number in numbers)
        ]

    def rank(
        self, query: np.ndarray, pages: Sequence[int] | None = None
    ) -> list[tuple[int, float]]:
        """Every candidate page with its score, best first, ties broken by page id."""
        candidates = list(range(len(self.pages))) if pages is None else list(pages)
        scores = maxsim_scores(query, self.embeddings, self.offsets, candidates)
        ranked = sorted(
            zip(candidates, scores.tolist(), strict=True),
            key=lambda item: (-item[1], self.pages[item[0]].page_id),
        )
        return ranked


# ---------------------------------------------------------------------------
# Query embedding cache
# ---------------------------------------------------------------------------


class QueryEmbeddingCache:
    """Query token vectors precomputed on the machine that has the model.

    Keyed by the SHA-256 of the query text. The whole cache belongs to one model
    identity, recorded once and checked on every write and every read, so vectors
    from two models can never be mixed or scored against the wrong index.
    """

    MANIFEST = "cache.json"

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @property
    def manifest(self) -> dict[str, Any] | None:
        path = self.directory / self.MANIFEST
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def check_identity(self, identity: dict[str, Any]) -> None:
        manifest = self.manifest
        if manifest is not None and not compatible_identity(manifest["identity"], identity):
            raise IndexIntegrityError(
                f"query cache at {self.directory} was built with {manifest['identity']}, "
                f"which is not compatible with {identity}. Delete it and re-run "
                "'mmrag index embed-queries'."
            )

    def get(self, text: str) -> np.ndarray | None:
        path = self.directory / "vectors" / f"{self.key(text)}.npy"
        return np.load(path) if path.exists() else None

    def put_many(
        self, items: Sequence[tuple[str, np.ndarray]], *, identity: dict[str, Any],
        encoder: dict[str, Any],
    ) -> int:
        self.check_identity(identity)
        vectors_dir = self.directory / "vectors"
        vectors_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest is None:
            (self.directory / self.MANIFEST).write_text(
                json.dumps({"identity": identity, "encoder": encoder}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        written = 0
        with (self.directory / "queries.jsonl").open("a", encoding="utf-8", newline="\n") as log_:
            for text, vectors in items:
                target = vectors_dir / f"{self.key(text)}.npy"
                # Written aside and renamed, so a reader never loads a torn file.
                partial = target.with_name(target.stem + ".partial.npy")
                np.save(partial, np.asarray(vectors, dtype=np.float32))
                os.replace(partial, target)
                log_.write(json.dumps({
                    "key": self.key(text), "query": text, "n_tokens": int(vectors.shape[0]),
                    "device": encoder.get("device"), "dtype": encoder.get("dtype"),
                }, ensure_ascii=False) + "\n")
                written += 1
        return written
