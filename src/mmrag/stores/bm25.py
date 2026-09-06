"""BM25 lexical index.

Kept in-process (``bm25s``) rather than delegated to Postgres full-text search,
for one reason: k1 and b have to be settable from the experiment config. They are
retrieval parameters the benchmark varies, and Postgres does not expose them.
Postgres FTS remains available as an independent cross-check.

BM25 earns its place beside a dense retriever because the two fail differently.
Dense retrieval is strong on paraphrase and weak on rare literal tokens -- a
register name like ``GPIO_OE``, a figure label like ``Figure 12``, an exact
figure like ``$22,360``. Those are precisely the queries a document benchmark
asks, and they are BM25's strength.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mmrag.logging_utils import get_logger

log = get_logger(__name__)

# Split on anything that is not alphanumeric, keeping intra-word punctuation that
# carries meaning in technical documents: GPIO_OE, 3.5, v1.2, 22,360.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._,][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, preserving technical identifiers.

    Deliberately not stemmed by default. Stemming conflates ``encoding`` and
    ``encoder``, which is usually helpful for prose and actively harmful for the
    identifier-heavy queries BM25 is here to answer.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Hit:
    chunk_id: str
    score: float
    rank: int


class BM25Index:
    """Persistable BM25 index over chunk texts."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self._retriever: Any | None = None

    # -- build ---------------------------------------------------------------

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        import bm25s

        if len(chunk_ids) != len(texts):
            raise ValueError(f"{len(chunk_ids)} ids but {len(texts)} texts")
        if not chunk_ids:
            raise ValueError("cannot build a BM25 index over zero chunks")

        self.chunk_ids = list(chunk_ids)
        corpus = [tokenize(t) for t in texts]
        empty = sum(1 for tokens in corpus if not tokens)
        if empty:
            log.warning("%d of %d chunks tokenised to nothing", empty, len(corpus))

        retriever = bm25s.BM25(k1=self.k1, b=self.b)
        retriever.index(corpus, show_progress=False)
        self._retriever = retriever
        log.info("built BM25 index over %d chunks", len(chunk_ids))

    # -- search --------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[BM25Hit]:
        if self._retriever is None:
            raise RuntimeError("BM25 index is not built or loaded")
        tokens = tokenize(query)
        if not tokens:
            return []

        k = min(k, len(self.chunk_ids))
        indices, scores = self._retriever.retrieve([tokens], k=k, show_progress=False, n_threads=1)
        hits: list[BM25Hit] = []
        for rank, (index, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1):
            # bm25s pads with zero-score entries when fewer than k documents
            # match; those are not results and must not enter fusion.
            if score <= 0:
                continue
            hits.append(BM25Hit(chunk_id=self.chunk_ids[int(index)], score=float(score), rank=rank))
        return hits

    # -- persistence ---------------------------------------------------------

    def save(self, directory: Path) -> None:
        if self._retriever is None:
            raise RuntimeError("nothing to save: the index is not built")
        directory.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(directory / "bm25s"), corpus=None)
        (directory / "meta.json").write_text(
            json.dumps({"chunk_ids": self.chunk_ids, "k1": self.k1, "b": self.b}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        import bm25s

        meta_path = directory / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no BM25 index at {directory}; run 'mmrag index build'")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        index = cls(k1=meta["k1"], b=meta["b"])
        index.chunk_ids = list(meta["chunk_ids"])
        index._retriever = bm25s.BM25.load(str(directory / "bm25s"), load_corpus=False)
        return index

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def describe(self) -> dict[str, Any]:
        return {"backend": "bm25s", "k1": self.k1, "b": self.b, "n_chunks": len(self.chunk_ids)}


def normalize_scores(scores: list[float]) -> np.ndarray:
    """Min-max normalise raw scores to [0, 1].

    BM25 scores are unbounded and corpus-dependent, so they are not comparable
    with cosine similarities. Only used for display and for weighted score
    fusion; rank-based RRF needs no normalisation, which is precisely why it is
    the default.
    """
    array = np.asarray(scores, dtype=np.float32)
    if array.size == 0:
        return array
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-9:
        return np.ones_like(array)
    return (array - low) / (high - low)
