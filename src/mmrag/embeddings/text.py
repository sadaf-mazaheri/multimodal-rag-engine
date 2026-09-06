"""Local dense text embeddings.

Open-source and local by design: retrieval quality must not depend on an API
key, or a change in a vendor's model would silently move every number in the
benchmark and there would be no way to tell that from a change in the pipeline.

The asymmetric prefix matters more than it looks. bge models are trained with an
instruction prefix on the *query* side only; omitting it costs several points of
recall, and applying it to passages as well costs a few more. Getting this
backwards is a common and near-invisible bug, so queries and passages go through
separate methods rather than one method with a flag.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mmrag.config import EmbeddingConfig
from mmrag.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device, honouring an explicit preference."""
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


class TextEmbedder:
    """Sentence-transformers encoder for chunks and queries.

    The model is loaded lazily so that importing this module -- which the CLI
    does for every command -- does not pull half a gigabyte of weights into
    memory just to print help text.
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading %s on %s", self.config.text_model, self.device)
            self._model = SentenceTransformer(self.config.text_model, device=self.device)
            if self.config.max_seq_length:
                self._model.max_seq_length = self.config.max_seq_length
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> float:
        """Load the model if needed; return how long that took, in ms.

        Callers time queries, and loading half a gigabyte of weights is a
        one-off startup cost, not part of answering a question. Folding it into
        the first query's latency would put a ~20s outlier into the benchmark's
        per-query timings and make the first query of every run look pathological.
        """
        if self.is_loaded:
            return 0.0
        started = time.perf_counter()
        _ = self.model
        return (time.perf_counter() - started) * 1000

    @property
    def dimension(self) -> int:
        """Embedding width, taken from the model rather than trusted from config.

        A config that disagrees with the loaded model would otherwise surface as
        a Qdrant dimension error thousands of vectors into an index build.
        """
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:  # pragma: no cover - not true of any supported model
            raise RuntimeError(f"{self.config.text_model} reports no embedding dimension")
        return int(dimension)

    def embed_passages(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        """Encode documents for indexing."""
        return self._encode(
            [f"{self.config.passage_prefix}{t}" for t in texts], show_progress=show_progress
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Encode queries, with the model's instruction prefix applied."""
        return self._encode([f"{self.config.query_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def _encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            # Normalising here means cosine similarity is a dot product, which
            # is what Qdrant is configured for.
            normalize_embeddings=self.config.normalize,
            # tqdm writes one line per batch when stderr is redirected, which
            # buries the actual log output of an index build in a wall of bars.
            show_progress_bar=show_progress and sys.stderr.isatty(),
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def describe(self) -> dict[str, Any]:
        """Identity of the encoder, recorded with every index for reproducibility."""
        return {
            "model": self.config.text_model,
            "dimension": self.dimension,
            "device": self.device,
            "normalized": self.config.normalize,
            "query_prefix": self.config.query_prefix,
            "max_seq_length": self.config.max_seq_length,
        }
