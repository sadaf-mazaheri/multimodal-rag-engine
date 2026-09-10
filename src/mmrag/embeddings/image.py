"""CLIP image embeddings.

**Ownership: unique to Method 2.** Method 1 has no image path at all -- a figure
reaches it only as caption plus OCR plus description, which is exactly what makes
it the baseline.

This is the component that recovers what flattening loses. On this corpus, 147
of 400 figures produce *no text whatsoever*: no caption, no OCR, no description.
Method 1 cannot retrieve them under any query. All 147 have a cropped image on
disk, so CLIP can reach every one of them -- which is the concrete mechanism
behind whatever advantage Method 2 shows on figure questions.

CLIP puts images and text in one shared space, so a text query can be scored
directly against an image with no textual intermediary. It is also genuinely
weaker than a text retriever on text-shaped questions, which is why its fusion
weight is 0.7 rather than 1.0 in ``configs/method2.yaml``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mmrag.config import EmbeddingConfig
from mmrag.embeddings.text import resolve_device
from mmrag.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)

# CLIP's text encoder truncates hard at 77 tokens. Queries are short, but a
# pasted paragraph would be silently cut, so it is trimmed knowingly instead.
CLIP_TEXT_CHAR_LIMIT = 300


class ImageEmbedder:
    """CLIP encoder for figure crops and text queries in one shared space."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading %s on %s", self.config.image_model, self.device)
            self._model = SentenceTransformer(self.config.image_model, device=self.device)
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> float:
        """Load the model if needed; return how long it took, in ms."""
        if self.is_loaded:
            return 0.0
        started = time.perf_counter()
        _ = self.model
        return (time.perf_counter() - started) * 1000

    @property
    def dimension(self) -> int:
        """Embedding width, probed from the model rather than trusted from config.

        CLIP models in sentence-transformers report ``None`` here: the CLIPModel
        wrapper is a single module with no pooling layer to ask, unlike the
        Transformer+Pooling stack a text model uses. So when the model declines
        to say, encode a throwaway string and measure the vector it returns --
        which is the only answer that cannot disagree with reality.

        Cached, because the fallback costs a forward pass.
        """
        if self._dimension is None:
            reported = self.model.get_sentence_embedding_dimension()
            if reported is None:
                reported = int(self.embed_queries(["probe"]).shape[1])
                log.debug("probed %s embedding dimension: %d", self.config.image_model, reported)
            self._dimension = int(reported)
        return self._dimension

    # -- encoding ------------------------------------------------------------

    def embed_images(
        self, paths: list[Path], *, show_progress: bool = False
    ) -> tuple[np.ndarray, list[int]]:
        """Encode figure crops.

        Returns the vectors and the indices they correspond to. Unreadable
        images are skipped rather than fatal -- a single truncated PNG must not
        abort an index build over hundreds of figures -- but they are counted,
        because a figure that fails to encode is a figure Method 2 cannot see
        either, and that has to stay visible in the numbers.
        """
        from PIL import Image, UnidentifiedImageError

        images = []
        kept: list[int] = []
        for index, path in enumerate(paths):
            try:
                with Image.open(path) as handle:
                    images.append(handle.convert("RGB"))
                kept.append(index)
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                log.warning("skipping unreadable figure %s: %s", path, exc)

        if not images:
            return np.empty((0, self.dimension), dtype=np.float32), []

        vectors = self.model.encode(
            # CLIP models in sentence-transformers accept PIL images here; the
            # published type stub only describes the text path.
            images,  # type: ignore[arg-type]
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress and sys.stderr.isatty(),
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32), kept

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Encode text queries into the same space as the images.

        No bge-style instruction prefix here: CLIP was not trained with one, and
        adding it would push the query away from the image manifold.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        trimmed = [t[:CLIP_TEXT_CHAR_LIMIT] for t in texts]
        vectors = self.model.encode(
            trimmed,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.config.image_model,
            "dimension": self.dimension,
            "device": self.device,
            "normalized": self.config.normalize,
        }
