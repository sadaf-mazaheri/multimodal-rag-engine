"""Cross-encoder reranking.

A bi-encoder embeds query and passage independently, so it can never model
interactions between them. A cross-encoder reads the pair together and scores
it directly -- far more accurate, and far too slow to run over a whole corpus.
Hence the standard arrangement used here: cheap retrieval produces a candidate
pool, the cross-encoder reorders only that pool.

Every method enables it with the same model and the same ``rerank_top_n``
(``retrieval.rerank_enabled``). Method 1 originally did not, which made the
reranker a second difference between the methods: a head-to-head result mixed
the modality-aware effect with the reranker effect, and neither was
attributable. Holding it constant costs the baseline some of its "simplest
thing that could work" character and buys a comparison in which retrieval is
the only variable.

For Method 2 the cross-encoder does a second job. RRF ranks by position and
cannot abstain, so an irrelevant modality still contributes its best candidate
at full strength; the cross-encoder reads the query and passage together and
can score that candidate down. It is therefore the arbiter across modalities,
which is why Method 2 guarantees every fired modality a floor in the pool it
sees -- ``retrieval.rerank_pool_per_modality``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mmrag.config import RetrievalConfig
from mmrag.embeddings.text import resolve_device
from mmrag.logging_utils import get_logger
from mmrag.schemas import ScoredChunk

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

log = get_logger(__name__)

# Cross-encoders truncate long inputs, so a chunk is trimmed before scoring
# rather than being silently cut mid-table by the model's own tokenizer.
MAX_PASSAGE_CHARS = 2000


class CrossEncoderReranker:
    """Reorders a candidate pool with a cross-encoder."""

    def __init__(self, config: RetrievalConfig, *, device: str = "auto"):
        self.config = config
        self.device = resolve_device(device)
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading reranker %s on %s", self.config.rerank_model, self.device)
            self._model = CrossEncoder(self.config.rerank_model, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk], *, top_k: int) -> list[ScoredChunk]:
        """Score and reorder candidates, returning the best ``top_k``.

        The fused rank is preserved in ``component_ranks`` so the reranker's
        effect stays measurable: without it, a reranker that reorders nothing
        and one that fixes everything look identical downstream.
        """
        if not candidates:
            return []

        pool = candidates[: self.config.rerank_top_n]
        pairs = [(query, c.chunk.text[:MAX_PASSAGE_CHARS]) for c in pool]
        scores = self.model.predict(pairs, show_progress_bar=False)

        order = sorted(zip(pool, scores, strict=True), key=lambda pair: -float(pair[1]))
        reranked = []
        for rank, (candidate, score) in enumerate(order[:top_k], start=1):
            reranked.append(
                candidate.model_copy(
                    update={
                        "score": float(score),
                        "rank": rank,
                        "component_ranks": {
                            **candidate.component_ranks,
                            "fused": candidate.rank,
                        },
                    }
                )
            )
        return reranked
