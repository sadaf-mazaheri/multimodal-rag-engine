"""The multimodal RAG engine: one pipeline, pluggable retrievers.

    query
      -> metadata resolution        (optional: "the IPCC report" -> doc_id filter)
      -> router                     (which modalities to fire)
      -> retrievers                 (any set implementing ``Retriever``)
      -> fusion                     (within modality, then weighted RRF across)
      -> modality-floored pool -> cross-encoder rerank
      -> answerer                   (generation.pipeline: v1 Answerer, or v2 evidence pack
                                     + one call + validation; citations either way)

The engine knows nothing about how its retrievers' indexes were built or where
they live. Index components (:mod:`mmrag.indexing`) open retrievers; the engine
composes whatever it is handed. Adding a retriever means implementing the
``Retriever`` protocol and registering it -- nothing downstream changes, because
fusion, pooling and reranking already operate on "some modalities, each with
some ranked lists".

The benchmark methods are configurations of this engine, not subclasses of each
other: Method 2 registers the text, table and figure retrievers; Method 3
registers the same four plus the visual page retriever and keeps it always on.
Method 1 predates the engine and keeps its own frozen single-index pipeline; see
``methods/method1_textified.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mmrag.config import ExperimentConfig
from mmrag.generation.pipeline import build_answerer
from mmrag.generation.providers.base import LLMProvider
from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import Retriever
from mmrag.retrieval.metadata import MetadataResolver
from mmrag.retrieval.modality import ModalityAwareRetriever, ModalityRetrievalResult
from mmrag.retrieval.router import HeuristicRouter
from mmrag.schemas import Answer, Chunk, Modality

log = get_logger(__name__)

ResolverFactory = Callable[[], MetadataResolver]


def build_reranker(config: ExperimentConfig) -> Any | None:
    """The configured cross-encoder, or None when reranking is off.

    The model itself loads on first use, so constructing this is cheap.
    """
    if not config.retrieval.rerank_enabled:
        return None
    from mmrag.retrieval.rerank import CrossEncoderReranker

    return CrossEncoderReranker(config.retrieval, device=config.embedding.device)


class RAGEngine:
    """Retrieve and answer over a set of registered retrievers."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        name: str,
        retrievers: dict[str, Retriever],
        chunks: dict[str, Chunk],
        router: HeuristicRouter | None = None,
        reranker: Any | None = None,
        resolver_factory: ResolverFactory | None = None,
    ):
        if not retrievers:
            raise ValueError("an engine needs at least one retriever")
        self.config = config
        self.name = name
        self._resolver_factory = resolver_factory
        self._resolver: MetadataResolver | None = None
        self.retriever = ModalityAwareRetriever(
            config.retrieval,
            router=router or HeuristicRouter(config.router),
            retrievers=retrievers,
            chunks=chunks,
            reranker=reranker,
        )

    # -- composition ---------------------------------------------------------

    @property
    def retrievers(self) -> dict[str, Retriever]:
        return self.retriever.retrievers

    @property
    def chunks(self) -> dict[str, Chunk]:
        return self.retriever.chunks

    @property
    def router(self) -> HeuristicRouter:
        return self.retriever.router

    @property
    def reranker(self) -> Any | None:
        return self.retriever.reranker

    @property
    def metadata_resolver(self) -> MetadataResolver | None:
        """Built on first use: it may reach Postgres."""
        if self._resolver is None and self._resolver_factory is not None:
            self._resolver = self._resolver_factory()
        return self._resolver

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "retrievers": {n: r.modality.value for n, r in self.retrievers.items()},
            "always_on": [m.value for m in self.router.always],
            "reranker": self.config.retrieval.rerank_model if self.reranker else None,
            "metadata_resolution": self._resolver_factory is not None,
            "n_chunks": len(self.chunks),
        }

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
        """Resolve, route and retrieve.

        When ``doc_ids`` is not given, the metadata resolver gets a chance to
        infer one from the query -- "what does the IPCC report say" becomes a
        document filter rather than a lexical hope. A caller's ``doc_ids`` are
        never widened.
        """
        resolved = list(doc_ids) if doc_ids else None
        if resolved is None and use_metadata and self.metadata_resolver is not None:
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

        Every engine configuration answers through the answerer
        ``generation.pipeline`` selects, with the same prompt and provider for a
        given pipeline, so answer-quality differences between configurations are
        attributable to retrieval. No images are attached.
        """
        retrieval = self.retrieve(query, top_k=top_k, doc_ids=doc_ids)
        answerer = build_answerer(self.config.generation, provider)
        answer = answerer.answer(query, retrieval.results, method=self.name)

        answer.latency_ms.update(retrieval.latency_ms)
        answer.metadata["retrieval"] = retrieval.diagnostics
        if retrieval.routing is not None:
            answer.metadata["routing"] = retrieval.routing.as_dict()
        return answer
