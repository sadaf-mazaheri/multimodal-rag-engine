"""What a benchmark method is: a named configuration of indexes plus an engine.

Every method exposes the same surface -- ``name``, ``chunks``, ``build_index``,
``retrieve`` and ``answer`` -- which is all the CLI, the evaluation runner and
generation evaluation ever call. :class:`EngineMethod` supplies that surface for
methods built on :class:`mmrag.engine.RAGEngine`; a subclass only says which
indexes it owns and which retrievers the engine gets.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mmrag.config import ExperimentConfig
from mmrag.engine import RAGEngine
from mmrag.generation.providers.base import LLMProvider
from mmrag.retrieval.metadata import MetadataResolver
from mmrag.retrieval.modality import ModalityAwareRetriever, ModalityRetrievalResult
from mmrag.schemas import Answer, Chunk, Modality


@runtime_checkable
class RAGMethod(Protocol):
    """The contract every method satisfies, including Method 1."""

    name: str
    config: ExperimentConfig

    @property
    def chunks(self) -> dict[str, Chunk]: ...

    def build_index(self, doc_ids: list[str] | None = None) -> Any: ...

    def retrieve(self, query: str, *, top_k: int | None = None) -> Any: ...

    def answer(self, query: str, provider: LLMProvider, *, top_k: int | None = None) -> Answer: ...


class EngineMethod:
    """A method whose retrieval and generation are a ``RAGEngine``."""

    name: str
    config: ExperimentConfig

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._engine: RAGEngine | None = None

    def _build_engine(self) -> RAGEngine:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def engine(self) -> RAGEngine:
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def reset(self) -> None:
        """Drop the engine, so the next query reopens the indexes."""
        self._engine = None

    @property
    def chunks(self) -> dict[str, Chunk]:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def retriever(self) -> ModalityAwareRetriever:
        return self.engine.retriever

    @property
    def metadata_resolver(self) -> MetadataResolver | None:
        return self.engine.metadata_resolver

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
        force_modalities: list[Modality] | None = None,
        use_metadata: bool = True,
    ) -> ModalityRetrievalResult:
        return self.engine.retrieve(
            query, top_k=top_k, doc_ids=doc_ids, force_modalities=force_modalities,
            use_metadata=use_metadata,
        )

    def answer(
        self,
        query: str,
        provider: LLMProvider,
        *,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> Answer:
        return self.engine.answer(query, provider, top_k=top_k, doc_ids=doc_ids)
