"""Shared retriever contract.

**Ownership: shared infrastructure.** Method 1's ``HybridRetriever`` predates
this protocol and calls BM25 and Qdrant inline; it is deliberately left alone so
that Method 1's behaviour is frozen while Method 2 is built. New retrievers
implement this protocol, and Method 2 composes them.

The small duplication between ``HybridRetriever``'s inline calls and
``BM25Retriever``/``DenseRetriever`` here is the deliberate price of that freeze:
a shared base class would have meant editing Method 1 and invalidating its
already-measured numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mmrag.schemas import Modality


@dataclass
class MetadataFilter:
    """Structural constraints applied *before* ranking.

    This is where the "metadata stays structured" rule pays off: these fields
    were kept out of the embedded text precisely so they could be filtered on
    rather than hoped for as a lexical match.
    """

    doc_ids: list[str] | None = None
    chunk_types: list[str] | None = None
    page_numbers: list[int] | None = None
    doc_types: list[str] | None = None
    domains: list[str] | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.doc_ids, self.chunk_types, self.page_numbers, self.doc_types, self.domains)
        )

    def merge(self, other: MetadataFilter) -> MetadataFilter:
        """Intersect two filters, narrowing rather than widening.

        A caller-supplied ``--doc-id`` must never be *broadened* by something the
        router inferred, so overlapping fields intersect.
        """

        def combine(a: list[Any] | None, b: list[Any] | None) -> list[Any] | None:
            if a is None:
                return b
            if b is None:
                return a
            return [v for v in a if v in set(b)]

        return MetadataFilter(
            doc_ids=combine(self.doc_ids, other.doc_ids),
            chunk_types=combine(self.chunk_types, other.chunk_types),
            page_numbers=combine(self.page_numbers, other.page_numbers),
            doc_types=combine(self.doc_types, other.doc_types),
            domains=combine(self.domains, other.domains),
        )

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class Hit:
    """One retrieved chunk id with the score and rank its retriever gave it."""

    chunk_id: str
    score: float
    rank: int


@dataclass
class RetrieverOutput:
    """A single retriever's ranked list, plus what it cost and how it behaved."""

    retriever: str
    modality: Modality
    hits: list[Hit] = field(default_factory=list)
    latency_ms: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [h.chunk_id for h in self.hits]

    def __len__(self) -> int:
        return len(self.hits)


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query into a ranked list of chunk ids."""

    name: str
    modality: Modality

    def retrieve(
        self, query: str, k: int, filters: MetadataFilter | None = None
    ) -> RetrieverOutput: ...
