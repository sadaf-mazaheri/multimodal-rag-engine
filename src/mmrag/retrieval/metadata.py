"""Document-level metadata retrieval, backed by Postgres.

**Ownership: unique to Method 2.** Method 1 has no metadata path -- every
constraint it can express has to be a lexical accident in the flattened text.

This is the fourth leg of the modality-aware design, and the one that most
directly justifies keeping metadata *structured* rather than concatenated into
embedding text. "What does the IPCC report say about sea level?" contains two
different kinds of information: a topic, which belongs to a retriever, and a
document constraint, which belongs to a ``WHERE`` clause. Method 1 must hope
that "IPCC" happens to appear in the right chunks. Method 2 resolves it to a
``doc_id`` and narrows every other retriever to that document.

Falls back to the parsed sidecars when Postgres is unreachable, so Method 2
stays runnable on a machine with no database -- the same property the index
build has.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import MetadataFilter
from mmrag.schemas import Document

log = get_logger(__name__)

# Words too common to identify a document. "Report" matches nine of fourteen
# corpus entries, so treating it as a document constraint would narrow to
# almost nothing useful while looking like precision.
STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "to",
        "with",
        "report",
        "annual",
        "paper",
        "document",
        "study",
        "review",
        "summary",
        "what",
        "which",
        "how",
        "does",
        "did",
        "is",
        "are",
        "was",
        "were",
        "say",
        "says",
        "about",
        "according",
        "data",
        "results",
        "analysis",
    }
)

# A term must be at least this long to count as a document signal, so stray
# acronyms and fragments do not narrow the corpus by accident.
MIN_TERM_CHARS = 3

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


@dataclass
class DocumentFacet:
    """The searchable identity of one document."""

    doc_id: str
    title: str
    publisher: str | None = None
    doc_type: str | None = None
    domain: str | None = None

    def terms(self) -> set[str]:
        """Distinctive lowercase terms that could name this document."""
        blob = " ".join(filter(None, [self.doc_id.replace("_", " "), self.title, self.publisher]))
        return {
            w.lower()
            for w in _WORD_RE.findall(blob)
            if len(w) >= MIN_TERM_CHARS and w.lower() not in STOPWORDS
        }


class MetadataResolver:
    """Turns document mentions in a query into a ``doc_id`` filter."""

    def __init__(self, facets: list[DocumentFacet]):
        self.facets = facets
        self._terms = {f.doc_id: f.terms() for f in facets}

    # -- construction --------------------------------------------------------

    @classmethod
    def from_postgres(cls) -> MetadataResolver | None:
        """Load facets from Postgres. Returns None if it is unreachable."""
        try:
            from mmrag.stores.postgres import PostgresStore

            with PostgresStore() as store:
                documents = store.list_documents()
        except Exception as exc:
            log.info("metadata resolver: Postgres unavailable (%s)", exc)
            return None
        return cls([_facet(d) for d in documents])

    @classmethod
    def from_documents(cls, documents: list[Document]) -> MetadataResolver:
        return cls([_facet(d) for d in documents])

    @classmethod
    def load(cls, fallback: list[Document] | None = None) -> MetadataResolver:
        """Postgres if available, otherwise the documents already in hand."""
        resolver = cls.from_postgres()
        if resolver is not None and resolver.facets:
            return resolver
        return cls.from_documents(fallback or [])

    # -- resolution ----------------------------------------------------------

    def resolve(self, query: str, *, min_terms: int = 1) -> MetadataFilter:
        """Narrow to the documents a query explicitly names.

        Requires a distinctive term match. A query naming no document returns an
        empty filter, which leaves every retriever unrestricted -- narrowing on a
        weak guess would silently hide the answer, and there is no way to
        recover from that downstream.
        """
        words = {
            w.lower()
            for w in _WORD_RE.findall(query)
            if len(w) >= MIN_TERM_CHARS and w.lower() not in STOPWORDS
        }
        if not words:
            return MetadataFilter()

        matched = [
            facet.doc_id
            for facet in self.facets
            if len(words & self._terms[facet.doc_id]) >= min_terms
        ]

        # Matching most of the corpus is not a constraint, it is noise.
        if not matched or len(matched) > max(1, len(self.facets) // 2):
            return MetadataFilter()
        return MetadataFilter(doc_ids=sorted(matched))

    def describe(self) -> dict[str, Any]:
        return {"n_documents": len(self.facets)}


def _facet(document: Document) -> DocumentFacet:
    return DocumentFacet(
        doc_id=document.doc_id,
        title=document.title,
        publisher=document.source or document.organization,
        doc_type=document.doc_type.value,
        domain=document.domain,
    )
