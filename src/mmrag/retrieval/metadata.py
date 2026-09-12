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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from mmrag.logging_utils import get_logger
from mmrag.retrieval.base import MetadataFilter
from mmrag.schemas import Chunk, Document

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

# A title word appearing in at least this fraction of the corpus *body* text is
# a topic word, not a name, and is discarded as a document signal.
#
# Titles alone cannot tell the two apart. "Architecture" appears in exactly one
# of fourteen titles, which made it look maximally discriminative, while
# appearing in nine of fourteen documents' text. "Table" is worse: unique to the
# TAPAS title and present in all fourteen bodies, so every question containing
# the word "table" was narrowed onto the table-parsing paper.
#
# 0.5 sits in the middle of a wide flat optimum -- measured over the gold set,
# any cut from 3/14 to 9/14 removes every mis-narrowing, and 7/14 through 9/14
# additionally keeps every correct one. It also reads as a rule rather than a
# tuned constant: a term in half the corpus is not a name.
MAX_DOCUMENT_FREQUENCY = 0.5

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def corpus_terms(chunks: Iterable[Chunk]) -> dict[str, set[str]]:
    """Body vocabulary per document, for judging which title words are generic.

    Built from indexed chunks rather than the raw PDFs so it describes exactly
    the text retrieval can actually see.
    """
    out: dict[str, set[str]] = {}
    for chunk in chunks:
        out.setdefault(chunk.doc_id, set()).update(
            w.lower() for w in _WORD_RE.findall(chunk.text)
        )
    return out


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

    def __init__(
        self,
        facets: list[DocumentFacet],
        *,
        body_terms: dict[str, set[str]] | None = None,
        max_document_frequency: float = MAX_DOCUMENT_FREQUENCY,
    ):
        self.facets = facets
        terms = {f.doc_id: f.terms() for f in facets}
        self.generic_terms = _generic_terms(terms, body_terms, max_document_frequency)
        # A generic term is not evidence about *which* document is meant, so it
        # is removed from every facet rather than from the one that owns it.
        self._terms = {doc_id: t - self.generic_terms for doc_id, t in terms.items()}
        if self.generic_terms:
            log.debug(
                "metadata resolver: %d title words are too common to identify a "
                "document and were dropped: %s",
                len(self.generic_terms),
                sorted(self.generic_terms),
            )

    # -- construction --------------------------------------------------------

    @classmethod
    def from_postgres(cls, **kwargs: Any) -> MetadataResolver | None:
        """Load facets from Postgres. Returns None if it is unreachable."""
        try:
            from mmrag.stores.postgres import PostgresStore

            with PostgresStore() as store:
                documents = store.list_documents()
        except Exception as exc:
            log.info("metadata resolver: Postgres unavailable (%s)", exc)
            return None
        return cls([_facet(d) for d in documents], **kwargs)

    @classmethod
    def from_documents(cls, documents: list[Document], **kwargs: Any) -> MetadataResolver:
        return cls([_facet(d) for d in documents], **kwargs)

    @classmethod
    def load(
        cls,
        fallback: list[Document] | None = None,
        *,
        corpus: Iterable[Chunk] | None = None,
        max_document_frequency: float = MAX_DOCUMENT_FREQUENCY,
    ) -> MetadataResolver:
        """Postgres if available, otherwise the documents already in hand.

        ``corpus`` supplies the indexed chunks. Without it the resolver cannot
        tell a document's name from one of its topic words, and falls back to
        the ``STOPWORDS`` list alone -- which is a hand-maintained approximation
        of the same judgement and does not scale.
        """
        options: dict[str, Any] = {"max_document_frequency": max_document_frequency}
        if corpus is not None:
            options["body_terms"] = corpus_terms(corpus)

        resolver = cls.from_postgres(**options)
        if resolver is not None and resolver.facets:
            return resolver
        return cls.from_documents(fallback or [], **options)

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
        return {
            "n_documents": len(self.facets),
            "n_generic_terms_dropped": len(self.generic_terms),
            # Absent corpus statistics the resolver is running on titles alone
            # and will narrow on topic words; worth seeing in a run record.
            "corpus_statistics": bool(self.generic_terms),
        }


def _generic_terms(
    facet_terms: dict[str, set[str]],
    body_terms: dict[str, set[str]] | None,
    max_document_frequency: float,
) -> frozenset[str]:
    """Title words too widespread in the corpus text to name a document.

    Uniqueness among titles is not discriminativeness. With fourteen documents
    a word can appear in exactly one title -- looking like a perfect identifier
    -- while appearing in every document's body. Document frequency over the
    indexed text is the missing signal, and it is the only one that scales:
    the alternative is enumerating generic words by hand forever.
    """
    if not body_terms:
        return frozenset()

    n_documents = len(body_terms)
    if n_documents == 0:
        return frozenset()

    cutoff = max_document_frequency * n_documents
    candidates = {term for terms in facet_terms.values() for term in terms}
    return frozenset(
        term
        for term in candidates
        if sum(1 for vocabulary in body_terms.values() if term in vocabulary) >= cutoff
    )


def _facet(document: Document) -> DocumentFacet:
    return DocumentFacet(
        doc_id=document.doc_id,
        title=document.title,
        publisher=document.source or document.organization,
        doc_type=document.doc_type.value,
        domain=document.domain,
    )
