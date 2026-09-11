"""The gold query set: what the benchmark counts as a correct retrieval.

Hand-authored and versioned, like ``configs/corpus.yaml`` -- never rewritten by
tooling, so its comments and per-entry justifications survive.

Why evidence is ``(doc_id, page, modality)``
--------------------------------------------
Two other identifiers were considered and rejected.

``chunk_id`` is disqualifying: ids are namespaced by variant and hashed over
content, so Method 1 and Method 2 have *different* ids for the same evidence by
construction. A chunk-level gold set cannot compare them at all.

``element_id`` is stable under content changes but not structural ones. It
encodes ``element_type`` and a per-type-per-page ordinal, so when the table
degeneracy fix reclassified 83 elements from chart to table it moved both the
type and the ordinals of everything after them on those pages. This project has
re-ingested three times; an element-keyed gold set would have been silently
invalidated twice.

``(doc_id, page, modality)`` survives all of that, and is the granularity a
citation already resolves to. It is also precise enough: a
``(doc, page, modality)`` key selects 2.09 chunks on average, and 1.36 for
figures.

The cost, stated plainly: page-level evidence **slightly over-credits**. A
retrieved chunk from the right page and modality counts as a hit even if that
particular chunk is not the sentence containing the answer. For a hand-verified
set of this size that bias is acceptable, but it is a bias, and it applies
equally to both methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mmrag.schemas import Chunk

Modality = Literal["text", "table", "figure"]
Stratum = Literal["text", "table", "figure", "natural"]


class Evidence(BaseModel):
    """One page that carries part of the answer."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    page: int = Field(ge=1)
    # None means "any modality on this page", used where the answer genuinely
    # spans a figure and its surrounding prose.
    modality: Modality | None = None
    # Free text recording *why* this page is the answer. Present so a reviewer
    # can check an entry without opening the PDF, and so a later re-ingest that
    # shifts a page number leaves a trail.
    note: str | None = None

    def matches(self, chunk: Chunk) -> bool:
        if chunk.doc_id != self.doc_id or chunk.page_number != self.page:
            return False
        return self.modality is None or chunk.chunk_type.value == self.modality


class GoldQuery(BaseModel):
    """One benchmark question and the evidence that answers it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    # How the question is phrased. `natural` is the honest slice: the router
    # audit showed explicit modality words hand the router its answer, so a
    # number pooled across strata flatters it.
    stratum: Stratum
    # The modality the answer actually lives in, independent of phrasing. A
    # `natural` question about a table is stratum=natural, requires=table.
    requires: Modality
    evidence: list[Evidence] = Field(min_length=1)
    notes: str | None = None

    @model_validator(mode="after")
    def _evidence_is_consistent(self) -> GoldQuery:
        typed = [e.modality for e in self.evidence if e.modality is not None]
        if typed and self.requires not in typed:
            raise ValueError(
                f"{self.id}: requires={self.requires!r} but no evidence entry has that "
                f"modality (found {sorted(set(typed))}). Either the question is "
                "mislabelled or the evidence is."
            )
        return self

    def match_index(self, chunk: Chunk) -> int | None:
        """Index of the evidence entry this chunk satisfies, or None.

        An index rather than a bool, because several retrieved chunks routinely
        satisfy the *same* evidence page and the metrics must count distinct
        evidence covered, not relevant chunks.
        """
        for index, entry in enumerate(self.evidence):
            if entry.matches(chunk):
                return index
        return None

    def matched(self, chunks: list[Chunk]) -> list[int | None]:
        """Per-rank evidence indices, the input contract for ``metrics``."""
        return [self.match_index(c) for c in chunks]


class GoldSet(BaseModel):
    """A versioned set of benchmark questions."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    description: str | None = None
    # SHA-256 of configs/corpus.lock.yaml at authoring time. Page numbers are
    # only meaningful against the corpus they were read from, so a mismatch is
    # a warning that the evidence may have moved.
    corpus_lock_sha256: str | None = None
    queries: list[GoldQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> GoldSet:
        seen: set[str] = set()
        duplicates = sorted({q.id for q in self.queries if q.id in seen or seen.add(q.id)})
        if duplicates:
            raise ValueError(f"duplicate query ids: {duplicates}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> GoldSet:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls.model_validate(payload)

    def by_stratum(self, stratum: Stratum) -> list[GoldQuery]:
        return [q for q in self.queries if q.stratum == stratum]

    def by_requires(self, modality: Modality) -> list[GoldQuery]:
        return [q for q in self.queries if q.requires == modality]

    def counts(self) -> dict[str, dict[str, int]]:
        """Composition of the set, for the run record and for review."""
        strata: dict[str, int] = {}
        requires: dict[str, int] = {}
        for query in self.queries:
            strata[query.stratum] = strata.get(query.stratum, 0) + 1
            requires[query.requires] = requires.get(query.requires, 0) + 1
        return {"stratum": strata, "requires": requires}


# ---------------------------------------------------------------------------
# Validation against a built index
# ---------------------------------------------------------------------------


class GoldProblem(BaseModel):
    """One unresolvable evidence entry."""

    query_id: str
    kind: Literal["no_such_page", "no_such_modality", "unknown_document"]
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.query_id}: {self.kind} -- {self.detail}"


def validate_against_chunks(gold: GoldSet, chunks: list[Chunk]) -> list[GoldProblem]:
    """Check every evidence entry resolves to real indexed content.

    Run before any evaluation. A gold set invalidated by re-ingestion otherwise
    reports itself as a retrieval *failure* -- the method looks broken when in
    fact the labels moved, which is the most expensive kind of wrong number to
    chase.
    """
    documents: set[str] = set()
    pages: set[tuple[str, int]] = set()
    typed: set[tuple[str, int, str]] = set()
    for chunk in chunks:
        documents.add(chunk.doc_id)
        pages.add((chunk.doc_id, chunk.page_number))
        typed.add((chunk.doc_id, chunk.page_number, chunk.chunk_type.value))

    problems: list[GoldProblem] = []
    for query in gold.queries:
        for entry in query.evidence:
            if entry.doc_id not in documents:
                problems.append(
                    GoldProblem(
                        query_id=query.id,
                        kind="unknown_document",
                        detail=f"{entry.doc_id!r} is not in the index",
                    )
                )
                continue
            if (entry.doc_id, entry.page) not in pages:
                problems.append(
                    GoldProblem(
                        query_id=query.id,
                        kind="no_such_page",
                        detail=f"{entry.doc_id} p{entry.page} has no indexed chunks",
                    )
                )
                continue
            if entry.modality and (entry.doc_id, entry.page, entry.modality) not in typed:
                problems.append(
                    GoldProblem(
                        query_id=query.id,
                        kind="no_such_modality",
                        detail=(
                            f"{entry.doc_id} p{entry.page} has no {entry.modality} chunk; "
                            "the page exists but the evidence does not"
                        ),
                    )
                )
    return problems


def describe(gold: GoldSet) -> dict[str, Any]:
    """Compact summary for the run record."""
    return {
        "version": gold.version,
        "n_queries": len(gold.queries),
        "corpus_lock_sha256": gold.corpus_lock_sha256,
        **gold.counts(),
    }
