"""Generation gold data: a sidecar to the retrieval gold set.

Retrieval gold says *where* an answer lives. Judging a generated answer also
needs *what* the answer is, and a way to test the refusal behaviour the prompt
asks for. Both live here rather than in ``v1.yaml``, which stays byte-identical:
the published retrieval numbers were computed against it.

Two things, kept minimal:

* ``required_facts`` per existing query -- short atomic facts copied from the
  gold evidence pages. Atomic rather than one reference answer, so completeness
  can be scored fact by fact and a correct paraphrase is not penalised for its
  wording. They give the judge a reference that does not depend on outside
  knowledge.
* ``unanswerable`` queries -- questions the corpus cannot answer, where the
  correct behaviour is to refuse. Without them refusal is observable only on the
  handful of queries where retrieval happened to miss.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mmrag.evaluation.gold import GoldSet

_UNANSWERABLE_ID = re.compile(r"^u\d{3}$")


class FactSet(BaseModel):
    """What a correct answer to one existing gold query must state."""

    model_config = ConfigDict(extra="forbid")

    required_facts: list[str] = Field(min_length=1)
    # How the facts were verified against the evidence page, and NEEDS REVIEW
    # where that verification was not verbatim.
    note: str | None = None

    @field_validator("required_facts")
    @classmethod
    def _facts_are_distinct_and_non_empty(cls, facts: list[str]) -> list[str]:
        cleaned = [f.strip() for f in facts]
        if any(not f for f in cleaned):
            raise ValueError("a required fact is empty")
        if len({f.lower() for f in cleaned}) != len(cleaned):
            raise ValueError("required facts contain a duplicate")
        return cleaned


class UnanswerableQuery(BaseModel):
    """A question the corpus cannot answer; the correct response is a refusal."""

    model_config = ConfigDict(extra="forbid")

    id: str
    query: str = Field(min_length=1)
    # Required: absence cannot be proven across the corpus, so every entry must
    # record why it is unanswerable and how that was checked.
    note: str = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _UNANSWERABLE_ID.match(value):
            raise ValueError(f"unanswerable ids look like u001, got {value!r}")
        return value


class GenerationGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    description: str | None = None
    # The retrieval gold these facts annotate, so a mismatch is detectable.
    retrieval_gold: str
    queries: dict[str, FactSet] = Field(min_length=1)
    unanswerable: list[UnanswerableQuery] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unanswerable_ids_are_unique(self) -> GenerationGold:
        ids = [u.id for u in self.unanswerable]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate unanswerable ids: {duplicates}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> GenerationGold:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls.model_validate(payload)

    def facts_for(self, query_id: str) -> list[str]:
        entry = self.queries.get(query_id)
        return list(entry.required_facts) if entry else []

    def needs_review(self) -> list[str]:
        flagged = [qid for qid, e in self.queries.items() if e.note and "NEEDS REVIEW" in e.note]
        flagged += [u.id for u in self.unanswerable if "NEEDS REVIEW" in u.note]
        return flagged


class GenerationGoldProblem(BaseModel):
    kind: Literal["missing_facts", "unknown_query", "id_collision"]
    query_id: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.query_id}: {self.kind} -- {self.detail}"


def validate_against_retrieval_gold(
    generation: GenerationGold, retrieval: GoldSet
) -> list[GenerationGoldProblem]:
    """Every retrieval query has facts, and nothing refers to a query that isn't there."""
    known = {q.id for q in retrieval.queries}
    problems: list[GenerationGoldProblem] = []

    for query_id in sorted(known - set(generation.queries)):
        problems.append(GenerationGoldProblem(
            kind="missing_facts", query_id=query_id,
            detail="retrieval gold query has no required_facts entry",
        ))
    for query_id in sorted(set(generation.queries) - known):
        problems.append(GenerationGoldProblem(
            kind="unknown_query", query_id=query_id,
            detail="not a query in the retrieval gold set",
        ))
    for entry in generation.unanswerable:
        if entry.id in known:
            problems.append(GenerationGoldProblem(
                kind="id_collision", query_id=entry.id,
                detail="unanswerable id collides with a retrieval gold id",
            ))
    return problems


def describe(generation: GenerationGold) -> dict[str, Any]:
    return {
        "version": generation.version,
        "n_answerable": len(generation.queries),
        "n_unanswerable": len(generation.unanswerable),
        "n_required_facts": sum(len(e.required_facts) for e in generation.queries.values()),
        "needs_review": generation.needs_review(),
    }
