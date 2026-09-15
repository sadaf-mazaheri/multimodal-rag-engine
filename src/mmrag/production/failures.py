"""Four-way failure decomposition over judged records.

A production view of the existing judged categories: every answerable query
lands in exactly one bucket, assigned in this order of precedence:

1. ``retrieval_failure``  -- the gold evidence was not retrieved (top-k);
2. ``context_failure``    -- retrieved, but dropped by the context budget;
3. ``generation_failure`` -- evidence in the prompt, answer not grounded-correct
   (wrong, incomplete, over-refusal or unsupported claims);
4. ``citation_failure``   -- grounded-correct, but with a citation problem
   (unresolved, missing, or not supporting the claim);
5. ``success``            -- grounded-correct with sound citations.

Queries whose generation or judging failed are ``error``, kept out of the four
buckets so an outage is never read as a quality failure.

Nothing is re-judged: every input is a field of an existing ``JudgedRecord``,
and each bucket keeps its query ids and original ``classify()`` category so it
traces back to the quality report. Two non-exclusive counts sit beside the
buckets, so precedence hides nothing: citation problems on any answer, and
retrieval failures that were nonetheless answered grounded-correct (usually a
gold-set gap, where the answer was found on a page the gold set does not list).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mmrag.evaluation.judged_eval import JudgedRecord
from mmrag.production.stats import rate

BUCKETS = ("retrieval_failure", "context_failure", "generation_failure",
           "citation_failure", "success")
UNANSWERABLE_BUCKETS = ("correct_refusal", "answered")


def bucket_of(record: JudgedRecord) -> str:
    """The single bucket for one judged record (answerable or not)."""
    gen = record.generation
    scores = record.scores
    if gen.status != "ok" or scores is None or record.judge.status != "ok":
        return "error"
    if not record.answerable:
        return "correct_refusal" if record.category == "correct_refusal" else "answered"
    if not gen.evidence_retrieved:
        return "retrieval_failure"
    if not gen.evidence_in_context:
        return "context_failure"
    if not scores.grounded_correct:
        return "generation_failure"
    if scores.citation_problem:
        return "citation_failure"
    return "success"


def decompose(records: Sequence[JudgedRecord]) -> dict[str, Any]:
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]

    def grouped(items: Sequence[JudgedRecord], names: Sequence[str]) -> dict[str, Any]:
        n = len(items)
        out: dict[str, Any] = {}
        for name in (*names, "error"):
            members = [r for r in items if bucket_of(r) == name]
            out[name] = {
                **rate(len(members), n),
                "query_ids": [r.query_id for r in members],
                "categories": sorted({r.category for r in members}),
            }
        return out

    scored = [r for r in answerable if r.scores is not None and r.generation.status == "ok"]
    leak = [r for r in scored
            if not r.generation.evidence_retrieved and r.scores.grounded_correct]
    citation_any = [r for r in scored if r.scores.citation_problem]
    return {
        "precedence": list(BUCKETS),
        "answerable": grouped(answerable, BUCKETS),
        "unanswerable": grouped(unanswerable, UNANSWERABLE_BUCKETS),
        "citation_problem_any": {
            **rate(len(citation_any), len(scored)),
            "query_ids": [r.query_id for r in citation_any],
        },
        "retrieval_failure_but_grounded_correct": {
            **rate(len(leak), len(scored)),
            "query_ids": [r.query_id for r in leak],
            "note": "counted as retrieval_failure by precedence; usually a gold-set gap",
        },
    }
