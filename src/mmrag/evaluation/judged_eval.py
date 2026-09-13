"""Scores, error categories and aggregates computed from judge verdicts.

Every number here is arithmetic over a verdict's observations plus the
deterministic fields of the generation record. The judge never assigns a score
itself, so any score can be traced back to the specific claims and facts that
produced it.

The report's central decomposition, over answerable queries:

    R    = P(gold evidence in the prompt)                   retrieval
    G    = P(grounded_correct | gold evidence in the prompt) generation
    E2E  = P(grounded_correct)                              end to end
    leak = P(grounded_correct | gold evidence NOT in prompt)

A non-zero leak is informative rather than suspicious: it means the answer was
found on a page the gold set does not list (the gold-set review flagged several)
or came from outside the sources, which the faithfulness check distinguishes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, Field

from mmrag.evaluation.generation_eval import GenerationRecord
from mmrag.evaluation.generation_metrics import normalise, key_tokens, mean, rate
from mmrag.evaluation.judge import JudgeOutcome, JudgeVerdict, judge_record
from mmrag.evaluation.llm_cache import CachingProvider

Correctness = Literal["correct", "partial", "incorrect", "refused", "not_assessable"]

CATEGORIES_ANSWERABLE = (
    "fully_correct", "correct_non_gold_page", "citation_error", "ungrounded_correct",
    "wrong_or_incomplete", "over_refusal", "context_truncated", "retrieval_miss_refused",
    "retrieval_miss_answered", "hallucinated_answer", "error",
)
CATEGORIES_UNANSWERABLE = ("correct_refusal", "answered_unanswerable", "hallucinated_answer",
                           "error")


class Scores(BaseModel):
    """Everything computed from one verdict."""

    n_claims: int
    n_supported: int
    n_cited: int
    n_citation_supported: int
    faithfulness: float | None
    has_unsupported: bool
    uncited_claim_rate: float | None
    citation_support: float | None
    n_facts: int
    n_covered: int
    completeness: float | None
    correctness: Correctness
    correctness_score: float | None
    refused: bool
    refusal_without_marker: bool
    context_sufficient: bool
    grounded_correct: bool | None
    citation_problem: bool
    judge_lexical_agreement: dict[str, int] = Field(default_factory=dict)


def correctness_label(verdict: JudgeVerdict, n_facts: int, refused: bool) -> Correctness:
    """Derived, not asked for: a label follows mechanically from the observations."""
    if refused:
        return "refused"
    if n_facts == 0:
        return "not_assessable"
    if verdict.contradicts_reference:
        return "incorrect"
    covered = sum(f.covered for f in verdict.required_facts)
    if covered == n_facts:
        return "correct"
    return "partial" if covered else "incorrect"


def score(record: GenerationRecord, verdict: JudgeVerdict) -> Scores:
    claims = verdict.claims
    n_claims = len(claims)
    n_supported = sum(c.supported_by_sources for c in claims)
    cited = [c for c in claims if c.cited_sources]
    n_citation_supported = sum(1 for c in cited if c.citation_supports is True)
    n_facts = len(record.required_facts)
    n_covered = sum(f.covered for f in verdict.required_facts)

    # The marker is the contract the prompt sets; a refusal in prose that omits
    # it still is a refusal, and is counted as a compliance failure separately.
    refused = bool(record.refused) or verdict.declines_to_answer
    label = correctness_label(verdict, n_facts, refused)
    has_unsupported = n_supported < n_claims

    citation_problem = (
        bool(record.unresolved_citations)
        or (n_claims > 0 and not record.citations)
        or (bool(cited) and n_citation_supported < len(cited))
    )

    agreement = {"checkable": 0, "agree": 0}
    haystack = normalise(record.answer or "")
    for fact, fv in zip(record.required_facts, verdict.required_facts, strict=True):
        tokens = key_tokens(fact)
        if not tokens:
            continue
        agreement["checkable"] += 1
        lexical = all(t in haystack for t in tokens)
        agreement["agree"] += int(lexical == fv.covered)

    return Scores(
        n_claims=n_claims,
        n_supported=n_supported,
        n_cited=len(cited),
        n_citation_supported=n_citation_supported,
        faithfulness=n_supported / n_claims if n_claims else None,
        has_unsupported=has_unsupported,
        uncited_claim_rate=(n_claims - len(cited)) / n_claims if n_claims else None,
        citation_support=n_citation_supported / len(cited) if cited else None,
        n_facts=n_facts,
        n_covered=n_covered,
        completeness=n_covered / n_facts if n_facts else None,
        correctness=label,
        correctness_score={"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(label),
        refused=refused,
        refusal_without_marker=verdict.declines_to_answer and not record.refused,
        context_sufficient=verdict.context_sufficient,
        grounded_correct=(label == "correct" and not has_unsupported) if record.answerable else None,
        citation_problem=citation_problem,
        judge_lexical_agreement=agreement,
    )


def classify(record: GenerationRecord, scores: Scores | None) -> str:
    """One error category per query. Precedence is deliberate and documented."""
    if record.status != "ok" or scores is None:
        return "error"

    if not record.answerable:
        if scores.refused and not record.mixed_refusal:
            return "correct_refusal"
        return "hallucinated_answer" if scores.has_unsupported else "answered_unanswerable"

    in_context = bool(record.evidence_in_context)
    truncated = bool(record.evidence_retrieved) and not in_context

    if scores.refused:
        if in_context:
            return "over_refusal"
        return "context_truncated" if truncated else "retrieval_miss_refused"

    if scores.correctness == "correct":
        if scores.has_unsupported:
            return "ungrounded_correct"
        if scores.citation_problem:
            return "citation_error"
        if not in_context or record.gold_page_cited is False:
            # Right and grounded, but not on the page the gold set lists: usually
            # a gold-set gap rather than a model error.
            return "correct_non_gold_page"
        return "fully_correct"

    if not in_context:
        if truncated:
            return "context_truncated"
        return "hallucinated_answer" if scores.has_unsupported else "retrieval_miss_answered"
    return "wrong_or_incomplete"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class JudgedRecord(BaseModel):
    query_id: str
    answerable: bool
    stratum: str | None = None
    requires: str | None = None
    generation: GenerationRecord
    judge: JudgeOutcome
    scores: Scores | None = None
    category: str


class JudgedRun(BaseModel):
    kind: Literal["judged"] = "judged"
    schema_version: int = 1
    method: str
    label: str
    created_at: str
    elapsed_s: float
    generation_run: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    judge: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    cache: dict[str, int] = Field(default_factory=dict)
    totals: dict[str, Any] = Field(default_factory=dict)
    records: list[JudgedRecord] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    def save(self, path):
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path) -> JudgedRun:
        from pathlib import Path

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def run_judging(
    records: Sequence[GenerationRecord],
    provider: CachingProvider,
    *,
    model: str,
    max_output_tokens: int,
    dry_run: bool = False,
    concurrency: int = 4,
    attempts: int = 2,
    backoff_s: float = 2.0,
    on_record: Callable[[JudgedRecord], None] | None = None,
) -> list[JudgedRecord]:
    def one(record: GenerationRecord) -> JudgedRecord:
        outcome = judge_record(record, provider, model=model,
                               max_output_tokens=max_output_tokens, dry_run=dry_run,
                               attempts=attempts, backoff_s=backoff_s)
        scores = score(record, outcome.verdict) if outcome.verdict is not None else None
        judged = JudgedRecord(
            query_id=record.query_id, answerable=record.answerable, stratum=record.stratum,
            requires=record.requires, generation=record, judge=outcome, scores=scores,
            category="dry_run" if dry_run else classify(record, scores),
        )
        if on_record is not None:
            on_record(judged)
        return judged

    if dry_run or concurrency <= 1:
        return [one(r) for r in records]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, records))


def _subset_metrics(items: Sequence[JudgedRecord]) -> dict[str, Any]:
    scored = [r for r in items if r.scores is not None]
    return {
        "n": len(items),
        "n_judged": len(scored),
        "evidence_in_context": rate(r.generation.evidence_in_context for r in items),
        "grounded_correct": rate(r.scores.grounded_correct for r in scored),
        "correctness": mean(r.scores.correctness_score for r in scored),
        "completeness": mean(r.scores.completeness for r in scored),
    }


def aggregate_judged(items: Sequence[JudgedRecord]) -> dict[str, Any]:
    answerable = [r for r in items if r.answerable]
    unanswerable = [r for r in items if not r.answerable]
    judged = [r for r in answerable if r.scores is not None]
    answered = [r for r in judged if not r.scores.refused]

    in_ctx = [r for r in judged if r.generation.evidence_in_context]
    out_ctx = [r for r in judged if not r.generation.evidence_in_context]

    labels = Counter(r.scores.correctness for r in judged)
    refusal_matrix = Counter(
        ("sufficient" if r.scores.context_sufficient else "insufficient",
         "refused" if r.scores.refused else "answered")
        for r in judged
    )
    agree = sum(r.scores.judge_lexical_agreement.get("agree", 0) for r in judged)
    checkable = sum(r.scores.judge_lexical_agreement.get("checkable", 0) for r in judged)
    proxy_agreement = rate(
        bool(r.generation.evidence_in_context) == r.scores.context_sufficient for r in judged
    )

    def by(field: str) -> dict[str, Any]:
        groups: dict[str, list[JudgedRecord]] = {}
        for r in answerable:
            groups.setdefault(getattr(r, field) or "-", []).append(r)
        return {k: _subset_metrics(v) for k, v in sorted(groups.items())}

    u_judged = [r for r in unanswerable if r.scores is not None]
    return {
        "coverage": {
            "answerable": {"n": len(answerable), "judged": len(judged)},
            "unanswerable": {"n": len(unanswerable), "judged": len(u_judged)},
            "judge_errors": sum(1 for r in items if r.judge.status == "error"),
            "generation_errors": sum(1 for r in items if r.generation.status != "ok"),
            "repaired": sum(1 for r in items if r.judge.repaired),
        },
        "decomposition": {
            "R_evidence_in_context": rate(r.generation.evidence_in_context for r in judged),
            "G_grounded_correct_given_evidence": rate(r.scores.grounded_correct for r in in_ctx),
            "E2E_grounded_correct": rate(r.scores.grounded_correct for r in judged),
            "leak_grounded_correct_without_evidence": rate(
                r.scores.grounded_correct for r in out_ctx),
        },
        "answerable": {
            "correctness": mean(r.scores.correctness_score for r in judged),
            "correctness_labels": {k: labels.get(k, 0) for k in
                                   ("correct", "partial", "incorrect", "refused")},
            "completeness": mean(r.scores.completeness for r in judged),
            "faithfulness": mean(r.scores.faithfulness for r in answered),
            "hallucination": rate(r.scores.has_unsupported for r in answered),
            "citation_support": mean(r.scores.citation_support for r in answered),
            "uncited_claim_rate": mean(r.scores.uncited_claim_rate for r in answered),
            "context_sufficient": rate(r.scores.context_sufficient for r in judged),
            "refusal_matrix": {f"{a}/{b}": refusal_matrix.get((a, b), 0)
                               for a in ("sufficient", "insufficient")
                               for b in ("answered", "refused")},
            "refusal_without_marker": rate(r.scores.refusal_without_marker for r in judged),
            "evidence_proxy_agrees_with_judge": proxy_agreement,
            "judge_lexical_fact_agreement": {
                "agree": agree, "checkable": checkable,
                "rate": round(agree / checkable, 4) if checkable else None},
            "by_requires": by("requires"),
            "by_stratum": by("stratum"),
        },
        "unanswerable": {
            "refused": rate(r.scores.refused for r in u_judged),
            "hallucination": rate(r.scores.has_unsupported for r in u_judged
                                  if not r.scores.refused),
            "context_sufficient": rate(r.scores.context_sufficient for r in u_judged),
        },
        "categories": dict(sorted(Counter(r.category for r in items).items())),
    }
