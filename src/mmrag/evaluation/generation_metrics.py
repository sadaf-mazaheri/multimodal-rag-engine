"""Deterministic answer metrics, and the aggregation helpers the reports share.

Everything here is computed without a model: refusal, citation resolution,
whether the gold evidence reached the prompt, and a lexical cross-check on
required facts. None of it is a headline quality metric on its own -- that is
the judge's job -- but all of it is free, exact and reproducible, and some of it
(evidence in context) is what connects retrieval quality to generation quality.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from mmrag.evaluation.gold import GoldQuery
from mmrag.generation.answerer import REFUSAL_MARKER
from mmrag.schemas import Chunk

# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------


def is_refusal(text: str | None) -> bool:
    return bool(text) and REFUSAL_MARKER in text


def is_mixed_refusal(text: str | None, n_resolved_citations: int) -> bool:
    """The refusal marker *and* cited content in one answer.

    A plain substring check counts this as a refusal, but the model has also
    asserted something -- so it is reported separately rather than silently
    folded into either bucket.
    """
    return is_refusal(text) and n_resolved_citations > 0


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def contains_gold_evidence(chunks: Iterable[Chunk], gold: GoldQuery | None) -> bool | None:
    """Whether any chunk satisfies a gold evidence entry. None when there is no gold."""
    if gold is None:
        return None
    return any(gold.match_index(chunk) is not None for chunk in chunks)


# ---------------------------------------------------------------------------
# Lexical fact check -- a cross-check on the judge, never a headline
# ---------------------------------------------------------------------------

# Tokens that must appear verbatim for a fact to be stated at all: numbers
# (2.9%, 22,360, 0.96) and identifiers containing digits (SPM.2, A100,
# Cortex-M0+). The optional letter prefix may contain dots and hyphens, or
# "SPM.2" would be reduced to a bare "2" that matches almost any answer.
_KEY_TOKEN = re.compile(r"(?:[A-Za-z][A-Za-z.\-]*)?\d[\w.,+%/-]*")


def normalise(text: str) -> str:
    return " ".join(text.lower().replace(" ", " ").split())


def key_tokens(fact: str) -> list[str]:
    tokens = []
    for match in _KEY_TOKEN.finditer(fact):
        token = match.group(0).strip(".,").lower()
        if any(ch.isdigit() for ch in token):
            tokens.append(token)
    return tokens


def fact_lexical_coverage(facts: Sequence[str], answer: str | None) -> dict[str, int]:
    """How many facts with checkable tokens have all of them in the answer.

    Facts with no numbers or identifiers are not lexically checkable and are
    excluded rather than counted as misses. The result is a lower bound on
    coverage and exists to measure agreement with the judge's own verdicts.
    """
    haystack = normalise(answer or "")
    checkable = found = 0
    for fact in facts:
        tokens = key_tokens(fact)
        if not tokens:
            continue
        checkable += 1
        if all(token in haystack for token in tokens):
            found += 1
    return {"checkable": checkable, "found": found}


# ---------------------------------------------------------------------------
# Aggregation: counts beside rates, always
# ---------------------------------------------------------------------------


def rate(values: Iterable[bool | None]) -> dict[str, Any]:
    """A proportion with its numerator and denominator. None values are excluded."""
    present = [bool(v) for v in values if v is not None]
    count = sum(present)
    return {
        "n": len(present),
        "count": count,
        "rate": round(count / len(present), 4) if present else None,
    }


def mean(values: Iterable[float | None]) -> dict[str, Any]:
    present = [float(v) for v in values if v is not None]
    return {
        "n": len(present),
        "mean": round(sum(present) / len(present), 4) if present else None,
    }


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(int(len(ordered) * fraction), len(ordered) - 1)], 1)
