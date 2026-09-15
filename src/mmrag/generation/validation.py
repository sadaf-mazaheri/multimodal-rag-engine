"""Deterministic claim and citation checks on a generated answer.

No model is called and the answer is never changed: the validator reports, and
what to do about a report is the caller's decision. It checks four things:

* **sentence citation coverage** -- every sentence that states something carries
  at least one ``[n]`` marker. Sentences that only say what the sources do not
  cover are not factual claims and are not required to cite;
* **unresolved citations** -- markers that point at no supplied source;
* **identifier grounding** -- every number or digit-bearing identifier in a
  sentence (``2.9%``, ``22,360``, ``SPM.6``, ``A100``) appears in at least one of
  the sources that sentence cites, or, for an uncited sentence, in any source.
  A number the model wrote that no cited source contains is the cheapest
  hallucination to catch exactly. Uncited refusals and statements about what the
  sources lack are skipped: the identifiers there name what is missing;
* **mixed refusal** -- the refusal marker alongside cited content.

These are checks, not verdicts. A paraphrased unit ("percent" for ``%``) can
fail grounding while being right, and a sentence can cite correctly while
misreading its source; the judge is still what scores answers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mmrag.generation.answerer import REFUSAL_MARKER, parse_citation_numbers
from mmrag.schemas import ScoredChunk
from mmrag.textify.tokens import split_sentences

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A fragment that is nothing but citations and punctuation, e.g. "[1][2]." left
# behind when a model puts its markers after the full stop.
_ONLY_CITATIONS = re.compile(r"^\s*(?:\[\d+(?:\s*,\s*\d+)*\]\s*)+[.;:,]?\s*$")
_LEADING_CITATIONS = re.compile(r"^\s*((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+[.;:,]?)")
# Numbers and identifiers that contain a digit: 2.9%, 22,360, SPM.6, Cortex-M0+.
_IDENTIFIER = re.compile(r"(?:[A-Za-z][A-Za-z.\-]*)?\d[\w.,+%/-]*")
NBSP = chr(0xA0)  # no-break space
_WORD = re.compile(r"[A-Za-z]{2,}")
# Statements about the evidence rather than about the world.
_ABOUT_EVIDENCE = re.compile(
    r"\b(?:sources?|context|documents?|provided (?:text|information))\b.*\b(?:do(?:es)? not|don't|"
    r"doesn't|not|no)\b|\b(?:not|never) (?:stated|specified|provided|mentioned|given|described|"
    r"included|covered)\b",
    re.IGNORECASE,
)


@dataclass
class SentenceCheck:
    index: int
    text: str
    factual: bool
    citations: list[int]
    unresolved: list[int]
    identifiers: list[str]
    ungrounded: list[str]


@dataclass
class ValidationReport:
    n_sentences: int = 0
    n_factual_sentences: int = 0
    n_cited_factual_sentences: int = 0
    sentence_citation_coverage: float | None = None
    uncited_sentences: list[int] = field(default_factory=list)
    unresolved_citations: list[int] = field(default_factory=list)
    identifiers_checked: int = 0
    ungrounded_identifiers: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False
    mixed_refusal: bool = False
    issues: list[str] = field(default_factory=list)
    passed: bool = True
    sentences: list[SentenceCheck] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerValidator:
    """Checks an answer against the numbered sources it was generated from."""

    def validate(self, answer: str, sources: Sequence[ScoredChunk]) -> ValidationReport:
        """``sources[n - 1]`` is source ``n``, exactly as given to the generator."""
        report = ValidationReport(refused=REFUSAL_MARKER in (answer or ""))
        texts = [_normalise(s.chunk.text) for s in sources]
        resolved_any = False

        for index, sentence in enumerate(answer_sentences(answer or "")):
            numbers = _citations_in(sentence)
            unresolved = [n for n in numbers if not 1 <= n <= len(sources)]
            valid = [n for n in numbers if 1 <= n <= len(sources)]
            resolved_any = resolved_any or bool(valid)
            factual = _is_factual(sentence)

            bare = _CITATION.sub(" ", sentence)
            # An uncited refusal or statement about the evidence ("the sources do not
            # mention the H100") names what is missing; its identifiers are not
            # claims, so they are not required to appear in a source. A sentence that
            # cites anything, or that is merely short, is still checked in full.
            meta = not numbers and _is_meta(sentence)
            identifiers = [] if meta else identifier_tokens(bare)
            pool = [texts[n - 1] for n in valid] if valid else texts
            ungrounded = [t for t in identifiers if not any(_grounded(t, text) for text in pool)]

            report.sentences.append(SentenceCheck(index, sentence, factual, numbers, unresolved,
                                                  identifiers, ungrounded))
            report.n_sentences += 1
            if factual:
                report.n_factual_sentences += 1
                if numbers:
                    report.n_cited_factual_sentences += 1
                else:
                    report.uncited_sentences.append(index)
            for n in unresolved:
                if n not in report.unresolved_citations:
                    report.unresolved_citations.append(n)
            report.identifiers_checked += len(identifiers)
            report.ungrounded_identifiers += [
                {"token": t, "sentence": index, "cited": valid} for t in ungrounded
            ]

        if report.n_factual_sentences:
            report.sentence_citation_coverage = round(
                report.n_cited_factual_sentences / report.n_factual_sentences, 4
            )
        report.mixed_refusal = report.refused and resolved_any

        if report.uncited_sentences and not report.refused:
            report.issues.append("uncited_sentence")
        if report.unresolved_citations:
            report.issues.append("unresolved_citation")
        if report.ungrounded_identifiers:
            report.issues.append("ungrounded_identifier")
        if report.mixed_refusal:
            report.issues.append("mixed_refusal")
        report.passed = not report.issues
        return report


def answer_sentences(text: str) -> list[str]:
    """Sentences, with citations placed after a full stop given back to that sentence.

    "It rose. [3] Then it fell. [2]" splits as "It rose." / "[3] Then it fell." /
    "[2]"; the leading or standalone markers belong to the sentence before.
    """
    out: list[str] = []
    for piece in split_sentences(text):
        leading = _LEADING_CITATIONS.match(piece)
        if out and leading:
            out[-1] = f"{out[-1]} {leading.group(1).strip()}"
            piece = piece[leading.end():].strip()
            if not piece or _ONLY_CITATIONS.match(piece):
                continue
        out.append(piece)
    return out


def identifier_tokens(text: str) -> list[str]:
    """Digit-bearing tokens, trailing punctuation removed, in order of appearance."""
    tokens: list[str] = []
    for match in _IDENTIFIER.finditer(text):
        token = match.group(0).rstrip(".,;:")
        if any(ch.isdigit() for ch in token) and token not in tokens:
            tokens.append(token)
    return tokens


def _citations_in(sentence: str) -> list[int]:
    return parse_citation_numbers(sentence)


def _is_meta(sentence: str) -> bool:
    """A refusal, or a statement about what the sources do or do not contain."""
    bare = _CITATION.sub(" ", sentence)
    return REFUSAL_MARKER in bare or bool(_ABOUT_EVIDENCE.search(bare))


def _is_factual(sentence: str) -> bool:
    bare = _CITATION.sub(" ", sentence)
    if REFUSAL_MARKER in bare:
        return False
    if len(_WORD.findall(bare)) < 3:
        return False
    return not _ABOUT_EVIDENCE.search(bare)


def _normalise(text: str) -> str:
    return " ".join(text.lower().replace(NBSP, " ").split())


def _grounded(token: str, source: str) -> bool:
    """A token counts as present if it, or its comma-free form, is in the source."""
    t = token.lower()
    if t in source:
        return True
    plain = t.replace(",", "")
    return plain != t and plain in source.replace(",", "")
