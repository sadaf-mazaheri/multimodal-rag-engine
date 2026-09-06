"""Flattening every modality into text: the defining step of Method 1.

This module is where the baseline's central assumption lives -- that a table is
adequately represented by its Markdown, and a chart by its caption plus whatever
OCR and a VLM recovered from it. Methods 2 and 3 exist to measure what that
assumption costs, so the cost has to be *observable* rather than implicit.

Hence :class:`FlattenReport`. Every element that survives flattening with no
text at all is counted and attributed, which turns "Method 1 did worse on chart
questions" into "Method 1 could not see 147 of the 400 figures, because nothing
textual was ever extracted from them".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from mmrag.schemas import Document, Element, ElementType

# Element types that carry no retrievable content of their own. Headers and
# footers repeat on every page; indexing them would put the document title at
# the top of the results for any query that happens to share a word with it.
SKIPPED_TYPES = frozenset({ElementType.HEADER, ElementType.FOOTER})


@dataclass
class FlattenReport:
    """What flattening recovered, and what it silently dropped."""

    total: int = 0
    kept: int = 0
    skipped_boilerplate: int = 0
    skipped_child: int = 0
    empty_by_type: Counter[str] = field(default_factory=Counter)
    empty_element_ids: list[str] = field(default_factory=list)

    @property
    def empty(self) -> int:
        return sum(self.empty_by_type.values())

    @property
    def invisible_figures(self) -> int:
        """Figures Method 1 cannot retrieve at all, because they have no text.

        The headline number for the Method 1 vs Method 3 comparison.
        """
        return sum(
            count
            for element_type, count in self.empty_by_type.items()
            if element_type in {"figure", "chart", "diagram"}
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "skipped_boilerplate": self.skipped_boilerplate,
            "skipped_child": self.skipped_child,
            "empty": self.empty,
            "empty_by_type": dict(sorted(self.empty_by_type.items())),
            "invisible_figures": self.invisible_figures,
        }


# A real section heading is short. Anything longer is a mis-detected block --
# a legal notice set large, or the vertical arXiv stamp down the page margin --
# and putting it in front of every chunk on the page would bury the actual
# content under boilerplate the embedder then has to see past.
MAX_HEADER_PART_CHARS = 80


def _usable_heading(value: str | None) -> str | None:
    if not value:
        return None
    text = " ".join(value.split())
    return text if 0 < len(text) <= MAX_HEADER_PART_CHARS else None


def context_header(document: Document, element: Element) -> str:
    """A minimal breadcrumb: document title, then section, then subsection.

    This is the one place structural metadata is deliberately allowed into
    embedded text, and it is kept minimal on purpose. Two reasons it earns its
    place, where a wholesale metadata dump would not:

    * A chunk from the middle of a document is often unintelligible alone
      ("It rose to 4.2% in the third quarter" -- what did, in which report?).
      The breadcrumb restores the referent that the page layout supplied visually.
    * It is a handful of tokens of genuinely disambiguating content, not
      identifiers or provenance. Ids, bounding boxes, extraction methods and
      confidences stay strictly in the structured payload, where they can be
      filtered and ranked on.

    Controlled by ``chunking.prepend_context_header`` so the ablation can be run.
    """
    parts = [document.title]
    section = _usable_heading(element.section)
    subsection = _usable_heading(element.subsection)
    if section and section != document.title:
        parts.append(section)
    if subsection and subsection != section:
        parts.append(subsection)
    return " > ".join(p.strip() for p in parts if p and p.strip())


def is_redundant_child(element: Element, by_id: dict[str, Element]) -> bool:
    """Whether an element's text is already carried by its parent.

    A caption is stored twice by design: as its own element (so it has a bounding
    box and a place in reading order) and on the parent's ``caption`` field (so
    the parent is self-describing). Indexing both would put the same sentence in
    two chunks, double-counting it in every recall figure.

    Only captions are redundant. Text *inside* a figure -- axis labels, in-plot
    annotations -- is also parented, but it is not repeated anywhere else, so it
    must still be indexed.
    """
    if element.element_type is not ElementType.CAPTION or not element.parent_id:
        return False
    parent = by_id.get(element.parent_id)
    if parent is None:
        return False
    return bool(parent.caption and parent.caption.strip())


def flatten_elements(
    elements: list[Element],
    *,
    skip_boilerplate: bool = True,
) -> tuple[list[Element], FlattenReport]:
    """Select the elements Method 1 will index, and report what it lost.

    Returns the retained elements in their original order, plus a report. The
    elements themselves are unchanged: flattening to a string happens per chunk,
    via :meth:`Element.best_text`.
    """
    by_id = {e.element_id: e for e in elements}
    report = FlattenReport(total=len(elements))
    kept: list[Element] = []

    for element in elements:
        if skip_boilerplate and element.element_type in SKIPPED_TYPES:
            report.skipped_boilerplate += 1
            continue
        if is_redundant_child(element, by_id):
            report.skipped_child += 1
            continue

        if not element.best_text().strip():
            # Recorded rather than merely dropped: an element that exists in the
            # corpus but contributes nothing to this index is exactly the
            # measurement Method 1 is here to produce.
            report.empty_by_type[element.element_type.value] += 1
            report.empty_element_ids.append(element.element_id)
            continue

        report.kept += 1
        kept.append(element)

    return kept, report
