"""Layout analysis: reading order, page furniture, and section tracking.

Three jobs, all of which exist to put *structure* onto a flat list of text
blocks so that later stages have something to filter and expand context by:

* :func:`assign_reading_order` -- column-aware ordering. A naive top-to-bottom
  sort interleaves the two columns of a research paper into nonsense, which
  silently destroys every chunk built from that page.
* :class:`HeaderFooterDetector` -- finds running page furniture by looking for
  text that *repeats across pages*. A single page cannot tell a header from a
  heading; the corpus-level view can.
* :class:`SectionTracker` -- carries the current section/subsection forward so
  a mid-chapter page still knows which chapter it belongs to.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from mmrag.ingestion.text_utils import collapse_whitespace, mask_digits
from mmrag.schemas import BBox

# ---------------------------------------------------------------------------
# Block abstraction
# ---------------------------------------------------------------------------


@dataclass
class LayoutBlock:
    """A parser-agnostic text block, with the typography needed for structure."""

    bbox: BBox
    text: str
    # Dominant font size in points; the primary heading signal.
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False
    block_index: int = 0
    # Filled in by assign_reading_order.
    column: int = 0
    reading_order: int = 0


# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------


def detect_columns(blocks: list[LayoutBlock], *, max_columns: int = 3) -> int:
    """Estimate how many text columns a page has.

    Works on the observation that in a genuinely multi-column layout, blocks
    cluster into disjoint horizontal bands with a gutter between them, and few
    blocks straddle the gutter. A single-column page has many blocks spanning
    the middle, so the gutter test fails and we return 1.
    """
    body = [b for b in blocks if b.bbox.area > 0]
    if len(body) < 4:
        return 1

    for n in range(max_columns, 1, -1):
        # Candidate gutters at the n-1 interior boundaries of an equal split.
        gutters = [i / n for i in range(1, n)]
        if all(_is_gutter(body, g) for g in gutters):
            return n
    return 1


def _is_gutter(blocks: list[LayoutBlock], x: float, *, tolerance: float = 0.02) -> bool:
    """Whether few enough blocks straddle the vertical line at ``x``."""
    straddling = sum(1 for b in blocks if b.bbox.x0 < x - tolerance < x + tolerance < b.bbox.x1)
    # Full-width titles and figures legitimately cross the gutter, so allow a
    # small fraction rather than requiring zero.
    return straddling / len(blocks) <= 0.15


def assign_reading_order(
    blocks: list[LayoutBlock],
    *,
    n_columns: int | None = None,
    line_tolerance: float = 0.01,
) -> list[LayoutBlock]:
    """Order blocks the way a human reads them, and stamp ``reading_order``.

    Column-major: all of column 0 top-to-bottom, then column 1, and so on.
    Full-width blocks (titles, wide figures, spanning tables) are kept in
    vertical position by assigning them to the column their left edge falls in,
    which keeps a page title ahead of the body text beneath it.

    Returns the same objects, sorted, with ``column`` and ``reading_order`` set.
    """
    if not blocks:
        return []

    n = n_columns if n_columns is not None else detect_columns(blocks)

    for b in blocks:
        b.column = 0 if n <= 1 else min(int(b.bbox.x0 * n), n - 1)

    # Round y to a tolerance so blocks on the same visual line do not flip order
    # over sub-point differences in their reported top edge.
    def key(b: LayoutBlock) -> tuple[int, float, float]:
        return (b.column, round(b.bbox.y0 / line_tolerance), b.bbox.x0)

    ordered = sorted(blocks, key=key)
    for i, b in enumerate(ordered):
        b.reading_order = i
    return ordered


# ---------------------------------------------------------------------------
# Headers and footers
# ---------------------------------------------------------------------------


@dataclass
class HeaderFooterDetector:
    """Finds running headers/footers by repetition across a document.

    Two passes: :meth:`observe` every page first, then :meth:`finalize`, then
    query with :meth:`classify`. Requiring the whole document before deciding is
    the point -- "Annual Report 2023" at the top of one page is a title; on
    forty pages it is furniture.
    """

    # Fraction of the page height treated as the header / footer band.
    band_ratio: float = 0.08
    # A candidate must appear on at least this fraction of pages to count.
    min_page_ratio: float = 0.3
    min_pages: int = 3

    _header_counts: Counter[str] = field(default_factory=Counter)
    _footer_counts: Counter[str] = field(default_factory=Counter)
    _n_pages: int = 0
    _headers: set[str] = field(default_factory=set)
    _footers: set[str] = field(default_factory=set)
    _finalized: bool = False

    def observe(self, blocks: list[LayoutBlock]) -> None:
        """Record one page's candidate furniture."""
        self._n_pages += 1
        for b in blocks:
            key = self._key(b.text)
            if not key:
                continue
            if b.bbox.y1 <= self.band_ratio:
                self._header_counts[key] += 1
            elif b.bbox.y0 >= 1.0 - self.band_ratio:
                self._footer_counts[key] += 1

    def finalize(self) -> None:
        """Decide which candidates are genuinely repeating."""
        threshold = max(self.min_pages, int(self._n_pages * self.min_page_ratio))
        self._headers = {k for k, c in self._header_counts.items() if c >= threshold}
        self._footers = {k for k, c in self._footer_counts.items() if c >= threshold}
        self._finalized = True

    def classify(self, block: LayoutBlock) -> str | None:
        """Return ``"header"``, ``"footer"``, or None for a block."""
        if not self._finalized:
            raise RuntimeError("call finalize() after observing every page")
        key = self._key(block.text)
        if not key:
            return None
        if key in self._headers and block.bbox.y1 <= self.band_ratio:
            return "header"
        if key in self._footers and block.bbox.y0 >= 1.0 - self.band_ratio:
            return "footer"
        return None

    @staticmethod
    def _key(text: str) -> str:
        """Normalised comparison key: whitespace collapsed, digits masked.

        Masking digits is what lets 'Page 3' and 'Page 17' be recognised as the
        same running footer.
        """
        collapsed = collapse_whitespace(text)
        if len(collapsed) < 2 or len(collapsed) > 200:
            return ""
        return mask_digits(collapsed).lower()

    @property
    def n_pages_observed(self) -> int:
        return self._n_pages


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class SectionTracker:
    """Tracks the running section and subsection through a document.

    Headings are identified by font size relative to the document's *body* size,
    which is estimated as the most common size weighted by character count.
    Relative sizing is what makes this work across a corpus where one document
    sets body text at 9pt and another at 12pt.
    """

    # A block must be this much larger than body text to be a heading.
    section_ratio: float = 1.35
    subsection_ratio: float = 1.12
    max_heading_chars: int = 200

    body_size: float = 0.0
    _size_weights: Counter[float] = field(default_factory=Counter)
    _section: str | None = None
    _subsection: str | None = None
    _finalized: bool = False

    def observe(self, blocks: list[LayoutBlock]) -> None:
        """Accumulate font-size statistics for body-size estimation."""
        for b in blocks:
            if b.font_size > 0 and b.text.strip():
                # Weight by length: a page of 10pt body text should outvote a
                # single 24pt title.
                self._size_weights[round(b.font_size, 1)] += len(b.text)

    def finalize(self) -> None:
        self.body_size = self._size_weights.most_common(1)[0][0] if self._size_weights else 0.0
        self._finalized = True

    def heading_level(self, block: LayoutBlock) -> int | None:
        """1 for a section heading, 2 for a subsection, None if not a heading."""
        if not self._finalized:
            raise RuntimeError("call finalize() after observing every page")
        text = block.text.strip()
        if not text or len(text) > self.max_heading_chars or self.body_size <= 0:
            return None
        # A heading is a short line, so a multi-line block is body text even if
        # it happens to be set large.
        if text.count("\n") > 1:
            return None

        ratio = block.font_size / self.body_size
        if ratio >= self.section_ratio:
            return 1
        if ratio >= self.subsection_ratio or (block.is_bold and ratio >= 1.0):
            return 2
        return None

    def update(self, block: LayoutBlock) -> int | None:
        """Advance the running section if ``block`` is a heading.

        Returns the heading level it was treated as, or None.
        """
        level = self.heading_level(block)
        text = collapse_whitespace(block.text)
        if level == 1:
            self._section = text
            self._subsection = None  # a new section resets its subsections
        elif level == 2:
            self._subsection = text
        return level

    def reset_document(self) -> None:
        self._section = None
        self._subsection = None

    @property
    def section(self) -> str | None:
        return self._section

    @property
    def subsection(self) -> str | None:
        return self._subsection
