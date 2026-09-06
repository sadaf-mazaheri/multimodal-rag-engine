"""Best-effort classification and extraction-confidence heuristics.

Everything here is a heuristic and is labelled as such. Two design rules:

1. **Abstain rather than guess.** ``UNKNOWN`` is a first-class answer. Method 2
   routes queries on ``figure_type``, so a confidently wrong label actively
   misdirects retrieval, whereas an honest ``UNKNOWN`` merely fails to help.
2. **Record the evidence.** Every classification returns *why* it decided what
   it did, which is stored on the element. When Step 6 finds a cluster of bad
   retrievals, the cause is then visible instead of having to be re-derived.

Nothing here calls a model. VLM-based figure description is a separate,
optional enrichment pass; this module must work offline and for free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mmrag.ingestion.text_utils import is_probably_numeric
from mmrag.schemas import FigureType, TableType


@dataclass(frozen=True)
class Classification:
    """A label, how much to trust it, and the evidence behind it."""

    label: str
    confidence: float
    evidence: str


# ---------------------------------------------------------------------------
# Figure type
# ---------------------------------------------------------------------------

# Caption wording is by far the most reliable cheap signal: a caption is a
# human's own description of what the figure is. Ordered most- to
# least-specific, since the first match wins.
_CAPTION_PATTERNS: list[tuple[FigureType, re.Pattern[str]]] = [
    (
        FigureType.SCREENSHOT,
        re.compile(r"\b(screenshot|screen ?capture|user interface|\bUI\b|dashboard)\b", re.I),
    ),
    (
        FigureType.MAP,
        re.compile(r"\b(map|geographic|spatial distribution|by (country|region))\b", re.I),
    ),
    (
        FigureType.CHART,
        re.compile(
            r"\b(chart|graph|plot|histogram|scatter|time[- ]series|trend|"
            r"distribution of|percent(age)?|share of|growth|index|per capita|"
            r"projections?|forecast)\b",
            re.I,
        ),
    ),
    (
        FigureType.DIAGRAM,
        re.compile(
            r"\b(diagram|architecture|schematic|block|flow ?chart|workflow|pipeline|"
            r"process|topology|overview of the (model|system)|framework)\b",
            re.I,
        ),
    ),
    (
        FigureType.PHOTO,
        re.compile(
            r"\b(photo(graph)?|image of|micrograph|specimen|structure of|rendering)\b", re.I
        ),
    ),
    (FigureType.EQUATION, re.compile(r"\b(equation|formula)\b", re.I)),
]


def classify_figure(
    *,
    caption: str | None = None,
    is_vector: bool = False,
    n_colors: int | None = None,
    mean_saturation: float | None = None,
    area_ratio: float | None = None,
    aspect_ratio: float | None = None,
) -> Classification:
    """Guess what a visual element depicts.

    ``is_vector`` says the figure was reconstructed from PDF drawing operators
    rather than an embedded raster. That is a strong structural signal: charts
    and diagrams in professionally typeset documents are almost always vector,
    while photographs are always raster.
    """
    # --- 1. caption wording, the most trustworthy evidence -----------------
    if caption:
        for figure_type, pattern in _CAPTION_PATTERNS:
            match = pattern.search(caption)
            if match:
                return Classification(figure_type.value, 0.8, f"caption matched {match.group(0)!r}")

    # --- 2. tiny wide marks are page furniture ------------------------------
    is_tiny = area_ratio is not None and area_ratio < 0.01
    is_strip = aspect_ratio is not None and (aspect_ratio > 4 or aspect_ratio < 0.25)
    if is_tiny and is_strip:
        return Classification(FigureType.LOGO.value, 0.5, "very small, extreme aspect ratio")

    # --- 3. vector vs raster ------------------------------------------------
    if is_vector:
        # Vector art with a small palette is a chart or a schematic. Without
        # a caption there is no reliable way to tell which, so say so.
        if n_colors is not None and n_colors <= 12:
            return Classification(FigureType.DIAGRAM.value, 0.4, "vector art, small palette")
        return Classification(FigureType.CHART.value, 0.35, "vector art, larger palette")

    if n_colors is not None and mean_saturation is not None:
        # Photographs have many distinct colours and continuous tone; synthetic
        # graphics have flat fills and a small palette.
        if n_colors > 5000 and mean_saturation > 0.15:
            return Classification(FigureType.PHOTO.value, 0.55, "many colours, high saturation")
        if n_colors < 64:
            return Classification(FigureType.DIAGRAM.value, 0.35, "raster with a tiny palette")

    return Classification(FigureType.UNKNOWN.value, 0.0, "no discriminating evidence")


# ---------------------------------------------------------------------------
# Table type
# ---------------------------------------------------------------------------

_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_CURRENCY = re.compile(r"[$€£¥]")
_PAREN_NEGATIVE = re.compile(r"\(\s*[\d,]+(\.\d+)?\s*\)")
_REGISTER_HEADERS = {"bits", "bit", "field", "offset", "reset", "register", "address", "rw", "r/w"}
_FINANCIAL_HEADERS = {
    "amount",
    "total",
    "revenue",
    "earnings",
    "assets",
    "liabilities",
    "cost",
    "expense",
    "income",
    "cash",
    "%",
    "usd",
    "eur",
}


def classify_table(columns: list[str], rows: list[list[str | None]]) -> Classification:
    """Guess a table's genre from its headers and cell content."""
    header_tokens = {c.strip().lower() for c in columns if c}
    cells = [c for row in rows for c in row if c and c.strip()]
    if not cells:
        return Classification(TableType.UNKNOWN.value, 0.0, "no populated cells")

    joined = " ".join(cells)
    n_cells = len(cells)

    # --- register maps: hex values or the canonical datasheet headers -------
    hex_hits = len(_HEX.findall(joined))
    if header_tokens & _REGISTER_HEADERS or hex_hits >= max(3, n_cells * 0.1):
        return Classification(
            TableType.REGISTER.value,
            0.75 if header_tokens & _REGISTER_HEADERS else 0.6,
            f"register headers={sorted(header_tokens & _REGISTER_HEADERS)} hex_cells={hex_hits}",
        )

    # --- financial statements ----------------------------------------------
    currency_hits = len(_CURRENCY.findall(joined))
    paren_negatives = len(_PAREN_NEGATIVE.findall(joined))
    if currency_hits >= 2 or paren_negatives >= 2 or header_tokens & _FINANCIAL_HEADERS:
        return Classification(
            TableType.FINANCIAL.value,
            0.7,
            f"currency={currency_hits} paren_negatives={paren_negatives}",
        )

    # --- numeric grids ------------------------------------------------------
    numeric_ratio = sum(1 for c in cells if is_probably_numeric(c)) / n_cells
    n_rows, n_cols = len(rows), len(columns)
    if numeric_ratio > 0.7 and n_rows >= 3 and n_cols >= 3:
        return Classification(
            TableType.MATRIX.value, 0.5, f"numeric_ratio={numeric_ratio:.2f}, {n_rows}x{n_cols}"
        )
    if 0.3 < numeric_ratio <= 0.7 and n_cols >= 3:
        return Classification(
            TableType.COMPARISON.value, 0.4, f"mixed labels and numbers, {n_rows}x{n_cols}"
        )

    return Classification(TableType.SIMPLE.value, 0.3, f"numeric_ratio={numeric_ratio:.2f}")


# ---------------------------------------------------------------------------
# Extraction confidence
# ---------------------------------------------------------------------------
#
# One scale, one meaning, across every modality:
#
#   1.0   the extractor's output is a faithful representation of the region
#   0.5   usable but degraded, or a structurally uncertain detection
#   0.0   the region was located but nothing meaningful was recovered
#
# Consumers use it as a filter and as a tie-break, and Step 6 groups error
# analysis by it -- so it must be comparable across extractors, not a per-module
# private score.


def text_confidence(*, replacement_ratio: float, n_chars: int, alpha_ratio: float) -> float:
    """Confidence for a text block.

    Penalises unmappable glyphs (a broken font map), very short fragments (often
    stray marks split off from a real block), and punctuation soup.
    """
    if n_chars == 0:
        return 0.0
    score = 1.0
    score -= min(1.0, replacement_ratio * 3.0)  # 33% unmappable -> zero
    if n_chars < 15:
        score -= 0.2
    if alpha_ratio < 0.2 and n_chars > 20:
        score -= 0.2  # plausible for numeric content, so only a mild penalty
    return max(0.0, min(1.0, score))


def table_confidence(
    *, fill_ratio: float, n_rows: int, n_cols: int, has_header: bool, ragged_rows: int = 0
) -> float:
    """Confidence for a detected table.

    Sparse detections are the dominant failure mode: PyMuPDF's ruling-line
    strategy readily turns a boxed callout or a chart's gridlines into a
    mostly-empty 'table'. Fill ratio is therefore the primary term.
    """
    if n_rows < 1 or n_cols < 1:
        return 0.0
    score = 0.25 + 0.6 * fill_ratio
    if has_header:
        score += 0.1
    if n_rows < 2 or n_cols < 2:
        score -= 0.3  # a 1xN strip is almost never a real table
    if ragged_rows:
        score -= min(0.2, 0.05 * ragged_rows)
    return max(0.0, min(1.0, score))


def figure_confidence(
    *, area_ratio: float, has_caption: bool, has_derived_text: bool, is_vector: bool
) -> float:
    """Confidence that a detected region is a real, meaningful figure.

    Note this scores *the detection*, not how well the figure is understood.
    A large captioned chart scores high even before any OCR or VLM pass has run;
    whether its content was recovered is ``FigureData.has_derived_text``.
    """
    score = 0.4
    # Bigger regions are much more likely to be real content than stray marks.
    score += min(0.3, area_ratio * 3.0)
    if has_caption:
        score += 0.25  # a caption is near-proof the region is a genuine figure
    if has_derived_text:
        score += 0.05
    if is_vector:
        # Vector clusters are reconstructed by us rather than declared by the
        # PDF, so they are inherently less certain than an embedded image.
        score -= 0.1
    return max(0.0, min(1.0, score))
