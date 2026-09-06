"""Figure detection, cropping, and caption attachment.

Two sources, because PDFs carry pictures in two entirely different ways:

* **Raster images** -- embedded bitmaps, reported directly by the parser.
* **Vector drawings** -- charts and schematics built from PDF drawing
  operators. These are invisible to ``get_images()``, and in a professionally
  typeset report they are the *majority* of the figures: the WHO situation
  report's map, the Fed's line charts and the IPCC's panels are all vector.
  Missing them would mean Method 3 has nothing to beat Method 1 on.

Vector figures have no declared extent, so one is reconstructed by clustering
nearby drawing primitives into connected regions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmrag.ingestion.classify import classify_figure, figure_confidence
from mmrag.ingestion.layout import LayoutBlock
from mmrag.logging_utils import get_logger
from mmrag.schemas import BBox, ElementType, FigureData, FigureType

log = get_logger(__name__)

# "Figure 3:", "Fig. 3.", "Chart 2 -", "Exhibit 4", "Panel (a)". Anchored to the
# start because a mid-sentence "see Figure 3" is a cross-reference, not a caption.
CAPTION_RE = re.compile(
    r"^\s*(figure|fig\.?|table|chart|exhibit|plate|panel|graph|diagram|map|scheme)\s*"
    r"[\s.:\-]*\(?[0-9IVXA-Za-z]{1,4}\)?[\s.:\-]",
    re.IGNORECASE,
)

# Figure captions in scientific documents often start with "Source:" instead.
SOURCE_RE = re.compile(r"^\s*(source|note)s?\s*:", re.IGNORECASE)


@dataclass
class FigureCandidate:
    """A detected visual region, before it becomes an Element."""

    bbox: BBox
    is_vector: bool
    # Raster only: the intrinsic pixel size the PDF declares for the image.
    native_width: int | None = None
    native_height: int | None = None
    n_primitives: int = 0

    @property
    def area(self) -> float:
        return self.bbox.area


# ---------------------------------------------------------------------------
# Vector clustering
# ---------------------------------------------------------------------------


def cluster_vector_drawings(
    rects: list[BBox],
    *,
    merge_gap: float = 0.02,
    min_primitives: int = 4,
    min_area_ratio: float = 0.01,
) -> list[FigureCandidate]:
    """Group scattered drawing primitives into figure-sized regions.

    Repeatedly merges any two clusters whose boxes are within ``merge_gap`` of
    each other (as a fraction of the page). A chart is hundreds of separate line
    and fill operations; only their union is a figure.

    Clusters made of very few primitives are dropped -- those are table rules,
    underlines, and page borders rather than pictures.
    """
    if not rects:
        return []

    clusters: list[tuple[BBox, int]] = [(r, 1) for r in rects]

    merged = True
    while merged:
        merged = False
        out: list[tuple[BBox, int]] = []
        while clusters:
            box, count = clusters.pop()
            grown = True
            while grown:
                grown = False
                remaining: list[tuple[BBox, int]] = []
                for other_box, other_count in clusters:
                    if _within_gap(box, other_box, merge_gap):
                        box = box.union(other_box)
                        count += other_count
                        grown = True
                        merged = True
                    else:
                        remaining.append((other_box, other_count))
                clusters = remaining
            out.append((box, count))
        clusters = out

    return [
        FigureCandidate(bbox=box, is_vector=True, n_primitives=count)
        for box, count in clusters
        if count >= min_primitives and box.area >= min_area_ratio
    ]


def text_coverage(region: BBox, blocks: list[LayoutBlock]) -> float:
    """Fraction of ``region`` covered by text blocks.

    The discriminator between a figure and a decorative background panel. A
    filled rectangle behind a block of prose clusters into a large, convincing
    "figure" -- the WHO situation report's blue Resources panel is exactly this
    -- and accepting it both invents a figure and hides the prose inside it.

    Real charts do contain text (axis labels, legends, titles) but it covers a
    small fraction of the plot area; a text panel is mostly text.
    """
    if region.area <= 0:
        return 0.0
    covered = sum(region.intersection_area(b.bbox) for b in blocks if b.text.strip())
    # Blocks can overlap slightly, so clamp rather than report >100% coverage.
    return min(1.0, covered / region.area)


def _within_gap(a: BBox, b: BBox, gap: float) -> bool:
    """Whether two boxes are close enough to belong to the same figure."""
    # Overlap in either axis plus proximity in the other is enough; a chart's
    # axis labels sit beside its plot area rather than on top of it.
    h_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    v_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return h_gap <= gap and v_gap <= gap


# ---------------------------------------------------------------------------
# Caption attachment
# ---------------------------------------------------------------------------


def is_caption_marked(text: str) -> bool:
    """Whether text opens with an explicit caption marker."""
    return bool(CAPTION_RE.match(text)) or bool(SOURCE_RE.match(text))


def find_caption(
    figure: BBox,
    blocks: list[LayoutBlock],
    *,
    max_gap: float = 0.06,
    min_horizontal_overlap: float = 0.3,
    body_font_size: float = 0.0,
    heading_size_ratio: float = 1.15,
) -> LayoutBlock | None:
    """Find the caption belonging to a figure.

    Scored on three signals, in priority order:

    1. Whether the text starts with a caption marker ("Figure 3:").
    2. Vertical proximity -- captions sit immediately below, occasionally above.
    3. Horizontal overlap -- which is what stops a caption in the left column
       being attached to a figure in the right column at the same height.

    Section headings are excluded outright when ``body_font_size`` is known.
    A heading sitting just above a chart is not its caption, and adopting one
    is actively harmful: the caption drives figure-type classification, so
    "Situation update:" would relabel a map as an unknown blob.
    """
    heading_threshold = body_font_size * heading_size_ratio if body_font_size > 0 else None
    best: tuple[float, LayoutBlock] | None = None

    for block in blocks:
        if not block.text.strip():
            continue
        marked = is_caption_marked(block.text)

        # Typographic headings are never captions -- unless they carry an
        # explicit marker, since some documents do set "Figure 3" in a larger face.
        if not marked and heading_threshold and block.font_size >= heading_threshold:
            continue

        gap = figure.vertical_gap(block.bbox)
        if gap > max_gap:
            continue
        overlap = figure.horizontal_overlap(block.bbox)
        if overlap < min_horizontal_overlap:
            continue

        is_below = block.bbox.y0 >= figure.y1 - 1e-6

        score = 0.0
        if marked:
            score += 10.0  # dominates: an explicit marker beats any geometry
        if is_below:
            score += 2.0  # captions are below far more often than above
        score += (1.0 - gap / max_gap) * 2.0
        score += overlap

        if best is None or score > best[0]:
            best = (score, block)

    if best is None:
        return None
    score, block = best
    if is_caption_marked(block.text):
        return block

    # Without a marker, demand both near-perfect geometry *and* the typographic
    # convention that captions are set smaller than body text. Geometry alone is
    # not enough: an ordinary paragraph immediately below a figure satisfies it
    # completely, and adopting one both invents a caption and poisons the
    # figure-type classification that reads it.
    if score < 4.0:
        return None
    if body_font_size > 0 and block.font_size >= body_font_size:
        return None
    return block


# ---------------------------------------------------------------------------
# Image statistics and cropping
# ---------------------------------------------------------------------------


def image_statistics(path: Path, *, sample_size: int = 200) -> dict[str, Any]:
    """Cheap colour statistics used by the figure-type heuristic.

    Downsampled first: exact statistics on a full-resolution crop cost far more
    than the signal is worth.
    """
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((sample_size, sample_size))
            colors = rgb.getcolors(maxcolors=1 << 24)
            n_colors = len(colors) if colors else None

            # ImageStat computes the per-band mean in C, and avoids materialising
            # the pixel list that Pillow has deprecated.
            mean_sat = ImageStat.Stat(rgb.convert("HSV")).mean[1] / 255.0
        return {"n_colors": n_colors, "mean_saturation": mean_sat}
    except Exception as exc:  # pragma: no cover - depends on the image
        log.debug("image statistics failed for %s: %s", path, exc)
        return {"n_colors": None, "mean_saturation": None}


def build_figure_data(
    candidate: FigureCandidate,
    *,
    image_path: Path | None,
    caption: str | None,
    dpi: int,
    page_area_ratio: float,
) -> tuple[FigureData, float]:
    """Assemble the structured figure payload and its extraction confidence."""
    stats: dict[str, Any] = {"n_colors": None, "mean_saturation": None}
    width_px = height_px = None
    if image_path is not None and image_path.exists():
        stats = image_statistics(image_path)
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width_px, height_px = img.size
        except Exception as exc:  # pragma: no cover
            log.debug("could not read crop size for %s: %s", image_path, exc)

    aspect = (width_px / height_px) if width_px and height_px else None
    classification = classify_figure(
        caption=caption,
        is_vector=candidate.is_vector,
        n_colors=stats["n_colors"],
        mean_saturation=stats["mean_saturation"],
        area_ratio=page_area_ratio,
        aspect_ratio=aspect,
    )

    figure = FigureData(
        image_path=str(image_path) if image_path else None,
        width_px=width_px,
        height_px=height_px,
        dpi=dpi,
        figure_type=FigureType(classification.label),
        figure_type_confidence=classification.confidence,
        figure_type_evidence=classification.evidence,
        n_colors=stats["n_colors"],
        mean_saturation=stats["mean_saturation"],
    )
    confidence = figure_confidence(
        area_ratio=page_area_ratio,
        has_caption=bool(caption),
        has_derived_text=figure.has_derived_text,
        is_vector=candidate.is_vector,
    )
    return figure, confidence


def element_type_for(figure_type: FigureType) -> ElementType:
    """Map a fine-grained figure type onto the coarse element taxonomy.

    The element type is what Method 2's router keys on, so it stays coarse;
    ``FigureData.figure_type`` keeps the detail.
    """
    if figure_type is FigureType.CHART:
        return ElementType.CHART
    if figure_type in {FigureType.DIAGRAM, FigureType.SCREENSHOT, FigureType.MAP}:
        return ElementType.DIAGRAM
    return ElementType.FIGURE
