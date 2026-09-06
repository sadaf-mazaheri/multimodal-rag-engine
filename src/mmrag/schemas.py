"""Core data model shared by all three retrieval methods.

The chain ``Document -> Page -> Element -> Chunk`` is deliberately explicit.
Every retrieval hit carries enough information to answer "which document, which
page, and where on that page did this come from?", which is what makes
element-level source attribution possible in the generated answers.

These models mirror ``scripts/sql/001_schema.sql``; the SQL is the storage
contract and this module is the in-process contract.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ElementType(str, Enum):
    """What kind of thing an element is, as decided by the layout parser."""

    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    DIAGRAM = "diagram"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"

    @property
    def is_visual(self) -> bool:
        return self in {ElementType.FIGURE, ElementType.CHART, ElementType.DIAGRAM}

    @property
    def is_textual(self) -> bool:
        return self in {
            ElementType.TEXT,
            ElementType.TITLE,
            ElementType.CAPTION,
            ElementType.HEADER,
            ElementType.FOOTER,
        }


class ChunkType(str, Enum):
    """The retrieval unit's modality.

    ``PAGE`` exists for Method 3, where a whole rendered page is the unit.
    """

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    PAGE = "page"


class Modality(str, Enum):
    """Retriever families that a query can be routed to (Methods 2 and 3)."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    VISUAL_PAGE = "visual_page"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Axis-aligned box, normalised to [0, 1] against the page width/height.

    Normalising decouples the box from the DPI a page image happened to be
    rendered at, so the same coordinates highlight the right region whether the
    consumer is a 72-dpi PDF viewer or a 200-dpi PNG.
    """

    model_config = ConfigDict(frozen=True)

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @field_validator("x1")
    @classmethod
    def _x_ordered(cls, v: float, info: Any) -> float:
        x0 = info.data.get("x0")
        if x0 is not None and v < x0:
            raise ValueError(f"x1 ({v}) must be >= x0 ({x0})")
        return v

    @field_validator("y1")
    @classmethod
    def _y_ordered(cls, v: float, info: Any) -> float:
        y0 = info.data.get("y0")
        if y0 is not None and v < y0:
            raise ValueError(f"y1 ({v}) must be >= y0 ({y0})")
        return v

    @classmethod
    def from_absolute(
        cls, x0: float, y0: float, x1: float, y1: float, *, width: float, height: float
    ) -> BBox:
        """Build from PDF-point coordinates, clamping to the page."""
        if width <= 0 or height <= 0:
            raise ValueError(f"page size must be positive, got {width}x{height}")

        def clamp(v: float) -> float:
            return min(1.0, max(0.0, v))

        # Some PDFs emit inverted boxes; normalise the ordering rather than reject.
        lo_x, hi_x = sorted((x0, x1))
        lo_y, hi_y = sorted((y0, y1))
        return cls(
            x0=clamp(lo_x / width),
            y0=clamp(lo_y / height),
            x1=clamp(hi_x / width),
            y1=clamp(hi_y / height),
        )

    def to_absolute(self, width: float, height: float) -> tuple[float, float, float, float]:
        return (self.x0 * width, self.y0 * height, self.x1 * width, self.y1 * height)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def union(self, other: BBox) -> BBox:
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def iou(self, other: BBox) -> float:
        """Intersection over union; used to match captions to figures."""
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------


def make_page_id(doc_id: str, page_number: int) -> str:
    return f"{doc_id}#p{page_number}"


def make_element_id(doc_id: str, page_number: int, element_type: str, ordinal: int) -> str:
    return f"{doc_id}#p{page_number}#{element_type}{ordinal:03d}"


def make_chunk_id(variant: str, *parts: str) -> str:
    """Stable, collision-resistant chunk id.

    Hashing the constituent parts means re-running ingestion on unchanged input
    reproduces the same ids, so indexes can be updated incrementally.
    """
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{variant}#{digest}"


# ---------------------------------------------------------------------------
# Corpus entities
# ---------------------------------------------------------------------------


class Document(BaseModel):
    doc_id: str
    title: str
    source_url: str | None = None
    publisher: str | None = None
    category: str | None = None
    license: str | None = None
    sha256: str
    n_pages: int
    file_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    page_id: str
    doc_id: str
    page_number: int = Field(ge=1, description="1-indexed, matching a PDF reader display")
    width: float
    height: float
    image_path: str | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Element(BaseModel):
    """One parsed region of a page.

    A single element carries every representation we managed to extract for that
    region. Method 1 reads the flattened text fields, Method 2 reads the
    modality-specific fields, and Method 3 mostly ignores elements in favour of
    the page image -- but all three cite the same element ids.
    """

    element_id: str
    doc_id: str
    page_id: str
    page_number: int = Field(ge=1)
    element_type: ElementType
    reading_order: int = 0
    bbox: BBox | None = None

    text: str | None = None
    table_markdown: str | None = None
    table_json: dict[str, Any] | None = None
    image_path: str | None = None
    caption: str | None = None
    ocr_text: str | None = None
    description: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def best_text(self) -> str:
        """The single most faithful textual rendering of this element.

        This method *is* Method 1: it is exactly the point at which non-text
        modalities are lossily collapsed into text. Method 2 exists to measure
        what that collapse costs.
        """
        if self.element_type is ElementType.TABLE and self.table_markdown:
            parts = [self.caption, self.table_markdown]
        elif self.element_type.is_visual:
            parts = [self.caption, self.description, self.ocr_text]
        else:
            parts = [self.text]
        return "\n".join(p.strip() for p in parts if p and p.strip())


class Chunk(BaseModel):
    """A retrieval unit: the thing that gets embedded, scored, and cited."""

    chunk_id: str
    doc_id: str
    page_number: int = Field(ge=1)
    chunk_type: ChunkType
    text: str
    element_ids: list[str] = Field(default_factory=list)
    token_count: int | None = None
    variant: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class ScoredChunk(BaseModel):
    """A chunk plus the score and provenance of *how it was retrieved*.

    ``retriever`` and ``rank`` are kept because fusion (RRF) is rank-based and
    the evaluation needs per-retriever contribution breakdowns.
    """

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    retriever: str
    modality: Modality = Modality.TEXT
    # Populated after fusion: the per-retriever ranks that produced the fused score.
    component_ranks: dict[str, int] = Field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class Citation(BaseModel):
    """Source attribution attached to a generated answer."""

    doc_id: str
    doc_title: str
    page_number: int
    element_ids: list[str] = Field(default_factory=list)
    chunk_id: str | None = None
    bbox: BBox | None = None
    snippet: str | None = None

    def human_label(self) -> str:
        return f"{self.doc_title}, p. {self.page_number}"


class Answer(BaseModel):
    """The end-to-end output of a method for one query."""

    query: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    method: str = "unknown"
    # Wall-clock and token accounting, filled in by the pipeline for the report.
    latency_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
