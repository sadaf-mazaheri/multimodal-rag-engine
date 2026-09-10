"""Core data model shared by all three retrieval methods.

The chain ``Document -> Page -> Element -> Chunk`` is deliberately explicit, and
metadata is a first-class part of it rather than a bag of strings bolted on at
the end. Every level carries structured fields so that later stages can
*filter* and *rank* on them, and so that any retrieval hit can answer "which
document, which page, which region, produced by which extractor, and how much
do we trust it?".

Two rules this module exists to enforce:

1. **Metadata stays structured.** It is never concatenated into the text that
   gets embedded. Embedding text is content; metadata is payload. Mixing them
   pollutes the vector space and makes it impossible to filter on a field
   afterwards.
2. **Provenance propagates.** A ``Chunk`` cannot exist without the element ids
   it came from, and those resolve to a page and a bounding box. See
   :func:`build_provenance`.

These models mirror ``scripts/sql/001_schema.sql``; the SQL is the storage
contract and this module is the in-process contract.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ElementType(str, Enum):
    """What kind of thing an element is, as decided by the layout parser."""

    TEXT = "text"
    TITLE = "title"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    DIAGRAM = "diagram"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    EQUATION = "equation"

    @property
    def is_visual(self) -> bool:
        return self in {ElementType.FIGURE, ElementType.CHART, ElementType.DIAGRAM}

    @property
    def is_textual(self) -> bool:
        return self in {
            ElementType.TEXT,
            ElementType.TITLE,
            ElementType.HEADING,
            ElementType.LIST,
            ElementType.CAPTION,
            ElementType.HEADER,
            ElementType.FOOTER,
            ElementType.FOOTNOTE,
            ElementType.EQUATION,
        }

    @property
    def is_boilerplate(self) -> bool:
        """Repeating page furniture: useful for provenance, noise for retrieval."""
        return self in {ElementType.HEADER, ElementType.FOOTER}


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


class DocumentType(str, Enum):
    """Coarse genre. Drives both metadata filtering and result slicing."""

    ANNUAL_REPORT = "annual_report"
    RESEARCH_PAPER = "research_paper"
    TECHNICAL_MANUAL = "technical_manual"
    DATASHEET = "datasheet"
    POLICY_REPORT = "policy_report"
    WHITEPAPER = "whitepaper"
    SITUATION_REPORT = "situation_report"
    OTHER = "other"


class FigureType(str, Enum):
    """Best-effort classification of a visual element.

    ``UNKNOWN`` is a legitimate, common answer. A confident wrong label is worse
    than an honest abstention, because Method 2 routes on this field.
    """

    PHOTO = "photo"
    CHART = "chart"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    MAP = "map"
    EQUATION = "equation"
    LOGO = "logo"
    UNKNOWN = "unknown"


class TableType(str, Enum):
    """Best-effort classification of a table's shape and content."""

    FINANCIAL = "financial"
    REGISTER = "register"
    COMPARISON = "comparison"
    MATRIX = "matrix"
    SIMPLE = "simple"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Which extractor produced an element.

    Recorded per element so that error analysis can ask "are the bad retrievals
    concentrated in one extractor?" -- a question that is impossible to answer
    once everything has been flattened into undifferentiated text.
    """

    PYMUPDF_TEXT = "pymupdf_text"
    PYMUPDF_TABLE = "pymupdf_table"
    PDFPLUMBER_TABLE = "pdfplumber_table"
    PYMUPDF_IMAGE = "pymupdf_image"
    PYMUPDF_DRAWING = "pymupdf_drawing"
    HEURISTIC_LAYOUT = "heuristic_layout"
    OCR_TESSERACT = "ocr_tesseract"
    VLM_CAPTION = "vlm_caption"


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

    def to_pixels(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """Integer pixel box for cropping or drawing on a rendered page image."""
        return (
            round(self.x0 * image_width),
            round(self.y0 * image_height),
            round(self.x1 * image_width),
            round(self.y1 * image_height),
        )

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def union(self, other: BBox) -> BBox:
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def iou(self, other: BBox) -> float:
        """Intersection over union; used to match captions to figures."""
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def intersection_area(self, other: BBox) -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def contains(self, other: BBox, *, tolerance: float = 0.0) -> bool:
        """Whether ``other`` sits inside this box, within a slack tolerance."""
        return (
            other.x0 >= self.x0 - tolerance
            and other.y0 >= self.y0 - tolerance
            and other.x1 <= self.x1 + tolerance
            and other.y1 <= self.y1 + tolerance
        )

    def vertical_gap(self, other: BBox) -> float:
        """Vertical whitespace between two boxes; 0 if they overlap vertically."""
        if self.y1 <= other.y0:
            return other.y0 - self.y1
        if other.y1 <= self.y0:
            return self.y0 - other.y1
        return 0.0

    def horizontal_overlap(self, other: BBox) -> float:
        """Shared x-extent as a fraction of the narrower box.

        Caption matching uses this: a caption belongs to the figure it sits
        directly under, not to one in the next column at a similar height.
        """
        ix0, ix1 = max(self.x0, other.x0), min(self.x1, other.x1)
        if ix1 <= ix0:
            return 0.0
        narrower = min(self.x1 - self.x0, other.x1 - other.x0)
        return (ix1 - ix0) / narrower if narrower > 0 else 0.0

    @staticmethod
    def union_all(boxes: list[BBox]) -> BBox | None:
        """Smallest box covering every input box, or None if there are none."""
        if not boxes:
            return None
        result = boxes[0]
        for b in boxes[1:]:
            result = result.union(b)
        return result


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------


def make_page_id(doc_id: str, page_number: int) -> str:
    return f"{doc_id}#p{page_number}"


def make_element_id(doc_id: str, page_number: int, element_type: str, ordinal: int) -> str:
    """Stable element id.

    Deterministic in (document, page, type, ordinal-within-type-on-page), so
    re-running ingestion over an unchanged PDF with an unchanged parser yields
    identical ids. That is what lets an evaluation gold set reference element
    ids without being invalidated by every re-ingest.
    """
    return f"{doc_id}#p{page_number}#{element_type}{ordinal:03d}"


def make_chunk_id(variant: str, *parts: str) -> str:
    """Stable, collision-resistant chunk id.

    Hashing the constituent parts means re-running ingestion on unchanged input
    reproduces the same ids, so indexes can be updated incrementally.
    """
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{variant}#{digest}"


def parse_element_id(element_id: str) -> tuple[str, int]:
    """Recover ``(doc_id, page_number)`` from an element id.

    Cheap provenance for debugging and for error analysis over logged runs,
    where the full element row may not be to hand.
    """
    try:
        doc_id, page_part, _ = element_id.split("#", 2)
        return doc_id, int(page_part.lstrip("p"))
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"malformed element_id: {element_id!r}") from exc


# ---------------------------------------------------------------------------
# Document level
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """Document-level metadata.

    Sourced from three places, in decreasing order of trust: the corpus manifest
    (human-curated), the PDF's own metadata dictionary, and heuristics over the
    first page. ``metadata['field_sources']`` records which won for each field,
    so a wrong title can be traced to the thing that produced it.
    """

    doc_id: str
    title: str
    source: str | None = Field(default=None, description="Publisher or issuing organisation")
    source_url: str | None = None
    authors: list[str] = Field(default_factory=list)
    organization: str | None = None
    publication_date: date | None = None
    version: str | None = None
    doc_type: DocumentType = DocumentType.OTHER
    domain: str | None = Field(default=None, description="Subject area, e.g. finance / science")
    language: str = "en"
    license: str | None = None

    file_name: str
    file_path: str
    sha256: str
    file_size_bytes: int | None = None

    n_pages: int = Field(ge=0, description="Pages in the source PDF")
    n_pages_ingested: int = Field(ge=0, description="Pages actually parsed, after any page_limit")

    ingested_at: datetime | None = None
    parser_version: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ingested_pages_fit(self) -> Document:
        if self.n_pages_ingested > self.n_pages:
            raise ValueError(
                f"n_pages_ingested ({self.n_pages_ingested}) exceeds "
                f"n_pages ({self.n_pages}) for {self.doc_id}"
            )
        return self

    @property
    def is_truncated(self) -> bool:
        return self.n_pages_ingested < self.n_pages

    def citation_label(self) -> str:
        """Short human-facing label, e.g. 'IPCC AR6 SPM (IPCC, 2021)'."""
        bits = [b for b in (self.organization or self.source, _year(self.publication_date)) if b]
        return f"{self.title} ({', '.join(bits)})" if bits else self.title


def _year(d: date | None) -> str | None:
    return str(d.year) if d else None


# ---------------------------------------------------------------------------
# Page level
# ---------------------------------------------------------------------------


class Page(BaseModel):
    """Page-level metadata.

    ``section``/``subsection`` are the *running* section at this point in the
    document, carried forward from the last heading seen -- so a page in the
    middle of a chapter still knows which chapter it is in, which is what makes
    section filtering and context expansion possible later.
    """

    page_id: str
    doc_id: str
    page_number: int = Field(ge=1, description="1-indexed, matching a PDF reader display")

    width: float = Field(gt=0, description="Page width in PDF points")
    height: float = Field(gt=0, description="Page height in PDF points")
    rotation: int = Field(default=0, description="Page rotation in degrees: 0, 90, 180 or 270")

    image_path: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_dpi: int | None = None

    section: str | None = None
    subsection: str | None = None
    header_text: str | None = None
    footer_text: str | None = None

    raw_text: str | None = Field(
        default=None, description="Full page text, for page-level fallback"
    )
    n_elements: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rotation")
    @classmethod
    def _valid_rotation(cls, v: int) -> int:
        normalised = v % 360
        if normalised not in {0, 90, 180, 270}:
            raise ValueError(f"rotation must be a multiple of 90, got {v}")
        return normalised

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def is_landscape(self) -> bool:
        return self.width > self.height


# ---------------------------------------------------------------------------
# Modality-specific structured payloads
# ---------------------------------------------------------------------------


# Ruling-line detection happily reports a rendered sentence as a 35-column
# table. Real tables in this corpus top out around a dozen columns; beyond this
# the detection is structural noise rather than data.
MAX_PLAUSIBLE_COLUMNS = 20


class TableData(BaseModel):
    """Structured table content.

    Kept as rows and columns, not only as Markdown. Method 1 uses ``markdown``
    (that flattening is the whole point of Method 1); Method 2 uses the
    structure. Keeping both is what makes the comparison possible.
    """

    n_rows: int = Field(ge=0)
    n_cols: int = Field(ge=0)
    header_rows: int = Field(default=1, ge=0)
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str | None]] = Field(default_factory=list)
    markdown: str = ""
    table_type: TableType = TableType.UNKNOWN
    # Fraction of cells that are non-empty. A sparse table is usually a
    # misdetected layout block rather than real tabular data.
    fill_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _shape_is_consistent(self) -> TableData:
        if self.columns and len(self.columns) != self.n_cols:
            raise ValueError(f"columns has {len(self.columns)} entries but n_cols is {self.n_cols}")
        if self.rows and len(self.rows) != self.n_rows:
            raise ValueError(f"rows has {len(self.rows)} entries but n_rows is {self.n_rows}")
        for i, row in enumerate(self.rows):
            if len(row) != self.n_cols:
                raise ValueError(f"row {i} has {len(row)} cells but n_cols is {self.n_cols}")
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_cols)

    @property
    def is_degenerate(self) -> bool:
        """A 'table' not worth indexing as one.

        Three failure modes, all common with ruling-line detection:

        * too small -- a 1xN strip is a boxed line of text;
        * too empty -- gridlines picked up from a chart;
        * a *word grid* -- the attention-visualisation figures in the
          Transformer paper come back as 14-column tables whose cells are
          single words of a sentence. Structurally they look like a perfect
          table (fill ratio 1.0), so only the cell content gives them away.
        """
        if self.n_cols < 2:
            return True
        # ``n_rows`` counts *body* rows: a promoted header lives in ``columns``
        # and is no longer one of them. Size must therefore be judged on
        # header + body, or a header-plus-one-data-row table is rejected --
        # which is the exact shape of a register description table, and made
        # every such table on a datasheet page reappear as a phantom chart.
        if self.n_rows < 1 or self.n_rows + self.header_rows < 2:
            return True
        if self.n_cols > MAX_PLAUSIBLE_COLUMNS:
            return True
        if self.fill_ratio < 0.3:
            return True
        return self.is_word_grid

    @property
    def is_word_grid(self) -> bool:
        """Whether the cells look like a sentence split across columns.

        Keyed on cells being *single alphabetic words* -- prose chopped at
        token boundaries. Deliberately not keyed on cell length alone: numeric
        tables also have short cells ("12", "7.1"), and rejecting those would
        throw away exactly the dense results grids this benchmark cares about
        most. Genuine table cells are numeric, multi-word, or carry punctuation
        and units; a column of bare English words across a very wide grid is
        a rendered sentence.
        """
        if self.n_cols < 6:
            return False
        cells = [c.strip() for row in self.rows for c in row if c and c.strip()]
        # Punctuation-only cells are part of the rendered sentence but are not
        # evidence either way, so they are excluded rather than counted against.
        informative = [c for c in cells if any(ch.isalnum() for ch in c)]
        if not informative:
            return False
        single_words = sum(1 for c in informative if c.isalpha() and " " not in c)
        return single_words / len(informative) > 0.6


class FigureData(BaseModel):
    """Structured figure/image content.

    ``ocr_text`` and ``description`` are *derived* representations added by the
    optional enrichment passes. They are stored separately from the caption
    because they have very different reliability, and the evaluation needs to be
    able to tell which one a retrieval actually matched on.
    """

    image_path: str | None = None
    width_px: int | None = Field(default=None, gt=0)
    height_px: int | None = Field(default=None, gt=0)
    dpi: int | None = None

    figure_type: FigureType = FigureType.UNKNOWN
    figure_type_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    figure_type_evidence: str | None = Field(
        default=None, description="Why this type was chosen, for error analysis"
    )

    ocr_text: str | None = None
    ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    description: str | None = Field(default=None, description="VLM-generated description")

    # Cheap image statistics used by the type heuristic; kept for debugging.
    n_colors: int | None = None
    mean_saturation: float | None = None

    @property
    def aspect_ratio(self) -> float | None:
        if self.width_px and self.height_px:
            return self.width_px / self.height_px
        return None

    @property
    def has_derived_text(self) -> bool:
        """Whether anything textual was recovered from this figure at all.

        When this is False, the figure is invisible to Method 1 by construction.
        """
        return bool((self.ocr_text or "").strip() or (self.description or "").strip())


# ---------------------------------------------------------------------------
# Element level
# ---------------------------------------------------------------------------


class Element(BaseModel):
    """One parsed region of a page: the atomic unit of the corpus.

    A single element carries every representation we managed to extract for that
    region. Method 1 reads the flattened text, Method 2 reads the structured
    payloads, and Method 3 mostly works from the page image -- but all three
    cite the same element ids, which is what makes the comparison fair.

    ``parent_id`` links an element to another *element*, not to its page: a
    caption is parented to the figure it describes. The page link is
    ``page_id``, which every element always has.
    """

    element_id: str
    doc_id: str
    page_id: str
    page_number: int = Field(ge=1)
    element_type: ElementType

    parent_id: str | None = Field(
        default=None, description="element_id of the owning element, e.g. caption -> figure"
    )
    reading_order: int = Field(default=0, ge=0, description="Position in page reading order")

    bbox: BBox | None = None
    section: str | None = None
    subsection: str | None = None

    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF_TEXT
    extraction_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How much to trust this extraction; heuristic, see ingestion.confidence",
    )

    text: str | None = None
    caption: str | None = Field(
        default=None, description="Caption text, for both tables and figures"
    )
    table: TableData | None = None
    figure: FigureData | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _payload_matches_type(self) -> Element:
        if self.table is not None and self.element_type is not ElementType.TABLE:
            raise ValueError(
                f"{self.element_id}: table payload on a {self.element_type.value} element"
            )
        if self.figure is not None and not self.element_type.is_visual:
            raise ValueError(
                f"{self.element_id}: figure payload on a {self.element_type.value} element"
            )
        if self.parent_id == self.element_id:
            raise ValueError(f"{self.element_id}: element cannot be its own parent")
        return self

    def best_text(self) -> str:
        """The single most faithful textual rendering of this element's *content*.

        This method *is* Method 1: it is exactly the point at which non-text
        modalities are lossily collapsed into text. Method 2 exists to measure
        what that collapse costs.

        Deliberately excludes structural metadata (section, document title, page
        number, extraction method). Those are payload for filtering and ranking,
        not content to be embedded -- concatenating them here would pollute the
        vector space and make the fields impossible to filter on afterwards.
        """
        if self.element_type is ElementType.TABLE and self.table is not None:
            parts = [self.caption, self.table.markdown]
        elif self.element_type.is_visual and self.figure is not None:
            parts = [self.caption, self.figure.description, self.figure.ocr_text]
        elif self.element_type.is_visual:
            parts = [self.caption]
        else:
            parts = [self.text]
        return "\n".join(p.strip() for p in parts if p and p.strip())

    @property
    def is_empty(self) -> bool:
        return not self.best_text()

    def structured_metadata(self) -> dict[str, Any]:
        """Filterable/rankable payload for this element.

        This is the counterpart to :meth:`best_text`: everything the retrieval
        layer may condition on, in structured form, kept strictly out of the
        embedded text.
        """
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "page_number": self.page_number,
            "element_id": self.element_id,
            "element_type": self.element_type.value,
            "extraction_method": self.extraction_method.value,
            "extraction_confidence": self.extraction_confidence,
        }
        if self.section:
            payload["section"] = self.section
        if self.subsection:
            payload["subsection"] = self.subsection
        if self.parent_id:
            payload["parent_id"] = self.parent_id
        if self.table is not None:
            payload["table_type"] = self.table.table_type.value
            payload["table_shape"] = list(self.table.shape)
        if self.figure is not None:
            payload["figure_type"] = self.figure.figure_type.value
            payload["has_derived_text"] = self.figure.has_derived_text
        return payload


# ---------------------------------------------------------------------------
# Chunk level
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A retrieval unit: the thing that gets embedded, scored, and cited.

    Built in Step 3, but defined here because its provenance contract belongs
    with the rest of the chain: a chunk must always name the elements it came
    from, so a hit resolves back to a region of a page.
    """

    chunk_id: str
    doc_id: str
    page_number: int = Field(ge=1)
    chunk_type: ChunkType
    text: str
    element_ids: list[str] = Field(default_factory=list)
    # Union of the source elements' boxes: the region to highlight for a citation.
    bbox: BBox | None = None
    section: str | None = None
    subsection: str | None = None
    token_count: int | None = None
    variant: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _elements_belong_to_this_chunk(self) -> Chunk:
        """A chunk may not silently mix documents or pages.

        Both would break citation: the chunk would claim a single (doc, page)
        that only some of its evidence actually came from.
        """
        for eid in self.element_ids:
            try:
                doc_id, page_number = parse_element_id(eid)
            except ValueError:
                continue  # synthetic ids in tests; the id format check is elsewhere
            if doc_id != self.doc_id:
                raise ValueError(
                    f"chunk {self.chunk_id} (doc {self.doc_id}) contains element {eid} "
                    f"from doc {doc_id}"
                )
            if page_number != self.page_number:
                raise ValueError(
                    f"chunk {self.chunk_id} (page {self.page_number}) contains element {eid} "
                    f"from page {page_number}"
                )
        return self


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """A fully resolved trace: Document -> Page -> Element -> Chunk.

    Produced by :func:`build_provenance`. This is what a citation renders from
    and what error analysis groups by.
    """

    doc_id: str
    doc_title: str
    page_number: int
    page_id: str
    chunk_id: str | None = None
    element_ids: list[str] = Field(default_factory=list)
    element_types: list[ElementType] = Field(default_factory=list)
    bbox: BBox | None = None
    section: str | None = None
    subsection: str | None = None
    page_image_path: str | None = None
    extraction_methods: list[ExtractionMethod] = Field(default_factory=list)
    min_extraction_confidence: float | None = None

    def human_label(self) -> str:
        base = f"{self.doc_title}, p. {self.page_number}"
        return f"{base} ({self.section})" if self.section else base


def build_provenance(
    chunk: Chunk,
    *,
    document: Document,
    page: Page,
    elements: dict[str, Element] | None = None,
) -> Provenance:
    """Resolve a chunk back to its document, page, and source regions.

    ``elements`` maps element_id -> Element. Ids that are missing from it are
    tolerated (an index may outlive an element table), but everything derived
    from them -- bbox, types, confidence -- is then simply absent rather than
    guessed at.
    """
    if page.doc_id != document.doc_id:
        raise ValueError(f"page {page.page_id} does not belong to document {document.doc_id}")
    if chunk.doc_id != document.doc_id:
        raise ValueError(f"chunk {chunk.chunk_id} does not belong to document {document.doc_id}")
    if chunk.page_number != page.page_number:
        raise ValueError(
            f"chunk {chunk.chunk_id} is on page {chunk.page_number}, "
            f"but page {page.page_id} was supplied"
        )

    resolved = [elements[eid] for eid in chunk.element_ids if elements and eid in elements]
    boxes = [e.bbox for e in resolved if e.bbox is not None]

    return Provenance(
        doc_id=document.doc_id,
        doc_title=document.title,
        page_number=page.page_number,
        page_id=page.page_id,
        chunk_id=chunk.chunk_id,
        element_ids=list(chunk.element_ids),
        element_types=[e.element_type for e in resolved],
        # Prefer the chunk's own box; fall back to the union of its elements'.
        bbox=chunk.bbox or BBox.union_all(boxes),
        section=chunk.section or page.section,
        subsection=chunk.subsection or page.subsection,
        page_image_path=page.image_path,
        extraction_methods=sorted({e.extraction_method for e in resolved}, key=lambda m: m.value),
        min_extraction_confidence=(
            min(e.extraction_confidence for e in resolved) if resolved else None
        ),
    )


# ---------------------------------------------------------------------------
# Retrieval (defined here so the provenance contract stays in one place;
# the retrieval *logic* lives in mmrag.retrieval)
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
    section: str | None = None
    snippet: str | None = None

    @classmethod
    def from_provenance(cls, prov: Provenance, *, snippet: str | None = None) -> Citation:
        return cls(
            doc_id=prov.doc_id,
            doc_title=prov.doc_title,
            page_number=prov.page_number,
            element_ids=prov.element_ids,
            chunk_id=prov.chunk_id,
            bbox=prov.bbox,
            section=prov.section,
            snippet=snippet,
        )

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
