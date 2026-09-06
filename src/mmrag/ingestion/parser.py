"""PyMuPDF document parser: PDF in, Document/Page/Element records out.

Runs in two passes over every document, and the two-pass structure is load
bearing rather than incidental:

* **Pass 1** collects typography and page-furniture statistics across the whole
  document. A single page cannot tell a running header from a section heading,
  nor a 14pt heading from 14pt body text -- both questions are only answerable
  once you have seen the rest of the document.
* **Pass 2** builds elements using those document-level statistics.

The parser emits no retrieval-specific structures. It produces the shared
representation; how it gets chunked and indexed is each method's business.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from mmrag.config import IngestionConfig
from mmrag.corpus.manifest import CorpusEntry
from mmrag.ingestion.classify import text_confidence
from mmrag.ingestion.figures import (
    CAPTION_RE,
    FigureCandidate,
    build_figure_data,
    cluster_vector_drawings,
    element_type_for,
    find_caption,
    text_coverage,
)
from mmrag.ingestion.layout import (
    HeaderFooterDetector,
    LayoutBlock,
    SectionTracker,
    assign_reading_order,
    detect_columns,
)
from mmrag.ingestion.metadata import build_document
from mmrag.ingestion.tables import build_table_data, confidence_for
from mmrag.ingestion.text_utils import collapse_whitespace, normalize_text, text_quality
from mmrag.logging_utils import get_logger
from mmrag.schemas import (
    BBox,
    Document,
    Element,
    ElementType,
    ExtractionMethod,
    Page,
    make_element_id,
    make_page_id,
)

log = get_logger(__name__)

PARSER_VERSION = f"pymupdf-{pymupdf.__version__}/mmrag-1"

# PyMuPDF span flag bit for a bold face.
_BOLD_FLAG = 1 << 4

# A block whose text is shorter than this is treated as a stray mark unless it
# sits inside a table or carries a caption marker.
_MIN_BLOCK_CHARS = 2

# A vector cluster whose area is more than this fraction covered by text blocks
# is a callout/background panel rather than a figure. Charts carry axis labels
# and legends, but those cover far less of the plot area than prose does.
_PANEL_TEXT_COVERAGE = 0.35


@dataclass
class ParsedDocument:
    """Everything one PDF produced."""

    document: Document
    pages: list[Page]
    elements: list[Element]

    def elements_by_page(self, page_number: int) -> list[Element]:
        return [e for e in self.elements if e.page_number == page_number]

    @property
    def stats(self) -> dict[str, Any]:
        by_type = Counter(e.element_type.value for e in self.elements)
        confidences = [e.extraction_confidence for e in self.elements]
        return {
            "n_pages": len(self.pages),
            "n_elements": len(self.elements),
            "by_type": dict(sorted(by_type.items())),
            "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
            "n_low_confidence": sum(1 for c in confidences if c < 0.5),
        }


class PdfParser:
    """Parses one PDF into the shared Document/Page/Element representation."""

    def __init__(self, config: IngestionConfig, *, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    # -- public API ---------------------------------------------------------

    def parse(self, entry: CorpusEntry, pdf_path: Path) -> ParsedDocument:
        # Rendered artefacts are keyed by position on the page, so a re-parse
        # that detects fewer figures would otherwise leave orphaned crops behind
        # that no element references -- confusing to debug and slowly filling
        # the disk. Clear them so the directory always matches the sidecar.
        self._clear_renders(entry.doc_id)

        doc = pymupdf.open(pdf_path)
        try:
            n_pages = doc.page_count
            limit = min(
                n_pages,
                entry.page_limit or n_pages,
                self.config.max_pages_per_doc or n_pages,
            )

            document = build_document(
                entry,
                file_path=pdf_path,
                pdf_metadata=doc.metadata or {},
                n_pages=n_pages,
                n_pages_ingested=limit,
                parser_version=PARSER_VERSION,
            )

            # --- pass 1: document-level statistics --------------------------
            furniture = HeaderFooterDetector()
            sections = SectionTracker()
            page_blocks: list[list[LayoutBlock]] = []
            for page_index in range(limit):
                blocks = self._text_blocks(doc[page_index])
                page_blocks.append(blocks)
                furniture.observe(blocks)
                sections.observe(blocks)
            furniture.finalize()
            sections.finalize()
            sections.reset_document()

            # --- pass 2: build the records ----------------------------------
            pages: list[Page] = []
            elements: list[Element] = []
            for page_index in range(limit):
                page, page_elements = self._build_page(
                    doc[page_index],
                    document=document,
                    blocks=page_blocks[page_index],
                    furniture=furniture,
                    sections=sections,
                )
                pages.append(page)
                elements.extend(page_elements)

            return ParsedDocument(document=document, pages=pages, elements=elements)
        finally:
            doc.close()

    # -- pass 1 -------------------------------------------------------------

    def _text_blocks(self, page: pymupdf.Page) -> list[LayoutBlock]:
        """Extract text blocks with the typography needed for layout analysis."""
        width, height = page.rect.width, page.rect.height
        if width <= 0 or height <= 0:
            return []

        raw = page.get_text("dict", sort=True)
        blocks: list[LayoutBlock] = []
        for index, block in enumerate(raw.get("blocks", [])):
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue
            spans = [s for line in block.get("lines", []) for s in line.get("spans", [])]
            if not spans:
                continue
            text = normalize_text("".join(s.get("text", "") for s in spans))
            if len(text) < _MIN_BLOCK_CHARS:
                continue

            # Dominant size weighted by characters, so a superscript marker
            # cannot outvote the line it is attached to.
            sizes: Counter[float] = Counter()
            for span in spans:
                sizes[round(float(span.get("size", 0.0)), 1)] += len(span.get("text", ""))
            font_size = sizes.most_common(1)[0][0] if sizes else 0.0
            dominant = max(spans, key=lambda s: len(s.get("text", "")))

            blocks.append(
                LayoutBlock(
                    bbox=BBox.from_absolute(*block["bbox"], width=width, height=height),
                    text=text,
                    font_size=font_size,
                    font_name=str(dominant.get("font", "")),
                    is_bold=bool(int(dominant.get("flags", 0)) & _BOLD_FLAG)
                    or "bold" in str(dominant.get("font", "")).lower(),
                    block_index=index,
                )
            )
        return blocks

    # -- pass 2 -------------------------------------------------------------

    def _build_page(
        self,
        page: pymupdf.Page,
        *,
        document: Document,
        blocks: list[LayoutBlock],
        furniture: HeaderFooterDetector,
        sections: SectionTracker,
    ) -> tuple[Page, list[Element]]:
        page_number = page.number + 1
        page_id = make_page_id(document.doc_id, page_number)
        width, height = page.rect.width, page.rect.height

        image_path, image_size = self._render_page(page, document.doc_id, page_number)

        # Order matters. Tables are detected and *validated* first, because a
        # rejected table's text must stay available as ordinary blocks; then
        # figures are detected knowing where the surviving tables are, because
        # a table's ruling lines are vector primitives and would otherwise
        # cluster into a phantom "chart" covering the very same region.
        tables = self._extract_tables(page, width, height) if self.config.extract_tables else []
        figures = (
            self._detect_figures(
                page, width, height, blocks=blocks, exclude=[t["bbox"] for t in tables]
            )
            if self.config.extract_figures
            else []
        )

        # Text inside a *table* is already carried by table_data.rows, so
        # emitting it again would double-count the same evidence and inflate
        # every recall number. Text inside a *figure* has no other
        # representation until an OCR or VLM pass runs, so it is kept and
        # parented to the figure instead -- dropping it would silently delete
        # real prose whenever figure detection over-reaches.
        table_regions = [t["bbox"] for t in tables]
        body_blocks = [b for b in blocks if not _is_contained(b.bbox, table_regions)]

        n_columns = detect_columns(blocks)
        ordered = assign_reading_order(body_blocks, n_columns=n_columns)

        header_text, footer_text = self._page_furniture(ordered, furniture)
        elements = self._build_elements(
            page=page,
            document=document,
            page_id=page_id,
            page_number=page_number,
            blocks=ordered,
            tables=tables,
            figures=figures,
            furniture=furniture,
            sections=sections,
            image_dpi=self.config.figure_image_dpi,
        )

        page_record = Page(
            page_id=page_id,
            doc_id=document.doc_id,
            page_number=page_number,
            width=width,
            height=height,
            rotation=page.rotation,
            image_path=str(image_path) if image_path else None,
            image_width=image_size[0] if image_size else None,
            image_height=image_size[1] if image_size else None,
            image_dpi=self.config.page_image_dpi if image_path else None,
            section=sections.section,
            subsection=sections.subsection,
            header_text=header_text,
            footer_text=footer_text,
            raw_text=normalize_text(page.get_text()) or None,
            n_elements=len(elements),
            metadata={
                "n_columns": n_columns,
                "n_text_blocks": len(blocks),
                "n_tables": len(tables),
                "n_figures": len(figures),
                "n_suppressed_blocks": len(blocks) - len(body_blocks),
            },
        )
        return page_record, elements

    # -- element construction ------------------------------------------------

    def _build_elements(
        self,
        *,
        page: pymupdf.Page,
        document: Document,
        page_id: str,
        page_number: int,
        blocks: list[LayoutBlock],
        tables: list[dict[str, Any]],
        figures: list[FigureCandidate],
        furniture: HeaderFooterDetector,
        sections: SectionTracker,
        image_dpi: int,
    ) -> list[Element]:
        ordinals: Counter[str] = Counter()
        elements: list[Element] = []
        # Captions are consumed by the figure/table they belong to, then emitted
        # as child elements; track them so they are not also emitted as body text.
        caption_blocks: dict[int, str] = {}

        def new_id(element_type: ElementType) -> str:
            ordinals[element_type.value] += 1
            return make_element_id(
                document.doc_id, page_number, element_type.value, ordinals[element_type.value]
            )

        # --- figures first: they claim their captions -------------------------
        # A caption belongs to exactly one object. Two figures stacked on a page
        # will both score the same nearby caption highly, and without this the
        # caption is duplicated into two elements with conflicting parents --
        # which then feeds the wrong text into figure-type classification.
        claimed: set[int] = set()

        def claim_caption(region: BBox) -> LayoutBlock | None:
            available = [b for b in blocks if id(b) not in claimed]
            block = find_caption(
                region,
                available,
                max_gap=self.config.caption_max_gap_ratio,
                body_font_size=sections.body_size,
            )
            if block is not None:
                claimed.add(id(block))
            return block

        figure_elements: list[tuple[Element, LayoutBlock | None]] = []
        for candidate in figures:
            caption_block = claim_caption(candidate.bbox)
            caption = collapse_whitespace(caption_block.text) if caption_block else None
            crop = self._render_figure(
                page, candidate, document.doc_id, page_number, len(figure_elements)
            )
            figure_data, confidence = build_figure_data(
                candidate,
                image_path=crop,
                caption=caption,
                dpi=image_dpi,
                page_area_ratio=candidate.bbox.area,
            )
            element_type = element_type_for(figure_data.figure_type)
            element_id = new_id(element_type)
            if caption_block is not None:
                caption_blocks[id(caption_block)] = element_id

            figure_elements.append(
                (
                    Element(
                        element_id=element_id,
                        doc_id=document.doc_id,
                        page_id=page_id,
                        page_number=page_number,
                        element_type=element_type,
                        bbox=candidate.bbox,
                        section=sections.section,
                        subsection=sections.subsection,
                        extraction_method=(
                            ExtractionMethod.PYMUPDF_DRAWING
                            if candidate.is_vector
                            else ExtractionMethod.PYMUPDF_IMAGE
                        ),
                        extraction_confidence=confidence,
                        caption=caption,
                        figure=figure_data,
                        metadata={
                            "is_vector": candidate.is_vector,
                            "n_primitives": candidate.n_primitives,
                            "area_ratio": round(candidate.bbox.area, 5),
                        },
                    ),
                    caption_block,
                )
            )

        # --- tables -----------------------------------------------------------
        table_elements: list[tuple[Element, LayoutBlock | None]] = []
        for table in tables:
            caption_block = claim_caption(table["bbox"])
            caption = collapse_whitespace(caption_block.text) if caption_block else None
            table_data, diagnostics = table["table_data"], table["diagnostics"]
            element_id = new_id(ElementType.TABLE)
            if caption_block is not None:
                caption_blocks[id(caption_block)] = element_id

            table_elements.append(
                (
                    Element(
                        element_id=element_id,
                        doc_id=document.doc_id,
                        page_id=page_id,
                        page_number=page_number,
                        element_type=ElementType.TABLE,
                        bbox=table["bbox"],
                        section=sections.section,
                        subsection=sections.subsection,
                        extraction_method=ExtractionMethod.PYMUPDF_TABLE,
                        extraction_confidence=confidence_for(table_data, diagnostics),
                        caption=caption,
                        table=table_data,
                        metadata=diagnostics,
                    ),
                    caption_block,
                )
            )

        # --- text blocks -------------------------------------------------------
        for block in blocks:
            if id(block) in caption_blocks:
                continue  # emitted below, parented to its figure/table

            kind = furniture.classify(block)
            if kind == "header":
                element_type = ElementType.HEADER
            elif kind == "footer":
                element_type = ElementType.FOOTER
            else:
                # Section state advances in reading order, so an element's
                # section is the heading that most recently preceded it.
                level = sections.update(block)
                if level == 1:
                    element_type = ElementType.TITLE
                elif level == 2:
                    element_type = ElementType.HEADING
                elif CAPTION_RE.match(block.text):
                    element_type = ElementType.CAPTION
                else:
                    element_type = ElementType.TEXT

            # Text lying inside a figure (axis labels, in-plot annotations) is
            # kept as its own element but parented to that figure, so it stays
            # retrievable while its relationship remains explicit.
            owner = next(
                (
                    parent.element_id
                    for parent, _ in figure_elements
                    if parent.bbox is not None
                    and parent.bbox.intersection_area(block.bbox) / max(block.bbox.area, 1e-9)
                    >= 0.7
                ),
                None,
            )

            quality = text_quality(block.text)
            elements.append(
                Element(
                    element_id=new_id(element_type),
                    doc_id=document.doc_id,
                    page_id=page_id,
                    page_number=page_number,
                    element_type=element_type,
                    parent_id=owner,
                    reading_order=block.reading_order,
                    bbox=block.bbox,
                    section=sections.section,
                    subsection=sections.subsection,
                    extraction_method=ExtractionMethod.PYMUPDF_TEXT,
                    extraction_confidence=text_confidence(
                        replacement_ratio=quality.replacement_ratio,
                        n_chars=quality.n_chars,
                        alpha_ratio=quality.alpha_ratio,
                    ),
                    text=block.text,
                    metadata={
                        "font_size": block.font_size,
                        "font_name": block.font_name,
                        "is_bold": block.is_bold,
                        "column": block.column,
                        "replacement_ratio": round(quality.replacement_ratio, 4),
                    },
                )
            )

        # --- figures/tables and their caption children ------------------------
        # A captioned object is placed immediately *before* its caption, using a
        # fractional key that the dense renumbering below resolves. Sharing the
        # caption's integer order would leave two elements tied, which makes
        # "the elements either side of this hit" ambiguous for context expansion.
        sort_key: dict[str, float] = {e.element_id: float(e.reading_order) for e in elements}
        next_order = float(len(blocks))
        for parent, caption_block in figure_elements + table_elements:
            if caption_block is not None:
                sort_key[parent.element_id] = caption_block.reading_order - 0.5
            else:
                sort_key[parent.element_id] = next_order
                next_order += 1
            elements.append(parent)

            if caption_block is not None:
                caption_element = Element(
                    element_id=new_id(ElementType.CAPTION),
                    doc_id=document.doc_id,
                    page_id=page_id,
                    page_number=page_number,
                    element_type=ElementType.CAPTION,
                    parent_id=parent.element_id,
                    reading_order=caption_block.reading_order,
                    bbox=caption_block.bbox,
                    section=sections.section,
                    subsection=sections.subsection,
                    extraction_method=ExtractionMethod.PYMUPDF_TEXT,
                    extraction_confidence=1.0,
                    text=caption_block.text,
                    metadata={"caption_for": parent.element_type.value},
                )
                sort_key[caption_element.element_id] = float(caption_block.reading_order)
                elements.append(caption_element)

        # Renumber densely from 0 so reading_order is a unique, gapless index
        # within the page -- the property context expansion relies on.
        elements.sort(key=lambda e: (sort_key[e.element_id], e.element_id))
        for position, element in enumerate(elements):
            element.reading_order = position
        return elements

    # -- extraction helpers ---------------------------------------------------

    def _extract_tables(
        self, page: pymupdf.Page, width: float, height: float
    ) -> list[dict[str, Any]]:
        """Detect tables via ruling lines.

        The 'text' strategy was evaluated and rejected: on a financial report it
        swallows the entire page, headings included, into one 66x13 pseudo-table.
        'lines' under-detects instead, which is the safer failure -- a missed
        table still has its text captured as ordinary blocks.
        """
        try:
            found = page.find_tables(strategy="lines")
        except Exception as exc:  # pragma: no cover - depends on the PDF
            log.debug("table detection failed on page %d: %s", page.number + 1, exc)
            return []

        out: list[dict[str, Any]] = []
        for table in found.tables:
            try:
                grid = table.extract()
            except Exception as exc:  # pragma: no cover
                log.debug("table extraction failed on page %d: %s", page.number + 1, exc)
                continue
            if not grid:
                continue

            # Build the structured form now, so a detection that turns out to be
            # a ruled callout rather than a table can be rejected here. Keeping
            # it would put a 27x1 pseudo-table into Method 2's table retriever
            # *and* swallow the real text inside it, which is the worse harm.
            table_data, diagnostics = build_table_data(
                grid, header_names=list(table.header.names) if table.header else None
            )
            if table_data.is_degenerate:
                log.debug(
                    "page %d: rejected %dx%d table (fill %.2f)",
                    page.number + 1,
                    table_data.n_rows,
                    table_data.n_cols,
                    table_data.fill_ratio,
                )
                continue

            out.append(
                {
                    "bbox": BBox.from_absolute(*table.bbox, width=width, height=height),
                    "table_data": table_data,
                    "diagnostics": diagnostics,
                }
            )
        return out

    def _detect_figures(
        self,
        page: pymupdf.Page,
        width: float,
        height: float,
        *,
        blocks: list[LayoutBlock],
        exclude: list[BBox] | None = None,
    ) -> list[FigureCandidate]:
        """Collect both embedded rasters and clustered vector artwork."""
        candidates: list[FigureCandidate] = []
        min_area = self.config.min_figure_area_ratio

        # --- embedded raster images -----------------------------------------
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 1:
                continue
            bbox = BBox.from_absolute(*block["bbox"], width=width, height=height)
            if bbox.area < min_area:
                continue
            candidates.append(
                FigureCandidate(
                    bbox=bbox,
                    is_vector=False,
                    native_width=block.get("width"),
                    native_height=block.get("height"),
                )
            )

        # --- vector artwork ---------------------------------------------------
        try:
            rects = [
                BBox.from_absolute(
                    d["rect"].x0,
                    d["rect"].y0,
                    d["rect"].x1,
                    d["rect"].y1,
                    width=width,
                    height=height,
                )
                for d in page.get_drawings()
                if d.get("rect") is not None and d["rect"].width > 0 and d["rect"].height > 0
            ]
        except Exception as exc:  # pragma: no cover
            log.debug("drawing extraction failed on page %d: %s", page.number + 1, exc)
            rects = []

        vector = cluster_vector_drawings(rects, min_area_ratio=min_area)
        excluded = exclude or []
        for candidate in vector:
            # A background panel or page border, not a figure.
            if candidate.bbox.area > 0.9:
                continue
            # A frame drawn around an embedded raster: the raster is the figure.
            if any(candidate.bbox.iou(existing.bbox) > 0.5 for existing in candidates):
                continue
            # A table's own ruling lines. Without this the same region is
            # emitted twice, once as a table and once as a phantom chart, and
            # every retrieval metric double-counts that evidence.
            if any(candidate.bbox.iou(region) > 0.5 for region in excluded):
                continue
            # A filled rectangle sitting behind prose: a callout panel, not a
            # figure. Accepting it would invent a figure *and* bury the text.
            coverage = text_coverage(candidate.bbox, blocks)
            if coverage > _PANEL_TEXT_COVERAGE:
                log.debug(
                    "page %d: rejected vector cluster as a text panel (coverage %.2f)",
                    page.number + 1,
                    coverage,
                )
                continue
            candidates.append(candidate)

        return candidates

    # -- rendering ------------------------------------------------------------

    def _render_page(
        self, page: pymupdf.Page, doc_id: str, page_number: int
    ) -> tuple[Path | None, tuple[int, int] | None]:
        """Render the full page. Method 3 retrieves over exactly these images."""
        target = self.output_dir / doc_id / "pages" / f"p{page_number:04d}.png"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            pixmap = page.get_pixmap(dpi=self.config.page_image_dpi)
            pixmap.save(target)
            return target, (pixmap.width, pixmap.height)
        except Exception as exc:  # pragma: no cover
            log.warning("page render failed for %s p%d: %s", doc_id, page_number, exc)
            return None, None

    def _render_figure(
        self,
        page: pymupdf.Page,
        candidate: FigureCandidate,
        doc_id: str,
        page_number: int,
        ordinal: int,
    ) -> Path | None:
        """Crop a figure at higher DPI than the page.

        Axis labels and legend text are the parts of a chart most likely to
        carry the answer, and they are illegible at page resolution.
        """
        target = self.output_dir / doc_id / "figures" / f"p{page_number:04d}_f{ordinal:03d}.png"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            rect = pymupdf.Rect(*candidate.bbox.to_absolute(page.rect.width, page.rect.height))
            pixmap = page.get_pixmap(clip=rect, dpi=self.config.figure_image_dpi)
            if pixmap.width < 8 or pixmap.height < 8:
                return None
            pixmap.save(target)
            return target
        except Exception as exc:  # pragma: no cover
            log.debug("figure crop failed for %s p%d: %s", doc_id, page_number, exc)
            return None

    def _clear_renders(self, doc_id: str) -> None:
        """Delete previously rendered pages and figure crops for one document."""
        for subdir in ("pages", "figures"):
            directory = self.output_dir / doc_id / subdir
            if not directory.is_dir():
                continue
            for path in directory.glob("*.png"):
                try:
                    path.unlink()
                except OSError as exc:  # pragma: no cover - locked file on Windows
                    log.debug("could not remove stale render %s: %s", path, exc)

    # -- misc ------------------------------------------------------------------

    @staticmethod
    def _page_furniture(
        blocks: list[LayoutBlock], furniture: HeaderFooterDetector
    ) -> tuple[str | None, str | None]:
        header = [b.text for b in blocks if furniture.classify(b) == "header"]
        footer = [b.text for b in blocks if furniture.classify(b) == "footer"]
        return (
            collapse_whitespace(" ".join(header)) or None,
            collapse_whitespace(" ".join(footer)) or None,
        )


def _is_contained(bbox: BBox, regions: list[BBox], *, min_overlap: float = 0.7) -> bool:
    """Whether a block lies mostly inside any of the given regions."""
    if bbox.area <= 0:
        return False
    return any(region.intersection_area(bbox) / bbox.area >= min_overlap for region in regions)
