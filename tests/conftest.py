"""Shared fixtures.

The synthetic PDF built here is the workhorse for ingestion tests. Building a
document with known structure -- a known number of pages, a ruled table at a
known position, a figure with a known caption, a footer repeated on every page
-- means the parser's output can be asserted against ground truth rather than
against whatever it happened to produce last time.

Tests that need the real corpus are marked ``slow`` and skip when it is absent,
so a fresh clone can run the suite before downloading 45 MB of PDFs.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from mmrag.corpus.manifest import CorpusEntry
from mmrag.schemas import (
    BBox,
    Chunk,
    ChunkType,
    Document,
    DocumentType,
    Element,
    ElementType,
    Page,
    make_element_id,
    make_page_id,
)

# Page geometry of the synthetic document, in PDF points (A4-ish).
PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
N_SYNTHETIC_PAGES = 3

FOOTER_TEMPLATE = "Synthetic Report 2024 | Page {n}"
FIGURE_CAPTION = "Figure 1: Quarterly revenue by region."
TABLE_CAPTION = "Table 1: Headcount by department."


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small PDF with known text, a ruled table, a figure, and a running footer."""
    path = tmp_path_factory.mktemp("pdfs") / "synthetic.pdf"
    doc = pymupdf.open()

    for page_number in range(1, N_SYNTHETIC_PAGES + 1):
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

        # Section heading, set noticeably larger than body text so the
        # SectionTracker's relative-size rule has something to find.
        page.insert_text((72, 90), f"Section {page_number}: Findings", fontsize=20)
        page.insert_text(
            (72, 130),
            "Revenue grew across every region during the period under review.",
            fontsize=10,
        )
        page.insert_text(
            (72, 150),
            "Operating costs remained broadly flat year on year.",
            fontsize=10,
        )

        # Running footer: identical on every page apart from the number, which
        # is exactly what header/footer detection keys on.
        page.insert_text((72, 810), FOOTER_TEMPLATE.format(n=page_number), fontsize=8)

        if page_number == 2:
            _draw_table(page)
        if page_number == 3:
            _draw_figure(page)

    doc.save(path)
    doc.close()
    return path


def _draw_table(page: pymupdf.Page) -> None:
    """A ruled 3x3 table, the layout PyMuPDF's 'lines' strategy detects."""
    x0, y0, cell_w, cell_h = 72.0, 300.0, 120.0, 24.0
    headers = ["Department", "Headcount", "Change"]
    rows = [["Engineering", "412", "+18"], ["Sales", "233", "-4"]]

    for row_index, values in enumerate([headers, *rows]):
        for col_index, value in enumerate(values):
            left = x0 + col_index * cell_w
            top = y0 + row_index * cell_h
            page.draw_rect(pymupdf.Rect(left, top, left + cell_w, top + cell_h), width=0.8)
            page.insert_text((left + 4, top + 16), value, fontsize=9)

    page.insert_text((x0, y0 + 3 * cell_h + 22), TABLE_CAPTION, fontsize=9)


def _draw_figure(page: pymupdf.Page) -> None:
    """A vector 'chart': enough primitives to survive cluster filtering."""
    x0, y0 = 150.0, 300.0
    page.draw_rect(pymupdf.Rect(x0, y0, x0 + 300, y0 + 200), width=1.0)
    for i in range(8):
        left = x0 + 20 + i * 34
        height = 20 + i * 18
        page.draw_rect(
            pymupdf.Rect(left, y0 + 180 - height, left + 24, y0 + 180),
            color=(0.1, 0.3, 0.7),
            fill=(0.1, 0.3, 0.7),
        )
    page.insert_text((x0, y0 + 225), FIGURE_CAPTION, fontsize=9)


@pytest.fixture
def synthetic_entry(synthetic_pdf: Path) -> CorpusEntry:
    """A manifest entry pointing at the synthetic PDF."""
    return CorpusEntry(
        doc_id="synthetic",
        title="Synthetic Report 2024",
        url="https://example.invalid/synthetic.pdf",
        publisher="Test Publisher",
        category="finance",
        license="CC0",
        modality_profile=["tables", "charts", "dense_text"],
        sha256="a" * 64,
        size_bytes=synthetic_pdf.stat().st_size,
    )


@pytest.fixture
def parsed_synthetic(synthetic_entry: CorpusEntry, synthetic_pdf: Path, tmp_path: Path):
    """The synthetic PDF, parsed. The default subject of the ingestion tests."""
    from mmrag.config import load_experiment_config
    from mmrag.ingestion.parser import PdfParser

    config = load_experiment_config("method1").ingestion
    parser = PdfParser(config, output_dir=tmp_path / "processed")
    return parser.parse(synthetic_entry, synthetic_pdf)


# ---------------------------------------------------------------------------
# Hand-built records, for tests that need exact control rather than a parse
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> Document:
    return Document(
        doc_id="doc1",
        title="Sample Document",
        source="Test Publisher",
        source_url="https://example.invalid/doc.pdf",
        authors=["A. Author"],
        organization="Test Publisher",
        doc_type=DocumentType.RESEARCH_PAPER,
        domain="science",
        language="en",
        license="CC BY 4.0",
        file_name="doc1.pdf",
        file_path="/tmp/doc1.pdf",
        sha256="b" * 64,
        n_pages=10,
        n_pages_ingested=10,
    )


@pytest.fixture
def sample_page() -> Page:
    return Page(
        page_id=make_page_id("doc1", 3),
        doc_id="doc1",
        page_number=3,
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
        image_path="/tmp/doc1/pages/p0003.png",
        section="Results",
        subsection="Ablations",
    )


@pytest.fixture
def sample_elements() -> dict[str, Element]:
    figure = Element(
        element_id=make_element_id("doc1", 3, "chart", 1),
        doc_id="doc1",
        page_id=make_page_id("doc1", 3),
        page_number=3,
        element_type=ElementType.CHART,
        reading_order=1,
        bbox=BBox(x0=0.1, y0=0.2, x1=0.6, y1=0.5),
        section="Results",
    )
    caption = Element(
        element_id=make_element_id("doc1", 3, "caption", 1),
        doc_id="doc1",
        page_id=make_page_id("doc1", 3),
        page_number=3,
        element_type=ElementType.CAPTION,
        parent_id=figure.element_id,
        reading_order=2,
        bbox=BBox(x0=0.1, y0=0.52, x1=0.6, y1=0.55),
        text="Figure 1: A chart.",
        section="Results",
    )
    return {e.element_id: e for e in (figure, caption)}


@pytest.fixture
def sample_chunk(sample_elements: dict[str, Element]) -> Chunk:
    return Chunk(
        chunk_id="method1#abc123",
        doc_id="doc1",
        page_number=3,
        chunk_type=ChunkType.FIGURE,
        text="Figure 1: A chart.",
        element_ids=list(sample_elements),
        variant="method1",
    )


@pytest.fixture(scope="session")
def real_corpus_available() -> bool:
    from mmrag.config import RAW_DIR

    return (RAW_DIR / "who_covid_sitrep_001.pdf").exists()
