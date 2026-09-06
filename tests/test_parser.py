"""End-to-end parser tests against a document with known ground truth.

The synthetic fixture has a deliberately known structure -- three pages, a ruled
table on page 2, a vector figure with a marked caption on page 3, and a footer
repeated on every page -- so these assert what the parser *should* find, not
whatever it found last time.

The metadata-consistency class is the important one: it checks the invariants
that hold the Document -> Page -> Element chain together. Every one of them is
something that, if it silently broke, would leave the pipeline apparently
working while producing uncitable evidence.
"""

from __future__ import annotations

from collections import Counter

import pytest
from tests.conftest import FIGURE_CAPTION, N_SYNTHETIC_PAGES, TABLE_CAPTION

from mmrag.ingestion.pipeline import read_sidecar, write_sidecar
from mmrag.schemas import ElementType, ExtractionMethod


class TestDocumentMetadata:
    def test_manifest_metadata_is_carried_through(self, parsed_synthetic):
        doc = parsed_synthetic.document
        assert doc.doc_id == "synthetic"
        assert doc.title == "Synthetic Report 2024"
        assert doc.source_url == "https://example.invalid/synthetic.pdf"
        assert doc.organization == "Test Publisher"
        assert doc.domain == "finance"
        assert doc.license == "CC0"
        assert doc.language == "en"

    def test_file_identity_is_recorded(self, parsed_synthetic):
        doc = parsed_synthetic.document
        assert doc.file_name == "synthetic.pdf"
        assert len(doc.sha256) == 64
        assert doc.parser_version and "pymupdf" in doc.parser_version
        assert doc.ingested_at is not None

    def test_page_counts_agree_with_what_was_parsed(self, parsed_synthetic):
        doc = parsed_synthetic.document
        assert doc.n_pages == N_SYNTHETIC_PAGES
        assert doc.n_pages_ingested == len(parsed_synthetic.pages)
        assert not doc.is_truncated

    def test_field_sources_are_recorded_for_traceability(self, parsed_synthetic):
        """Without this, a wrong title cannot be traced to what produced it."""
        sources = parsed_synthetic.document.metadata["field_sources"]
        assert sources["title"] == "manifest"
        assert "doc_type" in sources

    def test_page_limit_truncates_and_is_reported(self, synthetic_entry, synthetic_pdf, tmp_path):
        from mmrag.config import load_experiment_config
        from mmrag.ingestion.parser import PdfParser

        synthetic_entry.page_limit = 2
        config = load_experiment_config("method1").ingestion
        parsed = PdfParser(config, output_dir=tmp_path / "p").parse(synthetic_entry, synthetic_pdf)

        assert len(parsed.pages) == 2
        assert parsed.document.n_pages == N_SYNTHETIC_PAGES
        assert parsed.document.n_pages_ingested == 2
        assert parsed.document.is_truncated


class TestPageMetadata:
    def test_pages_are_numbered_from_one_without_gaps(self, parsed_synthetic):
        numbers = [p.page_number for p in parsed_synthetic.pages]
        assert numbers == list(range(1, N_SYNTHETIC_PAGES + 1))

    def test_geometry_is_recorded(self, parsed_synthetic):
        for page in parsed_synthetic.pages:
            assert page.width > 0 and page.height > 0
            assert page.rotation in (0, 90, 180, 270)
            assert not page.is_landscape

    def test_page_images_are_rendered_and_measured(self, parsed_synthetic):
        from pathlib import Path

        for page in parsed_synthetic.pages:
            assert page.image_path is not None, "Method 3 needs a page render"
            assert Path(page.image_path).exists()
            assert page.image_width and page.image_height
            assert page.image_dpi == 150

    def test_running_footer_is_detected(self, parsed_synthetic):
        """Text repeating on every page is furniture, not content."""
        footers = [p.footer_text for p in parsed_synthetic.pages]
        assert all(f and "Synthetic Report 2024" in f for f in footers), footers

    def test_section_is_carried_on_the_page(self, parsed_synthetic):
        assert any(p.section for p in parsed_synthetic.pages)

    def test_n_elements_matches_the_actual_count(self, parsed_synthetic):
        for page in parsed_synthetic.pages:
            actual = len(parsed_synthetic.elements_by_page(page.page_number))
            assert page.n_elements == actual, f"page {page.page_number} miscounts its elements"


class TestMetadataConsistency:
    """Cross-level invariants. These are the chain's structural integrity."""

    def test_every_element_belongs_to_the_parsed_document(self, parsed_synthetic):
        doc_id = parsed_synthetic.document.doc_id
        assert all(e.doc_id == doc_id for e in parsed_synthetic.elements)
        assert all(p.doc_id == doc_id for p in parsed_synthetic.pages)

    def test_every_element_page_id_resolves_to_a_real_page(self, parsed_synthetic):
        page_ids = {p.page_id for p in parsed_synthetic.pages}
        for element in parsed_synthetic.elements:
            assert element.page_id in page_ids, f"{element.element_id} has an orphan page_id"

    def test_element_page_number_agrees_with_its_page(self, parsed_synthetic):
        pages = {p.page_id: p for p in parsed_synthetic.pages}
        for element in parsed_synthetic.elements:
            assert element.page_number == pages[element.page_id].page_number

    def test_element_ids_are_unique(self, parsed_synthetic):
        ids = [e.element_id for e in parsed_synthetic.elements]
        duplicates = [i for i, n in Counter(ids).items() if n > 1]
        assert duplicates == [], f"duplicate element ids: {duplicates}"

    def test_page_ids_are_unique(self, parsed_synthetic):
        ids = [p.page_id for p in parsed_synthetic.pages]
        assert len(ids) == len(set(ids))

    def test_element_id_encodes_its_own_document_and_page(self, parsed_synthetic):
        """Provenance must be recoverable from the id alone, without a lookup."""
        from mmrag.schemas import parse_element_id

        for element in parsed_synthetic.elements:
            assert parse_element_id(element.element_id) == (element.doc_id, element.page_number)

    def test_reading_order_is_unique_within_a_page(self, parsed_synthetic):
        for page in parsed_synthetic.pages:
            orders = [e.reading_order for e in parsed_synthetic.elements_by_page(page.page_number)]
            assert len(orders) == len(set(orders)), f"page {page.page_number} has tied ordering"

    def test_every_element_declares_an_extraction_method_and_confidence(self, parsed_synthetic):
        for element in parsed_synthetic.elements:
            assert isinstance(element.extraction_method, ExtractionMethod)
            assert 0.0 <= element.extraction_confidence <= 1.0

    def test_structured_payloads_match_element_types(self, parsed_synthetic):
        for element in parsed_synthetic.elements:
            if element.table is not None:
                assert element.element_type is ElementType.TABLE
            if element.figure is not None:
                assert element.element_type.is_visual


class TestExtractedContent:
    def test_body_text_is_recovered(self, parsed_synthetic):
        text = " ".join(e.text or "" for e in parsed_synthetic.elements)
        assert "Revenue grew across every region" in text
        assert "Operating costs remained broadly flat" in text

    def test_headings_are_detected_as_titles_not_body(self, parsed_synthetic):
        headings = [
            e
            for e in parsed_synthetic.elements
            if e.element_type in (ElementType.TITLE, ElementType.HEADING)
        ]
        assert any("Findings" in (e.text or "") for e in headings)

    def test_the_ruled_table_is_found_and_structured(self, parsed_synthetic):
        tables = [e for e in parsed_synthetic.elements if e.element_type is ElementType.TABLE]
        assert len(tables) == 1, f"expected exactly one table, got {len(tables)}"

        table = tables[0].table
        assert table is not None
        assert tables[0].page_number == 2
        assert table.n_cols == 3
        assert "Department" in table.columns
        assert not table.is_degenerate
        # Both representations must be present: Method 1 reads the Markdown,
        # Method 2 reads the structure.
        assert "Engineering" in table.markdown
        assert any("412" in (c or "") for row in table.rows for c in row)

    def test_the_table_caption_is_attached(self, parsed_synthetic):
        tables = [e for e in parsed_synthetic.elements if e.element_type is ElementType.TABLE]
        assert tables[0].caption is not None
        assert "Headcount by department" in tables[0].caption

    def test_the_vector_figure_is_found_with_its_caption(self, parsed_synthetic):
        figures = [e for e in parsed_synthetic.elements if e.element_type.is_visual]
        assert figures, "the vector bar chart on page 3 must be detected"

        captioned = [f for f in figures if f.caption and "Quarterly revenue" in f.caption]
        assert captioned, f"captions found: {[f.caption for f in figures]}"

        figure = captioned[0]
        assert figure.page_number == 3
        assert figure.figure is not None
        assert figure.extraction_method is ExtractionMethod.PYMUPDF_DRAWING
        assert figure.metadata["is_vector"] is True

    def test_the_figure_caption_becomes_a_child_element(self, parsed_synthetic):
        captions = [
            e
            for e in parsed_synthetic.elements
            if e.element_type is ElementType.CAPTION and e.parent_id
        ]
        texts = [c.text for c in captions]
        assert any(FIGURE_CAPTION in (t or "") for t in texts), texts
        assert any(TABLE_CAPTION in (t or "") for t in texts), texts

    def test_table_cell_text_is_not_also_emitted_as_prose(self, parsed_synthetic):
        """Cells live in table_data; emitting them twice double-counts evidence."""
        prose = [
            e.text
            for e in parsed_synthetic.elements
            if e.element_type is ElementType.TEXT and e.text
        ]
        assert not any("Engineering" in t for t in prose)

    def test_footer_elements_are_typed_as_footers(self, parsed_synthetic):
        footers = [e for e in parsed_synthetic.elements if e.element_type is ElementType.FOOTER]
        assert len(footers) == N_SYNTHETIC_PAGES
        assert all(e.element_type.is_boilerplate for e in footers)


class TestDeterminism:
    def test_parsing_twice_produces_identical_output(
        self, synthetic_entry, synthetic_pdf, tmp_path
    ):
        """Stable ids are what let a gold set survive a re-ingest."""
        from mmrag.config import load_experiment_config
        from mmrag.ingestion.parser import PdfParser

        config = load_experiment_config("method1").ingestion
        first = PdfParser(config, output_dir=tmp_path / "a").parse(synthetic_entry, synthetic_pdf)
        second = PdfParser(config, output_dir=tmp_path / "b").parse(synthetic_entry, synthetic_pdf)

        assert [e.element_id for e in first.elements] == [e.element_id for e in second.elements]
        assert [e.element_type for e in first.elements] == [e.element_type for e in second.elements]
        assert [e.bbox for e in first.elements] == [e.bbox for e in second.elements]
        assert [e.parent_id for e in first.elements] == [e.parent_id for e in second.elements]


class TestSidecarRoundTrip:
    def test_round_trip_preserves_every_record(self, parsed_synthetic, tmp_path):
        path = write_sidecar(parsed_synthetic, tmp_path / "parsed.json")
        restored = read_sidecar(path)

        assert restored.document == parsed_synthetic.document
        assert restored.pages == parsed_synthetic.pages
        assert restored.elements == parsed_synthetic.elements

    def test_round_trip_preserves_structured_payloads(self, parsed_synthetic, tmp_path):
        """The nested table/figure payloads are the part most likely to be lost."""
        restored = read_sidecar(write_sidecar(parsed_synthetic, tmp_path / "p.json"))
        original_tables = [e.table for e in parsed_synthetic.elements if e.table]
        restored_tables = [e.table for e in restored.elements if e.table]
        assert original_tables == restored_tables and original_tables

    def test_a_sidecar_from_another_version_is_rejected(self, parsed_synthetic, tmp_path):
        """Silently trusting a stale sidecar would hide a parser change."""
        import json

        path = write_sidecar(parsed_synthetic, tmp_path / "parsed.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["sidecar_version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="sidecar version"):
            read_sidecar(path)


@pytest.mark.slow
class TestRealCorpus:
    """Spot checks against real documents, skipped without the corpus."""

    def test_transformer_figure_is_found_with_its_caption(self, real_corpus_available, tmp_path):
        if not real_corpus_available:
            pytest.skip("corpus not downloaded; run 'mmrag corpus download'")

        from mmrag.config import RAW_DIR, load_experiment_config
        from mmrag.corpus import CorpusManifest
        from mmrag.ingestion.parser import PdfParser

        entry = CorpusManifest.load().get("arxiv_attention")
        pdf = entry.local_path(RAW_DIR)
        if not pdf.exists():
            pytest.skip("arxiv_attention not downloaded")

        config = load_experiment_config("method1").ingestion
        parsed = PdfParser(config, output_dir=tmp_path / "p").parse(entry, pdf)

        # The canonical "the answer is only in the figure" case.
        diagrams = [
            e
            for e in parsed.elements
            if e.element_type.is_visual and e.caption and "Transformer" in e.caption
        ]
        assert diagrams, "Figure 1 of the Transformer paper must be detected"
        assert diagrams[0].page_number == 3

    def test_who_report_prose_is_not_swallowed_by_a_panel(self, real_corpus_available, tmp_path):
        """Regression: a filled background panel was detected as a chart,
        hiding the entire Resources section inside it."""
        if not real_corpus_available:
            pytest.skip("corpus not downloaded")

        from mmrag.config import RAW_DIR, load_experiment_config
        from mmrag.corpus import CorpusManifest
        from mmrag.ingestion.parser import PdfParser

        entry = CorpusManifest.load().get("who_covid_sitrep_001")
        config = load_experiment_config("method1").ingestion
        parsed = PdfParser(config, output_dir=tmp_path / "p").parse(
            entry, entry.local_path(RAW_DIR)
        )

        all_text = " ".join(e.text or "" for e in parsed.elements)
        assert "Technical interim guidance" in all_text
