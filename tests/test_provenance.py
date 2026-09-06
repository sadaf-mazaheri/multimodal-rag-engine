"""Provenance and metadata-integrity tests.

These are the tests that protect the architectural guarantee the whole benchmark
rests on: that any retrieval hit can be traced back through
``Chunk -> Element -> Page -> Document`` to a specific region of a specific
page. If that chain can silently break, citations become unverifiable and the
retrieval metrics in Step 6 measure nothing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mmrag.schemas import (
    BBox,
    Chunk,
    ChunkType,
    Citation,
    Document,
    Element,
    ElementType,
    ExtractionMethod,
    FigureData,
    Page,
    TableData,
    build_provenance,
    make_element_id,
    make_page_id,
    parse_element_id,
)


class TestIdentifierStability:
    """Ids must be reproducible: a gold set references them across re-ingests."""

    def test_element_id_is_deterministic(self):
        assert make_element_id("doc", 3, "table", 1) == make_element_id("doc", 3, "table", 1)

    def test_element_id_encodes_document_and_page(self):
        eid = make_element_id("mydoc", 42, "chart", 7)
        assert parse_element_id(eid) == ("mydoc", 42)

    def test_element_id_ordinal_is_zero_padded_for_sortability(self):
        ids = [make_element_id("d", 1, "text", n) for n in (1, 2, 10)]
        assert ids == sorted(ids), "lexicographic order must match numeric order"

    def test_page_id_round_trips(self):
        assert make_page_id("doc", 7) == "doc#p7"

    @pytest.mark.parametrize("bad", ["", "nodelimiters", "doc#pX#text001", "doc#"])
    def test_malformed_element_ids_are_rejected_loudly(self, bad):
        with pytest.raises(ValueError, match="malformed element_id"):
            parse_element_id(bad)


class TestBBoxNormalisation:
    """Boxes are normalised so they survive any rendering DPI."""

    def test_every_parsed_bbox_is_within_the_unit_square(self, parsed_synthetic):
        for element in parsed_synthetic.elements:
            if element.bbox is None:
                continue
            b = element.bbox
            assert 0.0 <= b.x0 <= b.x1 <= 1.0, f"{element.element_id} has bad x range"
            assert 0.0 <= b.y0 <= b.y1 <= 1.0, f"{element.element_id} has bad y range"

    def test_normalised_box_survives_a_dpi_change(self):
        """The same box must select the same region at any render resolution."""
        box = BBox.from_absolute(100, 200, 300, 400, width=600, height=800)
        at_72 = box.to_pixels(600, 800)
        at_300 = box.to_pixels(2500, 3333)
        # Same fractional position, different pixel grids.
        assert at_72 == (100, 200, 300, 400)
        assert at_300[0] / 2500 == pytest.approx(box.x0, abs=1e-3)
        assert at_300[3] / 3333 == pytest.approx(box.y1, abs=1e-3)

    def test_union_all_covers_every_input(self):
        boxes = [
            BBox(x0=0.1, y0=0.1, x1=0.3, y1=0.3),
            BBox(x0=0.5, y0=0.6, x1=0.9, y1=0.8),
        ]
        union = BBox.union_all(boxes)
        assert union == BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.8)
        assert all(union.contains(b) for b in boxes)

    def test_union_all_of_nothing_is_none(self):
        assert BBox.union_all([]) is None

    def test_horizontal_overlap_separates_columns(self):
        """The signal that stops a left-column caption adopting a right-column figure."""
        left = BBox(x0=0.05, y0=0.4, x1=0.45, y1=0.6)
        right = BBox(x0=0.55, y0=0.4, x1=0.95, y1=0.6)
        below_left = BBox(x0=0.05, y0=0.62, x1=0.45, y1=0.65)
        assert left.horizontal_overlap(right) == 0.0
        assert left.horizontal_overlap(below_left) == pytest.approx(1.0)

    def test_vertical_gap_is_zero_when_boxes_overlap(self):
        a = BBox(x0=0.0, y0=0.0, x1=1.0, y1=0.5)
        b = BBox(x0=0.0, y0=0.3, x1=1.0, y1=0.8)
        assert a.vertical_gap(b) == 0.0

    def test_vertical_gap_is_symmetric(self):
        a = BBox(x0=0.0, y0=0.0, x1=1.0, y1=0.2)
        b = BBox(x0=0.0, y0=0.5, x1=1.0, y1=0.7)
        assert a.vertical_gap(b) == pytest.approx(b.vertical_gap(a)) == pytest.approx(0.3)


class TestParentChildRelationships:
    def test_parent_must_not_be_self(self):
        with pytest.raises(ValidationError, match="cannot be its own parent"):
            Element(
                element_id="e1",
                doc_id="d",
                page_id="d#p1",
                page_number=1,
                element_type=ElementType.CAPTION,
                parent_id="e1",
            )

    def test_table_payload_rejected_on_a_non_table(self):
        with pytest.raises(ValidationError, match="table payload"):
            Element(
                element_id="e1",
                doc_id="d",
                page_id="d#p1",
                page_number=1,
                element_type=ElementType.TEXT,
                table=TableData(n_rows=1, n_cols=1),
            )

    def test_figure_payload_rejected_on_a_non_visual(self):
        with pytest.raises(ValidationError, match="figure payload"):
            Element(
                element_id="e1",
                doc_id="d",
                page_id="d#p1",
                page_number=1,
                element_type=ElementType.TABLE,
                figure=FigureData(),
            )

    def test_parsed_parents_all_resolve(self, parsed_synthetic):
        """No element may point at a parent that does not exist."""
        ids = {e.element_id for e in parsed_synthetic.elements}
        for element in parsed_synthetic.elements:
            if element.parent_id:
                assert element.parent_id in ids, (
                    f"{element.element_id} references missing parent {element.parent_id}"
                )

    def test_parsed_parents_are_on_the_same_page(self, parsed_synthetic):
        by_id = {e.element_id: e for e in parsed_synthetic.elements}
        for element in parsed_synthetic.elements:
            if element.parent_id:
                assert by_id[element.parent_id].page_number == element.page_number

    def test_parent_hierarchy_is_acyclic(self, parsed_synthetic):
        by_id = {e.element_id: e for e in parsed_synthetic.elements}
        for element in parsed_synthetic.elements:
            seen, current = set(), element
            while current.parent_id:
                assert current.parent_id not in seen, "cycle in parent chain"
                seen.add(current.parent_id)
                current = by_id[current.parent_id]

    def test_captions_are_parented_to_a_table_or_figure(self, parsed_synthetic):
        by_id = {e.element_id: e for e in parsed_synthetic.elements}
        captions = [
            e
            for e in parsed_synthetic.elements
            if e.element_type is ElementType.CAPTION and e.parent_id
        ]
        assert captions, "the synthetic document has captioned objects"
        for caption in captions:
            parent = by_id[caption.parent_id]
            assert parent.element_type is ElementType.TABLE or parent.element_type.is_visual

    def test_a_caption_is_claimed_by_at_most_one_parent(self, parsed_synthetic):
        """Two figures must not both adopt the same caption."""
        parents = [e.parent_id for e in parsed_synthetic.elements if e.parent_id]
        texts = [
            e.text
            for e in parsed_synthetic.elements
            if e.element_type is ElementType.CAPTION and e.text
        ]
        assert len(parents) == len(set(parents)) or len(texts) == len(set(texts))


class TestChunkProvenanceContract:
    def test_chunk_rejects_elements_from_another_document(self):
        with pytest.raises(ValidationError, match="from doc other"):
            Chunk(
                chunk_id="c1",
                doc_id="doc1",
                page_number=1,
                chunk_type=ChunkType.TEXT,
                text="x",
                element_ids=[make_element_id("other", 1, "text", 1)],
            )

    def test_chunk_rejects_elements_from_another_page(self):
        """A chunk claiming one page must not carry evidence from another."""
        with pytest.raises(ValidationError, match="from page 2"):
            Chunk(
                chunk_id="c1",
                doc_id="doc1",
                page_number=1,
                chunk_type=ChunkType.TEXT,
                text="x",
                element_ids=[make_element_id("doc1", 2, "text", 1)],
            )

    def test_chunk_accepts_consistent_elements(self):
        chunk = Chunk(
            chunk_id="c1",
            doc_id="doc1",
            page_number=1,
            chunk_type=ChunkType.TEXT,
            text="x",
            element_ids=[make_element_id("doc1", 1, "text", n) for n in (1, 2)],
        )
        assert len(chunk.element_ids) == 2


class TestBuildProvenance:
    def test_resolves_the_full_chain(
        self, sample_document, sample_page, sample_elements, sample_chunk
    ):
        prov = build_provenance(
            sample_chunk, document=sample_document, page=sample_page, elements=sample_elements
        )
        assert prov.doc_id == "doc1"
        assert prov.doc_title == "Sample Document"
        assert prov.page_number == 3
        assert prov.page_id == "doc1#p3"
        assert prov.chunk_id == sample_chunk.chunk_id
        assert set(prov.element_ids) == set(sample_elements)
        assert prov.page_image_path == "/tmp/doc1/pages/p0003.png"

    def test_bbox_is_the_union_of_the_source_elements(
        self, sample_document, sample_page, sample_elements, sample_chunk
    ):
        prov = build_provenance(
            sample_chunk, document=sample_document, page=sample_page, elements=sample_elements
        )
        assert prov.bbox == BBox(x0=0.1, y0=0.2, x1=0.6, y1=0.55)

    def test_explicit_chunk_bbox_wins_over_the_union(
        self, sample_document, sample_page, sample_elements, sample_chunk
    ):
        sample_chunk.bbox = BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
        prov = build_provenance(
            sample_chunk, document=sample_document, page=sample_page, elements=sample_elements
        )
        assert prov.bbox == BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)

    def test_section_falls_back_to_the_page(
        self, sample_document, sample_page, sample_elements, sample_chunk
    ):
        """A chunk with no section of its own inherits the page's running section."""
        assert sample_chunk.section is None
        prov = build_provenance(
            sample_chunk, document=sample_document, page=sample_page, elements=sample_elements
        )
        assert prov.section == "Results"
        assert prov.subsection == "Ablations"

    def test_reports_the_weakest_extraction_confidence(
        self, sample_document, sample_page, sample_elements, sample_chunk
    ):
        """A chunk is only as trustworthy as its least reliable evidence."""
        ids = list(sample_elements)
        sample_elements[ids[0]].extraction_confidence = 0.4
        sample_elements[ids[1]].extraction_confidence = 0.9
        prov = build_provenance(
            sample_chunk, document=sample_document, page=sample_page, elements=sample_elements
        )
        assert prov.min_extraction_confidence == pytest.approx(0.4)

    def test_collects_the_distinct_extraction_methods(
        self, sample_document, sample_page, sample_elements, sample_chunk
    ):
        ids = list(sample_elements)
        sample_elements[ids[0]].extraction_method = ExtractionMethod.PYMUPDF_DRAWING
        prov = build_provenance(
            sample_chunk, document=sample_document, page=sample_page, elements=sample_elements
        )
        assert set(prov.extraction_methods) == {
            ExtractionMethod.PYMUPDF_DRAWING,
            ExtractionMethod.PYMUPDF_TEXT,
        }

    def test_tolerates_unresolvable_element_ids(self, sample_document, sample_page, sample_chunk):
        """An index may outlive the element table; degrade, do not crash."""
        prov = build_provenance(
            sample_chunk, document=sample_document, page=sample_page, elements={}
        )
        assert prov.element_ids == sample_chunk.element_ids
        assert prov.bbox is None
        assert prov.min_extraction_confidence is None

    def test_mismatched_page_is_an_error(self, sample_document, sample_elements, sample_chunk):
        wrong_page = Page(page_id="doc1#p9", doc_id="doc1", page_number=9, width=595, height=842)
        with pytest.raises(ValueError, match="is on page 3"):
            build_provenance(
                sample_chunk,
                document=sample_document,
                page=wrong_page,
                elements=sample_elements,
            )

    def test_mismatched_document_is_an_error(self, sample_page, sample_elements, sample_chunk):
        other = Document(
            doc_id="other",
            title="Other",
            file_name="o.pdf",
            file_path="/tmp/o.pdf",
            sha256="c" * 64,
            n_pages=1,
            n_pages_ingested=1,
        )
        with pytest.raises(ValueError, match="does not belong to document other"):
            build_provenance(
                sample_chunk, document=other, page=sample_page, elements=sample_elements
            )


class TestCitation:
    def test_built_from_provenance_carries_the_region(
        self, sample_document, sample_page, sample_elements, sample_chunk
    ):
        prov = build_provenance(
            sample_chunk, document=sample_document, page=sample_page, elements=sample_elements
        )
        citation = Citation.from_provenance(prov, snippet="Revenue rose 8%.")
        assert citation.doc_title == "Sample Document"
        assert citation.page_number == 3
        assert citation.bbox is not None
        assert citation.section == "Results"
        assert citation.human_label() == "Sample Document, p. 3"
        assert citation.element_ids == prov.element_ids


class TestMetadataSeparation:
    """Metadata must stay structured, never folded into the embedded text."""

    def test_best_text_excludes_structural_metadata(self):
        element = Element(
            element_id="d#p1#text001",
            doc_id="d",
            page_id="d#p1",
            page_number=1,
            element_type=ElementType.TEXT,
            text="Revenue rose 8%.",
            section="Financial Review",
            subsection="Segment Results",
            extraction_method=ExtractionMethod.PYMUPDF_TEXT,
        )
        text = element.best_text()
        assert text == "Revenue rose 8%."
        for leaked in ("Financial Review", "Segment Results", "pymupdf", "d#p1"):
            assert leaked not in text

    def test_structured_metadata_exposes_the_filterable_fields(self):
        element = Element(
            element_id="d#p1#table001",
            doc_id="d",
            page_id="d#p1",
            page_number=1,
            element_type=ElementType.TABLE,
            section="Financial Review",
            table=TableData(n_rows=3, n_cols=4),
            extraction_confidence=0.75,
        )
        payload = element.structured_metadata()
        assert payload["element_type"] == "table"
        assert payload["section"] == "Financial Review"
        assert payload["table_shape"] == [3, 4]
        assert payload["extraction_confidence"] == 0.75

    def test_figure_metadata_flags_whether_any_text_was_recovered(self):
        blind = Element(
            element_id="d#p1#chart001",
            doc_id="d",
            page_id="d#p1",
            page_number=1,
            element_type=ElementType.CHART,
            figure=FigureData(),
        )
        # The failure mode Method 1 exists to expose: a figure with no text at
        # all is invisible to a purely textual pipeline.
        assert blind.structured_metadata()["has_derived_text"] is False
        assert blind.best_text() == ""
        assert blind.is_empty
