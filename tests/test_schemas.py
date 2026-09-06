"""Tests for the core data model.

The bounding-box arithmetic and ``Element.best_text`` get real attention here
because they are load-bearing: bboxes are what make citations point at a region
of a page, and ``best_text`` is the exact point where Method 1 flattens the
other modalities into text.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mmrag.schemas import (
    BBox,
    Chunk,
    ChunkType,
    Element,
    ElementType,
    make_chunk_id,
    make_element_id,
    make_page_id,
)


class TestBBox:
    def test_from_absolute_normalises(self):
        b = BBox.from_absolute(10, 20, 110, 220, width=200, height=400)
        assert b.x0 == pytest.approx(0.05)
        assert b.y0 == pytest.approx(0.05)
        assert b.x1 == pytest.approx(0.55)
        assert b.y1 == pytest.approx(0.55)

    def test_round_trips_through_absolute(self):
        b = BBox.from_absolute(10, 20, 110, 220, width=200, height=400)
        assert b.to_absolute(200, 400) == pytest.approx((10, 20, 110, 220))

    def test_inverted_input_is_reordered_not_rejected(self):
        """Some PDF generators emit y1 < y0; that must not blow up ingestion."""
        b = BBox.from_absolute(110, 220, 10, 20, width=200, height=400)
        assert (b.x0, b.y0, b.x1, b.y1) == pytest.approx((0.05, 0.05, 0.55, 0.55))

    def test_out_of_page_coordinates_are_clamped(self):
        b = BBox.from_absolute(-50, -50, 400, 800, width=200, height=400)
        assert (b.x0, b.y0, b.x1, b.y1) == (0.0, 0.0, 1.0, 1.0)

    def test_zero_page_size_is_an_error(self):
        with pytest.raises(ValueError, match="page size must be positive"):
            BBox.from_absolute(0, 0, 1, 1, width=0, height=100)

    def test_ordering_validator_rejects_bad_direct_construction(self):
        with pytest.raises(ValidationError):
            BBox(x0=0.5, y0=0.0, x1=0.2, y1=1.0)

    def test_area_and_union(self):
        a = BBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
        b = BBox(x0=0.5, y0=0.5, x1=1.0, y1=1.0)
        assert a.area == pytest.approx(0.25)
        assert a.union(b) == BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)

    def test_iou_disjoint_is_zero(self):
        a = BBox(x0=0.0, y0=0.0, x1=0.4, y1=0.4)
        b = BBox(x0=0.6, y0=0.6, x1=1.0, y1=1.0)
        assert a.iou(b) == 0.0

    def test_iou_touching_edges_is_zero(self):
        """Edge contact is not overlap; caption matching relies on this."""
        a = BBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
        b = BBox(x0=0.5, y0=0.0, x1=1.0, y1=0.5)
        assert a.iou(b) == 0.0

    def test_iou_identical_is_one(self):
        a = BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9)
        assert a.iou(a) == pytest.approx(1.0)

    def test_iou_half_overlap(self):
        a = BBox(x0=0.0, y0=0.0, x1=0.5, y1=1.0)
        b = BBox(x0=0.25, y0=0.0, x1=0.75, y1=1.0)
        # intersection 0.25, union 0.75
        assert a.iou(b) == pytest.approx(1 / 3)

    def test_is_hashable_and_frozen(self):
        a = BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
        assert len({a, BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)}) == 1
        with pytest.raises(ValidationError):
            a.x0 = 0.5  # type: ignore[misc]


class TestElementTypes:
    def test_visual_and_textual_are_disjoint(self):
        for t in ElementType:
            assert not (t.is_visual and t.is_textual)

    def test_table_is_neither_visual_nor_textual(self):
        """Tables are their own modality; that distinction is what Method 2 uses."""
        assert not ElementType.TABLE.is_visual
        assert not ElementType.TABLE.is_textual


class TestIdentifiers:
    def test_page_and_element_ids(self):
        assert make_page_id("doc", 3) == "doc#p3"
        assert make_element_id("doc", 3, "table", 1) == "doc#p3#table001"

    def test_chunk_id_is_deterministic(self):
        """Re-running ingestion on unchanged input must reproduce ids."""
        a = make_chunk_id("method1", "doc", "p3", "hello")
        b = make_chunk_id("method1", "doc", "p3", "hello")
        assert a == b and a.startswith("method1#")

    def test_chunk_id_varies_with_content_and_variant(self):
        base = make_chunk_id("method1", "doc", "p3", "hello")
        assert base != make_chunk_id("method1", "doc", "p3", "hello!")
        assert base != make_chunk_id("method2", "doc", "p3", "hello")

    def test_chunk_id_is_not_confused_by_part_boundaries(self):
        """('ab','c') and ('a','bc') must not collide -- hence the separator."""
        assert make_chunk_id("v", "ab", "c") != make_chunk_id("v", "a", "bc")


def _element(**kw) -> Element:
    defaults = dict(
        element_id="d#p1#text000",
        doc_id="d",
        page_id="d#p1",
        page_number=1,
        element_type=ElementType.TEXT,
    )
    return Element(**{**defaults, **kw})


class TestBestText:
    def test_plain_text_element(self):
        assert _element(text="  hello  ").best_text() == "hello"

    def test_table_prefers_markdown_and_keeps_caption(self):
        e = _element(
            element_type=ElementType.TABLE,
            caption="Table 1: Revenue",
            table_markdown="| a | b |\n|---|---|",
            text="ignored raw dump",
        )
        assert e.best_text() == "Table 1: Revenue\n| a | b |\n|---|---|"

    def test_table_without_markdown_falls_back_to_raw_text(self):
        e = _element(element_type=ElementType.TABLE, text="a b c")
        assert e.best_text() == "a b c"

    def test_figure_concatenates_caption_description_and_ocr(self):
        e = _element(
            element_type=ElementType.CHART,
            caption="Figure 2",
            description="A bar chart of revenue by year.",
            ocr_text="2021 2022 2023",
        )
        assert e.best_text() == "Figure 2\nA bar chart of revenue by year.\n2021 2022 2023"

    def test_figure_with_no_derived_text_is_empty(self):
        """The failure mode Method 1 is meant to expose: an invisible figure."""
        assert _element(element_type=ElementType.DIAGRAM).best_text() == ""

    def test_blank_and_whitespace_parts_are_dropped(self):
        e = _element(element_type=ElementType.FIGURE, caption="   ", description="real")
        assert e.best_text() == "real"


class TestChunk:
    def test_page_number_must_be_one_indexed(self):
        with pytest.raises(ValidationError):
            Chunk(
                chunk_id="c",
                doc_id="d",
                page_number=0,
                chunk_type=ChunkType.TEXT,
                text="x",
            )
