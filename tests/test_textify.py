"""Tests for flattening and chunking -- the mechanism that defines Method 1.

Two properties get the most attention, because they are the ones whose failure
would be invisible:

* **Provenance survives chunking.** Every chunk must name real elements from one
  page of one document. A chunk that quietly spans pages still produces a fluent
  answer and a plausible-looking citation that points at the wrong place.
* **Loss is measured, not merely incurred.** Method 1 exists to quantify the cost
  of flattening, so a figure with no recoverable text must be *counted*, not
  silently dropped.
"""

from __future__ import annotations

import pytest

from mmrag.config import ChunkingConfig
from mmrag.schemas import (
    BBox,
    ChunkType,
    Document,
    Element,
    ElementType,
    FigureData,
    TableData,
    make_element_id,
    make_page_id,
    parse_element_id,
)
from mmrag.textify.chunker import Chunker
from mmrag.textify.flatten import context_header, flatten_elements, is_redundant_child
from mmrag.textify.tokens import (
    HeuristicTokenCounter,
    get_token_counter,
    split_sentences,
)

DOC_ID = "doc1"


def _doc(title: str = "Annual Report 2024") -> Document:
    return Document(
        doc_id=DOC_ID,
        title=title,
        file_name="d.pdf",
        file_path="/tmp/d.pdf",
        sha256="a" * 64,
        n_pages=3,
        n_pages_ingested=3,
    )


def _el(
    ordinal: int,
    *,
    page: int = 1,
    kind: ElementType = ElementType.TEXT,
    text: str | None = "Some ordinary body text about revenue.",
    order: int = 0,
    **kw,
) -> Element:
    return Element(
        element_id=make_element_id(DOC_ID, page, kind.value, ordinal),
        doc_id=DOC_ID,
        page_id=make_page_id(DOC_ID, page),
        page_number=page,
        element_type=kind,
        reading_order=order,
        text=text,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.2),
        **kw,
    )


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


class TestSentenceSplitting:
    def test_splits_on_sentence_ends(self):
        assert split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]

    def test_does_not_split_after_an_abbreviation(self):
        """'See Fig. 3' is one sentence; splitting it strands a fragment."""
        assert split_sentences("See Fig. 3 for details. Next.") == [
            "See Fig. 3 for details.",
            "Next.",
        ]

    def test_does_not_split_after_an_initial(self):
        assert split_sentences("J. Smith et al. report gains. We agree.") == [
            "J. Smith et al. report gains.",
            "We agree.",
        ]

    def test_paragraphs_are_separate(self):
        assert split_sentences("First para.\n\nSecond para.") == ["First para.", "Second para."]

    def test_no_split_without_a_capital(self):
        assert split_sentences("Version 1.5 shipped") == ["Version 1.5 shipped"]

    @pytest.mark.parametrize("value", ["", "   ", "\n\n"])
    def test_blank_input(self, value):
        assert split_sentences(value) == []


class TestTokenCounters:
    def test_heuristic_scales_with_length(self):
        counter = HeuristicTokenCounter()
        assert counter.count("") == 0
        assert counter.count("one two three") < counter.count("one two three four five six")

    def test_heuristic_is_deterministic(self):
        counter = HeuristicTokenCounter()
        assert counter.count("Revenue rose 8%.") == counter.count("Revenue rose 8%.")

    def test_unknown_model_falls_back_rather_than_failing(self):
        """A machine with no network must still be able to chunk."""
        counter = get_token_counter("definitely/not-a-real-model-xyz")
        assert counter.name == "heuristic"
        assert counter.count("hello world") > 0


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


class TestContextHeader:
    def test_builds_a_breadcrumb(self):
        element = _el(1, section="Financial Review", subsection="Segments")
        assert context_header(_doc(), element) == "Annual Report 2024 > Financial Review > Segments"

    def test_omits_a_section_that_repeats_the_title(self):
        element = _el(1, section="Annual Report 2024")
        assert context_header(_doc(), element) == "Annual Report 2024"

    def test_drops_an_implausibly_long_heading(self):
        """A legal notice set in a large face is not a section heading."""
        element = _el(1, section="Provided proper attribution is provided, " * 6)
        assert context_header(_doc(), element) == "Annual Report 2024"

    def test_title_only_when_no_section(self):
        assert context_header(_doc(), _el(1)) == "Annual Report 2024"


class TestFlatten:
    def test_headers_and_footers_are_excluded(self):
        elements = [
            _el(1, kind=ElementType.HEADER, text="Annual Report 2024"),
            _el(1, kind=ElementType.FOOTER, text="Page 3"),
            _el(1, text="Real content here."),
        ]
        kept, report = flatten_elements(elements)
        assert len(kept) == 1
        assert report.skipped_boilerplate == 2

    def test_a_caption_already_on_its_parent_is_not_indexed_twice(self):
        figure = _el(
            1,
            kind=ElementType.CHART,
            text=None,
            caption="Figure 1: Revenue.",
            figure=FigureData(),
        )
        caption = _el(
            1,
            kind=ElementType.CAPTION,
            text="Figure 1: Revenue.",
            parent_id=figure.element_id,
        )
        kept, report = flatten_elements([figure, caption])
        assert report.skipped_child == 1
        assert [e.element_id for e in kept] == [figure.element_id]

    def test_text_inside_a_figure_is_still_indexed(self):
        """Axis labels are parented but exist nowhere else -- dropping them loses data."""
        figure = _el(1, kind=ElementType.CHART, text=None, caption="Figure 1", figure=FigureData())
        label = _el(2, text="0 10 20 30", parent_id=figure.element_id)
        kept, report = flatten_elements([figure, label])
        assert report.skipped_child == 0
        assert label.element_id in {e.element_id for e in kept}

    def test_a_figure_with_no_text_is_counted_not_silently_dropped(self):
        """The headline measurement: content Method 1 structurally cannot see."""
        blind = _el(1, kind=ElementType.DIAGRAM, text=None, figure=FigureData())
        kept, report = flatten_elements([blind])
        assert kept == []
        assert report.invisible_figures == 1
        assert report.empty_by_type["diagram"] == 1
        assert blind.element_id in report.empty_element_ids

    def test_a_captioned_figure_is_visible(self):
        seen = _el(
            1,
            kind=ElementType.CHART,
            text=None,
            caption="Figure 2: Revenue by region.",
            figure=FigureData(),
        )
        kept, report = flatten_elements([seen])
        assert len(kept) == 1
        assert report.invisible_figures == 0

    def test_report_totals_add_up(self):
        elements = [
            _el(1, kind=ElementType.HEADER, text="hdr"),
            _el(1, text="content"),
            _el(1, kind=ElementType.FIGURE, text=None, figure=FigureData()),
        ]
        _, report = flatten_elements(elements)
        assert (
            report.kept + report.skipped_boilerplate + report.skipped_child + report.empty
            == report.total
        )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@pytest.fixture
def chunker() -> Chunker:
    # The heuristic counter keeps these tests fast and offline; exactness is not
    # what is under test here, structure is.
    return Chunker(
        ChunkingConfig(target_tokens=64, overlap_tokens=16),
        variant="method1",
        token_counter=HeuristicTokenCounter(),
    )


class TestChunkProvenance:
    def test_no_chunk_spans_two_pages(self, chunker):
        elements = [
            _el(i, page=page, order=i, text=f"Sentence about topic {i} on page {page}.")
            for page in (1, 2)
            for i in range(1, 6)
        ]
        chunks, _ = chunker.chunk_document(_doc(), elements)
        for chunk in chunks:
            pages = {parse_element_id(e)[1] for e in chunk.element_ids}
            assert pages == {chunk.page_number}

    def test_every_chunk_names_real_elements(self, chunker):
        elements = [_el(i, order=i, text=f"Fact number {i}.") for i in range(1, 8)]
        known = {e.element_id for e in elements}
        chunks, _ = chunker.chunk_document(_doc(), elements)
        assert chunks
        for chunk in chunks:
            assert chunk.element_ids
            assert set(chunk.element_ids) <= known

    def test_chunk_bbox_covers_its_elements(self, chunker):
        elements = [_el(i, order=i, text=f"Fact {i}.") for i in range(1, 4)]
        chunks, _ = chunker.chunk_document(_doc(), elements)
        assert all(c.bbox is not None for c in chunks)

    def test_chunk_ids_are_deterministic(self, chunker):
        elements = [_el(i, order=i, text=f"Fact {i}.") for i in range(1, 6)]
        first, _ = chunker.chunk_document(_doc(), elements)
        second, _ = chunker.chunk_document(_doc(), elements)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_chunk_ids_are_unique(self, chunker):
        elements = [_el(i, order=i, text=f"Distinct fact number {i}.") for i in range(1, 30)]
        chunks, _ = chunker.chunk_document(_doc(), elements)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_identical_text_on_one_page_still_gets_distinct_ids(self, chunker):
        """Two charts on a page routinely share a caption like 'Source:'.

        Hashing content alone collided on the real corpus, and a collision maps
        a retrieval hit onto the wrong figure -- a silent provenance error, the
        exact failure this project is built to prevent.
        """
        shared = "Source:"
        figures = [
            _el(
                n,
                kind=ElementType.CHART,
                text=None,
                order=n,
                caption=shared,
                figure=FigureData(),
            )
            for n in (1, 2, 3)
        ]
        chunks, _ = chunker.chunk_document(_doc(), figures)

        assert len(chunks) == 3
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == 3, "distinct figures collapsed onto one id"
        # Each chunk still points at its own element.
        assert {c.element_ids[0] for c in chunks} == {f.element_id for f in figures}

    def test_variant_namespaces_the_ids(self):
        elements = [_el(1, text="Shared content.")]
        a, _ = Chunker(
            ChunkingConfig(), variant="method1", token_counter=HeuristicTokenCounter()
        ).chunk_document(_doc(), elements)
        b, _ = Chunker(
            ChunkingConfig(), variant="method2", token_counter=HeuristicTokenCounter()
        ).chunk_document(_doc(), elements)
        assert a[0].chunk_id != b[0].chunk_id


class TestChunkTypes:
    def test_a_table_becomes_its_own_chunk(self, chunker):
        table = _el(
            1,
            kind=ElementType.TABLE,
            text=None,
            order=1,
            caption="Table 1: Headcount.",
            table=TableData(
                n_rows=2,
                n_cols=2,
                columns=["Dept", "N"],
                rows=[["Eng", "412"], ["Sales", "233"]],
                markdown="| Dept | N |\n| --- | --- |\n| Eng | 412 |",
            ),
        )
        prose = _el(2, order=0, text="Prose before the table.")
        chunks, _ = chunker.chunk_document(_doc(), [prose, table])

        table_chunks = [c for c in chunks if c.chunk_type is ChunkType.TABLE]
        assert len(table_chunks) == 1
        assert table_chunks[0].element_ids == [table.element_id]
        assert "412" in table_chunks[0].text

    def test_prose_is_not_merged_into_a_table_chunk(self, chunker):
        table = _el(
            1,
            kind=ElementType.TABLE,
            text=None,
            order=1,
            table=TableData(
                n_rows=2,
                n_cols=2,
                columns=["a", "b"],
                rows=[["1", "2"], ["3", "4"]],
                markdown="| a | b |",
            ),
        )
        prose = _el(2, order=0, text="Distinctive prose sentence about margins.")
        chunks, _ = chunker.chunk_document(_doc(), [prose, table])
        for chunk in chunks:
            if chunk.chunk_type is ChunkType.TABLE:
                assert "margins" not in chunk.text

    def test_a_figure_becomes_its_own_chunk_with_its_provenance(self, chunker):
        figure = _el(
            1,
            kind=ElementType.CHART,
            text=None,
            caption="Figure 3: Revenue by region.",
            figure=FigureData(figure_type="chart", ocr_text="2021 2022"),
        )
        chunks, _ = chunker.chunk_document(_doc(), [figure])
        figure_chunks = [c for c in chunks if c.chunk_type is ChunkType.FIGURE]
        assert len(figure_chunks) == 1
        chunk = figure_chunks[0]
        assert chunk.metadata["figure_type"] == "chart"
        assert chunk.metadata["has_caption"] is True
        assert chunk.metadata["has_ocr"] is True
        assert chunk.metadata["has_description"] is False

    def test_a_huge_table_is_split_with_the_header_repeated(self):
        """A table fragment without its header is unreadable."""
        from mmrag.ingestion.tables import to_markdown

        columns = ["Item", "Amount", "Note"]
        rows = [[f"Row {i}", str(i * 1000), f"note {i}"] for i in range(120)]
        table = _el(
            1,
            kind=ElementType.TABLE,
            text=None,
            table=TableData(
                n_rows=len(rows),
                n_cols=3,
                columns=columns,
                rows=rows,
                # Real markdown, not a placeholder: the size check reads this
                # field, so a stub would make the split path unreachable.
                markdown=to_markdown(columns, rows),
            ),
        )
        chunker = Chunker(
            ChunkingConfig(max_table_tokens=200),
            token_counter=HeuristicTokenCounter(),
        )
        chunks, _ = chunker.chunk_document(_doc(), [table])
        assert len(chunks) > 1
        for chunk in chunks:
            assert "Item | Amount | Note" in chunk.text
            assert chunk.element_ids == [table.element_id]

    def test_split_table_parts_get_distinct_ids(self):
        from mmrag.ingestion.tables import to_markdown

        columns = ["Item", "N"]
        rows = [[f"Row {i}", str(i)] for i in range(120)]
        table = _el(
            1,
            kind=ElementType.TABLE,
            text=None,
            table=TableData(
                n_rows=len(rows),
                n_cols=2,
                columns=columns,
                rows=rows,
                markdown=to_markdown(columns, rows),
            ),
        )
        chunker = Chunker(
            ChunkingConfig(max_table_tokens=150), token_counter=HeuristicTokenCounter()
        )
        chunks, _ = chunker.chunk_document(_doc(), [table])
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestChunkSizing:
    def test_chunks_respect_the_token_target(self, chunker):
        elements = [
            _el(i, order=i, text=f"Sentence number {i} discussing quarterly performance in detail.")
            for i in range(1, 40)
        ]
        chunks, _ = chunker.chunk_document(_doc(), elements)
        assert len(chunks) > 1
        # Allow one sentence of overshoot: a chunk closes *after* the sentence
        # that crosses the target, rather than splitting mid-sentence.
        assert all(c.token_count <= chunker.config.target_tokens * 2 for c in chunks)

    def test_consecutive_chunks_overlap(self, chunker):
        elements = [
            _el(i, order=i, text=f"Distinct sentence {i} about operating margins and revenue.")
            for i in range(1, 30)
        ]
        chunks, _ = chunker.chunk_document(_doc(), elements)
        assert len(chunks) >= 2
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        assert first_words & second_words, "no overlap carried between chunks"

    def test_overlap_disabled_produces_no_repeat(self):
        chunker = Chunker(
            ChunkingConfig(target_tokens=64, overlap_tokens=0),
            token_counter=HeuristicTokenCounter(),
        )
        elements = [_el(i, order=i, text=f"Unique sentence {i} here.") for i in range(1, 20)]
        chunks, _ = chunker.chunk_document(_doc(), elements)
        # Every sentence appears exactly once across all chunks.
        joined = " ".join(c.text for c in chunks)
        assert joined.count("Unique sentence 5 here.") == 1

    def test_chunking_terminates_on_a_single_oversized_sentence(self):
        """A run-on line must not loop forever trying to fit the target."""
        chunker = Chunker(
            ChunkingConfig(target_tokens=64, overlap_tokens=8),
            token_counter=HeuristicTokenCounter(),
        )
        long_line = " ".join(f"token{i}" for i in range(500))
        chunks, _ = chunker.chunk_document(_doc(), [_el(1, text=long_line)])
        assert chunks


class TestChunkMetadata:
    def test_structural_metadata_is_a_payload_not_embedded_text(self, chunker):
        element = _el(1, section="Financial Review")
        chunks, _ = chunker.chunk_document(_doc(), [element])
        chunk = chunks[0]
        assert chunk.metadata["doc_title"] == "Annual Report 2024"
        assert chunk.metadata["doc_type"] == "other"
        # Identifiers and extraction provenance never enter the embedded text.
        for leaked in (element.element_id, "pymupdf", str(chunk.bbox)):
            assert leaked not in chunk.text

    def test_context_header_is_recorded_so_it_can_be_stripped(self, chunker):
        chunks, _ = chunker.chunk_document(_doc(), [_el(1, section="Results")])
        chunk = chunks[0]
        header = chunk.metadata["context_header"]
        assert header and chunk.text.startswith(header)

    def test_header_can_be_turned_off_for_an_ablation(self):
        chunker = Chunker(
            ChunkingConfig(prepend_context_header=False),
            token_counter=HeuristicTokenCounter(),
        )
        chunks, _ = chunker.chunk_document(_doc(), [_el(1, section="Results")])
        assert chunks[0].metadata["context_header"] == ""
        assert not chunks[0].text.startswith("Annual Report")


class TestChunkingReport:
    def test_report_counts_chunks_by_type(self, chunker):
        elements = [
            _el(1, order=0, text="Some prose."),
            _el(
                1,
                kind=ElementType.TABLE,
                text=None,
                order=1,
                table=TableData(
                    n_rows=1, n_cols=2, columns=["a", "b"], rows=[["1", "2"]], markdown="| a | b |"
                ),
            ),
            _el(
                1,
                kind=ElementType.CHART,
                text=None,
                order=2,
                caption="Figure 1: A chart.",
                figure=FigureData(),
            ),
        ]
        _, report = chunker.chunk_document(_doc(), elements)
        assert report.by_type == {"figure": 1, "table": 1, "text": 1}
        assert report.n_chunks == 3

    def test_report_carries_the_flattening_loss(self, chunker):
        blind = _el(1, kind=ElementType.DIAGRAM, text=None, figure=FigureData())
        _, report = chunker.chunk_document(_doc(), [_el(2, text="prose"), blind])
        assert report.flatten.invisible_figures == 1
        assert report.as_dict()["flatten"]["invisible_figures"] == 1

    def test_report_names_the_token_counter(self, chunker):
        _, report = chunker.chunk_document(_doc(), [_el(1)])
        assert report.token_counter == "heuristic"


def test_is_redundant_child_needs_a_real_parent():
    orphan = _el(1, kind=ElementType.CAPTION, text="Figure 1", parent_id="missing")
    assert not is_redundant_child(orphan, {})
