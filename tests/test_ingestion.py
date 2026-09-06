"""Ingestion component tests.

Each failure mode asserted here was observed on a real corpus document before it
was fixed, so these are regression tests rather than speculation:

* a blue background panel behind prose detected as a chart, hiding the prose;
* a table's ruling lines clustering into a phantom chart over the same region;
* an attention-visualisation figure parsed as a 35-column table of words;
* a section heading adopted as a figure's caption;
* one caption claimed by two different figures.
"""

from __future__ import annotations

import pytest

from mmrag.ingestion.classify import (
    classify_figure,
    classify_table,
    figure_confidence,
    table_confidence,
    text_confidence,
)
from mmrag.ingestion.figures import (
    cluster_vector_drawings,
    find_caption,
    is_caption_marked,
    text_coverage,
)
from mmrag.ingestion.layout import (
    HeaderFooterDetector,
    LayoutBlock,
    SectionTracker,
    assign_reading_order,
    detect_columns,
)
from mmrag.ingestion.metadata import infer_document_type, parse_authors, parse_pdf_date
from mmrag.ingestion.tables import build_table_data, to_markdown
from mmrag.ingestion.text_utils import (
    is_probably_numeric,
    mask_digits,
    normalize_text,
    text_quality,
)
from mmrag.schemas import BBox, DocumentType, FigureType, TableType


def block(x0, y0, x1, y1, text="body text", size=10.0, bold=False) -> LayoutBlock:
    return LayoutBlock(
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), text=text, font_size=size, is_bold=bold
    )


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_expands_ligatures(self):
        assert normalize_text("the ﬁrst ﬂag") == "the first flag"

    def test_straightens_quotes_and_dashes(self):
        assert normalize_text("“the company’s”—now") == '"the company\'s"-now'

    def test_joins_hyphenated_line_breaks(self):
        assert normalize_text("appli-\ncation") == "application"

    def test_leaves_genuine_compounds_alone(self):
        assert "state-of-the-art" in normalize_text("state-of-the-art results")

    def test_does_not_join_across_a_capital(self):
        """'Sales-\\nQ3' is two tokens, not the word 'SalesQ3'."""
        assert normalize_text("Sales-\nQ3") == "Sales-\nQ3"

    def test_removes_zero_width_and_soft_hyphens(self):
        assert normalize_text("a​b­c") == "abc"

    def test_collapses_runs_of_whitespace(self):
        assert normalize_text("a    b\n\n\n\nc") == "a b\n\nc"

    def test_is_idempotent(self):
        raw = "the ﬁrst “value”—appli-\ncation here"
        once = normalize_text(raw)
        assert normalize_text(once) == once

    def test_handles_none_and_empty(self):
        assert normalize_text(None) == ""
        assert normalize_text("   ") == ""

    def test_strips_nul_and_control_characters(self):
        """A single NUL fails an entire PostgreSQL write; both appear in the corpus."""
        assert normalize_text("ab\x00cd\x01ef") == "abcdef"

    def test_tabs_and_newlines_are_whitespace_not_damage(self):
        """Tab and newline survive control stripping.

        Tab is then collapsed to a space by the ordinary whitespace rule, which
        is what we want for retrieval text; the point is that it is treated as
        layout rather than deleted the way U+0000 is.
        """
        assert normalize_text("a\tb\nc") == "a b\nc"

    def test_normalises_crlf_before_stripping_controls(self):
        """\\r must become a newline, not vanish and join two lines into one."""
        assert normalize_text("line one\r\nline two\rline three") == (
            "line one\nline two\nline three"
        )

    def test_output_is_safe_for_a_postgres_text_column(self):
        raw = "Retrieval\x00Augmented\x01Generation\x0bTest"
        cleaned = normalize_text(raw)
        assert not any(ord(c) < 32 and c not in "\t\n" for c in cleaned)


class TestTextQuality:
    def test_counts_unmappable_glyphs(self):
        """Observed on the Berkshire report: a broken font map for apostrophes."""
        damaged = "Management�s Discussion"
        q = text_quality(damaged)
        assert q.n_replacement == 1
        assert q.replacement_ratio == pytest.approx(1 / len(damaged))

    def test_flags_heavily_damaged_text_as_unusable(self):
        assert not text_quality("�" * 5 + "abcde").is_usable

    def test_clean_text_is_usable(self):
        assert text_quality("A perfectly ordinary sentence.").is_usable

    def test_empty_text_is_not_usable(self):
        assert not text_quality("").is_usable


class TestTextHelpers:
    def test_mask_digits_makes_page_footers_comparable(self):
        assert mask_digits("Annual Report | Page 3") == mask_digits("Annual Report | Page 47")

    @pytest.mark.parametrize("value", ["1,234", "(45.6)", "$1,000", "12%"])
    def test_numeric_cells_detected(self, value):
        assert is_probably_numeric(value)

    @pytest.mark.parametrize("value", ["Engineering", "Net revenue", ""])
    def test_text_cells_not_numeric(self, value):
        assert not is_probably_numeric(value)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class TestColumnDetection:
    def test_single_column_body(self):
        blocks = [block(0.1, 0.1 + i * 0.1, 0.9, 0.15 + i * 0.1) for i in range(6)]
        assert detect_columns(blocks) == 1

    def test_two_column_paper(self):
        left = [block(0.05, 0.1 + i * 0.1, 0.45, 0.15 + i * 0.1) for i in range(5)]
        right = [block(0.55, 0.1 + i * 0.1, 0.95, 0.15 + i * 0.1) for i in range(5)]
        assert detect_columns(left + right) == 2

    def test_too_few_blocks_defaults_to_one_column(self):
        assert detect_columns([block(0.05, 0.1, 0.45, 0.2)]) == 1


class TestReadingOrder:
    def test_two_columns_are_not_interleaved(self):
        """The bug this prevents: left/right/left/right, which garbles every chunk."""
        left = [block(0.05, 0.1, 0.45, 0.2, text="L1"), block(0.05, 0.3, 0.45, 0.4, text="L2")]
        right = [block(0.55, 0.1, 0.95, 0.2, text="R1"), block(0.55, 0.3, 0.95, 0.4, text="R2")]
        ordered = assign_reading_order(left + right, n_columns=2)
        assert [b.text for b in ordered] == ["L1", "L2", "R1", "R2"]

    def test_single_column_is_top_to_bottom(self):
        blocks = [
            block(0.1, 0.5, 0.9, 0.6, text="second"),
            block(0.1, 0.1, 0.9, 0.2, text="first"),
        ]
        ordered = assign_reading_order(blocks, n_columns=1)
        assert [b.text for b in ordered] == ["first", "second"]

    def test_reading_order_is_dense_and_zero_based(self):
        blocks = [block(0.1, i * 0.1, 0.9, i * 0.1 + 0.05) for i in range(5)]
        ordered = assign_reading_order(blocks)
        assert [b.reading_order for b in ordered] == [0, 1, 2, 3, 4]

    def test_empty_input(self):
        assert assign_reading_order([]) == []


class TestHeaderFooterDetection:
    def test_text_repeating_across_pages_is_furniture(self):
        detector = HeaderFooterDetector()
        for n in range(1, 11):
            detector.observe(
                [
                    block(0.1, 0.01, 0.9, 0.04, text="Annual Report 2024"),
                    block(0.1, 0.4, 0.9, 0.5, text=f"Unique body text {n}"),
                    block(0.1, 0.96, 0.9, 0.99, text=f"Page {n}"),
                ]
            )
        detector.finalize()
        assert detector.classify(block(0.1, 0.01, 0.9, 0.04, text="Annual Report 2024")) == "header"
        assert detector.classify(block(0.1, 0.96, 0.9, 0.99, text="Page 7")) == "footer"
        assert detector.classify(block(0.1, 0.4, 0.9, 0.5, text="Unique body text 3")) is None

    def test_a_heading_appearing_once_is_not_furniture(self):
        # Varied by letter, not by digit: digit masking deliberately treats
        # "Page 3" and "Page 4" as the same footer, so numeric variation alone
        # would not distinguish a heading from furniture.
        detector = HeaderFooterDetector()
        for n in range(10):
            detector.observe(
                [block(0.1, 0.01, 0.9, 0.04, text=f"Chapter {chr(65 + n)}: A Distinct Heading")]
            )
        detector.finalize()
        assert (
            detector.classify(block(0.1, 0.01, 0.9, 0.04, text="Chapter A: A Distinct Heading"))
            is None
        )

    def test_digit_variation_alone_still_counts_as_furniture(self):
        """A running header differing only by chapter number is still furniture."""
        detector = HeaderFooterDetector()
        for n in range(10):
            detector.observe([block(0.1, 0.01, 0.9, 0.04, text=f"Chapter {n} | Annual Report")])
        detector.finalize()
        assert (
            detector.classify(block(0.1, 0.01, 0.9, 0.04, text="Chapter 3 | Annual Report"))
            == "header"
        )

    def test_text_in_the_middle_of_the_page_is_never_furniture(self):
        detector = HeaderFooterDetector()
        for _ in range(10):
            detector.observe([block(0.1, 0.45, 0.9, 0.55, text="Repeated mid-page note")])
        detector.finalize()
        assert detector.classify(block(0.1, 0.45, 0.9, 0.55, text="Repeated mid-page note")) is None

    def test_querying_before_finalize_is_an_error(self):
        with pytest.raises(RuntimeError, match="finalize"):
            HeaderFooterDetector().classify(block(0, 0, 1, 0.03))


class TestSectionTracker:
    def _tracker(self) -> SectionTracker:
        tracker = SectionTracker()
        # Lots of 10pt body text, so 10.0 wins the body-size vote.
        tracker.observe([block(0.1, 0.2, 0.9, 0.3, text="x" * 500, size=10.0)])
        tracker.finalize()
        return tracker

    def test_body_size_is_weighted_by_character_count(self):
        tracker = SectionTracker()
        tracker.observe(
            [
                block(0.1, 0.05, 0.9, 0.1, text="A Big Title", size=24.0),
                block(0.1, 0.2, 0.9, 0.8, text="y" * 2000, size=10.0),
            ]
        )
        tracker.finalize()
        assert tracker.body_size == 10.0

    def test_large_text_is_a_section_heading(self):
        tracker = self._tracker()
        assert tracker.heading_level(block(0.1, 0.05, 0.9, 0.1, text="Results", size=16.0)) == 1

    def test_slightly_larger_text_is_a_subsection(self):
        tracker = self._tracker()
        assert tracker.heading_level(block(0.1, 0.05, 0.9, 0.1, text="Ablations", size=12.0)) == 2

    def test_body_text_is_not_a_heading(self):
        tracker = self._tracker()
        assert (
            tracker.heading_level(block(0.1, 0.2, 0.9, 0.3, text="Ordinary prose.", size=10.0))
            is None
        )

    def test_a_long_block_is_never_a_heading(self):
        """Large-set pull quotes are not section headings."""
        tracker = self._tracker()
        long_text = "word " * 60
        assert tracker.heading_level(block(0.1, 0.2, 0.9, 0.4, text=long_text, size=18.0)) is None

    def test_a_new_section_resets_the_subsection(self):
        tracker = self._tracker()
        tracker.update(block(0.1, 0.05, 0.9, 0.1, text="Results", size=16.0))
        tracker.update(block(0.1, 0.15, 0.9, 0.2, text="Ablations", size=12.0))
        assert (tracker.section, tracker.subsection) == ("Results", "Ablations")
        tracker.update(block(0.1, 0.3, 0.9, 0.35, text="Discussion", size=16.0))
        assert (tracker.section, tracker.subsection) == ("Discussion", None)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


class TestVectorClustering:
    def test_nearby_primitives_form_one_figure(self):
        rects = [BBox(x0=0.2 + i * 0.01, y0=0.3, x1=0.21 + i * 0.01, y1=0.5) for i in range(10)]
        clusters = cluster_vector_drawings(rects, min_area_ratio=0.001)
        assert len(clusters) == 1
        assert clusters[0].n_primitives == 10
        assert clusters[0].is_vector

    def test_distant_groups_stay_separate(self):
        left = [BBox(x0=0.05 + i * 0.01, y0=0.1, x1=0.06 + i * 0.01, y1=0.3) for i in range(6)]
        right = [BBox(x0=0.75 + i * 0.01, y0=0.1, x1=0.76 + i * 0.01, y1=0.3) for i in range(6)]
        assert len(cluster_vector_drawings(left + right, min_area_ratio=0.001)) == 2

    def test_stray_rules_are_dropped(self):
        """Two lines are an underline or a table rule, not a chart."""
        rects = [BBox(x0=0.1, y0=0.5, x1=0.9, y1=0.502), BBox(x0=0.1, y0=0.51, x1=0.9, y1=0.512)]
        assert cluster_vector_drawings(rects, min_primitives=4) == []

    def test_tiny_clusters_are_dropped(self):
        rects = [BBox(x0=0.5, y0=0.5, x1=0.505, y1=0.505) for _ in range(10)]
        assert cluster_vector_drawings(rects, min_area_ratio=0.01) == []

    def test_no_input(self):
        assert cluster_vector_drawings([]) == []


class TestTextCoverage:
    def test_prose_panel_is_almost_fully_covered(self):
        """The WHO 'Resources' panel: a filled rectangle behind body text."""
        region = BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9)
        blocks = [block(0.12, 0.12 + i * 0.09, 0.88, 0.20 + i * 0.09) for i in range(8)]
        assert text_coverage(region, blocks) > 0.7

    def test_a_chart_has_low_text_coverage(self):
        """Axis labels and a legend cover only a little of a plot."""
        region = BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.9)
        blocks = [
            block(0.11, 0.85, 0.3, 0.88, text="0  10  20"),
            block(0.11, 0.11, 0.25, 0.14, text="Revenue"),
        ]
        assert text_coverage(region, blocks) < 0.1

    def test_no_text_is_zero(self):
        assert text_coverage(BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0), []) == 0.0

    def test_coverage_never_exceeds_one(self):
        """Overlapping blocks must not report impossible coverage."""
        region = BBox(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
        blocks = [block(0.0, 0.0, 1.0, 1.0) for _ in range(5)]
        assert text_coverage(region, blocks) == 1.0


class TestCaptionMatching:
    FIGURE = BBox(x0=0.2, y0=0.2, x1=0.8, y1=0.6)

    def test_marked_caption_below_is_found(self):
        caption = block(0.2, 0.62, 0.8, 0.65, text="Figure 3: Revenue by region.")
        assert find_caption(self.FIGURE, [caption]) is caption

    def test_caption_in_another_column_is_rejected(self):
        far = block(0.85, 0.62, 0.99, 0.65, text="unrelated column text")
        assert find_caption(self.FIGURE, [far]) is None

    def test_distant_text_is_rejected(self):
        far_below = block(0.2, 0.9, 0.8, 0.95, text="A distant paragraph of body text.")
        assert find_caption(self.FIGURE, [far_below]) is None

    def test_a_heading_is_not_adopted_as_a_caption(self):
        """'Situation update:' sat above a chart and was being stolen as its caption."""
        heading = block(0.2, 0.16, 0.8, 0.19, text="Situation update:", size=16.0)
        assert find_caption(self.FIGURE, [heading], body_font_size=10.0) is None

    def test_a_marked_caption_set_large_is_still_accepted(self):
        marked = block(0.2, 0.62, 0.8, 0.65, text="Figure 4: Big caption.", size=16.0)
        assert find_caption(self.FIGURE, [marked], body_font_size=10.0) is marked

    def test_unmarked_prose_directly_below_is_rejected(self):
        """Perfect geometry is not enough: body text set at body size is prose."""
        prose = block(
            0.2, 0.605, 0.8, 0.68, text="The following section discusses results.", size=10.0
        )
        assert find_caption(self.FIGURE, [prose], body_font_size=10.0) is None

    def test_unmarked_small_type_directly_below_is_accepted(self):
        """Captions are conventionally set smaller than body text."""
        caption = block(0.2, 0.605, 0.8, 0.63, text="Revenue by region, 2019-2023.", size=8.0)
        assert find_caption(self.FIGURE, [caption], body_font_size=10.0) is caption

    def test_marked_caption_wins_over_a_closer_unmarked_block(self):
        closer = block(0.2, 0.601, 0.8, 0.62, text="some adjacent prose")
        marked = block(0.2, 0.63, 0.8, 0.66, text="Figure 5: The real caption.")
        assert find_caption(self.FIGURE, [closer, marked]) is marked

    @pytest.mark.parametrize(
        "text",
        ["Figure 1: x", "Fig. 2. y", "Table 3 - z", "Chart 4: w", "Source: WHO", "Panel (a) —"],
    )
    def test_caption_markers_recognised(self, text):
        assert is_caption_marked(text)

    @pytest.mark.parametrize("text", ["see Figure 3 for details", "The figure shows", "Revenue"])
    def test_cross_references_are_not_captions(self, text):
        assert not is_caption_marked(text)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class TestTableStructuring:
    def test_header_row_becomes_columns(self):
        grid = [["Dept", "Count"], ["Eng", "412"], ["Sales", "233"]]
        table, _ = build_table_data(grid)
        assert table.columns == ["Dept", "Count"]
        assert table.n_rows == 2 and table.n_cols == 2
        assert table.header_rows == 1

    def test_empty_columns_are_dropped(self):
        """PyMuPDF's 'lines' strategy emits an empty column per alignment gap."""
        grid = [["Dept", None, "Count", None], ["Eng", None, "412", None]]
        table, diagnostics = build_table_data(grid)
        assert table.n_cols == 2
        assert diagnostics["dropped_cols"] >= 2

    def test_currency_symbol_columns_are_merged_into_their_value(self):
        """'$' | '22,360' is one cell that alignment split into two columns."""
        grid = [
            ["Item", "", "Amount"],
            ["Premiums written", "$", "22,360"],
            ["Losses incurred", "$", "15,120"],
        ]
        table, diagnostics = build_table_data(grid)
        assert table.n_cols == 2
        assert diagnostics["merged_symbol_cols"] == 1
        assert any("$22,360" in (c or "") for row in table.rows for c in row)

    def test_empty_rows_are_dropped(self):
        grid = [["A", "B"], [None, None], ["1", "2"]]
        _, diagnostics = build_table_data(grid)
        assert diagnostics["dropped_rows"] == 1

    def test_fill_ratio_reflects_sparsity(self):
        grid = [["A", "B", "C"], ["1", None, None], ["2", None, None]]
        table, _ = build_table_data(grid)
        assert 0.0 < table.fill_ratio < 0.5

    def test_completely_empty_grid_yields_an_empty_table(self):
        table, diagnostics = build_table_data([[None, None], [None, None]])
        assert table.n_rows == 0 and table.is_degenerate
        assert "reason" in diagnostics


class TestTableDegeneracy:
    def test_single_column_strip_is_degenerate(self):
        table, _ = build_table_data([["a"], ["b"], ["c"]])
        assert table.is_degenerate

    def test_word_grid_is_degenerate(self):
        """The attention-visualisation figures parsed as 27-column word tables."""
        sentence = [
            "The",
            "Law",
            "will",
            "never",
            "be",
            "perfect",
            "but",
            "its",
            "application",
            "should",
            "be",
            "just",
        ]
        table, _ = build_table_data([sentence, sentence])
        assert table.is_word_grid
        assert table.is_degenerate

    def test_very_wide_grid_is_degenerate(self):
        """No real table in this corpus has 25 columns; that is a rendered line."""
        row = [f"c{i}" for i in range(25)]
        wide, _ = build_table_data([row, row])
        assert wide.is_degenerate

    def test_a_numeric_matrix_is_not_a_word_grid(self):
        """Short cells alone must not condemn a table -- results grids look like this."""
        header = ["model", "d", "h", "ppl", "bleu", "params"]
        rows = [
            ["base", "512", "8", "4.9", "25.8", "65"],
            ["big", "1024", "16", "4.3", "26.4", "213"],
        ]
        table, _ = build_table_data([header, *rows])
        assert not table.is_word_grid
        assert not table.is_degenerate

    def test_a_real_multiword_table_is_kept(self):
        header = ["Energy Source", "Entity", "Location", "Capacity (MW)", "Owned", "Share", "Notes"]
        rows = [
            ["Wind", "PacifiCorp, MEC", "Iowa, Wyoming", "12,524", "12,524", "41%", "n/a"],
            ["Coal", "PacifiCorp, MEC", "Utah, Nevada", "12,174", "7,483", "24%", "n/a"],
        ]
        table, _ = build_table_data([header, *rows])
        assert not table.is_degenerate


class TestMarkdown:
    def test_escapes_pipes_so_a_cell_cannot_break_the_grid(self):
        md = to_markdown(["a", "b"], [["x|y", "z"]])
        assert "x\\|y" in md
        # Every row must still declare exactly two columns: count only the
        # unescaped delimiters, since the escaped one is cell content.
        for line in md.splitlines():
            delimiters = len(line.replace("\\|", "")) - len(
                line.replace("\\|", "").replace("|", "")
            )
            assert delimiters == 3, line

    def test_newlines_in_cells_are_flattened(self):
        assert "\n" not in to_markdown(["a"], [["x\ny"]]).splitlines()[2]

    def test_none_cells_render_blank(self):
        assert "|  |" in to_markdown(["a", "b"], [[None, None]])

    def test_empty_table_is_empty_string(self):
        assert to_markdown([], []) == ""


class TestTableClassification:
    def test_register_map(self):
        result = classify_table(["Bits", "Field", "Reset"], [["31:16", "DATA", "0x0000"]])
        assert result.label == TableType.REGISTER.value

    def test_financial_statement(self):
        result = classify_table(
            ["Item", "2023", "2022"], [["Revenue", "$1,000", "$900"], ["Loss", "(45)", "(30)"]]
        )
        assert result.label == TableType.FINANCIAL.value

    def test_numeric_matrix(self):
        rows = [[str(i * j) for j in range(4)] for i in range(1, 5)]
        result = classify_table(["a", "b", "c", "d"], rows)
        assert result.label == TableType.MATRIX.value

    def test_empty_table_is_unknown_with_zero_confidence(self):
        result = classify_table([], [])
        assert result.label == TableType.UNKNOWN.value and result.confidence == 0.0

    def test_every_classification_records_its_evidence(self):
        assert classify_table(["Bits"], [["0x1"]]).evidence


# ---------------------------------------------------------------------------
# Figure classification and confidence
# ---------------------------------------------------------------------------


class TestFigureClassification:
    def test_caption_wording_dominates(self):
        result = classify_figure(caption="Figure 2: Revenue growth chart", is_vector=False)
        assert result.label == FigureType.CHART.value
        assert result.confidence >= 0.8

    def test_diagram_from_caption(self):
        result = classify_figure(caption="Figure 1: The Transformer - model architecture.")
        assert result.label == FigureType.DIAGRAM.value

    def test_map_from_caption(self):
        assert (
            classify_figure(caption="Figure 4: Cases by country (map)").label
            == FigureType.MAP.value
        )

    def test_abstains_without_evidence(self):
        """An honest UNKNOWN beats a confident wrong label: Method 2 routes on this."""
        result = classify_figure()
        assert result.label == FigureType.UNKNOWN.value
        assert result.confidence == 0.0

    def test_photograph_from_colour_statistics(self):
        result = classify_figure(is_vector=False, n_colors=20000, mean_saturation=0.4)
        assert result.label == FigureType.PHOTO.value

    def test_vector_art_is_never_called_a_photo(self):
        result = classify_figure(is_vector=True, n_colors=30000, mean_saturation=0.9)
        assert result.label != FigureType.PHOTO.value

    def test_small_wide_marks_are_logos(self):
        result = classify_figure(area_ratio=0.004, aspect_ratio=8.0)
        assert result.label == FigureType.LOGO.value


class TestConfidenceScores:
    def test_clean_text_scores_high(self):
        assert text_confidence(replacement_ratio=0.0, n_chars=200, alpha_ratio=0.8) == 1.0

    def test_unmappable_glyphs_drive_confidence_down(self):
        assert text_confidence(replacement_ratio=0.34, n_chars=200, alpha_ratio=0.8) == 0.0

    def test_short_fragments_are_penalised(self):
        assert text_confidence(replacement_ratio=0.0, n_chars=5, alpha_ratio=0.9) < 1.0

    def test_empty_text_is_zero(self):
        assert text_confidence(replacement_ratio=0.0, n_chars=0, alpha_ratio=0.0) == 0.0

    def test_sparse_tables_score_below_dense_ones(self):
        sparse = table_confidence(fill_ratio=0.2, n_rows=5, n_cols=5, has_header=True)
        dense = table_confidence(fill_ratio=1.0, n_rows=5, n_cols=5, has_header=True)
        assert sparse < dense <= 1.0

    def test_one_by_n_strips_are_penalised(self):
        assert table_confidence(fill_ratio=1.0, n_rows=1, n_cols=5, has_header=False) < 0.8

    def test_a_captioned_figure_outscores_an_uncaptioned_one(self):
        with_caption = figure_confidence(
            area_ratio=0.2, has_caption=True, has_derived_text=False, is_vector=False
        )
        without = figure_confidence(
            area_ratio=0.2, has_caption=False, has_derived_text=False, is_vector=False
        )
        assert with_caption > without

    @pytest.mark.parametrize("area", [0.0, 0.01, 0.5, 1.0])
    def test_all_confidences_stay_in_range(self, area):
        value = figure_confidence(
            area_ratio=area, has_caption=True, has_derived_text=True, is_vector=True
        )
        assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("D:20240705120000+04'00'", (2024, 7, 5)),
            ("D:20200121", (2020, 1, 21)),
            ("D:2019", (2019, 1, 1)),
        ],
    )
    def test_pdf_dates_parsed(self, raw, expected):
        parsed = parse_pdf_date(raw)
        assert (parsed.year, parsed.month, parsed.day) == expected

    @pytest.mark.parametrize("raw", [None, "", "not a date", "D:18000101"])
    def test_unusable_dates_return_none(self, raw):
        assert parse_pdf_date(raw) is None

    def test_authors_split_on_common_separators(self):
        assert parse_authors("Jane Doe; John Smith and Ada Lovelace") == [
            "Jane Doe",
            "John Smith",
            "Ada Lovelace",
        ]

    @pytest.mark.parametrize(
        "raw", ["Adobe Acrobat", "Microsoft Word", "pdfTeX-1.40", "LaTeX with hyperref"]
    )
    def test_typesetting_tools_are_not_authors(self, raw):
        assert parse_authors(raw) == []

    @pytest.mark.parametrize(
        ("title", "domain", "expected"),
        [
            ("Berkshire Hathaway 2023 Annual Report", "finance", DocumentType.ANNUAL_REPORT),
            ("RP2040 Datasheet", "technical", DocumentType.DATASHEET),
            ("NASA Systems Engineering Handbook", "technical", DocumentType.TECHNICAL_MANUAL),
            ("Situation Report 1", "health", DocumentType.SITUATION_REPORT),
            ("Attention Is All You Need", "science", DocumentType.RESEARCH_PAPER),
        ],
    )
    def test_document_type_inference(self, title, domain, expected):
        assert infer_document_type(title, domain)[0] is expected

    def test_inference_always_explains_itself(self):
        _, evidence = infer_document_type("Some Untyped Paper", None)
        assert evidence
