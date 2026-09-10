"""Tests for Method 2's query router.

The router is the component that most easily fails *silently*. A router that
always fires everything still returns good results -- it has just stopped being
a router, and Method 2 has quietly become "Method 1 with more indexes". So these
tests check not only that the right modalities fire, but that the router
actually discriminates.
"""

from __future__ import annotations

import pytest

from mmrag.config import RouterConfig
from mmrag.retrieval.base import MetadataFilter
from mmrag.retrieval.router import HeuristicRouter
from mmrag.schemas import Modality


@pytest.fixture
def router() -> HeuristicRouter:
    return HeuristicRouter(RouterConfig(strategy="heuristic", fallback_to_all=True))


@pytest.fixture
def strict() -> HeuristicRouter:
    """No fallback, so routing decisions are visible in isolation."""
    return HeuristicRouter(RouterConfig(strategy="heuristic", fallback_to_all=False))


class TestModalitySelection:
    def test_text_always_fires(self, strict):
        """Text is the safety net; a missed answer cannot be recovered later."""
        for query in ("show me figure 3", "what is in the revenue table", "summarize this"):
            assert Modality.TEXT in strict.route(query).modalities

    @pytest.mark.parametrize(
        "query",
        [
            "which table lists headcount by department",
            "how many rows are in the results table",
            "what is the total revenue broken down per segment",
        ],
    )
    def test_table_queries_route_to_tables(self, strict, query):
        assert Modality.TABLE in strict.route(query).modalities

    @pytest.mark.parametrize(
        "query",
        [
            "what does the architecture diagram show",
            "describe the chart of emissions over time",
            "which figure illustrates the attention mechanism",
        ],
    )
    def test_visual_queries_route_to_images(self, strict, query):
        assert Modality.IMAGE in strict.route(query).modalities

    def test_prose_question_does_not_fire_table_or_image(self, strict):
        """The discriminating case: if this fans out, the router is doing nothing."""
        decision = strict.route("Why did the committee revise its guidance?")
        assert decision.modalities == [Modality.TEXT]

    def test_explicit_figure_number_is_conclusive(self, strict):
        decision = strict.route("What is shown in Figure 3?")
        assert Modality.IMAGE in decision.modalities
        assert decision.scores[Modality.IMAGE.value] == 1.0

    def test_explicit_table_number_routes_to_tables_not_images(self, strict):
        decision = strict.route("What is in Table 2?")
        assert Modality.TABLE in decision.modalities
        assert Modality.IMAGE not in decision.modalities

    def test_strategy_all_fires_everything(self):
        router = HeuristicRouter(RouterConfig(strategy="all"))
        decision = router.route("anything at all")
        assert set(decision.modalities) == {Modality.TEXT, Modality.TABLE, Modality.IMAGE}
        assert decision.strategy == "all"


class TestFallback:
    def test_ambiguous_query_fans_out(self, router):
        """A false negative is unrecoverable; a false positive costs latency."""
        decision = router.route("2019-nCoV case counts")
        assert len(decision.modalities) == 3
        assert decision.fell_back

    def test_clearly_textual_query_suppresses_the_fallback(self, router):
        decision = router.route("Summarize the committee's reasoning")
        assert decision.modalities == [Modality.TEXT]
        assert not decision.fell_back

    def test_a_confident_signal_suppresses_the_fallback(self, router):
        decision = router.route("which table shows revenue by segment")
        assert not decision.fell_back
        assert set(decision.modalities) == {Modality.TEXT, Modality.TABLE}

    def test_fallback_disabled_leaves_text_alone(self, strict):
        assert strict.route("2019-nCoV case counts").modalities == [Modality.TEXT]


class TestScoringAndSignals:
    def test_signals_are_recorded_for_analysis(self, strict):
        decision = strict.route("which chart shows the trend")
        assert "chart" in " ".join(decision.signals[Modality.IMAGE.value]).lower()

    def test_scores_saturate_at_one(self, strict):
        decision = strict.route("table rows columns cells totals breakdown per segment")
        assert decision.scores[Modality.TABLE.value] == 1.0

    def test_many_weak_hints_do_not_outrank_one_strong_word(self, strict):
        """Saturation exists so 'shows' five times cannot beat 'diagram' once."""
        weak = strict.route("it shows the trend").scores[Modality.IMAGE.value]
        strong = strict.route("the diagram").scores[Modality.IMAGE.value]
        assert strong > weak

    def test_confidence_ignores_the_always_on_text_score(self, strict):
        assert strict.route("Why did this happen?").confidence == 0.0

    def test_decision_serialises_for_the_run_record(self, strict):
        payload = strict.route("which figure shows revenue").as_dict()
        assert set(payload) >= {"modalities", "scores", "signals", "strategy", "confidence"}


class TestFilters:
    def test_explicit_page_reference_becomes_a_filter(self, router):
        assert router.route("what is on page 12").filters.page_numbers == [12]

    def test_a_bare_number_is_not_a_page_filter(self, router):
        """Treating '12' as a page would silently discard every other page."""
        assert router.route("what happened in 12 countries").filters.page_numbers is None

    def test_caller_document_filter_is_preserved(self, router):
        base = MetadataFilter(doc_ids=["arxiv_attention"])
        decision = router.route("what is attention", base_filters=base)
        assert decision.filters.doc_ids == ["arxiv_attention"]


class TestMetadataFilterMerge:
    def test_merge_intersects_rather_than_widens(self):
        """A caller's --doc-id must never be broadened by an inferred filter."""
        caller = MetadataFilter(doc_ids=["a"])
        inferred = MetadataFilter(doc_ids=["a", "b"])
        assert inferred.merge(caller).doc_ids == ["a"]

    def test_merge_keeps_fields_only_one_side_sets(self):
        merged = MetadataFilter(page_numbers=[3]).merge(MetadataFilter(doc_ids=["a"]))
        assert merged.page_numbers == [3]
        assert merged.doc_ids == ["a"]

    def test_disjoint_filters_intersect_to_nothing(self):
        merged = MetadataFilter(doc_ids=["a"]).merge(MetadataFilter(doc_ids=["b"]))
        assert merged.doc_ids == []

    def test_is_empty(self):
        assert MetadataFilter().is_empty
        assert not MetadataFilter(doc_ids=["a"]).is_empty
