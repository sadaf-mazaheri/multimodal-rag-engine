"""Tests for BM25, rank fusion, and the hybrid retriever.

Fusion gets the closest attention. It is pure arithmetic with no model in the
loop, so it can be verified exactly rather than merely smoke-tested -- and a
fusion bug is the kind that degrades results without ever raising, which would
quietly invalidate every comparison in Step 6.
"""

from __future__ import annotations

import pytest

from mmrag.retrieval.fusion import (
    RankedList,
    fusion_diagnostics,
    reciprocal_rank_fusion,
)
from mmrag.stores.bm25 import BM25Index, normalize_scores, tokenize

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Revenue GREW") == ["revenue", "grew"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("GPIO_OE register", ["gpio_oe", "register"]),
            ("version 1.5.2 shipped", ["version", "1.5.2", "shipped"]),
            ("total was 22,360 dollars", ["total", "was", "22,360", "dollars"]),
        ],
    )
    def test_preserves_technical_identifiers(self, text, expected):
        """These are exactly the tokens BM25 exists to match literally."""
        assert tokenize(text) == expected

    def test_drops_punctuation(self):
        assert tokenize("Hello, world! (again)") == ["hello", "world", "again"]

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("!!! ???") == []


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


@pytest.fixture
def bm25() -> BM25Index:
    index = BM25Index(k1=1.5, b=0.75)
    index.build(
        ["c1", "c2", "c3", "c4"],
        [
            "The Transformer architecture uses multi-head self attention.",
            "Revenue grew to 22,360 million dollars in the fourth quarter.",
            "The GPIO_OE register controls output enable for each pin.",
            "Climate projections show warming under every emissions scenario.",
        ],
    )
    return index


class TestBM25:
    def test_finds_the_lexically_matching_chunk(self, bm25):
        hits = bm25.search("multi-head attention", k=3)
        assert hits and hits[0].chunk_id == "c1"

    def test_matches_an_exact_identifier(self, bm25):
        """The case dense retrieval is weakest on and BM25 is strongest on."""
        hits = bm25.search("GPIO_OE", k=3)
        assert hits[0].chunk_id == "c3"

    def test_matches_an_exact_figure(self, bm25):
        hits = bm25.search("22,360", k=3)
        assert hits[0].chunk_id == "c2"

    def test_ranks_are_sequential_from_one(self, bm25):
        hits = bm25.search("the", k=4)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_scores_are_descending(self, bm25):
        hits = bm25.search("attention transformer architecture", k=4)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_nothing_rather_than_padding(self, bm25):
        """Zero-score padding entering fusion would fabricate false evidence."""
        assert bm25.search("zzzzz nonexistent term", k=4) == []

    def test_empty_query(self, bm25):
        assert bm25.search("!!!", k=3) == []

    def test_k_larger_than_the_corpus_is_safe(self, bm25):
        assert len(bm25.search("the", k=100)) <= 4

    def test_building_over_nothing_is_an_error(self):
        with pytest.raises(ValueError, match="zero chunks"):
            BM25Index().build([], [])

    def test_mismatched_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="2 ids but 1 texts"):
            BM25Index().build(["a", "b"], ["only one"])

    def test_searching_before_building_is_an_error(self):
        with pytest.raises(RuntimeError, match="not built"):
            BM25Index().search("anything")

    def test_round_trips_through_disk(self, bm25, tmp_path):
        bm25.save(tmp_path / "idx")
        loaded = BM25Index.load(tmp_path / "idx")
        assert loaded.chunk_ids == bm25.chunk_ids
        assert (loaded.k1, loaded.b) == (bm25.k1, bm25.b)
        assert loaded.search("GPIO_OE", k=2)[0].chunk_id == "c3"

    def test_loading_a_missing_index_says_what_to_do(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="mmrag index build"):
            BM25Index.load(tmp_path / "nope")


class TestNormalizeScores:
    def test_maps_to_unit_range(self):
        out = normalize_scores([1.0, 3.0, 5.0])
        assert out.min() == 0.0 and out.max() == 1.0

    def test_identical_scores_do_not_divide_by_zero(self):
        assert list(normalize_scores([2.0, 2.0])) == [1.0, 1.0]

    def test_empty(self):
        assert len(normalize_scores([])) == 0


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------


class TestRRF:
    def test_score_matches_the_formula(self):
        lists = [RankedList("bm25", ["a"]), RankedList("dense", ["a"])]
        result = reciprocal_rank_fusion(lists, k=60)[0]
        assert result.score == pytest.approx(1 / 61 + 1 / 61)

    def test_agreement_beats_a_single_strong_vote(self):
        """The property that makes hybrid retrieval worth having."""
        lists = [
            RankedList("bm25", ["solo", "agreed"]),
            RankedList("dense", ["other", "agreed"]),
        ]
        results = reciprocal_rank_fusion(lists, k=60)
        assert results[0].chunk_id == "agreed"

    def test_records_which_retriever_supplied_each_rank(self):
        lists = [
            RankedList("bm25", ["a", "b"]),
            RankedList("dense", ["b"]),
        ]
        by_id = {r.chunk_id: r for r in reciprocal_rank_fusion(lists)}
        assert by_id["b"].component_ranks == {"bm25": 2, "dense": 1}
        assert by_id["a"].component_ranks == {"bm25": 1}
        assert by_id["b"].retrievers == ["bm25", "dense"]

    def test_weights_scale_a_retriever_s_influence(self):
        lists = [RankedList("bm25", ["a"]), RankedList("dense", ["b"])]
        results = reciprocal_rank_fusion(lists, weights={"bm25": 2.0, "dense": 1.0})
        assert results[0].chunk_id == "a"

    def test_zero_weight_excludes_a_retriever_entirely(self):
        lists = [RankedList("bm25", ["a"]), RankedList("dense", ["b"])]
        results = reciprocal_rank_fusion(lists, weights={"bm25": 0.0})
        assert [r.chunk_id for r in results] == ["b"]

    def test_lower_k_sharpens_the_advantage_of_rank_one(self):
        lists = [RankedList("bm25", ["a", "b", "c", "d", "e"])]
        gap_small_k = (
            reciprocal_rank_fusion(lists, k=1)[0].score
            - reciprocal_rank_fusion(lists, k=1)[1].score
        )
        gap_large_k = (
            reciprocal_rank_fusion(lists, k=200)[0].score
            - reciprocal_rank_fusion(lists, k=200)[1].score
        )
        assert gap_small_k > gap_large_k

    def test_ranks_are_sequential(self):
        lists = [RankedList("bm25", ["a", "b", "c"]), RankedList("dense", ["c", "d"])]
        results = reciprocal_rank_fusion(lists)
        assert [r.rank for r in results] == [1, 2, 3, 4]

    def test_output_is_deterministic(self):
        """Two runs of one config must produce identical metrics."""
        lists = [
            RankedList("bm25", ["a", "b", "c", "d"]),
            RankedList("dense", ["d", "c", "b", "a"]),
        ]
        first = [r.chunk_id for r in reciprocal_rank_fusion(lists)]
        second = [r.chunk_id for r in reciprocal_rank_fusion(lists)]
        assert first == second

    def test_ties_break_deterministically_by_best_rank_then_id(self):
        lists = [RankedList("bm25", ["b", "a"]), RankedList("dense", ["a", "b"])]
        results = reciprocal_rank_fusion(lists)
        assert results[0].score == pytest.approx(results[1].score)
        assert [r.chunk_id for r in results] == ["a", "b"]

    def test_duplicate_ids_keep_the_best_rank(self):
        lists = [RankedList("bm25", ["a", "b", "a"])]
        by_id = {r.chunk_id: r for r in reciprocal_rank_fusion(lists)}
        assert by_id["a"].component_ranks["bm25"] == 1

    def test_top_k_truncates_after_ordering(self):
        lists = [RankedList("bm25", ["a", "b", "c", "d", "e"])]
        assert len(reciprocal_rank_fusion(lists, top_k=2)) == 2

    def test_empty_lists_produce_no_results(self):
        assert reciprocal_rank_fusion([RankedList("bm25", [])]) == []
        assert reciprocal_rank_fusion([]) == []

    def test_invalid_k_is_rejected(self):
        with pytest.raises(ValueError, match="rrf k must be >= 1"):
            reciprocal_rank_fusion([RankedList("bm25", ["a"])], k=0)


class TestFusionDiagnostics:
    def test_attributes_results_to_their_retrievers(self):
        lists = [
            RankedList("bm25", ["a", "b"]),
            RankedList("dense", ["b", "c"]),
        ]
        results = reciprocal_rank_fusion(lists)
        diagnostics = fusion_diagnostics(results, lists)

        assert diagnostics["contributed"] == {"bm25": 2, "dense": 2}
        assert diagnostics["found_only_by"] == {"bm25": 1, "dense": 1}
        assert diagnostics["found_by_all"] == 1

    def test_reveals_a_hybrid_that_has_degenerated_to_one_retriever(self):
        """Without this, a fusion doing nothing looks like a working hybrid."""
        lists = [RankedList("bm25", ["a", "b", "c"]), RankedList("dense", [])]
        results = reciprocal_rank_fusion(lists)
        diagnostics = fusion_diagnostics(results, lists)
        assert diagnostics["contributed"]["dense"] == 0
        assert diagnostics["found_only_by"]["bm25"] == 3
