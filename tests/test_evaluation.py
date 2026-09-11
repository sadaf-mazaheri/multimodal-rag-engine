"""Gold-set schema and retrieval metrics.

Metric expectations are hand-computed and written out, so the tests fail when
the arithmetic changes rather than restating whatever the implementation does.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from mmrag.evaluation.gold import (
    Evidence,
    GoldQuery,
    GoldSet,
    validate_against_chunks,
)
from mmrag.evaluation.metrics import (
    aggregate,
    hit_rate_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)
from mmrag.schemas import BBox, Chunk, ChunkType


def chunk(doc="arxiv_attention", page=3, kind=ChunkType.TEXT, cid="c1") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        page_number=page,
        chunk_type=kind,
        text="body",
        element_ids=[f"{doc}#p{page}#text000"],
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
        variant="method1",
    )


# ---------------------------------------------------------------- metrics


class TestRecall:
    def test_counts_distinct_gold_entries_not_relevant_results(self):
        """Two chunks from one gold page are one unit of evidence, not two.

        The bug this guards: counting relevant *results* reports recall 2.0 for
        a single-evidence query, because a page yields several chunks.
        """
        assert recall_at_k([0, 0, 0], n_gold=1, k=10) == 1.0

    def test_partial_coverage(self):
        assert recall_at_k([0, None, 1], n_gold=4, k=10) == 0.5

    def test_respects_the_cutoff(self):
        assert recall_at_k([None, None, 0], n_gold=1, k=2) == 0.0
        assert recall_at_k([None, None, 0], n_gold=1, k=3) == 1.0

    def test_nothing_relevant_is_zero(self):
        assert recall_at_k([None, None], n_gold=2, k=10) == 0.0

    def test_rejects_a_gold_query_with_no_evidence(self):
        with pytest.raises(ValueError, match="at least one evidence"):
            recall_at_k([0], n_gold=0, k=5)

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k([0], n_gold=1, k=0)


class TestPrecision:
    def test_counts_results_not_distinct_evidence(self):
        """Deliberately unlike recall: two useful results are two useful results."""
        assert precision_at_k([0, 0, None, None], k=4) == 0.5

    def test_shorter_result_list_than_k_uses_what_there_is(self):
        assert precision_at_k([0, None], k=10) == 0.5

    def test_empty_results(self):
        assert precision_at_k([], k=10) == 0.0


class TestHitRate:
    def test_any_relevant_hit_counts(self):
        assert hit_rate_at_k([None, None, 2], k=3) == 1.0

    def test_outside_the_cutoff_does_not(self):
        assert hit_rate_at_k([None, None, 2], k=2) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank([0, None]) == 1.0

    def test_third_position(self):
        assert reciprocal_rank([None, None, 1]) == pytest.approx(1 / 3)

    def test_no_relevant_result(self):
        assert reciprocal_rank([None, None]) == 0.0

    def test_uses_the_first_hit_not_the_best_entry(self):
        assert reciprocal_rank([None, 1, 0]) == pytest.approx(0.5)


class TestNdcg:
    def test_perfect_single_evidence_ranking(self):
        assert ndcg_at_k([0, None, None], n_gold=1, k=10) == pytest.approx(1.0)

    def test_discounts_by_position(self):
        # one gold entry found at rank 3: dcg = 1/log2(4), ideal = 1/log2(2)
        expected = (1 / math.log2(4)) / (1 / math.log2(2))
        assert ndcg_at_k([None, None, 0], n_gold=1, k=10) == pytest.approx(expected)

    def test_a_repeated_gold_entry_earns_no_second_gain(self):
        once = ndcg_at_k([0, None], n_gold=1, k=10)
        twice = ndcg_at_k([0, 0], n_gold=1, k=10)
        assert once == pytest.approx(twice)

    def test_two_entries_in_order_is_perfect(self):
        assert ndcg_at_k([0, 1], n_gold=2, k=10) == pytest.approx(1.0)

    def test_ideal_is_capped_by_k(self):
        """With k=1 the best reachable is one entry, so finding one is perfect."""
        assert ndcg_at_k([0], n_gold=5, k=1) == pytest.approx(1.0)


class TestSummarizeAndAggregate:
    def test_summarize_keys_are_named_by_k(self):
        out = summarize([0, None], n_gold=1, k_values=[1, 5])
        assert set(out) == {
            "mrr",
            "recall@1", "precision@1", "hit_rate@1", "ndcg@1",
            "recall@5", "precision@5", "hit_rate@5", "ndcg@5",
        }

    def test_aggregate_is_a_macro_average(self):
        """Every query counts once, whatever its evidence count."""
        out = aggregate([{"recall@5": 1.0}, {"recall@5": 0.0}])
        assert out["recall@5"] == 0.5

    def test_aggregate_treats_a_missing_key_as_zero(self):
        out = aggregate([{"recall@5": 1.0}, {"mrr": 1.0}])
        assert out["recall@5"] == 0.5

    def test_aggregate_of_nothing_is_empty(self):
        assert aggregate([]) == {}

    def test_mean_of_an_empty_slice_is_zero(self):
        """A slice with no queries is a normal state, not an error."""
        assert mean([]) == 0.0


# ---------------------------------------------------------------- gold set


class TestEvidenceMatching:
    def test_matches_on_document_page_and_modality(self):
        entry = Evidence(doc_id="arxiv_attention", page=3, modality="figure")
        assert entry.matches(chunk(page=3, kind=ChunkType.FIGURE))

    def test_rejects_the_wrong_modality_on_the_right_page(self):
        entry = Evidence(doc_id="arxiv_attention", page=3, modality="figure")
        assert not entry.matches(chunk(page=3, kind=ChunkType.TEXT))

    def test_rejects_the_right_modality_on_the_wrong_page(self):
        entry = Evidence(doc_id="arxiv_attention", page=3, modality="figure")
        assert not entry.matches(chunk(page=4, kind=ChunkType.FIGURE))

    def test_rejects_another_document(self):
        entry = Evidence(doc_id="arxiv_attention", page=3)
        assert not entry.matches(chunk(doc="arxiv_rag", page=3))

    def test_omitting_modality_accepts_any(self):
        entry = Evidence(doc_id="arxiv_attention", page=3)
        assert entry.matches(chunk(page=3, kind=ChunkType.TABLE))
        assert entry.matches(chunk(page=3, kind=ChunkType.TEXT))


class TestGoldQuery:
    def test_matched_returns_evidence_indices_in_rank_order(self):
        query = GoldQuery(
            id="q1", query="?", stratum="natural", requires="figure",
            evidence=[
                Evidence(doc_id="arxiv_attention", page=3, modality="figure"),
                Evidence(doc_id="arxiv_attention", page=4, modality="figure"),
            ],
        )
        chunks = [
            chunk(page=9, kind=ChunkType.TEXT),
            chunk(page=4, kind=ChunkType.FIGURE),
            chunk(page=3, kind=ChunkType.FIGURE),
        ]
        assert query.matched(chunks) == [None, 1, 0]

    def test_requires_must_appear_in_the_evidence(self):
        with pytest.raises(ValidationError, match="no evidence entry has that modality"):
            GoldQuery(
                id="q1", query="?", stratum="table", requires="table",
                evidence=[Evidence(doc_id="d", page=1, modality="text")],
            )

    def test_untyped_evidence_does_not_trip_the_consistency_check(self):
        query = GoldQuery(
            id="q1", query="?", stratum="natural", requires="figure",
            evidence=[Evidence(doc_id="d", page=1)],
        )
        assert query.requires == "figure"

    def test_evidence_is_required(self):
        with pytest.raises(ValidationError):
            GoldQuery(id="q1", query="?", stratum="text", requires="text", evidence=[])

    def test_unknown_fields_are_rejected(self):
        """A typo in a hand-authored file must fail loudly, not be ignored."""
        with pytest.raises(ValidationError):
            Evidence(doc_id="d", page=1, modalty="figure")


class TestGoldSet:
    def build(self, **overrides):
        payload = {
            "version": 1,
            "queries": [
                {
                    "id": "q1", "query": "a", "stratum": "text", "requires": "text",
                    "evidence": [{"doc_id": "arxiv_attention", "page": 6, "modality": "text"}],
                },
                {
                    "id": "q2", "query": "b", "stratum": "figure", "requires": "figure",
                    "evidence": [{"doc_id": "arxiv_attention", "page": 3, "modality": "figure"}],
                },
            ],
        }
        payload.update(overrides)
        return GoldSet.model_validate(payload)

    def test_duplicate_ids_are_rejected(self):
        with pytest.raises(ValidationError, match="duplicate query ids"):
            self.build(queries=[
                {"id": "q1", "query": "a", "stratum": "text", "requires": "text",
                 "evidence": [{"doc_id": "d", "page": 1, "modality": "text"}]},
                {"id": "q1", "query": "b", "stratum": "text", "requires": "text",
                 "evidence": [{"doc_id": "d", "page": 2, "modality": "text"}]},
            ])

    def test_slicing_by_stratum_and_requirement(self):
        gold = self.build()
        assert [q.id for q in gold.by_stratum("figure")] == ["q2"]
        assert [q.id for q in gold.by_requires("text")] == ["q1"]

    def test_counts_report_composition(self):
        assert self.build().counts() == {
            "stratum": {"text": 1, "figure": 1},
            "requires": {"text": 1, "figure": 1},
        }

    def test_roundtrips_through_yaml(self, tmp_path):
        import yaml

        path = tmp_path / "gold.yaml"
        path.write_text(yaml.safe_dump(self.build().model_dump(mode="json")), encoding="utf-8")
        assert len(GoldSet.load(path).queries) == 2

    def test_a_non_mapping_file_is_rejected(self, tmp_path):
        path = tmp_path / "gold.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            GoldSet.load(path)


class TestValidationAgainstChunks:
    gold = GoldSet.model_validate({
        "version": 1,
        "queries": [{
            "id": "q1", "query": "a", "stratum": "figure", "requires": "figure",
            "evidence": [{"doc_id": "arxiv_attention", "page": 3, "modality": "figure"}],
        }],
    })

    def test_resolvable_evidence_reports_no_problems(self):
        chunks = [chunk(page=3, kind=ChunkType.FIGURE)]
        assert validate_against_chunks(self.gold, chunks) == []

    def test_unknown_document_is_reported(self):
        problems = validate_against_chunks(self.gold, [chunk(doc="other", page=3)])
        assert [p.kind for p in problems] == ["unknown_document"]

    def test_missing_page_is_reported(self):
        problems = validate_against_chunks(self.gold, [chunk(page=99, kind=ChunkType.FIGURE)])
        assert [p.kind for p in problems] == ["no_such_page"]

    def test_page_present_but_wrong_modality_is_reported(self):
        """The failure mode a re-ingest actually produces."""
        problems = validate_against_chunks(self.gold, [chunk(page=3, kind=ChunkType.TEXT)])
        assert [p.kind for p in problems] == ["no_such_modality"]
        assert "the page exists but the evidence does not" in problems[0].detail
