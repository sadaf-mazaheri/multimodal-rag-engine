"""The evaluation runner and report.

Driven by a stub method, so the runner's behaviour is asserted without a built
index, a model, or twenty minutes of cross-encoder time.
"""

from __future__ import annotations

import pytest

from mmrag.config import load_experiment_config
from mmrag.evaluation.gold import GoldSet
from mmrag.evaluation.report import (
    disagreements,
    headline_table,
    latency_table,
    load_run,
    save_run,
    to_markdown,
)
from mmrag.evaluation.retrieval_eval import RetrievalRun, evaluate_query, run_evaluation
from mmrag.schemas import BBox, Chunk, ChunkType, Modality, ScoredChunk


def make_chunk(doc: str, page: int, kind: ChunkType, cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        page_number=page,
        chunk_type=kind,
        text=f"{doc} p{page}",
        element_ids=[f"{doc}#p{page}#x000"],
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
        variant="method1",
    )


class StubResult:
    def __init__(self, chunks, latency=None, routing=None):
        self.results = [
            ScoredChunk(chunk=c, score=1.0 / i, rank=i, retriever="stub",
                        modality=Modality.TEXT)
            for i, c in enumerate(chunks, 1)
        ]
        self.latency_ms = latency or {"total_ms": 10.0}
        self.routing = routing


class StubMethod:
    """Returns a fixed ranked list. Records whether use_metadata was passed."""

    name = "stub"

    def __init__(self, chunks, *, accepts_metadata=False, latency=None):
        self._chunks = chunks
        self.seen_metadata: bool | None = None
        self._latency = latency
        if accepts_metadata:
            self.retrieve = self._with_metadata  # type: ignore[method-assign]

    def retrieve(self, query: str, *, top_k: int | None = None):  # noqa: ARG002
        return StubResult(self._chunks[: top_k or 10], self._latency)

    def _with_metadata(self, query: str, *, top_k=None, use_metadata=True):  # noqa: ARG002
        self.seen_metadata = use_metadata
        return StubResult(self._chunks[: top_k or 10], self._latency)


GOLD = GoldSet.model_validate({
    "version": 1,
    "queries": [
        {
            "id": "q1", "query": "a figure question", "stratum": "figure",
            "requires": "figure",
            "evidence": [{"doc_id": "d1", "page": 3, "modality": "figure"}],
        },
        {
            "id": "q2", "query": "a text question", "stratum": "natural",
            "requires": "text",
            "evidence": [{"doc_id": "d1", "page": 7, "modality": "text"}],
        },
    ],
})


@pytest.fixture
def config():
    return load_experiment_config("method1")


class TestEvaluateQuery:
    def test_scores_a_hit_at_rank_one(self, config):
        target = make_chunk("d1", 3, ChunkType.FIGURE, "c1")
        method = StubMethod([target])
        result = evaluate_query(
            method, GOLD.queries[0], top_k=10, k_values=[1, 5, 10]
        )
        assert result.metrics["recall@10"] == 1.0
        assert result.metrics["mrr"] == 1.0
        assert result.retrieved[0].matched == 0

    def test_scores_a_miss(self, config):
        wrong = make_chunk("d1", 99, ChunkType.FIGURE, "c1")
        result = evaluate_query(
            StubMethod([wrong]), GOLD.queries[0], top_k=10, k_values=[10]
        )
        assert result.metrics["recall@10"] == 0.0
        assert result.retrieved[0].matched is None

    def test_carries_the_gold_labels_through(self, config):
        result = evaluate_query(
            StubMethod([]), GOLD.queries[1], top_k=10, k_values=[10]
        )
        assert (result.stratum, result.requires) == ("natural", "text")

    def test_use_metadata_is_passed_only_when_accepted(self):
        """Method 1 has no such parameter; passing it would be a TypeError."""
        chunks = [make_chunk("d1", 3, ChunkType.FIGURE, "c1")]
        plain = StubMethod(chunks)
        evaluate_query(plain, GOLD.queries[0], top_k=5, k_values=[5], use_metadata=False)
        assert plain.seen_metadata is None

        aware = StubMethod(chunks, accepts_metadata=True)
        evaluate_query(aware, GOLD.queries[0], top_k=5, k_values=[5], use_metadata=False)
        assert aware.seen_metadata is False

    def test_routing_is_recorded_when_present(self, config):
        class Routed(StubMethod):
            def retrieve(self, query, *, top_k=None):  # noqa: ARG002
                class R:
                    def as_dict(self):
                        return {"fell_back": True, "modalities": ["text", "image"]}
                return StubResult(self._chunks, routing=R())

        result = evaluate_query(
            Routed([make_chunk("d1", 3, ChunkType.FIGURE, "c1")]),
            GOLD.queries[0], top_k=5, k_values=[5],
        )
        assert result.routing == {"fell_back": True, "modalities": ["text", "image"]}


class TestRunEvaluation:
    def build(self, config, **kwargs):
        chunks = [
            make_chunk("d1", 3, ChunkType.FIGURE, "c1"),
            make_chunk("d1", 7, ChunkType.TEXT, "c2"),
        ]
        return run_evaluation(
            StubMethod(chunks, **kwargs), GOLD, config, config_name="method1"
        )

    def test_scores_every_query(self, config):
        run = self.build(config)
        assert [q.query_id for q in run.per_query] == ["q1", "q2"]

    def test_slices_by_both_axes(self, config):
        run = self.build(config)
        assert set(run.by_stratum) == {"figure", "natural"}
        assert set(run.by_requires) == {"figure", "text"}
        assert run.by_stratum["figure"]["n"] == 1.0

    def test_records_the_config_and_gold_description(self, config):
        run = self.build(config)
        assert run.gold["n_queries"] == 2
        assert run.config["retrieval"]["rerank_enabled"] is True

    def test_reports_latency_percentiles_per_stage(self, config):
        run = self.build(config, latency={"total_ms": 5.0, "rerank_ms": 3.0})
        assert run.latency["total_ms_median"] == 5.0
        assert run.latency["rerank_ms_p90"] == 3.0

    def test_progress_callback_is_invoked_per_query(self, config):
        seen = []
        chunks = [make_chunk("d1", 3, ChunkType.FIGURE, "c1")]
        run_evaluation(
            StubMethod(chunks), GOLD, config, config_name="method1",
            on_query=lambda i, total, r: seen.append((i, total, r.query_id)),
        )
        assert seen == [(1, 2, "q1"), (2, 2, "q2")]

    def test_overrides_are_recorded_in_the_run(self, config):
        run = run_evaluation(
            StubMethod([]), GOLD, config, config_name="method1",
            overrides={"rerank_enabled": False},
        )
        assert run.overrides == {"rerank_enabled": False}


class TestRunPersistence:
    def test_roundtrips_through_json(self, config, tmp_path):
        chunks = [make_chunk("d1", 3, ChunkType.FIGURE, "c1")]
        run = run_evaluation(StubMethod(chunks), GOLD, config, config_name="method1", tag="x")
        path = save_run(run, tmp_path / "run.json")
        again = load_run(path)
        assert again.label() == "stub/x"
        assert again.metrics == run.metrics

    def test_label_without_a_tag_is_just_the_method(self, config):
        run = run_evaluation(StubMethod([]), GOLD, config, config_name="method1")
        assert run.label() == "stub"


class TestReportRendering:
    def runs(self, config):
        good = run_evaluation(
            StubMethod([make_chunk("d1", 3, ChunkType.FIGURE, "c1"),
                        make_chunk("d1", 7, ChunkType.TEXT, "c2")]),
            GOLD, config, config_name="method1", tag="good",
        )
        bad = run_evaluation(
            StubMethod([make_chunk("d1", 99, ChunkType.TEXT, "c9")]),
            GOLD, config, config_name="method1", tag="bad",
        )
        return [good, bad]

    def test_headline_table_has_a_row_per_run(self, config):
        table = headline_table(self.runs(config))
        assert table.row_count == 2

    def test_latency_table_renders(self, config):
        assert latency_table(self.runs(config)).row_count == 2

    def test_disagreements_lists_queries_where_runs_differ(self, config):
        table = disagreements(self.runs(config))
        assert table.row_count >= 1

    def test_disagreements_is_empty_for_a_single_run(self, config):
        assert disagreements(self.runs(config)[:1]).row_count == 0

    def test_markdown_export_has_a_row_per_run(self, config):
        text = to_markdown(self.runs(config))
        assert text.count("\n") == 3  # header, separator, two runs

    def test_rendering_an_empty_run_list_does_not_raise(self):
        from rich.console import Console

        from mmrag.evaluation.report import render

        render([], console=Console(quiet=True))


class TestRunSchema:
    def test_a_run_requires_its_provenance_fields(self):
        with pytest.raises(Exception):
            RetrievalRun()  # type: ignore[call-arg]
