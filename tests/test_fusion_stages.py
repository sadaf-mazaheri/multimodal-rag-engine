"""Two-stage fusion and the modality-floored rerank pool.

These are the guards on the fix for a defect that was invisible to every
existing test: figures could not reach the fused top-k on a query that was not
explicitly visual, because RRF is additive and text supplied two ranked lists
where every other modality supplied one.

Driven by stub retrievers rather than the real corpus, so the arithmetic is
asserted directly and the suite stays fast.
"""

from __future__ import annotations

import pytest

from mmrag.config import RetrievalConfig, RouterConfig
from mmrag.retrieval.base import Hit, MetadataFilter, RetrieverOutput
from mmrag.retrieval.modality import (
    ModalityAwareRetriever,
    _balanced_pool,
    _fuse_within_modalities,
    _modality_weights,
)
from mmrag.retrieval.router import HeuristicRouter
from mmrag.schemas import BBox, Chunk, ChunkType, Modality, ScoredChunk


def make_chunk(chunk_id: str, chunk_type: ChunkType, *, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_number=page,
        chunk_type=chunk_type,
        text=f"text of {chunk_id}",
        element_ids=[f"doc#p{page:04d}#{chunk_id}"],
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.5),
        variant="method2",
    )


class StubRetriever:
    """Returns a fixed ranked list, so fusion inputs are exactly known."""

    def __init__(self, name: str, modality: Modality, chunk_ids: list[str]):
        self.name = name
        self.modality = modality
        self.chunk_ids = chunk_ids

    def retrieve(self, query, k, filters=None):  # noqa: ARG002
        hits = [Hit(chunk_id=c, score=1.0, rank=i) for i, c in enumerate(self.chunk_ids[:k], 1)]
        return RetrieverOutput(retriever=self.name, modality=self.modality, hits=hits)


class StubReranker:
    """Reorders by a supplied preference, standing in for the cross-encoder."""

    def __init__(self, prefer: list[str] | None = None):
        self.prefer = prefer or []
        self.saw: list[str] = []

    def rerank(self, query, candidates, *, top_k):  # noqa: ARG002
        self.saw = [c.chunk.chunk_id for c in candidates]
        ranked = sorted(
            candidates,
            key=lambda c: (self.prefer.index(c.chunk.chunk_id)
                           if c.chunk.chunk_id in self.prefer else len(self.prefer)),
        )
        return [c.model_copy(update={"rank": i}) for i, c in enumerate(ranked[:top_k], 1)]


TEXT_IDS = [f"t{i}" for i in range(1, 21)]
FIG_IDS = [f"f{i}" for i in range(1, 21)]
TAB_IDS = [f"b{i}" for i in range(1, 21)]

ALL_CHUNKS = {
    **{c: make_chunk(c, ChunkType.TEXT) for c in TEXT_IDS},
    **{c: make_chunk(c, ChunkType.FIGURE) for c in FIG_IDS},
    **{c: make_chunk(c, ChunkType.TABLE) for c in TAB_IDS},
}


def build(*, rerank_enabled: bool, reranker=None, floor: int = 8, weights=None):
    config = RetrievalConfig(
        top_k=10,
        candidates_per_retriever=50,
        rerank_enabled=rerank_enabled,
        rerank_top_n=25,
        rerank_pool_per_modality=floor,
        fusion_weights=weights or {"bm25": 1.0, "dense": 1.0, "table": 1.0, "image": 0.7},
    )
    retrievers = {
        "bm25": StubRetriever("bm25", Modality.TEXT, TEXT_IDS),
        "dense": StubRetriever("dense", Modality.TEXT, TEXT_IDS),
        "table": StubRetriever("table", Modality.TABLE, TAB_IDS),
        "image": StubRetriever("image", Modality.IMAGE, FIG_IDS),
    }
    return ModalityAwareRetriever(
        config,
        router=HeuristicRouter(RouterConfig(strategy="all")),
        retrievers=retrievers,
        chunks=ALL_CHUNKS,
        reranker=reranker if rerank_enabled else None,
    )


class TestTextHasNoStructuralDoubleVote:
    """The defect itself: text supplied two lists, every other modality one."""

    def test_each_modality_becomes_exactly_one_ranked_list(self):
        outputs = [
            StubRetriever("bm25", Modality.TEXT, TEXT_IDS).retrieve("q", 20),
            StubRetriever("dense", Modality.TEXT, TEXT_IDS).retrieve("q", 20),
            StubRetriever("image", Modality.IMAGE, FIG_IDS).retrieve("q", 20),
        ]
        fused = _fuse_within_modalities(outputs, k=60)
        assert set(fused) == {"text", "image"}

    def test_text_weight_is_not_the_sum_of_its_retrievers(self):
        """bm25 1.0 + dense 1.0 must resolve to text 1.0, never 2.0."""
        outputs = [
            StubRetriever("bm25", Modality.TEXT, TEXT_IDS).retrieve("q", 5),
            StubRetriever("dense", Modality.TEXT, TEXT_IDS).retrieve("q", 5),
            StubRetriever("image", Modality.IMAGE, FIG_IDS).retrieve("q", 5),
        ]
        weights = _modality_weights(outputs, {"bm25": 1.0, "dense": 1.0, "image": 0.7})
        assert weights == {"text": 1.0, "image": 0.7}

    def test_an_explicit_modality_key_wins(self):
        outputs = [StubRetriever("bm25", Modality.TEXT, TEXT_IDS).retrieve("q", 5)]
        assert _modality_weights(outputs, {"text": 0.4, "bm25": 1.0}) == {"text": 0.4}

    def test_a_rank_one_figure_beats_a_rank_two_text_chunk_at_equal_weight(self):
        """The double-vote, isolated.

        At equal weights a figure at image-rank 1 scores 1/61 while a text chunk
        at rank 2 scores 1/62, so the figure wins. Before the split that text
        chunk appeared in *both* text lists and scored 2/62 -- beating the
        figure purely on list count.

        Note this is the ceiling being equalised, not figures being surfaced:
        the weighted case still favours text, which is what the modality floor
        and the cross-encoder exist to handle.
        """
        retriever = build(
            rerank_enabled=False,
            weights={"bm25": 1.0, "dense": 1.0, "table": 0.0, "image": 1.0},
        )
        result = retriever.retrieve("anything", top_k=4)
        ids = [h.chunk.chunk_id for h in result.results]
        assert ids.index("f1") < ids.index("t2")


class TestRerankPoolSurvival:
    def test_every_fired_modality_reaches_the_pool(self):
        reranker = StubReranker()
        retriever = build(rerank_enabled=True, reranker=reranker, floor=8)
        retriever.retrieve("anything", top_k=10)
        seen = {ALL_CHUNKS[c].chunk_type for c in reranker.saw}
        assert seen == {ChunkType.TEXT, ChunkType.TABLE, ChunkType.FIGURE}

    def test_the_floor_is_honoured_per_modality(self):
        reranker = StubReranker()
        retriever = build(rerank_enabled=True, reranker=reranker, floor=5)
        retriever.retrieve("anything", top_k=10)
        counts: dict[ChunkType, int] = {}
        for c in reranker.saw:
            counts[ALL_CHUNKS[c].chunk_type] = counts.get(ALL_CHUNKS[c].chunk_type, 0) + 1
        for kind in (ChunkType.TEXT, ChunkType.TABLE, ChunkType.FIGURE):
            assert counts.get(kind, 0) >= 5

    def test_the_reranker_still_decides_the_final_order(self):
        """The floor governs membership only; relevance is the reranker's call."""
        reranker = StubReranker(prefer=["t1", "t2", "t3"])
        retriever = build(rerank_enabled=True, reranker=reranker, floor=8)
        result = retriever.retrieve("anything", top_k=3)
        assert [h.chunk.chunk_id for h in result.results] == ["t1", "t2", "t3"]

    def test_pool_composition_is_reported(self):
        retriever = build(rerank_enabled=True, reranker=StubReranker(), floor=8)
        result = retriever.retrieve("anything", top_k=10)
        assert result.diagnostics["rerank_pool_by_modality"]


class TestPoolIsBoundedAndDeduplicated:
    def test_pool_never_exceeds_the_limit(self):
        reranker = StubReranker()
        retriever = build(rerank_enabled=True, reranker=reranker, floor=8)
        retriever.retrieve("anything", top_k=10)
        assert len(reranker.saw) <= 25

    def test_pool_has_no_duplicates(self):
        reranker = StubReranker()
        retriever = build(rerank_enabled=True, reranker=reranker, floor=8)
        retriever.retrieve("anything", top_k=10)
        assert len(reranker.saw) == len(set(reranker.saw))

    def test_a_chunk_found_by_two_modalities_is_pooled_once(self):
        shared = make_chunk("shared", ChunkType.FIGURE)
        chunks = {**ALL_CHUNKS, "shared": shared}
        config = RetrievalConfig(
            top_k=5, candidates_per_retriever=50, rerank_enabled=True,
            rerank_top_n=25, rerank_pool_per_modality=4,
        )
        reranker = StubReranker()
        retriever = ModalityAwareRetriever(
            config,
            router=HeuristicRouter(RouterConfig(strategy="all")),
            retrievers={
                "bm25": StubRetriever("bm25", Modality.TEXT, ["shared", *TEXT_IDS]),
                "dense": StubRetriever("dense", Modality.TEXT, ["shared", *TEXT_IDS]),
                "image": StubRetriever("image", Modality.IMAGE, ["shared", *FIG_IDS]),
            },
            chunks=chunks,
            reranker=reranker,
        )
        retriever.retrieve("anything", top_k=5)
        assert reranker.saw.count("shared") == 1

    def test_floor_of_zero_disables_the_quota(self):
        pool = _balanced_pool(
            [ScoredChunk(chunk=ALL_CHUNKS[c], score=1.0, rank=i, retriever="x",
                         modality=Modality.TEXT)
             for i, c in enumerate(TEXT_IDS[:12], 1)],
            {"text": TEXT_IDS},
            per_modality_floor=0,
            limit=5,
        )
        assert len(pool) == 5

    def test_pool_keeps_fused_order(self):
        scored = [
            ScoredChunk(chunk=ALL_CHUNKS[c], score=1.0, rank=i, retriever="x",
                        modality=Modality.TEXT)
            for i, c in enumerate([*TEXT_IDS[:5], *FIG_IDS[:5]], 1)
        ]
        pool = _balanced_pool(
            scored, {"text": TEXT_IDS[:5], "image": FIG_IDS[:5]},
            per_modality_floor=3, limit=10,
        )
        assert [h.rank for h in pool] == sorted(h.rank for h in pool)


class TestRerankerDisabledFallback:
    def test_no_reranker_means_no_quota_injection(self):
        """Without an arbiter a floor would promote evidence nothing vouched for."""
        retriever = build(rerank_enabled=False)
        result = retriever.retrieve("anything", top_k=10)
        assert len(result.results) == 10
        assert result.diagnostics["reranked"] is False
        assert result.diagnostics["rerank_pool_by_modality"] == {}

    def test_results_are_still_ranked_contiguously(self):
        retriever = build(rerank_enabled=False)
        result = retriever.retrieve("anything", top_k=10)
        assert [h.rank for h in result.results] == list(range(1, 11))

    def test_two_stage_fusion_applies_with_or_without_a_reranker(self):
        for enabled in (True, False):
            retriever = build(rerank_enabled=enabled, reranker=StubReranker())
            result = retriever.retrieve("anything", top_k=10)
            stages = result.diagnostics["fusion_stages"]["within_modality"]
            assert set(stages) == {"text", "table", "image"}


class TestBothMethodsShareTheRerankerSetup:
    """A head-to-head is only attributable while this holds."""

    @pytest.mark.parametrize("field", ["rerank_enabled", "rerank_model", "rerank_top_n"])
    def test_method1_and_method2_agree(self, field):
        from mmrag.config import load_experiment_config

        one = load_experiment_config("method1").retrieval
        two = load_experiment_config("method2").retrieval
        assert getattr(one, field) == getattr(two, field)

    def test_reranking_is_actually_enabled(self):
        from mmrag.config import load_experiment_config

        assert load_experiment_config("method1").retrieval.rerank_enabled is True
        assert load_experiment_config("method2").retrieval.rerank_enabled is True
