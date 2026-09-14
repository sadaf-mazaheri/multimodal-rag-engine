"""The engine, and the methods as configurations of it.

CPU-only, no services, no model downloads: retrievers are fixed ranked lists and
generation uses the echo provider.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mmrag.config import RouterConfig, load_experiment_config
from mmrag.engine import RAGEngine, build_reranker
from mmrag.generation.providers import EchoProvider
from mmrag.indexing.modality import CHUNKS_FILE, ModalityIndex, _atomic_write_lines
from mmrag.methods import (
    EngineMethod,
    Method1Textified,
    Method2ModalityAware,
    Method3HybridVisual,
    RAGMethod,
    build_method,
)
from mmrag.retrieval.base import Hit, MetadataFilter, RetrieverOutput
from mmrag.retrieval.router import HeuristicRouter
from mmrag.schemas import BBox, Chunk, ChunkType, Modality
from mmrag.stores.multivector import PageIndexWriter, PageRecord


def chunk(cid: str, doc: str = "doc_a", page: int = 1, kind: ChunkType = ChunkType.TEXT) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, page_number=page, chunk_type=kind,
                 text=f"{cid} says something useful", element_ids=[f"{doc}#p{page}#text000"],
                 bbox=BBox(x0=0, y0=0, x1=1, y1=0.5), variant="method2",
                 metadata={"doc_title": doc.upper()})


CHUNKS = {c.chunk_id: c for c in [
    chunk("t1"), chunk("t2", page=2), chunk("tb1", page=3, kind=ChunkType.TABLE),
    chunk("f1", doc="doc_b", kind=ChunkType.FIGURE),
]}


class ListRetriever:
    """A retriever returning a fixed ranked list, recording what it was asked."""

    def __init__(self, name: str, modality: Modality, ids: list[str]):
        self.name, self.modality, self.ids = name, modality, ids
        self.calls: list[MetadataFilter | None] = []

    def retrieve(self, query, k, filters=None):
        self.calls.append(filters)
        ids = [i for i in self.ids if not (filters and filters.doc_ids)
               or CHUNKS[i].doc_id in filters.doc_ids]
        return RetrieverOutput(self.name, self.modality,
                               [Hit(c, 1.0 / r, r) for r, c in enumerate(ids[:k], 1)])


class Resolver:
    def __init__(self, doc_ids):
        self.doc_ids, self.calls = doc_ids, 0

    def resolve(self, query):
        self.calls += 1
        return MetadataFilter(doc_ids=self.doc_ids)


def config(**retrieval):
    return load_experiment_config(
        "method2", overrides={"retrieval": {"rerank_enabled": False, **retrieval}}
    )


def engine(*, resolver=None, always=None, retrievers=None):
    cfg = config()
    retrievers = retrievers or {
        "bm25": ListRetriever("bm25", Modality.TEXT, ["t1", "t2"]),
        "table": ListRetriever("table", Modality.TABLE, ["tb1"]),
        "image": ListRetriever("image", Modality.IMAGE, ["f1"]),
    }
    return RAGEngine(
        cfg, name="test", retrievers=retrievers, chunks=CHUNKS,
        router=HeuristicRouter(cfg.router, always=always),
        resolver_factory=(lambda: resolver) if resolver is not None else None,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:
    def test_needs_a_retriever(self):
        with pytest.raises(ValueError, match="at least one retriever"):
            RAGEngine(config(), name="x", retrievers={}, chunks=CHUNKS)

    def test_routes_fans_out_and_fuses_across_registered_retrievers(self):
        result = engine().retrieve("what happened", top_k=4, use_metadata=False)
        assert result.routing.fell_back
        assert result.diagnostics["retrievers_fired"] == ["bm25", "table", "image"]
        assert {h.chunk_id for h in result.results} == set(CHUNKS)
        assert [h.rank for h in result.results] == [1, 2, 3, 4]

    def test_a_new_retriever_needs_no_change_downstream(self):
        extra = ListRetriever("visual_page", Modality.VISUAL_PAGE, ["t2", "tb1"])
        built = engine(always=[Modality.VISUAL_PAGE], retrievers={
            "bm25": ListRetriever("bm25", Modality.TEXT, ["t1"]), "visual_page": extra,
        })
        result = built.retrieve("explain it", top_k=3, use_metadata=False)
        assert result.routing.modalities == [Modality.TEXT, Modality.VISUAL_PAGE]
        assert result.diagnostics["retrievers_fired"] == ["bm25", "visual_page"]
        assert {h.chunk_id for h in result.results} == {"t1", "t2", "tb1"}
        assert built.describe()["always_on"] == ["visual_page"]

    def test_metadata_resolution_narrows_and_is_loaded_lazily(self):
        resolver = Resolver(["doc_b"])
        built = engine(resolver=resolver)
        assert resolver.calls == 0 and built._resolver is None
        result = built.retrieve("what happened", top_k=4)
        assert resolver.calls == 1
        assert {h.chunk.doc_id for h in result.results} == {"doc_b"}

    def test_metadata_resolution_can_be_ablated(self):
        resolver = Resolver(["doc_b"])
        result = engine(resolver=resolver).retrieve("what happened", top_k=4, use_metadata=False)
        assert resolver.calls == 0
        assert {h.chunk.doc_id for h in result.results} == {"doc_a", "doc_b"}

    def test_a_callers_documents_are_never_replaced_by_inference(self):
        resolver = Resolver(["doc_b"])
        result = engine(resolver=resolver).retrieve("what happened", top_k=4, doc_ids=["doc_a"])
        assert resolver.calls == 0
        assert {h.chunk.doc_id for h in result.results} == {"doc_a"}

    def test_no_resolver_means_no_narrowing(self):
        result = engine().retrieve("what happened", top_k=4)
        assert len(result.results) == 4

    def test_answers_with_citations_routing_and_the_engine_name(self):
        answer = engine().answer("what happened", EchoProvider(), top_k=3)
        assert answer.method == "test"
        assert answer.citations and all(c.chunk_id in CHUNKS for c in answer.citations)
        assert "routing" in answer.metadata and "retrieval" in answer.metadata
        assert answer.metadata["n_images"] == 0

    def test_reranker_follows_the_config(self):
        assert build_reranker(config()) is None
        assert build_reranker(config(rerank_enabled=True)) is not None


# ---------------------------------------------------------------------------
# Router: always-on modalities
# ---------------------------------------------------------------------------


class TestAlwaysOnModalities:
    @pytest.mark.parametrize("strategy,fallback,query", [
        ("heuristic", False, "explain the method"),       # text only
        ("heuristic", True, "what was the result"),       # fell back to everything
        ("heuristic", False, "which table lists totals"),  # text + table
        ("all", True, "anything"),
    ])
    def test_always_on_is_appended_to_every_decision(self, strategy, fallback, query):
        cfg = RouterConfig(strategy=strategy, fallback_to_all=fallback)
        plain = HeuristicRouter(cfg).route(query)
        routed = HeuristicRouter(cfg, always=[Modality.VISUAL_PAGE]).route(query)
        assert routed.modalities == [*plain.modalities, Modality.VISUAL_PAGE]
        assert (routed.signals, routed.fell_back, routed.strategy) == \
            (plain.signals, plain.fell_back, plain.strategy)

    def test_an_available_modality_is_not_duplicated(self):
        router = HeuristicRouter(RouterConfig(), always=[Modality.TEXT])
        assert router.always == []
        assert router.route("explain it").modalities.count(Modality.TEXT) == 1


# ---------------------------------------------------------------------------
# Methods as configurations
# ---------------------------------------------------------------------------


class TestMethods:
    def test_registry_dispatches_on_the_config(self):
        assert isinstance(build_method(load_experiment_config("method1")), Method1Textified)
        assert isinstance(build_method(load_experiment_config("method2")), Method2ModalityAware)
        assert isinstance(build_method(load_experiment_config("method3")), Method3HybridVisual)

    def test_registry_rejects_an_unknown_method(self):
        cfg = load_experiment_config("method2").model_copy(update={"method": "method9"})
        with pytest.raises(ValueError, match="method9"):
            build_method(cfg)

    def test_methods_share_a_contract_not_a_hierarchy(self):
        m1, m2, m3 = (build_method(load_experiment_config(f"method{i}")) for i in (1, 2, 3))
        for method in (m1, m2, m3):
            assert isinstance(method, RAGMethod)
        assert isinstance(m2, EngineMethod) and isinstance(m3, EngineMethod)
        assert not isinstance(m3, Method2ModalityAware)

    def test_method2_is_the_engine_over_the_modality_index(self, tmp_path, monkeypatch):
        cfg = config()
        method = Method2ModalityAware(cfg, index_base=tmp_path)
        (tmp_path / "method2").mkdir()
        (tmp_path / "method2" / CHUNKS_FILE).write_text(
            "".join(c.model_dump_json() + "\n" for c in CHUNKS.values()), encoding="utf-8")
        monkeypatch.setattr(ModalityIndex, "retrievers", lambda self: {
            "bm25": ListRetriever("bm25", Modality.TEXT, ["t1", "t2"])})

        assert method.index_dir == tmp_path / "method2"
        assert method.chunks.keys() == CHUNKS.keys()
        assert list(method.engine.retrievers) == ["bm25"]
        assert method.engine.router.always == []
        result = method.retrieve("explain", top_k=2, use_metadata=False)
        assert [h.chunk_id for h in result.results] == ["t1", "t2"]
        method.reset()
        assert method._engine is None

    def test_a_variant_namespaces_the_whole_index(self, tmp_path):
        method = Method2ModalityAware(config(), index_base=tmp_path, variant="scratch")
        assert method.index_dir == tmp_path / "scratch"
        assert method.collection("text") == "scratch_text"


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    def test_line_writer_leaves_no_partial_file(self, tmp_path):
        target = tmp_path / "chunks.jsonl"
        target.write_text("old\n", encoding="utf-8")
        _atomic_write_lines(target, (json.dumps({"n": i}) for i in range(3)))
        assert target.read_text(encoding="utf-8").splitlines() == [
            '{"n": 0}', '{"n": 1}', '{"n": 2}']
        assert list(tmp_path.iterdir()) == [target]

    def test_a_failed_write_keeps_the_previous_file(self, tmp_path):
        target = tmp_path / "chunks.jsonl"
        target.write_text("old\n", encoding="utf-8")

        def lines():
            yield "new"
            raise RuntimeError("interrupted")

        with pytest.raises(RuntimeError):
            _atomic_write_lines(target, lines())
        assert target.read_text(encoding="utf-8") == "old\n"

    def test_page_index_rebuild_replaces_the_previous_index_cleanly(self, tmp_path):
        def build(n_tokens):
            writer = PageIndexWriter(tmp_path / "index", dim=4)
            writer.add(PageRecord(page_id="d#p1", doc_id="d", page_number=1, image="d/p.png",
                                  image_sha256="0" * 64, image_width=1, image_height=1,
                                  image_dpi=150, n_tokens=n_tokens), np.ones((n_tokens, 4)))
            return writer.finalize({"kind": "visual_page_index", "corpus": {"complete": True},
                                    "model": {"identity": {}}})

        build(2)
        index = build(3)
        assert index.manifest["layout"]["n_tokens"] == 3
        assert sorted(p.name for p in tmp_path.iterdir()) == ["index"]
