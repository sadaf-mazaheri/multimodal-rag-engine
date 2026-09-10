"""End-to-end tests for Method 2.

Builds real per-modality indexes over a small corpus slice -- real BM25, real
bge embeddings, real CLIP image vectors, real Qdrant -- then routes, retrieves
and answers against them. Generation uses the echo provider, so nothing here
costs money or depends on a vendor.

The tests that matter most are the ones checking that Method 2 does something
Method 1 *structurally cannot*: retrieve a figure that carries no text at all.

Marked ``integration`` and ``slow``; skips cleanly without Qdrant or a parsed
corpus.
"""

from __future__ import annotations

import uuid

import pytest

from mmrag.config import load_experiment_config
from mmrag.generation.providers import EchoProvider
from mmrag.retrieval.views import has_figure_text
from mmrag.schemas import ChunkType, Modality, parse_element_id

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Same slice Method 1's tests use, so the two suites are directly comparable.
TEST_DOCS = ["who_covid_sitrep_001", "arxiv_attention"]


@pytest.fixture(scope="module")
def method(tmp_path_factory):
    """A Method 2 instance with real, throwaway per-modality indexes."""
    from mmrag.config import PROCESSED_DIR
    from mmrag.methods import Method2ModalityAware
    from mmrag.stores.qdrant import QdrantStore

    missing = [d for d in TEST_DOCS if not (PROCESSED_DIR / d / "parsed.json").exists()]
    if missing:
        pytest.skip(f"not ingested: {missing}; run 'mmrag ingest run'")

    config = load_experiment_config("method2")
    variant = f"test2-{uuid.uuid4().hex[:8]}"
    instance = Method2ModalityAware(config, index_base=tmp_path_factory.mktemp("indexes"))
    instance.variant = variant
    instance.index_dir = instance.index_dir.parent / variant

    try:
        QdrantStore(variant).exists()
    except Exception as exc:
        pytest.skip(f"Qdrant unavailable ({exc}); run 'docker compose up -d'")

    instance.build_index(doc_ids=TEST_DOCS)
    yield instance
    for name in ("text", "table_schema", "image"):
        QdrantStore(instance.collection(name)).delete()


@pytest.fixture(scope="module")
def report(method):
    import json

    from mmrag.methods.method2_modality import MANIFEST_FILE

    return json.loads((method.index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))["report"]


class TestIndexBuild:
    def test_builds_one_index_per_modality(self, report):
        assert report["n_text_indexed"] > 0
        assert report["n_tables_indexed"] > 0, "the WHO report has case-count tables"
        assert report["n_figures_indexed"] > 0

    def test_figure_images_are_embedded(self, report):
        assert report["n_figure_images_embedded"] > 0
        assert report["n_figures_without_image"] == 0, "every figure should have a crop on disk"

    def test_reports_what_only_the_image_index_can_reach(self, report):
        """The headline Method 2 number, recorded at build time."""
        assert report["text_invisible_recoverable"] >= 0
        assert (
            report["text_invisible_recoverable"]
            == report["n_figures_without_text"] - report["n_figures_without_image"]
        )

    def test_each_qdrant_collection_holds_exactly_what_was_written(self, method, report):
        from mmrag.stores.qdrant import QdrantStore

        assert QdrantStore(method.collection("text")).count() == report["n_text_indexed"]
        assert QdrantStore(method.collection("table_schema")).count() == report["n_tables_indexed"]
        assert QdrantStore(method.collection("image")).count() == report["n_figure_images_embedded"]

    def test_uses_its_own_variant_so_method1_is_untouched(self, method):
        assert method.variant.startswith("test2-")
        assert all(c.variant == method.variant for c in method.chunks.values())

    def test_provenance_survives_chunking(self, method):
        for chunk in method.chunks.values():
            for element_id in chunk.element_ids:
                doc_id, page = parse_element_id(element_id)
                assert doc_id == chunk.doc_id
                assert page == chunk.page_number


class TestRouting:
    def test_prose_question_fires_only_text_retrievers(self, method):
        result = method.retrieve("Why was the guidance revised?", top_k=5)
        assert result.routing is not None
        assert set(result.diagnostics["retrievers_fired"]) == {"bm25", "dense"}

    def test_visual_question_fires_the_image_retriever(self, method):
        result = method.retrieve("What does the architecture diagram show?", top_k=5)
        assert "image" in result.diagnostics["retrievers_fired"]

    def test_table_question_fires_the_table_retriever(self, method):
        result = method.retrieve("Which table lists confirmed cases by country?", top_k=5)
        assert "table" in result.diagnostics["retrievers_fired"]

    def test_routing_decision_is_recorded_on_the_result(self, method):
        result = method.retrieve("Which figure shows the model architecture?", top_k=5)
        payload = result.routing.as_dict()
        assert Modality.IMAGE.value in payload["modalities"]
        assert payload["signals"]


class TestRetrieval:
    def test_finds_the_transformer_architecture_figure(self, method):
        result = method.retrieve("Transformer model architecture diagram", top_k=10)
        assert any(
            hit.chunk.chunk_type is ChunkType.FIGURE and hit.chunk.doc_id == "arxiv_attention"
            for hit in result.results
        )

    def test_recovers_a_figure_that_carries_no_text(self, method):
        """The capability Method 1 structurally lacks.

        A figure with no caption, OCR or description is unreachable by any
        text index. If Method 2 can surface one, the image vector is the only
        thing that could have done it.
        """
        blind = [
            c
            for c in method.chunks.values()
            if c.chunk_type is ChunkType.FIGURE and not has_figure_text(c)
        ]
        if not blind:
            pytest.skip("this corpus slice has no text-invisible figures")

        found = False
        for query in (
            "a diagram of stacked layers with arrows",
            "a plot with labelled axes",
            "an architecture schematic",
        ):
            result = method.retrieve(query, top_k=25, force_modalities=[Modality.IMAGE])
            if any(hit.chunk_id in {c.chunk_id for c in blind} for hit in result.results):
                found = True
                break
        assert found, "no text-invisible figure was reachable via the image index"

    def test_image_retriever_reports_what_it_recovered(self, method):
        result = method.retrieve("a schematic diagram", top_k=10, force_modalities=[Modality.IMAGE])
        stats = result.diagnostics["per_retriever"]["image"]
        assert stats["clip_hits"] > 0
        assert "text_invisible_recovered" in stats

    def test_table_retriever_uses_both_of_its_views(self, method):
        result = method.retrieve(
            "confirmed cases by country", top_k=10, force_modalities=[Modality.TABLE]
        )
        stats = result.diagnostics["per_retriever"]["table"]
        assert stats["content_hits"] > 0 or stats["schema_hits"] > 0

    def test_rare_literal_token_is_found(self, method):
        result = method.retrieve("2019-nCoV", top_k=10)
        assert any("nCoV" in hit.chunk.text for hit in result.results)

    def test_results_are_ranked_and_deduplicated(self, method):
        result = method.retrieve("attention mechanism", top_k=10)
        assert [h.rank for h in result.results] == list(range(1, len(result.results) + 1))
        ids = [h.chunk_id for h in result.results]
        assert len(ids) == len(set(ids))

    def test_document_filter_is_honoured(self, method):
        result = method.retrieve("attention", top_k=10, doc_ids=["arxiv_attention"])
        assert result.results
        assert {h.chunk.doc_id for h in result.results} == {"arxiv_attention"}

    def test_no_stale_index_references(self, method):
        result = method.retrieve("attention", top_k=10)
        assert result.diagnostics["missing_chunk_records"] == 0

    def test_latency_is_broken_down(self, method):
        result = method.retrieve("attention", top_k=5)
        assert "routing_ms" in result.latency_ms
        assert "fusion_ms" in result.latency_ms
        assert result.latency_ms["total_ms"] > 0

    def test_nonsense_query_does_not_crash(self, method):
        assert isinstance(method.retrieve("zzzqqq xyzzy plugh", top_k=5).results, list)


class TestAnswering:
    def test_produces_an_answer_with_resolvable_citations(self, method):
        answer = method.answer("What is the Transformer architecture?", EchoProvider(), top_k=5)
        assert answer.method == "method2"
        assert answer.citations
        for citation in answer.citations:
            assert citation.doc_id in TEST_DOCS
            assert citation.chunk_id in method.chunks

    def test_answer_records_the_routing_decision(self, method):
        """Step 6 needs to attribute an answer to the route that produced it."""
        answer = method.answer("Which figure shows the architecture?", EchoProvider(), top_k=5)
        assert "routing" in answer.metadata
        assert "retrieval" in answer.metadata

    def test_method2_attaches_no_images_to_the_model(self, method):
        """Passing page renders is Method 3's defining move, not Method 2's."""
        answer = method.answer("Describe Figure 1.", EchoProvider(), top_k=5)
        assert answer.metadata["n_images"] == 0

    def test_citations_carry_a_region_of_a_page(self, method):
        answer = method.answer("What is self-attention?", EchoProvider(), top_k=5)
        assert any(c.bbox is not None for c in answer.citations)


class TestMethod1IsUnaffected:
    """Method 2 must not disturb Method 1's index or behaviour."""

    def test_method1_collection_still_exists_and_is_separate(self, method):
        from mmrag.stores.qdrant import QdrantStore

        store = QdrantStore("method1")
        if not store.exists():
            pytest.skip("method1 index not built on this machine")
        assert store.count() > 0
        assert method.collection("text") != "method1"
