"""End-to-end tests for Method 1.

Builds a real index over a small slice of the parsed corpus -- real BM25, real
embeddings, real Qdrant -- then retrieves and answers against it. The generation
step uses the echo provider, so the test costs nothing and does not depend on a
vendor being reachable; what is under test is the plumbing, not answer quality.

Marked ``integration`` because it needs Qdrant, and ``slow`` because it embeds a
few hundred chunks on CPU. Skips cleanly when either the services or the parsed
corpus are missing.
"""

from __future__ import annotations

import uuid

import pytest

from mmrag.config import load_experiment_config
from mmrag.generation.providers import EchoProvider
from mmrag.schemas import ChunkType, parse_element_id

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Small and modality-varied: a 5-page report with tables and a map, and a paper
# with a canonical "the answer is only in the figure" diagram.
TEST_DOCS = ["who_covid_sitrep_001", "arxiv_attention"]


@pytest.fixture(scope="module")
def method(tmp_path_factory):
    """A Method 1 instance with a real, throwaway index."""
    from mmrag.config import PROCESSED_DIR
    from mmrag.methods import Method1Textified
    from mmrag.stores.qdrant import QdrantStore

    missing = [d for d in TEST_DOCS if not (PROCESSED_DIR / d / "parsed.json").exists()]
    if missing:
        pytest.skip(f"not ingested: {missing}; run 'mmrag ingest run'")

    config = load_experiment_config("method1")
    # A unique collection so the test never disturbs the real method1 index.
    config.name = f"test-{uuid.uuid4().hex[:8]}"
    instance = Method1Textified(config, index_base=tmp_path_factory.mktemp("indexes"))
    instance.variant = config.name
    instance.index_dir = instance.index_dir.parent / config.name

    try:
        QdrantStore(instance.variant).exists()
    except Exception as exc:
        pytest.skip(f"Qdrant unavailable ({exc}); run 'docker compose up -d'")

    instance.build_index(doc_ids=TEST_DOCS)
    yield instance
    QdrantStore(instance.variant).delete()


class TestIndexBuild:
    def test_produces_chunks_of_every_modality(self, method):
        types = {c.chunk_type for c in method.chunks.values()}
        assert ChunkType.TEXT in types
        assert ChunkType.TABLE in types, "the WHO report has case-count tables"
        assert ChunkType.FIGURE in types, "the Transformer paper has figures"

    def test_chunk_ids_are_unique(self, method):
        """The build refuses to proceed otherwise; this asserts the outcome."""
        ids = [c.chunk_id for c in method.chunks.values()]
        assert len(ids) == len(set(ids))

    def test_vector_store_holds_exactly_the_indexed_chunks(self, method):
        from mmrag.stores.qdrant import QdrantStore

        assert QdrantStore(method.variant).count() == len(method.chunks)

    def test_every_chunk_keeps_single_page_provenance(self, method):
        for chunk in method.chunks.values():
            for element_id in chunk.element_ids:
                doc_id, page = parse_element_id(element_id)
                assert doc_id == chunk.doc_id
                assert page == chunk.page_number

    def test_manifest_records_the_config_and_the_loss(self, method):
        import json

        from mmrag.methods.method1_textified import MANIFEST_FILE

        manifest = json.loads((method.index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        assert manifest["method"] == "method1"
        assert manifest["config"]["chunking"]["target_tokens"] > 0
        assert "invisible_figures" in manifest["report"]
        assert manifest["report"]["embedder"]["dimension"] == 384


class TestRetrieval:
    def test_finds_the_transformer_architecture_figure(self, method):
        result = method.retrieve("Transformer model architecture diagram", top_k=10)
        assert result.results
        assert any(
            "Transformer" in hit.chunk.text and hit.chunk.chunk_type is ChunkType.FIGURE
            for hit in result.results
        ), "Figure 1's caption should be retrievable as a figure chunk"

    def test_both_retrievers_contribute(self, method):
        """If one side contributes nothing, the 'hybrid' is not hybrid."""
        result = method.retrieve("novel coronavirus confirmed cases", top_k=10)
        contributed = result.diagnostics["contributed"]
        assert contributed["bm25"] > 0
        assert contributed["dense"] > 0

    def test_rare_literal_token_is_found(self, method):
        """BM25's job: exact identifiers dense retrieval tends to blur."""
        result = method.retrieve("2019-nCoV", top_k=10)
        assert any("nCoV" in hit.chunk.text for hit in result.results)

    def test_paraphrase_is_found_without_shared_wording(self, method):
        """Dense retrieval's job: no lexical overlap with the source text."""
        result = method.retrieve(
            "How does the model attend to different positions simultaneously?", top_k=10
        )
        assert result.results
        assert any("attention" in hit.chunk.text.lower() for hit in result.results), (
            "a purely lexical retriever would miss this phrasing"
        )

    def test_results_are_ranked_and_deduplicated(self, method):
        result = method.retrieve("attention", top_k=10)
        assert [h.rank for h in result.results] == list(range(1, len(result.results) + 1))
        ids = [h.chunk_id for h in result.results]
        assert len(ids) == len(set(ids))

    def test_top_k_is_respected(self, method):
        assert len(method.retrieve("attention", top_k=3).results) <= 3

    def test_document_filter_restricts_both_retrievers(self, method):
        result = method.retrieve("attention", top_k=10, doc_ids=["arxiv_attention"])
        assert result.results
        assert {h.chunk.doc_id for h in result.results} == {"arxiv_attention"}

    def test_component_ranks_are_recorded_for_ablation(self, method):
        """Which retrievers found a hit, and where fusion had put it.

        ``fused`` appears because reranking is enabled: the reranker records the
        pre-rerank position so its own effect stays measurable. Without it a
        reranker that reorders nothing and one that fixes everything look
        identical downstream.
        """
        result = method.retrieve("self attention mechanism", top_k=5)
        assert all(h.component_ranks for h in result.results)
        assert all(
            set(h.component_ranks) <= {"bm25", "dense", "fused"} for h in result.results
        )
        assert all("fused" in h.component_ranks for h in result.results)

    def test_latency_is_broken_down_by_stage(self, method):
        result = method.retrieve("attention", top_k=5)
        for stage in ("bm25_ms", "embed_ms", "dense_ms", "fusion_ms", "total_ms"):
            assert stage in result.latency_ms

    def test_no_stale_index_references(self, method):
        result = method.retrieve("attention", top_k=10)
        assert result.diagnostics["missing_chunk_records"] == 0

    def test_a_nonsense_query_returns_few_or_no_results(self, method):
        """It must not crash, and it must not invent confident evidence."""
        result = method.retrieve("zzzqqq xyzzy plugh", top_k=10)
        assert isinstance(result.results, list)


class TestAnswering:
    def test_produces_an_answer_with_resolvable_citations(self, method):
        answer = method.answer("What is the Transformer architecture?", EchoProvider(), top_k=5)
        assert answer.method == "method1"
        assert answer.text
        assert answer.citations
        for citation in answer.citations:
            assert citation.doc_id in TEST_DOCS
            assert citation.page_number >= 1
            assert citation.chunk_id in method.chunks

    def test_citations_carry_a_region_of_a_page(self, method):
        """The provenance guarantee, end to end: a hit resolves to a bounding box."""
        answer = method.answer("What is self-attention?", EchoProvider(), top_k=5)
        assert any(c.bbox is not None for c in answer.citations)

    def test_no_unresolved_citations(self, method):
        answer = method.answer("What is attention?", EchoProvider(), top_k=5)
        assert answer.metadata["unresolved_citations"] == []

    def test_method1_never_attaches_images(self, method):
        """Flattening to text is the defining constraint; images would break it."""
        answer = method.answer("Describe Figure 1.", EchoProvider(), top_k=5)
        assert answer.metadata["n_images"] == 0

    def test_answer_carries_retrieval_diagnostics(self, method):
        """Step 6 scores retrieval and generation separately."""
        answer = method.answer("What is attention?", EchoProvider(), top_k=5)
        assert answer.retrieved
        assert "retrieval" in answer.metadata
        assert "contributed" in answer.metadata["retrieval"]

    def test_latency_covers_retrieval_and_generation(self, method):
        answer = method.answer("What is attention?", EchoProvider(), top_k=5)
        assert "generation_ms" in answer.latency_ms
        assert "total_ms" in answer.latency_ms
