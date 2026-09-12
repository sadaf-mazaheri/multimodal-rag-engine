"""Tests for Method 2's views, retrievers, metadata resolver, and orchestrator.

The retrievers are exercised against in-memory fakes rather than live services,
so the routing/fusion/filtering logic is tested deterministically and for free.
``test_method2.py`` covers the same code against real indexes.
"""

from __future__ import annotations

import pytest

from mmrag.config import RetrievalConfig, RouterConfig
from mmrag.retrieval.base import Hit, MetadataFilter, RetrieverOutput
from mmrag.retrieval.metadata import MetadataResolver
from mmrag.retrieval.modality import ModalityAwareRetriever
from mmrag.retrieval.modality_retrievers import _apply_chunk_filters
from mmrag.retrieval.router import HeuristicRouter
from mmrag.retrieval.views import (
    figure_text_view,
    has_figure_text,
    table_content_view,
    table_schema_view,
)
from mmrag.schemas import (
    BBox,
    Chunk,
    ChunkType,
    Document,
    DocumentType,
    Element,
    ElementType,
    Modality,
    TableData,
    make_element_id,
    make_page_id,
)

DOC_ID = "annual_2024"


def _chunk(
    n: int,
    *,
    chunk_id: str | None = None,
    chunk_type: ChunkType = ChunkType.TEXT,
    text: str = "Some prose about revenue.",
    doc_id: str = DOC_ID,
    page: int = 1,
    header: str = "",
    section: str | None = None,
) -> Chunk:
    body = f"{header}\n\n{text}" if header else text
    return Chunk(
        chunk_id=chunk_id or f"method2#{n:04d}",
        doc_id=doc_id,
        page_number=page,
        chunk_type=chunk_type,
        text=body,
        element_ids=[make_element_id(doc_id, page, chunk_type.value, n)],
        section=section,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
        variant="method2",
        metadata={"doc_title": "Annual Report 2024", "context_header": header},
    )


def _table_element(columns: list[str], rows: list[list[str | None]], caption: str) -> Element:
    return Element(
        element_id=make_element_id(DOC_ID, 1, "table", 1),
        doc_id=DOC_ID,
        page_id=make_page_id(DOC_ID, 1),
        page_number=1,
        element_type=ElementType.TABLE,
        caption=caption,
        table=TableData(
            n_rows=len(rows),
            n_cols=len(columns),
            columns=columns,
            rows=rows,
            markdown="| " + " | ".join(columns) + " |",
            table_type="financial",
        ),
    )


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


class TestTableViews:
    def test_schema_view_describes_the_table_not_its_cells(self):
        """The whole point: headers stop competing with hundreds of digits."""
        element = _table_element(
            ["Segment", "Revenue", "Change"],
            [["Insurance", "22360", "+4%"], ["Rail", "5910", "-1%"]],
            "Table 1: Revenue by segment.",
        )
        view = table_schema_view(_chunk(1, chunk_type=ChunkType.TABLE), element)

        assert "Table 1: Revenue by segment." in view
        assert "Columns: Segment, Revenue, Change" in view
        assert "Table type: financial" in view
        assert "2 rows by 3 columns" in view
        # Cell values must not leak in, or the view stops being a schema.
        assert "22360" not in view

    def test_schema_view_includes_section_context(self):
        element = _table_element(["A"], [["1"]], "Table 2")
        chunk = _chunk(1, chunk_type=ChunkType.TABLE, section="Segment Results")
        assert "Section: Segment Results" in table_schema_view(chunk, element)

    def test_schema_view_falls_back_to_the_leading_caption_line(self):
        chunk = _chunk(1, chunk_type=ChunkType.TABLE, text="Table 9: Headcount.\n| a | b |")
        assert "Table 9: Headcount." in table_schema_view(chunk, None)

    def test_schema_view_ignores_a_markdown_row_as_a_caption(self):
        chunk = _chunk(1, chunk_type=ChunkType.TABLE, text="| a | b |\n| 1 | 2 |")
        assert "| a | b |" not in table_schema_view(chunk, None)

    def test_schema_view_strips_the_breadcrumb_header(self):
        chunk = _chunk(
            1,
            chunk_type=ChunkType.TABLE,
            text="Table 4: Costs.\n| a |",
            header="Annual Report 2024 > Costs",
        )
        assert table_schema_view(chunk, None).startswith("Table 4: Costs.")

    def test_content_view_is_the_cells(self):
        chunk = _chunk(1, chunk_type=ChunkType.TABLE, text="| Segment | 22360 |")
        assert "22360" in table_content_view(chunk)


class TestFigureViews:
    def test_a_figure_with_a_caption_has_text(self):
        chunk = _chunk(
            1,
            chunk_type=ChunkType.FIGURE,
            text="Figure 2: Emissions by scenario.",
            header="Annual Report 2024",
        )
        assert has_figure_text(chunk)
        assert figure_text_view(chunk) == "Figure 2: Emissions by scenario."

    def test_a_figure_whose_only_text_is_the_header_is_invisible(self):
        """Counting the breadcrumb as content would understate the Method 1 gap."""
        chunk = _chunk(1, chunk_type=ChunkType.FIGURE, text="", header="Annual Report 2024")
        assert not has_figure_text(chunk)

    def test_whitespace_only_body_is_invisible(self):
        chunk = _chunk(1, chunk_type=ChunkType.FIGURE, text="   \n  ", header="Annual Report")
        assert not has_figure_text(chunk)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestChunkFilters:
    @pytest.fixture
    def chunks(self):
        return {
            "method2#0001": _chunk(1, doc_id="a", page=1),
            "method2#0002": _chunk(2, doc_id="b", page=5),
            "method2#0003": _chunk(3, doc_id="a", page=5, chunk_type=ChunkType.TABLE),
        }

    def _hits(self):
        return [Hit(f"method2#{n:04d}", 1.0 / n, n) for n in (1, 2, 3)]

    def test_no_filter_is_a_passthrough(self, chunks):
        hits = self._hits()
        assert _apply_chunk_filters(hits, chunks, None) == hits
        assert _apply_chunk_filters(hits, chunks, MetadataFilter()) == hits

    def test_doc_filter(self, chunks):
        kept = _apply_chunk_filters(self._hits(), chunks, MetadataFilter(doc_ids=["a"]))
        assert [h.chunk_id for h in kept] == ["method2#0001", "method2#0003"]

    def test_page_filter(self, chunks):
        kept = _apply_chunk_filters(self._hits(), chunks, MetadataFilter(page_numbers=[5]))
        assert [h.chunk_id for h in kept] == ["method2#0002", "method2#0003"]

    def test_chunk_type_filter(self, chunks):
        kept = _apply_chunk_filters(self._hits(), chunks, MetadataFilter(chunk_types=["table"]))
        assert [h.chunk_id for h in kept] == ["method2#0003"]

    def test_ranks_are_renumbered_without_gaps(self, chunks):
        """RRF reads rank position; a gap would silently distort every score."""
        kept = _apply_chunk_filters(self._hits(), chunks, MetadataFilter(doc_ids=["a"]))
        assert [h.rank for h in kept] == [1, 2]

    def test_unknown_chunk_ids_are_dropped(self, chunks):
        hits = [Hit("method2#9999", 1.0, 1)]
        assert _apply_chunk_filters(hits, chunks, MetadataFilter(doc_ids=["a"])) == []


# ---------------------------------------------------------------------------
# Metadata resolver
# ---------------------------------------------------------------------------


def _document(doc_id: str, title: str, publisher: str | None = None) -> Document:
    return Document(
        doc_id=doc_id,
        title=title,
        source=publisher,
        doc_type=DocumentType.POLICY_REPORT,
        file_name=f"{doc_id}.pdf",
        file_path=f"/tmp/{doc_id}.pdf",
        sha256="a" * 64,
        n_pages=1,
        n_pages_ingested=1,
    )


class TestMetadataResolver:
    @pytest.fixture
    def resolver(self):
        return MetadataResolver.from_documents(
            [
                _document("ipcc_ar6_wg1_spm", "IPCC AR6 WGI Summary", "IPCC"),
                _document("berkshire_ar_2023", "Berkshire Hathaway 2023 Annual Report"),
                _document("rp2040_datasheet", "RP2040 Datasheet", "Raspberry Pi"),
                _document("arxiv_attention", "Attention Is All You Need"),
                _document("who_covid_sitrep_001", "WHO Situation Report 1", "WHO"),
                _document("nasa_seh", "NASA Systems Engineering Handbook", "NASA"),
            ]
        )

    def test_resolves_a_named_document(self, resolver):
        assert resolver.resolve("what does the IPCC report say about warming").doc_ids == [
            "ipcc_ar6_wg1_spm"
        ]

    def test_resolves_by_a_distinctive_identifier(self, resolver):
        assert resolver.resolve("what is the RP2040 clock speed").doc_ids == ["rp2040_datasheet"]

    def test_a_query_naming_no_document_does_not_narrow(self, resolver):
        """Narrowing on a weak guess hides the answer with no way to recover."""
        assert resolver.resolve("what were the total revenues").doc_ids is None

    def test_generic_words_do_not_narrow(self, resolver):
        """'annual report' matches too many documents to be a constraint."""
        assert resolver.resolve("what does the annual report say").doc_ids is None

    def test_matching_most_of_the_corpus_is_treated_as_noise(self, resolver):
        assert resolver.resolve("report summary data analysis").doc_ids is None

    def test_empty_query(self, resolver):
        assert resolver.resolve("").doc_ids is None

    def test_an_empty_corpus_resolves_nothing(self):
        assert MetadataResolver.from_documents([]).resolve("IPCC").doc_ids is None


class TestGenericTitleWordsDoNotIdentifyADocument:
    """A title word that is common in the corpus text is a topic, not a name.

    Uniqueness among titles is not discriminativeness. On the real corpus,
    "architecture" appeared in exactly one of fourteen titles -- looking like a
    perfect identifier -- while appearing in nine documents' body text, and
    "table" was unique to the TAPAS title while appearing in all fourteen
    bodies. Eight of thirty narrowing decisions over the gold set went to the
    wrong document as a result.
    """

    DOCUMENTS = [
        _document("nvidia_ampere_wp", "NVIDIA A100 Tensor Core GPU Architecture Whitepaper"),
        _document("arxiv_attention", "Attention Is All You Need"),
        _document("arxiv_tapas", "TAPAS: Weakly Supervised Table Parsing via Pre-training"),
        _document("ipcc_ar6_wg1_spm", "IPCC AR6 WGI Summary", "IPCC"),
        _document("rp2040_datasheet", "RP2040 Datasheet", "Raspberry Pi"),
        _document("nasa_seh", "NASA Systems Engineering Handbook", "NASA"),
    ]

    # "architecture" and "table" are everywhere; the names are not.
    BODIES = {
        "nvidia_ampere_wp": {"architecture", "table", "gpu", "ampere", "streaming"},
        "arxiv_attention": {"architecture", "table", "transformer", "encoder"},
        "arxiv_tapas": {"architecture", "table", "parsing", "denotation"},
        "ipcc_ar6_wg1_spm": {"architecture", "table", "warming", "emissions"},
        "rp2040_datasheet": {"architecture", "table", "register", "gpio"},
        "nasa_seh": {"architecture", "table", "verification", "lifecycle"},
    }

    @pytest.fixture
    def resolver(self):
        return MetadataResolver.from_documents(self.DOCUMENTS, body_terms=self.BODIES)

    @pytest.fixture
    def blind(self):
        """The old behaviour: titles only, no corpus statistics."""
        return MetadataResolver.from_documents(self.DOCUMENTS)

    def test_architecture_no_longer_narrows(self, resolver):
        """The defect that broke retrieval of the Transformer architecture figure."""
        assert resolver.resolve("Transformer model architecture diagram").doc_ids is None

    def test_table_no_longer_narrows_onto_the_table_parsing_paper(self, resolver):
        """The defect that halved Method 2's table recall.

        Every question containing the word "table" was filtered to TAPAS.
        """
        assert resolver.resolve("Which table lists confirmed cases?").doc_ids is None

    def test_without_corpus_statistics_the_defect_is_reproduced(self, blind):
        """Documents the degradation, so the fix is not silently bypassable."""
        assert blind.resolve("Transformer model architecture diagram").doc_ids == [
            "nvidia_ampere_wp"
        ]
        assert blind.resolve("Which table lists confirmed cases?").doc_ids == ["arxiv_tapas"]

    def test_a_real_identifier_still_narrows(self, resolver):
        assert resolver.resolve("what does the IPCC say about warming").doc_ids == [
            "ipcc_ar6_wg1_spm"
        ]
        assert resolver.resolve("what is the RP2040 clock speed").doc_ids == [
            "rp2040_datasheet"
        ]

    def test_a_rare_title_word_is_still_a_signal(self, resolver):
        """Only *widespread* words are dropped; distinctive ones survive."""
        assert resolver.resolve("weakly supervised parsing").doc_ids == ["arxiv_tapas"]

    def test_dropped_terms_are_reported(self, resolver):
        assert {"architecture", "table"} <= resolver.generic_terms
        assert resolver.describe()["corpus_statistics"] is True
        assert resolver.describe()["n_generic_terms_dropped"] >= 2

    def test_no_corpus_statistics_is_reported_too(self, blind):
        assert blind.generic_terms == frozenset()
        assert blind.describe()["corpus_statistics"] is False

    @pytest.mark.parametrize(
        ("n_documents_containing", "expected_dropped"),
        [(2, False), (3, True), (4, True)],
    )
    def test_the_threshold_boundary(self, n_documents_containing, expected_dropped):
        """At the default 0.5, a term in half of six documents is dropped."""
        docs = [_document(f"d{i}", f"Doc {i} Widget") for i in range(6)]
        bodies = {
            f"d{i}": ({"widget"} if i < n_documents_containing else set()) | {f"unique{i}"}
            for i in range(6)
        }
        resolver = MetadataResolver.from_documents(docs, body_terms=bodies)
        assert ("widget" in resolver.generic_terms) is expected_dropped

    def test_the_threshold_is_configurable(self):
        docs = [_document("a", "Alpha Widget"), _document("b", "Beta Gadget")]
        bodies = {"a": {"widget"}, "b": {"widget"}}
        assert "widget" in MetadataResolver.from_documents(docs, body_terms=bodies).generic_terms
        loose = MetadataResolver.from_documents(
            docs, body_terms=bodies, max_document_frequency=1.5
        )
        assert "widget" not in loose.generic_terms

    def test_corpus_terms_builds_a_vocabulary_per_document(self):
        from mmrag.retrieval.metadata import corpus_terms
        from mmrag.schemas import BBox, Chunk, ChunkType

        def chunk(doc, text, cid):
            return Chunk(
                chunk_id=cid, doc_id=doc, page_number=1, chunk_type=ChunkType.TEXT,
                text=text, element_ids=[f"{doc}#p1#t000"],
                bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4), variant="method2",
            )

        vocab = corpus_terms([chunk("a", "Alpha Beta", "c1"), chunk("a", "Gamma", "c2"),
                              chunk("b", "Beta", "c3")])
        assert vocab == {"a": {"alpha", "beta", "gamma"}, "b": {"beta"}}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FakeRetriever:
    """Returns a fixed ranked list, and records how it was called."""

    def __init__(self, name: str, modality: Modality, chunk_ids: list[str]):
        self.name = name
        self.modality = modality
        self.chunk_ids = chunk_ids
        self.calls: list[tuple[str, int, MetadataFilter | None]] = []

    def retrieve(self, query, k, filters=None):
        self.calls.append((query, k, filters))
        return RetrieverOutput(
            retriever=self.name,
            modality=self.modality,
            hits=[Hit(cid, 1.0 / (i + 1), i + 1) for i, cid in enumerate(self.chunk_ids[:k])],
        )


class TestModalityAwareRetriever:
    @pytest.fixture
    def chunks(self):
        return {
            "t1": _chunk(1, chunk_id="t1", chunk_type=ChunkType.TEXT),
            "t2": _chunk(2, chunk_id="t2", chunk_type=ChunkType.TEXT),
            "tab1": _chunk(3, chunk_id="tab1", chunk_type=ChunkType.TABLE),
            "fig1": _chunk(4, chunk_id="fig1", chunk_type=ChunkType.FIGURE),
        }

    def _build(self, chunks, *, fallback=False):
        retrievers = {
            "bm25": FakeRetriever("bm25", Modality.TEXT, ["t1", "t2"]),
            "dense": FakeRetriever("dense", Modality.TEXT, ["t2", "t1"]),
            "table": FakeRetriever("table", Modality.TABLE, ["tab1"]),
            "image": FakeRetriever("image", Modality.IMAGE, ["fig1"]),
        }
        retriever = ModalityAwareRetriever(
            RetrievalConfig(top_k=5, candidates_per_retriever=10),
            router=HeuristicRouter(RouterConfig(fallback_to_all=fallback)),
            retrievers=retrievers,
            chunks=chunks,
        )
        return retriever, retrievers

    def test_only_routed_retrievers_fire(self, chunks):
        retriever, fakes = self._build(chunks)
        retriever.retrieve("Why did the committee revise its guidance?")

        assert fakes["bm25"].calls and fakes["dense"].calls
        assert not fakes["table"].calls, "table retriever fired on a prose question"
        assert not fakes["image"].calls, "image retriever fired on a prose question"

    def test_table_query_fires_the_table_retriever(self, chunks):
        retriever, fakes = self._build(chunks)
        retriever.retrieve("which table shows revenue by segment")
        assert fakes["table"].calls
        assert not fakes["image"].calls

    def test_visual_query_fires_the_image_retriever(self, chunks):
        retriever, fakes = self._build(chunks)
        retriever.retrieve("what does the architecture diagram show")
        assert fakes["image"].calls
        assert not fakes["table"].calls

    def test_forced_modalities_override_the_router(self, chunks):
        """The ablation hook: fire an exact retriever set regardless of routing."""
        retriever, fakes = self._build(chunks)
        retriever.retrieve("anything", force_modalities=[Modality.IMAGE])
        assert fakes["image"].calls
        assert not fakes["bm25"].calls

    def test_results_are_fused_and_ranked(self, chunks):
        retriever, _ = self._build(chunks, fallback=True)
        result = retriever.retrieve("2019-nCoV case counts")
        assert [h.rank for h in result.results] == list(range(1, len(result.results) + 1))
        assert len({h.chunk_id for h in result.results}) == len(result.results)

    def test_diagnostics_name_what_fired_and_what_contributed(self, chunks):
        retriever, _ = self._build(chunks, fallback=True)
        result = retriever.retrieve("2019-nCoV case counts")
        diagnostics = result.diagnostics

        assert set(diagnostics["retrievers_fired"]) == {"bm25", "dense", "table", "image"}
        assert diagnostics["contributed"]["table"] >= 1
        assert diagnostics["returned_by_modality"]["table"] == 1
        assert "per_retriever" in diagnostics

    def test_doc_filter_reaches_every_retriever(self, chunks):
        retriever, fakes = self._build(chunks, fallback=True)
        retriever.retrieve("2019-nCoV case counts", doc_ids=["annual_2024"])
        for fake in fakes.values():
            if fake.calls:
                assert fake.calls[0][2].doc_ids == ["annual_2024"]

    def test_routing_decision_is_returned(self, chunks):
        retriever, _ = self._build(chunks)
        result = retriever.retrieve("which table shows revenue")
        assert result.routing is not None
        assert Modality.TABLE in result.routing.modalities

    def test_latency_is_broken_down_per_retriever(self, chunks):
        retriever, _ = self._build(chunks, fallback=True)
        result = retriever.retrieve("2019-nCoV case counts")
        assert "routing_ms" in result.latency_ms
        assert "fusion_ms" in result.latency_ms
        assert "total_ms" in result.latency_ms

    def test_stale_index_references_are_reported(self, chunks):
        retriever, fakes = self._build(chunks)
        fakes["bm25"].chunk_ids = ["does-not-exist", "t1"]
        result = retriever.retrieve("Why did the committee revise its guidance?")
        assert result.diagnostics["missing_chunk_records"] >= 1

    def test_output_is_deterministic(self, chunks):
        retriever, _ = self._build(chunks, fallback=True)
        first = [h.chunk_id for h in retriever.retrieve("2019-nCoV case counts").results]
        second = [h.chunk_id for h in retriever.retrieve("2019-nCoV case counts").results]
        assert first == second

    def test_reported_modality_reflects_what_a_chunk_is(self, chunks):
        retriever, _ = self._build(chunks, fallback=True)
        result = retriever.retrieve("2019-nCoV case counts")
        by_id = {h.chunk_id: h for h in result.results}
        assert by_id["tab1"].modality is Modality.TABLE
        assert by_id["fig1"].modality is Modality.IMAGE


class TestMissingCollections:
    """A modality the corpus does not contain must degrade, not crash.

    The router fans out by design, so it will routinely ask for a modality this
    corpus happens to lack -- a slice with no figures leaves the image
    collection absent. That has to return "found nothing", not abort the query.
    """

    class _AbsentStore:
        collection = "method2_image"

        def exists(self):
            return False

        def search(self, *args, **kwargs):  # pragma: no cover - must not be called
            raise AssertionError("search called on a collection reported absent")

    class _BrokenStore:
        collection = "method2_image"

        def exists(self):
            return True

        def search(self, *args, **kwargs):
            raise ConnectionError("qdrant is down")

    def _embedder(self):
        class Fake:
            def ensure_loaded(self):
                return 0.0

            def embed_query(self, text):
                import numpy as np

                return np.zeros(4, dtype="float32")

        return Fake()

    def test_absent_collection_returns_no_hits(self):
        from mmrag.retrieval.modality_retrievers import ImageRetriever

        retriever = ImageRetriever(
            image_store=self._AbsentStore(),
            image_embedder=self._embedder(),
            text_index=None,
            chunks={},
        )
        output = retriever.retrieve("a diagram", 5)
        assert output.hits == []
        assert output.diagnostics["unavailable"] == "collection missing"

    def test_unreachable_service_is_reported_not_raised(self):
        from mmrag.retrieval.modality_retrievers import ImageRetriever

        retriever = ImageRetriever(
            image_store=self._BrokenStore(),
            image_embedder=self._embedder(),
            text_index=None,
            chunks={},
        )
        output = retriever.retrieve("a diagram", 5)
        assert output.hits == []
        assert "ConnectionError" in output.diagnostics["unavailable"]

    def test_dense_retriever_survives_a_missing_collection(self):
        from mmrag.retrieval.modality_retrievers import DenseRetriever

        retriever = DenseRetriever(self._AbsentStore(), self._embedder())
        output = retriever.retrieve("anything", 5)
        assert output.hits == []
        assert output.diagnostics["unavailable"] == "collection missing"


class TestImageEmbedderDimension:
    """CLIP declines to report its own embedding width.

    ``get_sentence_embedding_dimension()`` returns None for CLIP models in
    sentence-transformers -- the CLIPModel wrapper has no pooling layer to ask.
    Treating that as an error failed a full index build after the text and table
    indexes had already been written, so the fallback probes the model instead.
    """

    class _SilentModel:
        """A model that will not say how wide its vectors are."""

        def __init__(self, width: int = 512):
            self.width = width
            self.encode_calls = 0

        def get_sentence_embedding_dimension(self):
            return None

        def encode(self, texts, **kwargs):
            import numpy as np

            self.encode_calls += 1
            return np.zeros((len(texts), self.width), dtype="float32")

    class _TalkativeModel:
        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, **kwargs):  # pragma: no cover - must not be needed
            raise AssertionError("probed a model that reported its dimension")

    def _embedder(self, model):
        from mmrag.config import EmbeddingConfig
        from mmrag.embeddings.image import ImageEmbedder

        embedder = ImageEmbedder(EmbeddingConfig())
        embedder._model = model
        return embedder

    def test_probes_when_the_model_reports_nothing(self):
        embedder = self._embedder(self._SilentModel(512))
        assert embedder.dimension == 512

    def test_probe_result_is_cached(self):
        model = self._SilentModel(512)
        embedder = self._embedder(model)
        assert embedder.dimension == embedder.dimension == 512
        assert model.encode_calls == 1, "the probe ran more than once"

    def test_a_reported_dimension_is_used_without_probing(self):
        assert self._embedder(self._TalkativeModel()).dimension == 384

    def test_empty_input_returns_a_correctly_shaped_array(self):
        """This path reads .dimension, so it must not recurse into probing."""
        embedder = self._embedder(self._SilentModel(512))
        assert embedder.embed_queries([]).shape == (0, 512)

