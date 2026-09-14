"""Method 3: config, device handling, page index, retrieval and composition.

Everything here runs on a CPU with no GPU, no model download and no network. The
real ColQwen2 path is exercised by one test marked ``slow`` that skips unless the
visual extra is installed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
import typing
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from mmrag.config import VisualRetrievalConfig, load_experiment_config
from mmrag.embeddings.visual import (
    ColQwen2Encoder,
    DeviceUnavailableError,
    VisualEncodingError,
    VisualModelUnavailableError,
    compatible_identity,
    resolve_dtype,
    resolve_visual_device,
)
from mmrag.evaluation.gold import GoldQuery
from mmrag.evaluation.retrieval_eval import evaluate_query
from mmrag.indexing.modality import ModalityIndex
from mmrag.ingestion.parser import ParsedDocument
from mmrag.ingestion.pipeline import write_sidecar
from mmrag.methods.method2_modality import Method2ModalityAware
from mmrag.methods.method3_visual import (
    CpuIndexingRefusedError,
    Method3HybridVisual,
    prepare_pages,
    resolve_page_image,
)
from mmrag.retrieval.base import Hit, MetadataFilter, RetrieverOutput
from mmrag.retrieval.modality import ModalityAwareRetriever
from mmrag.retrieval.router import HeuristicRouter
from mmrag.retrieval.visual_page import VisualPageRetriever
from mmrag.schemas import BBox, Chunk, ChunkType, Document, Modality, Page
from mmrag.stores.multivector import (
    EMBEDDINGS_FILE,
    MANIFEST_FILE,
    IndexIntegrityError,
    PageEmbeddingIndex,
    PageIndexWriter,
    PageRecord,
    QueryEmbeddingCache,
    maxsim_scores,
)

DIM = 8
COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "white": (255, 255, 255)}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _unit(index: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[index] = 1.0
    return v


class FakeEncoder:
    """Deterministic stand-in for ColQwen2.

    A page's tokens point along the axis of its dominant colour channel; a query
    token for "red"/"green"/"blue" points along the same axis. Pages get a
    different number of tokens each, so offsets are genuinely exercised.
    """

    def __init__(self, device: str = "cpu", *, model: str = "vidore/colqwen2-v1.0",
                 commit: str = "abc123"):
        self.device = device
        self.model = model
        self.commit = commit
        self.image_calls = 0
        self.query_calls = 0

    def encode_images(self, images):
        self.image_calls += 1
        out = []
        for image in images:
            r, g, b = image.getpixel((0, 0))
            axis = int(np.argmax([r, g, b])) if (r, g, b) != (255, 255, 255) else 3
            n = 3 + axis
            out.append(np.stack([_unit(axis)] + [_unit(4 + (i % 4)) * 0.1 for i in range(n - 1)]))
        return out

    def encode_queries(self, texts):
        self.query_calls += 1
        axes = {"red": 0, "green": 1, "blue": 2}
        return [
            np.stack([_unit(axes[w]) for w in t.lower().split() if w in axes] or [_unit(7)])
            for t in texts
        ]

    def identity(self):
        return {"model": self.model, "revision": None, "commit": self.commit, "dim": DIM}

    def describe(self):
        return {**self.identity(), "device": self.device, "dtype": "float32",
                "processor": {"class": "Fake"}}


class FakeTorch:
    def __init__(self, *, cuda=False, count=0, mps=False, bf16=True):
        self.cuda = types.SimpleNamespace(
            is_available=lambda: cuda, device_count=lambda: count, is_bf16_supported=lambda: bf16
        )
        self.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))


class ListRetriever:
    """A Method 2 retriever stand-in returning a fixed ranked list."""

    def __init__(self, name, modality, ids):
        self.name, self.modality, self.ids = name, modality, ids

    def retrieve(self, query, k, filters=None):
        return RetrieverOutput(self.name, self.modality,
                               [Hit(c, 1.0 / r, r) for r, c in enumerate(self.ids[:k], 1)])


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

# doc_a: p1 red, p2 green, p3 white (no chunks); doc_b: p1 blue
PAGES = {"doc_a": ["red", "green", "white"], "doc_b": ["blue"]}


def chunk(cid, doc, page, kind=ChunkType.TEXT):
    return Chunk(chunk_id=cid, doc_id=doc, page_number=page, chunk_type=kind, text=f"{cid} text",
                 element_ids=[f"{doc}#p{page}#text000"], bbox=BBox(x0=0, y0=0, x1=1, y1=0.5),
                 variant="method2", metadata={"doc_title": doc})


CHUNKS = [
    chunk("m2#a1t", "doc_a", 1), chunk("m2#a1f", "doc_a", 1, ChunkType.FIGURE),
    chunk("m2#a2t", "doc_a", 2), chunk("m2#a2b", "doc_a", 2, ChunkType.TABLE),
    chunk("m2#b1t", "doc_b", 1),
]


@pytest.fixture
def corpus(tmp_path):
    processed = tmp_path / "processed"
    shas = {}
    for doc_id, colours in PAGES.items():
        sha = hashlib.sha256(doc_id.encode()).hexdigest()
        shas[doc_id] = sha
        pages = []
        for number, colour in enumerate(colours, start=1):
            path = processed / doc_id / "pages" / f"p{number:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 50), COLORS[colour]).save(path)
            pages.append(Page(page_id=f"{doc_id}#p{number}", doc_id=doc_id, page_number=number,
                              width=612, height=792, image_path=str(path), image_width=40,
                              image_height=50, image_dpi=150))
        document = Document(doc_id=doc_id, title=doc_id.upper(), file_name=f"{doc_id}.pdf",
                            file_path=f"raw/{doc_id}.pdf", sha256=sha, n_pages=len(colours),
                            n_pages_ingested=len(colours))
        write_sidecar(ParsedDocument(document=document, pages=pages, elements=[]),
                      processed / doc_id / "parsed.json", stats={})

    lock = tmp_path / "corpus.lock.yaml"
    lock.write_text(yaml.safe_dump({"entries": {
        d: {"n_pages": len(PAGES[d]), "sha256": s, "size_bytes": 1} for d, s in shas.items()
    }}), encoding="utf-8")

    indexes = tmp_path / "indexes"
    (indexes / "method2").mkdir(parents=True)
    (indexes / "method2" / "chunks.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in CHUNKS), encoding="utf-8")
    return types.SimpleNamespace(processed=processed, lock=lock, indexes=indexes, tmp=tmp_path)


def make_method(corpus, *, encoder=None, **overrides):
    cfg = load_experiment_config("method3", overrides={
        "embedding": {"visual_dim": DIM}, "retrieval": {"rerank_enabled": False}, **overrides})
    encoder = encoder or FakeEncoder()
    method = Method3HybridVisual(cfg, index_base=corpus.indexes, processed_dir=corpus.processed,
                                 lock_path=corpus.lock, encoder_factory=lambda device: encoder)
    return method, encoder


def built(corpus, **kwargs):
    method, encoder = make_method(corpus)
    method.build_index(device="cpu", allow_cpu=True, **kwargs)
    return method, encoder


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_method3_is_method2_plus_the_page_signal(self):
        m2, m3 = load_experiment_config("method2"), load_experiment_config("method3")
        assert m3.method == "method3"
        assert {k: v for k, v in m3.retrieval.fusion_weights.items() if k != "visual_page"} == \
            m2.retrieval.fusion_weights
        assert m3.retrieval.fusion_weights["visual_page"] == 1.0
        for field in ("top_k", "candidates_per_retriever", "rrf_k", "rerank_enabled",
                      "rerank_model", "rerank_top_n"):
            assert getattr(m3.retrieval, field) == getattr(m2.retrieval, field)
        assert m3.router == m2.router and m3.enrichment == m2.enrichment
        assert m3.chunking == m2.chunking
        assert m3.embedding.text_model == m2.embedding.text_model

    def test_the_rerank_pool_fits_every_modality_floor(self):
        r = load_experiment_config("method3").retrieval
        assert 4 * r.rerank_pool_per_modality <= r.rerank_top_n

    def test_generator_judge_and_metrics_are_shared(self):
        m2, m3 = load_experiment_config("method2"), load_experiment_config("method3")
        assert m3.generation == m2.generation and m3.evaluation == m2.evaluation
        assert m3.evaluation.llm_judge_model == m3.generation.text_model == "gpt-4o-mini"

    def test_methods_1_and_2_are_unchanged(self):
        m1, m2 = load_experiment_config("method1"), load_experiment_config("method2")
        assert m1.retrieval.fusion_weights == {"bm25": 1.0, "dense": 1.0}
        assert m2.retrieval.fusion_weights == {"bm25": 1.0, "dense": 1.0, "table": 1.0,
                                               "image": 0.7}
        assert m2.retrieval.rerank_pool_per_modality == 8
        assert m1.visual == m2.visual == VisualRetrievalConfig()

    @pytest.mark.parametrize("device", ["gpu", "cuda0", "CUDA", ""])
    def test_unrecognised_devices_are_rejected_by_the_config(self, device):
        with pytest.raises(ValueError, match=r"visual\.device"):
            VisualRetrievalConfig(device=device)

    @pytest.mark.parametrize("device", ["auto", "cpu", "mps", "cuda", "cuda:1"])
    def test_recognised_devices(self, device):
        assert VisualRetrievalConfig(device=device).device == device


# ---------------------------------------------------------------------------
# Device and dtype
# ---------------------------------------------------------------------------


class TestDeviceSelection:
    def test_cpu_never_needs_torch(self):
        class Exploding:
            def __getattr__(self, name):
                raise AssertionError("cpu must not probe torch")

        assert resolve_visual_device("cpu", torch_module=Exploding()) == "cpu"

    def test_auto_prefers_cuda_then_mps_then_cpu(self):
        assert resolve_visual_device("auto", torch_module=FakeTorch(cuda=True, count=1)) == "cuda"
        assert resolve_visual_device("auto", torch_module=FakeTorch(mps=True)) == "mps"
        assert resolve_visual_device("auto", torch_module=FakeTorch()) == "cpu"

    def test_explicit_cuda_without_a_gpu_is_an_error_not_a_fallback(self):
        with pytest.raises(DeviceUnavailableError, match="CUDA is not available"):
            resolve_visual_device("cuda", torch_module=FakeTorch())

    def test_cuda_index_must_exist(self):
        torch = FakeTorch(cuda=True, count=1)
        assert resolve_visual_device("cuda:0", torch_module=torch) == "cuda:0"
        with pytest.raises(DeviceUnavailableError, match="only 1 CUDA"):
            resolve_visual_device("cuda:1", torch_module=torch)

    def test_mps_must_exist(self):
        with pytest.raises(DeviceUnavailableError, match="mps"):
            resolve_visual_device("mps", torch_module=FakeTorch())

    def test_nonsense_is_rejected(self):
        with pytest.raises(DeviceUnavailableError, match="unrecognised"):
            resolve_visual_device("gpu")

    def test_dtype_follows_the_hardware(self):
        assert resolve_dtype("auto", "cuda", torch_module=FakeTorch(bf16=True)) == "bfloat16"
        assert resolve_dtype("auto", "cuda:0", torch_module=FakeTorch(bf16=False)) == "float16"
        assert resolve_dtype("auto", "cpu") == "float32"
        assert resolve_dtype("auto", "mps") == "float32"
        assert resolve_dtype("bfloat16", "cpu") == "bfloat16"

    def test_this_machine_resolves_auto_without_error(self):
        assert resolve_visual_device("auto") in {"cpu", "cuda", "mps"}


# ---------------------------------------------------------------------------
# ColQwen2 wrapper
# ---------------------------------------------------------------------------


class TestColQwen2Encoder:
    def test_missing_package_is_a_clear_requirement(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "colpali_engine", None)
        monkeypatch.setitem(sys.modules, "colpali_engine.models", None)
        encoder = ColQwen2Encoder("vidore/colqwen2-v1.0", VisualRetrievalConfig(device="cpu"),
                                  expected_dim=128)
        with pytest.raises(VisualModelUnavailableError, match=r"\.\[visual\]"):
            encoder.encode_queries(["q"])

    def _fake_colpali(self, monkeypatch, output):
        torch = pytest.importorskip("torch")

        class Batch(dict):
            def to(self, device):
                return self

        class Processor:
            image_processor = types.SimpleNamespace(min_pixels=1, max_pixels=2)

            @classmethod
            def from_pretrained(cls, name, **kw):
                return cls()

            def process_queries(self, texts):
                return Batch(attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]])[: len(texts)])

            process_images = process_queries

        class Model:
            device = "cpu"
            config = types.SimpleNamespace(_commit_hash="c0ffee")
            loaded_with: typing.ClassVar[dict] = {}

            @classmethod
            def from_pretrained(cls, name, **kw):
                cls.loaded_with = kw
                return cls()

            def eval(self):
                return self

            def __call__(self, **batch):
                return output(torch)

        module = types.ModuleType("colpali_engine.models")
        module.ColQwen2, module.ColQwen2Processor = Model, Processor
        monkeypatch.setitem(sys.modules, "colpali_engine", types.ModuleType("colpali_engine"))
        monkeypatch.setitem(sys.modules, "colpali_engine.models", module)
        return Model

    def test_padding_is_dropped_and_vectors_are_float32(self, monkeypatch):
        def output(torch):
            out = torch.ones(2, 3, 4, dtype=torch.bfloat16)
            out[0, 2] = 0  # padding position
            return out

        model = self._fake_colpali(monkeypatch, output)
        encoder = ColQwen2Encoder("m", VisualRetrievalConfig(device="cpu", model_revision="r1"),
                                  expected_dim=4)
        vectors = encoder.encode_queries(["a", "b"])
        assert [v.shape for v in vectors] == [(2, 4), (3, 4)]
        assert all(v.dtype == np.float32 for v in vectors)
        assert model.loaded_with["device_map"] == "cpu" and model.loaded_with["revision"] == "r1"
        assert encoder.identity() == {"model": "m", "revision": "r1", "commit": "c0ffee", "dim": 4}
        assert encoder.describe()["dtype"] == "float32"

    def test_nan_output_is_refused_with_advice(self, monkeypatch):
        self._fake_colpali(monkeypatch, lambda torch: torch.full((2, 3, 4), float("nan")))
        encoder = ColQwen2Encoder("m", VisualRetrievalConfig(device="cpu"), expected_dim=4)
        with pytest.raises(VisualEncodingError, match="float32"):
            encoder.encode_queries(["a", "b"])

    def test_wrong_width_is_refused(self, monkeypatch):
        self._fake_colpali(monkeypatch, lambda torch: torch.ones(2, 3, 5))
        encoder = ColQwen2Encoder("m", VisualRetrievalConfig(device="cpu"), expected_dim=4)
        with pytest.raises(VisualEncodingError, match="width 4"):
            encoder.encode_queries(["a", "b"])

    def test_requesting_cuda_on_a_cpu_machine_fails_at_construction(self, monkeypatch):
        import mmrag.embeddings.visual as visual

        monkeypatch.setattr(visual, "_torch", lambda: FakeTorch())
        with pytest.raises(DeviceUnavailableError):
            ColQwen2Encoder("m", VisualRetrievalConfig(device="cuda"), expected_dim=4)

    def test_identity_compatibility(self):
        a = {"model": "m", "revision": None, "commit": "x", "dim": 4}
        assert compatible_identity(a, {**a, "revision": "r"})
        assert not compatible_identity(a, {**a, "commit": "y"})
        assert not compatible_identity(a, {**a, "dim": 8})
        assert not compatible_identity(a, {**a, "model": "other"})


# ---------------------------------------------------------------------------
# MaxSim and the page index
# ---------------------------------------------------------------------------


class TestMaxSim:
    def test_matches_a_naive_implementation_across_blocks(self):
        rng = np.random.default_rng(0)
        sizes = [5, 1, 9, 3, 7]
        embeddings = rng.normal(size=(sum(sizes), DIM)).astype(np.float16)
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        query = rng.normal(size=(4, DIM)).astype(np.float32)

        naive = [
            (embeddings[offsets[i]:offsets[i + 1]].astype(np.float32) @ query.T).max(0).sum()
            for i in range(len(sizes))
        ]
        for block in (1, 6, 1_000):
            got = maxsim_scores(query, embeddings, offsets, block_tokens=block)
            np.testing.assert_allclose(got, naive, rtol=1e-5)
        np.testing.assert_allclose(maxsim_scores(query, embeddings, offsets, [3, 0]),
                                   [naive[3], naive[0]], rtol=1e-5)


def record(page_id, n_tokens, doc="d", number=1):
    return PageRecord(page_id=page_id, doc_id=doc, page_number=number, image=f"{doc}/pages/x.png",
                      image_sha256="0" * 64, image_width=1, image_height=1, image_dpi=150,
                      n_tokens=n_tokens)


class TestPageIndexStorage:
    def _write(self, directory):
        writer = PageIndexWriter(directory, dim=DIM)
        writer.add(record("d#p1", 2, number=1), np.ones((2, DIM)))
        writer.add(record("d#p2", 3, number=2), np.zeros((3, DIM)) + _unit(1))
        return writer.finalize({"method": "method3", "model": {"identity": {"model": "m"}},
                                "corpus": {"complete": True}})

    def test_roundtrip_with_manifest(self, tmp_path):
        index = self._write(tmp_path / "visual")
        assert index.manifest["index_version"] == 1
        assert index.manifest["layout"] == {"dim": DIM, "storage_dtype": "float16",
                                            "n_pages": 2, "n_tokens": 5}
        assert set(index.manifest["files"]) == {"embeddings.npy", "offsets.npy", "pages.jsonl"}
        assert index.embeddings.dtype == np.float16
        assert index.page_index("d", 2) == 1 and index.is_complete
        assert not (tmp_path / "visual.tmp").exists()

    def test_tampered_embeddings_are_refused(self, tmp_path):
        index = self._write(tmp_path / "visual")
        np.save(index.directory / EMBEDDINGS_FILE, np.zeros((5, DIM), dtype=np.float16))
        with pytest.raises(IndexIntegrityError, match="checksum"):
            PageEmbeddingIndex.load(tmp_path / "visual")

    def test_a_different_index_version_is_refused(self, tmp_path):
        index = self._write(tmp_path / "visual")
        manifest = json.loads((index.directory / MANIFEST_FILE).read_text())
        manifest["index_version"] = 99
        (index.directory / MANIFEST_FILE).write_text(json.dumps(manifest))
        with pytest.raises(IndexIntegrityError, match="version 99"):
            PageEmbeddingIndex.load(tmp_path / "visual")

    def test_writer_refuses_bad_input(self, tmp_path):
        writer = PageIndexWriter(tmp_path / "v", dim=DIM)
        with pytest.raises(IndexIntegrityError, match="width"):
            writer.add(record("d#p1", 2), np.ones((2, DIM + 1)))
        with pytest.raises(IndexIntegrityError, match="vectors for a record"):
            writer.add(record("d#p1", 2), np.ones((3, DIM)))
        writer.add(record("d#p1", 2), np.ones((2, DIM)))
        with pytest.raises(IndexIntegrityError, match="twice"):
            writer.add(record("d#p1", 2), np.ones((2, DIM)))
        with pytest.raises(IndexIntegrityError, match="empty"):
            PageIndexWriter(tmp_path / "w", dim=DIM).finalize({})

    def test_query_cache_is_bound_to_one_model(self, tmp_path):
        cache = QueryEmbeddingCache(tmp_path / "qc")
        identity = {"model": "m", "commit": "a", "dim": DIM}
        cache.put_many([("what is red", np.ones((2, DIM)))], identity=identity,
                       encoder={"device": "cuda", "dtype": "bfloat16"})
        assert cache.get("what is red").shape == (2, DIM) and cache.get("other") is None
        with pytest.raises(IndexIntegrityError, match="not compatible"):
            cache.put_many([("x", np.ones((1, DIM)))], identity={**identity, "commit": "b"},
                           encoder={})


# ---------------------------------------------------------------------------
# Page preparation and index build
# ---------------------------------------------------------------------------


class TestPagePreparation:
    def test_a_windows_path_from_another_machine_still_resolves(self, corpus):
        page = Page(page_id="doc_a#p1", doc_id="doc_a", page_number=1, width=1, height=1,
                    image_path=r"C:\Users\someone\rag\data\processed\doc_a\pages\p0001.png")
        assert resolve_page_image(page, corpus.processed) == \
            corpus.processed / "doc_a" / "pages" / "p0001.png"

    def test_pages_carry_exact_provenance(self, corpus):
        pages, documents, problems = prepare_pages(corpus.processed)
        assert problems == [] and set(documents) == {"doc_a", "doc_b"}
        assert [(p.record.doc_id, p.record.page_number) for p in pages] == \
            [("doc_a", 1), ("doc_a", 2), ("doc_a", 3), ("doc_b", 1)]
        first = pages[0].record
        assert first.image == "doc_a/pages/p0001.png" and first.image_dpi == 150
        assert first.image_sha256 == sha(corpus.processed / "doc_a" / "pages" / "p0001.png")

    def test_a_missing_render_is_reported(self, corpus):
        (corpus.processed / "doc_b" / "pages" / "p0001.png").unlink()
        _, _, problems = prepare_pages(corpus.processed)
        assert len(problems) == 1 and "doc_b#p1" in problems[0]


class TestIndexBuild:
    def test_cpu_is_refused_unless_allowed(self, corpus):
        method, encoder = make_method(corpus)
        with pytest.raises(CpuIndexingRefusedError, match="GPU"):
            method.build_index(device="cpu")
        assert encoder.image_calls == 0

    def test_manifest_records_model_device_preprocessing_and_corpus(self, corpus):
        method, _ = built(corpus, batch_size=3)
        manifest = method.visual_index.manifest
        assert manifest["kind"] == "visual_page_index"
        assert manifest["model"]["identity"] == FakeEncoder().identity()
        assert manifest["model"]["description"]["device"] == "cpu"
        assert manifest["preprocessing"]["image_dpi"] == [150]
        assert manifest["corpus"]["lock_sha256"] == sha(corpus.lock)
        assert set(manifest["corpus"]["documents"]) == {"doc_a", "doc_b"}
        assert manifest["corpus"]["complete"] is True
        assert manifest["chunk_set"]["path"].endswith("method2/chunks.jsonl")
        assert manifest["chunk_set"]["indexed_pages_without_chunks"] == 1  # doc_a p3
        assert manifest["config"]["visual_dim"] == DIM
        assert [p.n_tokens for p in method.visual_index.pages] == [3, 4, 6, 5]

    def test_method2_artifacts_are_never_written(self, corpus):
        before = {p: sha(p) for p in (corpus.indexes / "method2").rglob("*") if p.is_file()}
        built(corpus)
        after = {p: sha(p) for p in (corpus.indexes / "method2").rglob("*") if p.is_file()}
        assert before == after
        assert (corpus.indexes / "visual_pages" / "index" / MANIFEST_FILE).exists()

    def test_subset_builds_are_marked_partial(self, corpus):
        method, _ = built(corpus, max_pages=2)
        assert not method.visual_index.is_complete
        assert method.visual_index.manifest["corpus"]["subset"] == {"doc_ids": None, "max_pages": 2}

    def test_a_corpus_that_drifted_from_the_lock_is_refused(self, corpus):
        payload = yaml.safe_load(corpus.lock.read_text())
        payload["entries"]["doc_b"]["sha256"] = "f" * 64
        corpus.lock.write_text(yaml.safe_dump(payload))
        method, _ = make_method(corpus)
        with pytest.raises(IndexIntegrityError, match="doc_b"):
            method.build_index(device="cpu", allow_cpu=True)

    def test_out_of_memory_halves_the_batch(self, corpus):
        class Flaky(FakeEncoder):
            def encode_images(self, images):
                if len(images) > 1:
                    raise RuntimeError("CUDA out of memory")
                return super().encode_images(images)

        method, _ = make_method(corpus, encoder=Flaky())
        method.build_index(device="cpu", allow_cpu=True, batch_size=4)
        assert len(method.visual_index.pages) == 4

    def test_a_complete_index_missing_a_chunk_page_is_refused_on_load(self, corpus):
        built(corpus)
        extra = chunk("m2#b9t", "doc_b", 9)
        with (corpus.indexes / "method2" / "chunks.jsonl").open("a", encoding="utf-8") as f:
            f.write(extra.model_dump_json() + "\n")
        method, _ = make_method(corpus)
        with pytest.raises(IndexIntegrityError, match="doc_b#p9"):
            _ = method.visual_index


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestVisualPageRetriever:
    def _retriever(self, corpus, **kwargs):
        method, encoder = built(corpus)
        kwargs.setdefault("encoder_factory", lambda: encoder)
        return VisualPageRetriever(method.visual_index, method.chunks, **kwargs), encoder

    def test_a_page_hit_expands_to_that_pages_own_chunks(self, corpus):
        retriever, _ = self._retriever(corpus)
        out = retriever.retrieve("green", k=10)
        assert out.modality is Modality.VISUAL_PAGE
        assert out.chunk_ids[:2] == ["m2#a2t", "m2#a2b"]
        assert [h.rank for h in out.hits] == list(range(1, len(out.hits) + 1))
        assert out.hits[0].score == out.hits[1].score > out.hits[2].score

    def test_every_hit_is_a_real_chunk_on_a_single_page(self, corpus):
        retriever, _ = self._retriever(corpus)
        by_id = {c.chunk_id: c for c in CHUNKS}
        out = retriever.retrieve("blue", k=10)
        assert set(out.chunk_ids) <= set(by_id)
        assert len(out.chunk_ids) == len(set(out.chunk_ids)) == 5
        assert out.diagnostics["pages_without_chunks_skipped"] == 1

    def test_k_truncates_chunks_not_pages(self, corpus):
        retriever, _ = self._retriever(corpus)
        assert retriever.retrieve("red", k=1).chunk_ids == ["m2#a1t"]

    def test_filters(self, corpus):
        retriever, _ = self._retriever(corpus)
        only_b = retriever.retrieve("red", 10, MetadataFilter(doc_ids=["doc_b"]))
        assert set(only_b.chunk_ids) == {"m2#b1t"}
        tables = retriever.retrieve("red", 10, MetadataFilter(chunk_types=["table"]))
        assert tables.chunk_ids == ["m2#a2b"]
        paged = retriever.retrieve("red", 10, MetadataFilter(page_numbers=[2]))
        assert {c.split("#")[1][:2] for c in paged.chunk_ids} == {"a2"}

    def test_cached_queries_never_load_the_model(self, corpus):
        method, encoder = built(corpus)
        cache = QueryEmbeddingCache(method.query_cache_dir)
        cache.put_many([("red", encoder.encode_queries(["red"])[0])],
                       identity=encoder.identity(), encoder=encoder.describe())
        calls = encoder.query_calls

        def refuse():
            raise AssertionError("model loaded despite a cached query")

        retriever = VisualPageRetriever(method.visual_index, method.chunks, cache=cache,
                                        encoder_factory=refuse)
        out = retriever.retrieve("red", k=2)
        assert out.chunk_ids == ["m2#a1t", "m2#a1f"]
        assert out.diagnostics["query_embedding"] == "cache"
        assert encoder.query_calls == calls

    def test_uncached_query_without_a_model_is_a_clear_error(self, corpus):
        method, _ = built(corpus)
        retriever = VisualPageRetriever(method.visual_index, method.chunks, encoder_factory=None)
        with pytest.raises(VisualModelUnavailableError, match="embed-queries"):
            retriever.retrieve("red", k=2)

    def test_a_different_query_model_is_refused(self, corpus):
        retriever, _ = self._retriever(corpus, encoder_factory=lambda: FakeEncoder(commit="zzz"))
        with pytest.raises(IndexIntegrityError, match="not the model"):
            retriever.retrieve("red", k=2)
        # A refused encoder is not kept, so a second query is refused too.
        with pytest.raises(IndexIntegrityError, match="not the model"):
            retriever.retrieve("red", k=2)

    def test_pages_are_ranked_before_they_are_expanded(self, corpus):
        retriever, _ = self._retriever(corpus)
        ranking = retriever.rank_pages("green")
        assert ranking.pages_scored == 4 and ranking.query_embedding == "encoded"
        assert [p.rank for p in ranking.pages] == [1, 2, 3, 4]
        assert (ranking.pages[0].doc_id, ranking.pages[0].page_number) == ("doc_a", 2)
        only_b = retriever.rank_pages("green", MetadataFilter(doc_ids=["doc_b"]))
        assert [p.page_id for p in only_b.pages] == ["doc_b#p1"]

    def test_a_cache_from_another_model_is_refused(self, corpus):
        method, _ = built(corpus)
        cache = QueryEmbeddingCache(method.query_cache_dir)
        other = FakeEncoder(commit="zzz")
        cache.put_many([("red", np.ones((1, DIM)))], identity=other.identity(), encoder={})
        with pytest.raises(IndexIntegrityError, match="not compatible"):
            VisualPageRetriever(method.visual_index, method.chunks, cache=cache)

    def test_embed_queries_fills_the_cache_once(self, corpus):
        method, _ = built(corpus)
        assert method.embed_queries(["red", "blue", "red"]) == \
            {"requested": 2, "already_cached": 0, "encoded": 2}
        assert method.embed_queries(["red", "green"]) == \
            {"requested": 2, "already_cached": 1, "encoded": 1}


# ---------------------------------------------------------------------------
# Composition with Method 2 and the evaluation schema
# ---------------------------------------------------------------------------


class TestComposition:
    @pytest.fixture
    def method(self, corpus, monkeypatch):
        method, _ = built(corpus)

        def base(self):
            # Stands in for the Qdrant-backed text, table and figure retrievers.
            return {"bm25": ListRetriever("bm25", Modality.TEXT, ["m2#b1t", "m2#a1t"])}

        monkeypatch.setattr(ModalityIndex, "retrievers", base)
        return method

    def test_method3_composes_rather_than_inherits_method2(self):
        assert not issubclass(Method3HybridVisual, Method2ModalityAware)

    def test_page_retriever_is_added_and_always_fires(self, method):
        retriever = method.retriever
        assert list(retriever.retrievers) == ["bm25", "visual_page"]
        assert isinstance(retriever, ModalityAwareRetriever)
        assert isinstance(retriever.router, HeuristicRouter)
        assert retriever.router.always == [Modality.VISUAL_PAGE]
        # "explain" is a clearly textual query: Method 2's router picks text only.
        result = method.retrieve("explain green", top_k=5, use_metadata=False)
        assert result.routing.modalities == [Modality.TEXT, Modality.VISUAL_PAGE]
        assert "visual_page" in result.diagnostics["retrievers_fired"]

    def test_results_use_the_shared_schema_and_score_against_gold(self, method):
        gold = GoldQuery(id="q1", query="explain green", stratum="natural", requires="table",
                         evidence=[{"doc_id": "doc_a", "page": 2, "modality": "table"}])
        outcome = evaluate_query(method, gold, top_k=5, k_values=[1, 5], use_metadata=False)
        assert outcome.metrics["recall@5"] == 1.0
        assert all(r.chunk_id.startswith("m2#") for r in outcome.retrieved)
        hit = next(r for r in outcome.retrieved if r.matched == 0)
        assert (hit.doc_id, hit.page, hit.chunk_type) == ("doc_a", 2, "table")
        assert outcome.routing["modalities"] == ["text", "visual_page"]

    def test_method_reads_method2_and_writes_only_the_visual_index(self, method, corpus):
        assert method.index_dir == corpus.indexes / "method2"
        assert method.collection("text") == "method2_text"
        assert method.visual_dir == corpus.indexes / "visual_pages" / "index"
        assert method.query_cache_dir == corpus.indexes / "visual_pages" / "query_cache"
        assert method.name == "method3"

    def test_a_method2_config_is_rejected(self, corpus):
        with pytest.raises(ValueError, match="method3 config"):
            Method3HybridVisual(load_experiment_config("method2"), index_base=corpus.indexes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def _patch(self, monkeypatch, corpus):
        import mmrag.methods as methods

        def factory(cfg, **kwargs):
            method, _ = make_method(corpus)
            return method

        monkeypatch.setattr(methods, "Method3HybridVisual", factory)

    def test_index_build_refuses_cpu_with_a_clear_message(self, monkeypatch, corpus):
        import mmrag.cli as cli

        self._patch(monkeypatch, corpus)
        result = CliRunner().invoke(cli.app, ["index", "build", "-c", "method3", "--device", "cpu"])
        assert result.exit_code == 1 and "CpuIndexingRefusedError" in result.output

    def test_visual_options_are_rejected_for_other_methods(self):
        import mmrag.cli as cli

        result = CliRunner().invoke(cli.app, ["index", "build", "-c", "method1", "--device", "cpu"])
        assert result.exit_code == 2

    def test_eval_run_refuses_a_partial_page_index(self, monkeypatch, corpus):
        import mmrag.cli as cli

        built(corpus, max_pages=2)
        self._patch(monkeypatch, corpus)
        result = CliRunner().invoke(cli.app, ["eval", "run", "-c", "method3"])
        assert result.exit_code == 1 and "partial" in result.output


# ---------------------------------------------------------------------------
# Real model (GPU machine)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_colqwen2_encodes_a_page_and_a_query():
    """Downloads ColQwen2 (~4.5 GB). Run on the GPU machine: pytest -m slow -k real_colqwen2."""
    pytest.importorskip("colpali_engine")
    cfg = load_experiment_config("method3")
    encoder = ColQwen2Encoder(cfg.embedding.visual_model, cfg.visual,
                              expected_dim=cfg.embedding.visual_dim)
    page = Image.new("RGB", (850, 1100), "white")
    [page_vectors] = encoder.encode_images([page])
    [query_vectors] = encoder.encode_queries(["What does the chart show?"])
    assert page_vectors.shape[1] == query_vectors.shape[1] == cfg.embedding.visual_dim
    assert page_vectors.shape[0] > 100 and np.isfinite(page_vectors).all()
