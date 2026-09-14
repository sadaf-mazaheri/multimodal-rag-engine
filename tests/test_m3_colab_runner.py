"""The Colab bundle and GPU runner, end to end on CPU with a fake encoder.

The runner writes the page index and query cache on a machine without the
repository, so the test that matters is the round trip: pack a corpus, run every
stage, then open the artefacts with the main repository's own Method 3 loader,
validator and retriever.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from mmrag.config import PROJECT_ROOT, load_experiment_config
from mmrag.methods.method3_visual import Method3HybridVisual
from tests.test_method3 import DIM, FakeEncoder, corpus  # noqa: F401 - corpus is a fixture

SCRIPTS = PROJECT_ROOT / "scripts"
QUERIES = ["red", "blue green"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pack_module(monkeypatch):
    module = _load("pack_m3_colab")
    monkeypatch.setattr(
        module.GoldSet,
        "load",
        classmethod(
            lambda cls, path: types.SimpleNamespace(
                queries=[
                    types.SimpleNamespace(id="q001", query=QUERIES[0]),
                    types.SimpleNamespace(id="q002", query=QUERIES[1]),
                ]
            )
        ),
    )
    monkeypatch.setattr(
        module.GenerationGold,
        "load",
        classmethod(
            lambda cls, path: types.SimpleNamespace(
                unanswerable=[types.SimpleNamespace(id="u001", query=QUERIES[0])]
            )
        ),
    )
    return module


@pytest.fixture
def runner():
    return _load("run_m3_gpu")


def config():
    return load_experiment_config(
        "method3",
        overrides={"embedding": {"visual_dim": DIM}, "retrieval": {"rerank_enabled": False}},
    )


@pytest.fixture
def bundle(pack_module, corpus, tmp_path):  # noqa: F811
    archive = pack_module.pack(
        tmp_path / "dist",
        processed_dir=corpus.processed,
        lock_path=corpus.lock,
        chunks_path=corpus.indexes / "method2" / "chunks.jsonl",
        config=config(),
    )
    extracted = tmp_path / "colab"
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    return extracted / "m3_colab"


def run(runner, bundle, out, stage, *args, encoder=None):
    encoder = encoder or FakeEncoder()
    argv = [
        stage,
        "--bundle",
        str(bundle),
        "--out",
        str(out),
        "--device",
        "cpu",
        "--allow-cpu",
        "--batch-size",
        "2",
        *args,
    ]
    return runner.main(argv, encoder_factory=lambda device: encoder)


class TestBundle:
    def test_contents(self, bundle):
        meta = json.loads((bundle / "inputs" / "bundle.json").read_text())
        assert meta["pages"]["count"] == 4 and meta["queries"]["count"] == 3
        assert meta["config"]["visual_dim"] == DIM
        assert sorted(map(tuple, meta["chunk_set"]["pages"])) == [
            ("doc_a", 1),
            ("doc_a", 2),
            ("doc_b", 1),
        ]
        assert (bundle / "inputs" / "renders" / "doc_a" / "pages" / "p0001.png").exists()
        for name in (
            "run_m3_gpu.py",
            "requirements-gpu.txt",
            "mmrag/stores/multivector.py",
            "mmrag/embeddings/visual.py",
            "mmrag/config.py",
            "mmrag/logging_utils.py",
        ):
            assert (bundle / name).exists(), name

    def test_shipped_package_imports_without_the_heavy_stack(self, bundle):
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import mmrag.stores.multivector, mmrag.embeddings.visual, mmrag.config;"
            "import mmrag; assert mmrag.__file__.startswith(sys.argv[1]), mmrag.__file__;"
            "heavy = [m for m in ('qdrant_client', 'psycopg', 'sentence_transformers', 'bm25s')"
            " if m in sys.modules]; assert not heavy, heavy; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe, str(bundle)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"


class TestRunner:
    def test_all_stages_produce_artifacts_the_main_repository_accepts(
        self,
        runner,
        bundle,
        corpus,  # noqa: F811
        tmp_path,
    ):
        out = tmp_path / "out"
        assert run(runner, bundle, out, "all", "--smoke-pages", "2") == 0

        reports = out / "data" / "indexes" / "visual_pages" / "build"
        smoke = json.loads((reports / "smoke_report.json").read_text())
        assert smoke["passed"] and smoke["pages"] == 2
        build = json.loads((reports / "build_report.json").read_text())
        assert build["n_pages"] == 4 and build["indexed_pages_without_chunks"] == 1
        assert json.loads((reports / "verify_report.json").read_text())["passed"]
        assert (out / "m3_visual_pages.zip").exists()

        # Unzip at a "repository root" next to the method2 chunk set, then open it
        # with the main repository's loader, validator and retriever.
        repo = tmp_path / "repo"
        with zipfile.ZipFile(out / "m3_visual_pages.zip") as zf:
            zf.extractall(repo)
        (repo / "data" / "indexes" / "method2").mkdir(parents=True)
        shutil.copy2(
            corpus.indexes / "method2" / "chunks.jsonl",
            repo / "data" / "indexes" / "method2" / "chunks.jsonl",
        )

        def refuse(device):
            raise AssertionError("the main repository must not need the model")

        method = Method3HybridVisual(
            config(),
            index_base=repo / "data" / "indexes",
            processed_dir=corpus.processed,
            lock_path=corpus.lock,
            encoder_factory=refuse,
        )
        info = method.describe_visual_index()
        assert info["complete"] and info["cached_queries"] == 2
        assert method.visual_index.manifest["kind"] == "visual_page_index"
        retriever = method.visual.retriever(method.chunks)
        out_red = retriever.retrieve("red", k=2)
        assert out_red.diagnostics["query_embedding"] == "cache"
        assert out_red.chunk_ids == ["m2#a1t", "m2#a1f"]

    def test_an_interrupted_build_resumes_from_checkpoints(self, runner, bundle, tmp_path):
        out = tmp_path / "out"
        assert run(runner, bundle, out, "smoke", "--smoke-pages", "1") == 0

        class Crashes(FakeEncoder):
            def encode_images(self, images):
                if self.image_calls >= 1:
                    raise RuntimeError("session lost")
                return super().encode_images(images)

        with pytest.raises(RuntimeError, match="session lost"):
            run(runner, bundle, out, "build", encoder=Crashes())
        survivor = FakeEncoder()
        assert run(runner, bundle, out, "build", encoder=survivor) == 0
        report = json.loads((out / "data/indexes/visual_pages/build/build_report.json").read_text())
        assert report["pages_skipped"] == 2 and report["pages_encoded"] == 2
        assert report["n_pages"] == 4

    def test_checkpoints_from_another_model_are_refused(self, runner, bundle, tmp_path):
        out = tmp_path / "out"
        assert run(runner, bundle, out, "build", "--skip-smoke") == 0
        (out / "data/indexes/visual_pages/build/build_report.json").unlink()
        assert (
            run(runner, bundle, out, "build", "--skip-smoke", encoder=FakeEncoder(commit="other"))
            == 1
        )

    def test_build_needs_a_passing_smoke_test(self, runner, bundle, tmp_path):
        assert run(runner, bundle, tmp_path / "out", "build") == 1

    def test_cpu_needs_explicit_permission(self, runner, bundle, tmp_path):
        with pytest.raises(SystemExit):
            runner.main(
                ["check", "--bundle", str(bundle), "--out", str(tmp_path / "o"), "--device", "cpu"]
            )

    def test_a_corrupted_render_is_detected(self, runner, bundle, tmp_path):
        render = bundle / "inputs" / "renders" / "doc_b" / "pages" / "p0001.png"
        render.write_bytes(render.read_bytes() + b"x")
        assert run(runner, bundle, tmp_path / "out", "smoke", "--smoke-pages", "4") == 1

    def test_package_requires_verification(self, runner, bundle, tmp_path):
        assert run(runner, bundle, tmp_path / "out", "package") == 1


def test_repository_scripts_reference_existing_modules():
    module = _load("pack_m3_colab")
    package = Path(importlib.util.find_spec("mmrag").origin).parent
    for relative in module.MODULES:
        assert (package / relative).exists(), relative
