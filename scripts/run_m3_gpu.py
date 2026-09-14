#!/usr/bin/env python
"""Method 3 GPU runner: build the ColQwen2 page index and query cache from a bundle.

Runs on a CUDA machine (Google Colab) from the self-contained bundle made by
``scripts/pack_m3_colab.py``. It needs no Postgres, Qdrant, API key, repository
checkout or development environment -- only ``requirements-gpu.txt``.

It writes exactly the on-disk formats the main repository reads, because it
uses the repository's own storage and encoder code (``mmrag.stores.multivector``
and ``mmrag.embeddings.visual``, shipped inside the bundle), and the main
repository re-verifies everything when the artefacts are copied back.

Stages (each can be run on its own, and each is safe to re-run)::

    check          CUDA, GPU, library versions, bundle integrity
    smoke          4-8 pages: load the model, encode, write + reload a partial
                   index, encode queries, rank; estimates the full build time
    build          every page, checkpointed per page to --work-dir (resumable),
                   then the index is written in one atomic step
    embed-queries  every evaluation query into the query cache (resumable)
    verify         the finished artefacts against the bundle
    package        m3_visual_pages.zip, laid out to unzip at the repository root
    all            the above in order, skipping stages already complete

Usage::

    python run_m3_gpu.py all --out /content/m3_out \\
        --work-dir /content/drive/MyDrive/m3_work
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import platform
import statistics
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if (HERE / "mmrag").is_dir():  # bundle layout: use the shipped subset of the package
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from mmrag.config import VisualRetrievalConfig  # noqa: E402
from mmrag.embeddings.visual import (  # noqa: E402
    ColQwen2Encoder,
    DeviceUnavailableError,
    VisualEncodingError,
    VisualModelUnavailableError,
    compatible_identity,
)
from mmrag.stores.multivector import (  # noqa: E402
    IndexIntegrityError,
    PageEmbeddingIndex,
    PageIndexWriter,
    PageRecord,
    QueryEmbeddingCache,
    sha256_file,
)

log = logging.getLogger("run_m3_gpu")

BUNDLE_VERSION = 1
# Must equal mmrag.indexing.visual_pages.INDEX_KIND; tests/test_m3_colab_runner.py
# loads this runner's output with the main repository's validator.
INDEX_KIND = "visual_page_index"
# Layout of the outputs, relative to --out, mirroring the main repository.
INDEX_REL = Path("data/indexes/visual_pages/index")
CACHE_REL = Path("data/indexes/visual_pages/query_cache")
REPORTS_REL = Path("data/indexes/visual_pages/build")
SMOKE_REL = Path("smoke/visual_pages/index")
PACKAGE_NAME = "m3_visual_pages.zip"


class RunnerError(RuntimeError):
    """A precondition failed; the message says what to do."""


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Bundle:
    """The inputs packed by scripts/pack_m3_colab.py."""

    def __init__(self, root: Path):
        self.root = Path(root)
        meta_path = self.root / "inputs" / "bundle.json"
        if not meta_path.exists():
            raise RunnerError(
                f"no bundle at {self.root} (missing inputs/bundle.json). Unzip m3_colab_bundle.zip "
                "and pass --bundle to the extracted m3_colab/ directory."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.id = sha256_file(meta_path)
        if self.meta.get("bundle_version") != BUNDLE_VERSION:
            raise RunnerError(
                f"bundle version {self.meta.get('bundle_version')} != {BUNDLE_VERSION}"
            )
        self.pages: list[dict[str, Any]] = [
            json.loads(line)
            for line in (self.root / self.meta["pages"]["file"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.queries: list[str] = list(
            dict.fromkeys(
                q["query"]
                for q in json.loads(
                    (self.root / self.meta["queries"]["file"]).read_text(encoding="utf-8")
                )
                if q["query"].strip()
            )
        )

    @property
    def visual_model(self) -> str:
        return self.meta["config"]["visual_model"]

    @property
    def visual_dim(self) -> int:
        return int(self.meta["config"]["visual_dim"])

    def visual_config(self, **updates: Any) -> VisualRetrievalConfig:
        return VisualRetrievalConfig(**{**self.meta["config"]["visual"], **updates})

    def image_path(self, page: dict[str, Any]) -> Path:
        return self.root / "inputs" / "renders" / page["image"]

    def verify(self) -> dict[str, Any]:
        """Lockfile, documents, page count and every render present."""
        lock = self.root / self.meta["corpus"]["lock_file"]
        if sha256_file(lock) != self.meta["corpus"]["lock_sha256"]:
            raise RunnerError(f"{lock} does not match the checksum recorded in bundle.json")
        pinned = {
            d: e["sha256"] for d, e in yaml.safe_load(lock.read_text("utf-8"))["entries"].items()
        }
        drifted = sorted(
            d for d, s in self.meta["corpus"]["documents"].items() if pinned.get(d) != s
        )
        if drifted:
            raise RunnerError(f"bundled documents do not match the corpus lockfile: {drifted}")
        if len(self.pages) != self.meta["pages"]["count"]:
            raise RunnerError(
                f"pages.jsonl has {len(self.pages)} pages, bundle.json says "
                f"{self.meta['pages']['count']}"
            )
        missing = [p["page_id"] for p in self.pages if not self.image_path(p).exists()]
        if missing:
            raise RunnerError(
                f"{len(missing)} page render(s) missing from the bundle, e.g. {missing[:3]}"
            )
        return {
            "bundle_id": self.id,
            "pages": len(self.pages),
            "queries": len(self.queries),
            "documents": len(self.meta["corpus"]["documents"]),
            "created_at": self.meta["created_at"],
            "source": self.meta.get("source"),
        }


# ---------------------------------------------------------------------------
# Environment and encoder
# ---------------------------------------------------------------------------


def environment() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for package in (
        "torch",
        "transformers",
        "colpali-engine",
        "peft",
        "accelerate",
        "numpy",
        "pillow",
        "pydantic",
    ):
        try:
            info[package] = version(package)
        except PackageNotFoundError:
            info[package] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gib"] = round(props.total_memory / 2**30, 1)
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        info["cuda_available"] = False
    return info


def make_encoder(bundle: Bundle, device: str, dtype: str | None) -> ColQwen2Encoder:
    config = bundle.visual_config(**({"dtype": dtype} if dtype else {}))
    return ColQwen2Encoder(
        bundle.visual_model, config, expected_dim=bundle.visual_dim, device=device
    )


def _encoder_config(encoder: Any, bundle: Bundle) -> dict[str, Any]:
    config = getattr(encoder, "config", None)
    if isinstance(config, VisualRetrievalConfig):
        return config.model_dump(mode="json")
    return bundle.visual_config().model_dump(mode="json")


def _is_out_of_memory(exc: BaseException) -> bool:
    return "OutOfMemory" in type(exc).__name__ or "out of memory" in str(exc).lower()


def _empty_cuda_cache() -> None:
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _peak_memory_gib() -> float | None:
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return None


# ---------------------------------------------------------------------------
# Resumable per-page checkpoints
# ---------------------------------------------------------------------------


def _atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class WorkDir:
    """Per-page vectors written as they are produced, so a lost session resumes.

    Bound to one model identity, storage precision and bundle: a checkpoint from
    another model or corpus is refused rather than mixed in. A page counts as
    done only if its render still has the checksum it was encoded from.
    """

    def __init__(
        self,
        directory: Path,
        *,
        identity: dict[str, Any],
        dim: int,
        storage_dtype: str,
        bundle_id: str,
    ):
        self.directory = Path(directory)
        self.pages_dir = self.directory / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dtype = np.dtype(storage_dtype)
        manifest = self.directory / "work.json"
        expected = {
            "identity": identity,
            "dim": dim,
            "storage_dtype": self.storage_dtype.name,
            "bundle_id": bundle_id,
        }
        if manifest.exists():
            found = json.loads(manifest.read_text(encoding="utf-8"))
            if (
                not compatible_identity(found["identity"], identity)
                or found["dim"] != dim
                or found["storage_dtype"] != expected["storage_dtype"]
                or found["bundle_id"] != bundle_id
            ):
                raise RunnerError(
                    f"checkpoints in {self.directory} were made with {found}, not {expected}. "
                    "Use a new --work-dir, or delete this one to start over."
                )
        else:
            _atomic_bytes(manifest, json.dumps(expected, indent=2, sort_keys=True).encode())

    def _paths(self, page_id: str) -> tuple[Path, Path]:
        key = hashlib.sha1(page_id.encode("utf-8")).hexdigest()
        return self.pages_dir / f"{key}.npy", self.pages_dir / f"{key}.json"

    def done(self, page_id: str, image_sha256: str) -> bool:
        vectors, record = self._paths(page_id)
        if not (vectors.exists() and record.exists()):
            return False
        return json.loads(record.read_text(encoding="utf-8")).get("image_sha256") == image_sha256

    def save(self, record: PageRecord, vectors: np.ndarray) -> None:
        vectors_path, record_path = self._paths(record.page_id)
        tmp = vectors_path.with_name(vectors_path.stem + ".partial.npy")
        np.save(tmp, np.asarray(vectors, dtype=self.storage_dtype))
        os.replace(tmp, vectors_path)
        # The record is written last: it is what marks the page as done.
        _atomic_bytes(record_path, json.dumps(asdict(record), sort_keys=True).encode())

    def load(self, page_id: str) -> tuple[PageRecord, np.ndarray]:
        vectors_path, record_path = self._paths(page_id)
        record = PageRecord(**json.loads(record_path.read_text(encoding="utf-8")))
        vectors = np.load(vectors_path)
        if vectors.shape[0] != record.n_tokens:
            raise IndexIntegrityError(f"checkpoint for {page_id} is inconsistent; delete it")
        return record, vectors


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _record(page: dict[str, Any], image_sha256: str, n_tokens: int) -> PageRecord:
    return PageRecord(
        page_id=page["page_id"],
        doc_id=page["doc_id"],
        page_number=int(page["page_number"]),
        image=page["image"],
        image_sha256=image_sha256,
        image_width=int(page.get("image_width") or 0),
        image_height=int(page.get("image_height") or 0),
        image_dpi=page.get("image_dpi"),
        n_tokens=n_tokens,
    )


def encode_pages(
    encoder: Any,
    bundle: Bundle,
    pages: list[dict[str, Any]],
    *,
    batch_size: int,
    sink: Any,
    skip: Any = None,
) -> dict[str, Any]:
    """Encode pages in batches, halving the batch on out-of-memory.

    ``sink(record, vectors)`` receives each page; ``skip(page_id, sha)`` says
    whether a page is already done.
    """
    from PIL import Image

    todo: list[tuple[dict[str, Any], str]] = []
    skipped = 0
    for page in pages:
        data = bundle.image_path(page).read_bytes()
        digest = _sha256_bytes(data)
        if page.get("image_sha256") and page["image_sha256"] != digest:
            raise RunnerError(
                f"{page['page_id']}: render differs from the one packed; re-upload it"
            )
        if skip is not None and skip(page["page_id"], digest):
            skipped += 1
            continue
        todo.append((page, digest))
    if skipped:
        log.info("resuming: %d of %d pages already checkpointed", skipped, len(pages))

    started = time.perf_counter()
    position, encoded, tokens = 0, 0, []
    while position < len(todo):
        batch = todo[position : position + batch_size]
        images = []
        for page, _ in batch:
            with Image.open(bundle.image_path(page)) as handle:
                images.append(handle.convert("RGB"))
        try:
            vectors = encoder.encode_images(images)
        except Exception as exc:
            if _is_out_of_memory(exc) and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                log.warning("out of memory; retrying with batch size %d", batch_size)
                _empty_cuda_cache()
                continue
            raise
        if len(vectors) != len(batch):
            raise IndexIntegrityError(
                f"encoder returned {len(vectors)} results for {len(batch)} pages"
            )
        for (page, digest), page_vectors in zip(batch, vectors, strict=True):
            sink(_record(page, digest, int(page_vectors.shape[0])), page_vectors)
            tokens.append(int(page_vectors.shape[0]))
        position += len(batch)
        encoded += len(batch)
        elapsed = time.perf_counter() - started
        rate = encoded / elapsed if elapsed else 0.0
        eta = (len(todo) - position) / rate if rate else float("nan")
        log.info(
            "encoded %d/%d pages (%.2f pages/s, ETA %.0f min, batch %d)",
            position,
            len(todo),
            rate,
            eta / 60,
            batch_size,
        )
    elapsed = time.perf_counter() - started
    return {
        "pages_requested": len(pages),
        "pages_skipped": skipped,
        "pages_encoded": encoded,
        "encode_seconds": round(elapsed, 1),
        "seconds_per_page": round(elapsed / encoded, 3) if encoded else None,
        "final_batch_size": batch_size,
        "tokens": tokens,
    }


def build_manifest(
    bundle: Bundle,
    encoder: Any,
    records: list[PageRecord],
    *,
    complete: bool,
    subset: dict[str, Any] | None,
    build: dict[str, Any],
) -> dict[str, Any]:
    """The same manifest mmrag.indexing.visual_pages.VisualPageIndexer.build writes."""
    description = encoder.describe()
    chunk_pages = {(doc, int(page)) for doc, page in bundle.meta["chunk_set"]["pages"]}
    return {
        "kind": INDEX_KIND,
        "model": {"identity": encoder.identity(), "description": description},
        "preprocessing": {
            "image_source": "ingestion page renders (data/processed/<doc>/pages)",
            "image_dpi": sorted({r.image_dpi for r in records if r.image_dpi}),
            "color_mode": "RGB",
            "resize": "model processor defaults",
            "processor": description.get("processor"),
        },
        "corpus": {
            "lock_sha256": bundle.meta["corpus"]["lock_sha256"],
            "documents": bundle.meta["corpus"]["documents"],
            "complete": complete,
            "subset": subset,
        },
        "chunk_set": {
            "path": bundle.meta["chunk_set"]["path"],
            "chunks_sha256": bundle.meta["chunk_set"]["chunks_sha256"],
            "indexed_pages_without_chunks": sum(
                1 for r in records if (r.doc_id, r.page_number) not in chunk_pages
            ),
        },
        "config": {
            "visual": _encoder_config(encoder, bundle),
            "visual_model": bundle.visual_model,
            "visual_dim": bundle.visual_dim,
        },
        "build": build,
    }


def _token_stats(tokens: list[int]) -> dict[str, float]:
    return (
        {"min": min(tokens), "median": statistics.median(tokens), "max": max(tokens)}
        if tokens
        else {}
    )


def _write_report(out: Path, name: str, payload: dict[str, Any]) -> Path:
    path = out / REPORTS_REL / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True, default=str).encode())
    return path


def _read_report(out: Path, name: str) -> dict[str, Any] | None:
    path = out / REPORTS_REL / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_check(bundle: Bundle, out: Path, *, require_cuda: bool) -> dict[str, Any]:
    env = environment()
    log.info("environment: %s", env)
    if require_cuda and not env.get("cuda_available"):
        raise RunnerError("CUDA is not available. In Colab: Runtime > Change runtime type > GPU.")
    summary = bundle.verify()
    log.info("bundle ok: %s", summary)
    report = {"checked_at": _now(), "environment": env, "bundle": summary}
    _write_report(out, "environment.json", report)
    return report


def stage_smoke(
    bundle: Bundle, out: Path, encoder: Any, *, pages: int, batch_size: int
) -> dict[str, Any]:
    if not 1 <= pages <= 16:
        raise RunnerError("--smoke-pages must be between 1 and 16")
    selected = bundle.pages[:pages]
    started = time.perf_counter()
    identity = encoder.identity()  # loads the model
    load_s = time.perf_counter() - started
    log.info("model loaded in %.1fs: %s", load_s, identity)

    writer = PageIndexWriter(
        out / SMOKE_REL, dim=bundle.visual_dim, storage_dtype=bundle.visual_config().storage_dtype
    )
    raw: list[np.ndarray] = []

    def sink(record: PageRecord, vectors: np.ndarray) -> None:
        raw.append(vectors)
        writer.add(record, vectors)

    stats = encode_pages(encoder, bundle, selected, batch_size=batch_size, sink=sink)

    # A page's vectors must not depend on what it was batched with.
    from PIL import Image

    with Image.open(bundle.image_path(selected[0])) as handle:
        [alone] = encoder.encode_images([handle.convert("RGB")])
    if alone.shape != raw[0].shape:
        raise RunnerError(f"page 1 encoded alone has shape {alone.shape}, batched {raw[0].shape}")
    cosine = float(
        np.min(
            np.sum(alone * raw[0], axis=1)
            / (np.linalg.norm(alone, axis=1) * np.linalg.norm(raw[0], axis=1))
        )
    )
    if cosine < 0.98:
        raise RunnerError(
            f"batched and unbatched page vectors disagree (min token cosine {cosine:.4f})"
        )

    index = writer.finalize(
        build_manifest(
            bundle,
            encoder,
            writer.records,
            complete=False,
            subset={"max_pages": pages},
            build={"runner": "run_m3_gpu.py", "stage": "smoke", "bundle_id": bundle.id},
        )
    )
    reloaded = PageEmbeddingIndex.load(index.directory)  # checksums verified

    probes = bundle.queries[:3]
    query_vectors = encoder.encode_queries(probes)
    ranked = []
    for text, vectors in zip(probes, query_vectors, strict=True):
        top = reloaded.rank(vectors)[:3]
        ranked.append(
            {
                "query": text,
                "query_tokens": int(vectors.shape[0]),
                "top_pages": [[reloaded.pages[i].page_id, round(s, 3)] for i, s in top],
            }
        )
    if not all(np.isfinite(v).all() for v in query_vectors):
        raise RunnerError("query vectors are not finite")

    per_page = stats["seconds_per_page"] or 0.0
    report = {
        "passed": True,
        "finished_at": _now(),
        "identity": identity,
        "description": encoder.describe(),
        "model_load_seconds": round(load_s, 1),
        "pages": pages,
        "batch_size": batch_size,
        **{k: v for k, v in stats.items() if k != "tokens"},
        "tokens_per_page": _token_stats(stats["tokens"]),
        "batched_vs_unbatched_min_cosine": round(cosine, 5),
        "peak_gpu_memory_gib": _peak_memory_gib(),
        "probe_queries": ranked,
        "estimated_full_build_minutes": round(per_page * len(bundle.pages) / 60, 1),
        "index_dir": str(index.directory),
        "bundle_id": bundle.id,
    }
    _write_report(out, "smoke_report.json", report)
    log.info(
        "smoke test passed: %.2fs/page, estimated full build %.0f min, report %s",
        per_page,
        report["estimated_full_build_minutes"],
        out / REPORTS_REL,
    )
    return report


def stage_build(
    bundle: Bundle, out: Path, work_dir: Path, encoder: Any, *, batch_size: int, skip_smoke: bool
) -> dict[str, Any]:
    smoke = _read_report(out, "smoke_report.json")
    identity = encoder.identity()
    if not skip_smoke and not (
        smoke
        and smoke.get("passed")
        and compatible_identity(smoke["identity"], identity)
        and smoke.get("bundle_id") == bundle.id
    ):
        raise RunnerError("run the smoke stage first (or pass --skip-smoke)")

    config = bundle.visual_config()
    work = WorkDir(
        work_dir,
        identity=identity,
        dim=bundle.visual_dim,
        storage_dtype=config.storage_dtype,
        bundle_id=bundle.id,
    )
    started = time.perf_counter()
    stats = encode_pages(
        encoder, bundle, bundle.pages, batch_size=batch_size, sink=work.save, skip=work.done
    )

    writer = PageIndexWriter(
        out / INDEX_REL, dim=bundle.visual_dim, storage_dtype=config.storage_dtype
    )
    for page in bundle.pages:
        record, vectors = work.load(page["page_id"])
        writer.add(record, vectors)
    index = writer.finalize(
        build_manifest(
            bundle,
            encoder,
            writer.records,
            complete=True,
            subset=None,
            build={
                "runner": "run_m3_gpu.py",
                "stage": "build",
                "bundle_id": bundle.id,
                "finished_at": _now(),
                "gpu": environment().get("gpu"),
                "pages_resumed_from_checkpoints": stats["pages_skipped"],
            },
        )
    )

    tokens = [p.n_tokens for p in index.pages]
    index_bytes = sum(f.stat().st_size for f in index.directory.iterdir())
    report = {
        "finished_at": _now(),
        "identity": identity,
        "description": encoder.describe(),
        "n_documents": len(bundle.meta["corpus"]["documents"]),
        "n_pages": len(index.pages),
        "n_tokens": int(sum(tokens)),
        "tokens_per_page": _token_stats(tokens),
        "indexed_pages_without_chunks": index.manifest["chunk_set"]["indexed_pages_without_chunks"],
        "index_bytes": index_bytes,
        "index_mib": round(index_bytes / 2**20, 1),
        "embeddings_sha256": index.manifest["files"]["embeddings.npy"],
        "elapsed_seconds_this_session": round(time.perf_counter() - started, 1),
        **{k: v for k, v in stats.items() if k != "tokens"},
        "peak_gpu_memory_gib": _peak_memory_gib(),
        "environment": environment(),
        "bundle_id": bundle.id,
    }
    _write_report(out, "build_report.json", report)
    log.info(
        "page index written: %d pages, %d token vectors, %.0f MiB",
        report["n_pages"],
        report["n_tokens"],
        report["index_mib"],
    )
    return report


def stage_embed_queries(
    bundle: Bundle, out: Path, encoder: Any, *, batch_size: int
) -> dict[str, Any]:
    index_dir = out / INDEX_REL
    if not PageEmbeddingIndex.exists(index_dir):
        raise RunnerError("no page index yet; run the build stage first")
    index = PageEmbeddingIndex.load(index_dir)
    identity = encoder.identity()
    if not compatible_identity(identity, index.identity):
        raise IndexIntegrityError(
            f"encoder {identity} did not build {index_dir} ({index.identity})"
        )

    cache = QueryEmbeddingCache(out / CACHE_REL)
    cache.check_identity(identity)
    todo = [t for t in bundle.queries if cache.get(t) is None]
    description = encoder.describe()
    started = time.perf_counter()
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        cache.put_many(
            list(zip(batch, encoder.encode_queries(batch), strict=True)),
            identity=identity,
            encoder=description,
        )
        log.info("encoded %d/%d queries", min(start + batch_size, len(todo)), len(todo))

    lengths = []
    for text in bundle.queries:
        vectors = cache.get(text)
        if (
            vectors is None
            or vectors.shape[1] != bundle.visual_dim
            or not np.isfinite(vectors).all()
        ):
            raise IndexIntegrityError(f"query cache entry for {text[:60]!r} is missing or invalid")
        lengths.append(int(vectors.shape[0]))
    report = {
        "finished_at": _now(),
        "identity": identity,
        "queries": len(bundle.queries),
        "already_cached": len(bundle.queries) - len(todo),
        "encoded": len(todo),
        "encode_seconds": round(time.perf_counter() - started, 2),
        "tokens_per_query": _token_stats(lengths),
        "bundle_id": bundle.id,
    }
    _write_report(out, "queries_report.json", report)
    return report


def stage_verify(bundle: Bundle, out: Path) -> dict[str, Any]:
    """Everything the main repository will check, checked here first."""
    index = PageEmbeddingIndex.load(out / INDEX_REL)
    manifest = index.manifest
    problems = []
    if manifest.get("kind") != INDEX_KIND:
        problems.append("manifest kind")
    if not index.is_complete or len(index.pages) != len(bundle.pages):
        problems.append(f"index holds {len(index.pages)} of {len(bundle.pages)} pages")
    if (index.identity.get("model"), index.identity.get("dim")) != (
        bundle.visual_model,
        bundle.visual_dim,
    ):
        problems.append(
            f"identity {index.identity} vs config {bundle.visual_model}/{bundle.visual_dim}"
        )
    if manifest["corpus"]["lock_sha256"] != bundle.meta["corpus"]["lock_sha256"]:
        problems.append("corpus lock checksum")
    missing = sorted(
        f"{d}#p{p}"
        for d, p in bundle.meta["chunk_set"]["pages"]
        if index.page_index(d, int(p)) is None
    )
    if missing:
        problems.append(f"{len(missing)} chunk-bearing pages missing, e.g. {missing[:3]}")
    cache = QueryEmbeddingCache(out / CACHE_REL)
    try:
        cache.check_identity(index.identity)
    except IndexIntegrityError as exc:
        problems.append(str(exc))
    uncached = [q for q in bundle.queries if cache.get(q) is None]
    if uncached:
        problems.append(f"{len(uncached)} queries not in the cache")
    if problems:
        raise RunnerError("verification failed: " + "; ".join(problems))
    report = {
        "verified_at": _now(),
        "passed": True,
        "pages": len(index.pages),
        "queries_cached": len(bundle.queries),
        "identity": index.identity,
        "embeddings_sha256": manifest["files"]["embeddings.npy"],
        "bundle_id": bundle.id,
    }
    _write_report(out, "verify_report.json", report)
    log.info("verified: %d pages, %d cached queries", len(index.pages), len(bundle.queries))
    return report


def stage_package(out: Path) -> Path:
    """Zip the artefacts with repository-relative paths, plus their checksums."""
    root = out / "data" / "indexes" / "visual_pages"
    if not (root / "build" / "verify_report.json").exists():
        raise RunnerError("run the verify stage before packaging")
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".partial"))
    checksums = {p.relative_to(out).as_posix(): sha256_file(p) for p in files}
    _write_report(out, "artifacts.json", {"packaged_at": _now(), "files": checksums})
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".partial"))
    target = out / PACKAGE_NAME
    tmp = target.with_name(target.name + ".partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(out).as_posix())
    os.replace(tmp, target)
    log.info(
        "packaged %d files into %s (%.0f MiB)", len(files), target, target.stat().st_size / 2**20
    )
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(out / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None, *, encoder_factory: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "stage", choices=["check", "smoke", "build", "embed-queries", "verify", "package", "all"]
    )
    parser.add_argument("--bundle", type=Path, default=HERE, help="extracted m3_colab/ directory")
    parser.add_argument("--out", type=Path, default=None, help="outputs (default <bundle>/outputs)")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="per-page checkpoints; put this on Google Drive to survive disconnects",
    )
    parser.add_argument(
        "--device", default="cuda", help="cuda | cuda:N (cpu only with --allow-cpu)"
    )
    parser.add_argument(
        "--allow-cpu", action="store_true", help="tests only; never for the full build"
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default=None,
        help="override visual.dtype, e.g. float32 if half precision yields NaNs",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--smoke-pages", type=int, default=8)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args(argv)

    bundle_root = args.bundle.resolve()
    out = (args.out or bundle_root / "outputs").resolve()
    work_dir = (args.work_dir or out / "work").resolve()
    _configure_logging(out)

    if args.device.startswith("cpu") and not args.allow_cpu:
        parser.error("--device cpu needs --allow-cpu; ColQwen2 on CPU is for tiny tests only")

    try:
        bundle = Bundle(bundle_root)
        config = bundle.visual_config()
        batch_size = args.batch_size or config.batch_size
        query_batch_size = args.query_batch_size or config.query_batch_size
        stages = (
            ["check", "smoke", "build", "embed-queries", "verify", "package"]
            if args.stage == "all"
            else [args.stage]
        )

        encoder = None

        def get_encoder() -> Any:
            nonlocal encoder
            if encoder is None:
                encoder = (
                    encoder_factory(args.device)
                    if encoder_factory
                    else make_encoder(bundle, args.device, args.dtype)
                )
            return encoder

        for stage in stages:
            log.info("== %s", stage)
            if stage == "check":
                stage_check(bundle, out, require_cuda=not args.allow_cpu)
            elif stage == "smoke":
                smoke = _read_report(out, "smoke_report.json")
                if (
                    args.stage == "all"
                    and smoke
                    and smoke.get("passed")
                    and smoke.get("bundle_id") == bundle.id
                ):
                    log.info("smoke test already passed; skipping")
                    continue
                stage_smoke(
                    bundle, out, get_encoder(), pages=args.smoke_pages, batch_size=batch_size
                )
            elif stage == "build":
                built = _read_report(out, "build_report.json")
                if (
                    args.stage == "all"
                    and built
                    and built.get("bundle_id") == bundle.id
                    and PageEmbeddingIndex.exists(out / INDEX_REL)
                ):
                    log.info("page index already built; skipping")
                    continue
                if args.device.startswith("cpu") and len(bundle.pages) > 16:
                    raise RunnerError("refusing to build more than 16 pages on CPU")
                stage_build(
                    bundle,
                    out,
                    work_dir,
                    get_encoder(),
                    batch_size=batch_size,
                    skip_smoke=args.skip_smoke,
                )
            elif stage == "embed-queries":
                stage_embed_queries(bundle, out, get_encoder(), batch_size=query_batch_size)
            elif stage == "verify":
                stage_verify(bundle, out)
            elif stage == "package":
                print(stage_package(out))
    except (
        RunnerError,
        IndexIntegrityError,
        FileNotFoundError,
        DeviceUnavailableError,
        VisualEncodingError,
        VisualModelUnavailableError,
    ) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
