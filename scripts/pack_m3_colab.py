#!/usr/bin/env python
"""Pack everything the Method 3 GPU runner needs into one uploadable bundle.

Run on the main (ingestion/evaluation) machine, in the project environment::

    python scripts/pack_m3_colab.py            # -> dist/m3_colab_bundle.zip

The bundle is self-contained: the runner, its requirements, the small subset of
``mmrag`` it imports, the corpus lockfile, every page render, a slim page list
taken from the ingestion sidecars, the pages that carry chunks in the method2
chunk set, and every evaluation query. No full sidecars, PDFs, Qdrant or
Postgres data are included.

Layout inside the zip::

    m3_colab/
      run_m3_gpu.py  requirements-gpu.txt  COLAB.md
      mmrag/                      config, logging, visual encoder, multi-vector store
      inputs/bundle.json          what was packed, from which commit, with checksums
      inputs/corpus.lock.yaml     the pinned corpus
      inputs/pages.jsonl          one row per page render
      inputs/queries.json         retrieval gold + unanswerable generation queries
      inputs/renders/<doc>/pages/<page>.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from mmrag.config import (
    CONFIG_DIR,
    INDEX_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    ExperimentConfig,
    load_experiment_config,
)
from mmrag.evaluation.generation_gold import GenerationGold
from mmrag.evaluation.gold import GoldSet
from mmrag.indexing.visual_pages import (
    check_documents_against_lock,
    corpus_lock,
    resolve_page_image,
)
from mmrag.ingestion.pipeline import read_sidecar
from mmrag.schemas import Chunk
from mmrag.stores.multivector import sha256_file

SCRIPTS = Path(__file__).resolve().parent
BUNDLE_VERSION = 1

# The only package modules the runner imports. Package __init__ files are
# replaced by stubs, because the real ones import Qdrant, Postgres and
# sentence-transformers, none of which the runner needs.
MODULES = [
    "config.py",
    "logging_utils.py",
    "embeddings/visual.py",
    "stores/multivector.py",
]
STUBS = {
    "__init__.py": (
        '"""Subset of mmrag shipped with the Method 3 GPU runner."""\n\n'
        '__version__ = "{version}"\n'
    ),
    "embeddings/__init__.py": '"""Bundle stub: only visual.py is shipped."""\n',
    "stores/__init__.py": '"""Bundle stub: only multivector.py is shipped."""\n',
}


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack(
    out_dir: Path,
    *,
    config_name: str = "method3",
    processed_dir: Path = PROCESSED_DIR,
    lock_path: Path = CONFIG_DIR / "corpus.lock.yaml",
    chunks_path: Path = INDEX_DIR / "method2" / "chunks.jsonl",
    gold_path: Path = PROJECT_ROOT / "data/eval/gold/v1.yaml",
    generation_gold_path: Path = PROJECT_ROOT / "data/eval/gold/generation_v1.yaml",
    make_zip: bool = True,
    config: ExperimentConfig | None = None,
) -> Path:
    config = config or load_experiment_config(config_name)
    bundle = out_dir / "m3_colab"
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "inputs" / "renders").mkdir(parents=True)

    # --- code -------------------------------------------------------------
    import mmrag

    package_root = Path(mmrag.__file__).resolve().parent
    code: dict[str, str] = {}
    for relative in MODULES:
        target = bundle / "mmrag" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(package_root / relative, target)
        code[f"mmrag/{relative}"] = sha256_file(target)
    for relative, text in STUBS.items():
        (bundle / "mmrag" / relative).write_text(
            text.format(version=mmrag.__version__), encoding="utf-8"
        )
    for name in ("run_m3_gpu.py", "requirements-gpu.txt"):
        shutil.copy2(SCRIPTS / name, bundle / name)
        code[name] = sha256_file(bundle / name)
    guide = PROJECT_ROOT / "docs" / "m3_colab.md"
    if guide.exists():
        shutil.copy2(guide, bundle / "COLAB.md")

    # --- corpus: lockfile, documents, page renders -------------------------
    shutil.copy2(lock_path, bundle / "inputs" / "corpus.lock.yaml")
    lock_sha, pinned = corpus_lock(lock_path)

    documents: dict[str, str] = {}
    pages: list[dict] = []
    problems: list[str] = []
    for sidecar in sorted(processed_dir.glob("*/parsed.json")):
        parsed = read_sidecar(sidecar)
        documents[parsed.document.doc_id] = parsed.document.sha256
        for page in sorted(parsed.pages, key=lambda p: p.page_number):
            source = resolve_page_image(page, processed_dir)
            if source is None:
                problems.append(page.page_id)
                continue
            relative = f"{page.doc_id}/pages/{source.name}"
            target = bundle / "inputs" / "renders" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            pages.append(
                {
                    "page_id": page.page_id,
                    "doc_id": page.doc_id,
                    "page_number": page.page_number,
                    "image": relative,
                    "image_sha256": _sha256(target.read_bytes()),
                    "image_width": page.image_width,
                    "image_height": page.image_height,
                    "image_dpi": page.image_dpi,
                }
            )
    if not documents:
        raise SystemExit(f"no parsed documents under {processed_dir}; run 'mmrag ingest run'")
    if problems:
        raise SystemExit(f"{len(problems)} page(s) have no render, e.g. {problems[:3]}")
    check_documents_against_lock(documents, pinned)
    (bundle / "inputs" / "pages.jsonl").write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in pages), encoding="utf-8"
    )

    # --- the chunk set pages expand into: only which pages carry chunks ----
    if not chunks_path.exists():
        raise SystemExit(f"no chunk set at {chunks_path}; run 'mmrag index build --config method2'")
    chunk_pages: set[tuple[str, int]] = set()
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunk = Chunk.model_validate_json(line)
                chunk_pages.add((chunk.doc_id, chunk.page_number))

    # --- evaluation queries: the same set as 'mmrag index embed-queries' ---
    queries = [
        {"id": q.id, "source": "retrieval_gold", "query": q.query}
        for q in GoldSet.load(gold_path).queries
    ]
    queries += [
        {"id": u.id, "source": "generation_gold_unanswerable", "query": u.query}
        for u in GenerationGold.load(generation_gold_path).unanswerable
    ]
    (bundle / "inputs" / "queries.json").write_text(json.dumps(queries, indent=2), encoding="utf-8")

    meta = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": sys.version.split()[0],
        },
        "config": {
            "name": config_name,
            "visual": config.visual.model_dump(mode="json"),
            "visual_model": config.embedding.visual_model,
            "visual_dim": config.embedding.visual_dim,
        },
        "corpus": {
            "lock_file": "inputs/corpus.lock.yaml",
            "lock_sha256": lock_sha,
            "documents": documents,
        },
        "chunk_set": {
            "path": "data/indexes/method2/chunks.jsonl",
            "chunks_sha256": sha256_file(chunks_path),
            "pages": sorted([doc, page] for doc, page in chunk_pages),
        },
        "pages": {
            "file": "inputs/pages.jsonl",
            "count": len(pages),
            "image_dpi": sorted({p["image_dpi"] for p in pages if p["image_dpi"]}),
        },
        "queries": {
            "file": "inputs/queries.json",
            "count": len(queries),
            "unique": len({q["query"] for q in queries}),
            "sources": {
                "retrieval_gold": gold_path.relative_to(PROJECT_ROOT).as_posix(),
                "generation_gold": generation_gold_path.relative_to(PROJECT_ROOT).as_posix(),
            },
        },
        "code": code,
    }
    (bundle / "inputs" / "bundle.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"bundle: {len(pages)} pages from {len(documents)} documents, {len(queries)} queries, "
        f"{len(chunk_pages)} chunk-bearing pages -> {bundle}"
    )

    if not make_zip:
        return bundle
    archive_path = out_dir / "m3_colab_bundle.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zf:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                # PNGs are already compressed; storing them is faster and no larger.
                kind = zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED
                zf.write(path, path.relative_to(out_dir).as_posix(), compress_type=kind)
    print(f"zip: {archive_path} ({archive_path.stat().st_size / 2**20:.0f} MiB)")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "dist")
    parser.add_argument("--config", default="method3")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    pack(args.out, config_name=args.config, make_zip=not args.no_zip)


if __name__ == "__main__":
    main()
