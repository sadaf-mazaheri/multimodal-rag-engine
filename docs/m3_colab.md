# Method 3 on a Colab GPU

The ColQwen2 page index is built on a GPU with a self-contained runner. Colab
needs **one uploaded zip** — no repository clone, database, Qdrant or API key.
Everything else (evaluation, generation, judging) stays on the main machine.

```
main machine                     Colab (GPU)                        main machine
pack_m3_colab.py  ──zip──▶  run_m3_gpu.py all  ──zip──▶  unzip at repo root
                            check → smoke → build →        mmrag index status -c method3
                            embed-queries → verify →       mmrag eval run -c method3
                            package
```

## What goes where

| | Contents |
|---|---|
| **Code on Colab** | `m3_colab/run_m3_gpu.py`, `m3_colab/requirements-gpu.txt`, and `m3_colab/mmrag/` — only `config.py`, `logging_utils.py`, `embeddings/visual.py`, `stores/multivector.py` (the repository's own encoder and index format), with stub `__init__.py` files |
| **Corpus on Colab** | `m3_colab/inputs/renders/<doc>/pages/*.png` (951 page renders), `inputs/pages.jsonl` (page ids, sizes, DPI, checksums), `inputs/corpus.lock.yaml`, and in `inputs/bundle.json` the document hashes and which pages carry method2 chunks |
| **Queries on Colab** | `m3_colab/inputs/queries.json`: the 42 retrieval gold queries and 8 unanswerable generation queries, as `mmrag index embed-queries` would select them |
| **Generated on Colab** | `data/indexes/visual_pages/index/` (page index), `data/indexes/visual_pages/query_cache/`, `data/indexes/visual_pages/build/` (reports), checkpoints in `--work-dir`, `smoke/` (partial test index), `run.log` |
| **Copied back** | `m3_visual_pages.zip` only; it contains `data/indexes/visual_pages/` with repository-relative paths |

## 1. On the main machine: make the bundle

```bash
python scripts/pack_m3_colab.py
```

Writes `dist/m3_colab_bundle.zip` (a few hundred MB, mostly PNGs). It refuses
to pack if a render is missing or the parsed documents drift from
`configs/corpus.lock.yaml`. Upload the zip to Google Drive, into
`MyDrive/mmrag_m3/`. A browser upload to Drive is more reliable than
uploading into the Colab session.

## 2. On Colab

Runtime → Change runtime type → **GPU**. L4 or A100 run bfloat16; a T4 runs
float16. Or open `notebooks/m3_colab_gpu.ipynb`, which contains these cells.

**Setup**

```python
from google.colab import drive
drive.mount("/content/drive")
DRIVE = "/content/drive/MyDrive/mmrag_m3"       # holds m3_colab_bundle.zip
OUT = "/content/m3_out"                         # fast local disk
WORK = f"{DRIVE}/work"                          # per-page checkpoints, survive disconnects
RUN = "python /content/m3/m3_colab/run_m3_gpu.py"
!mkdir -p /content/m3 && unzip -q -o {DRIVE}/m3_colab_bundle.zip -d /content/m3
```

**Install** (keeps Colab's CUDA torch)

```python
!pip install -q -r /content/m3/m3_colab/requirements-gpu.txt
```

**Check** — CUDA, GPU, versions, bundle integrity

```python
!{RUN} check --out {OUT}
```

**Smoke test** — loads ColQwen2 (~4.5 GB download), encodes 8 pages, checks
batched and unbatched vectors agree, writes and reloads a partial index,
encodes and ranks 3 queries, estimates the full build time

```python
!{RUN} smoke --out {OUT} --smoke-pages 8
!cat {OUT}/data/indexes/visual_pages/build/smoke_report.json
!mkdir -p {DRIVE}/reports && cp {OUT}/data/indexes/visual_pages/build/*.json {DRIVE}/reports/
```

**Full index** — all 951 pages, checkpointed to Drive per page

```python
!{RUN} build --out {OUT} --work-dir {WORK}
```

**Query embeddings, verification, packaging**

```python
!{RUN} embed-queries --out {OUT}
!{RUN} verify --out {OUT}
!{RUN} package --out {OUT}
!cp {OUT}/m3_visual_pages.zip {OUT}/run.log {DRIVE}/
```

Or run every stage in one go: `!{RUN} all --out {OUT} --work-dir {WORK}`.

**If the session disconnects during the build**, rerun *Setup* and *Install*,
then:

```python
!{RUN} build --out {OUT} --work-dir {WORK} --skip-smoke
```

Pages already in `WORK` are skipped. `--skip-smoke` is needed only because the
smoke report was on the lost local disk. Checkpoints are bound to the model
identity, storage precision and bundle, so a checkpoint from anything else is
refused rather than mixed in.

**Troubleshooting**

- *NaN or infinite values* (half precision overflow, usually on a T4): add `--dtype float32`.
- *Out of memory*: the batch halves automatically; `--batch-size 2` starts smaller.
- *CUDA is not available*: the runtime is not a GPU runtime.

## 3. Back on the main machine

Download `m3_visual_pages.zip` from Drive and extract it at the repository
root:

```powershell
Expand-Archive m3_visual_pages.zip -DestinationPath . -Force
```

```bash
unzip -o m3_visual_pages.zip -d .
```

That creates `data/indexes/visual_pages/{index,query_cache,build}`. Then:

```bash
mmrag index status --config method3     # re-verifies checksums, model identity, corpus lock, coverage
mmrag eval run --config method3 --tag rerank
```

`mmrag eval run` refuses a partial index. Retrieval runs on CPU from the query
cache and never loads ColQwen2.
