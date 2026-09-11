# Component ownership

Which parts of the system are shared across methods, and which belong to exactly
one. This is the document to read before changing anything: editing a shared
component changes every method's numbers at once, which is how a benchmark
quietly stops being a controlled experiment.

## The rule

> **One ingestion, one chunker, one generation path. Methods differ only in how
> chunks are indexed and searched.**

If two methods disagree on a result, the cause must be traceable to retrieval.
That is only true while everything upstream and downstream of retrieval is
literally the same code.

---

## Shared infrastructure

Used unchanged by every method. Changing any of these affects all methods
simultaneously and invalidates prior measurements.

| Component | Module | Role |
|---|---|---|
| Core schemas | `schemas.py` | `Document → Page → Element → Chunk`, `BBox`, `ScoredChunk`, `Citation` |
| Config system | `config.py` | `Settings` (env) vs `ExperimentConfig` (YAML), `extends` inheritance |
| Corpus | `corpus/` | Pinned manifest + lockfile, verifying downloader |
| Ingestion | `ingestion/` | PDF → elements with page/bbox provenance |
| Chunker | `textify/chunker.py` | Elements → page-bounded chunks, per `variant`. One opt-in flag added for Method 2 (see below); Method 1's output is unchanged, and frozen by `tests/test_chunk_freeze.py` |
| Flattening | `textify/flatten.py` | Element selection + loss reporting; `keep_textless_visuals` opt-in |
| Token counting | `textify/tokens.py` | HF tokenizer with heuristic fallback |
| Text embedder | `embeddings/text.py` | `bge-small-en-v1.5`, asymmetric query prefix |
| BM25 | `stores/bm25.py` | Lexical index with configurable k1/b |
| Qdrant | `stores/qdrant.py` | Dense vector store, filterable payload |
| Postgres | `stores/postgres.py` | Document/page/element metadata |
| Rank fusion | `retrieval/fusion.py` | Weighted RRF + contribution diagnostics |
| Reranking | `retrieval/rerank.py` | Cross-encoder |
| Generation | `generation/` | `Answerer`, prompt, citation resolution, providers |
| CLI | `cli.py` | Same commands for every method, dispatched on `cfg.method` |

---

## Method 1 — Textified

**Frozen.** Method 1 established the baseline numbers, so it is not modified
while later methods are built.

| Component | Module | Notes |
|---|---|---|
| Pipeline | `methods/method1_textified.py` | Chunk → one flat index → hybrid retrieve → answer |
| Retriever | `retrieval/hybrid.py` | BM25 + dense over a single index, inline |

Indexes: BM25 at `data/indexes/method1/bm25`, Qdrant collection `method1`.

---

## Method 2 — Modality-Aware

Everything below is new. None of it touches Method 1.

| Component | Module | Role |
|---|---|---|
| Pipeline | `methods/method2_modality.py` | Per-modality index build, routed retrieval |
| Query router | `retrieval/router.py` | Lexical signals → which modalities to fire |
| Orchestrator | `retrieval/modality.py` | Route → fan out → fuse → rerank |
| Retrievers | `retrieval/modality_retrievers.py` | `bm25`, `dense`, `table`, `image` |
| Modality views | `retrieval/views.py` | Table schema vs content view; figure text view |
| Image embedder | `embeddings/image.py` | CLIP, shared image/text space |
| Metadata resolver | `retrieval/metadata.py` | Postgres → `doc_id` filters |
| Retriever contract | `retrieval/base.py` | `Retriever` protocol, `MetadataFilter`, `Hit` |

Indexes, all namespaced by variant so both methods coexist:

```
data/indexes/method2/
  text/              BM25 over text chunks
  table_content/     BM25 over table cells
  figure_text/       BM25 over figure captions/OCR/descriptions
Qdrant collections:
  method2_text          dense over text chunks
  method2_table_schema  dense over table schema views
  method2_image         CLIP vectors for figure crops
```

---

## The one shared-code change Method 2 required

`flatten_elements()` gained an opt-in `keep_textless_visuals` flag, and
`Chunker` threads it through. **Method 1 uses the default (`False`) and its
chunk set is unaffected** — both methods produce an identical 2,190 text and
446 table chunks, and Method 2's extra chunks are exactly the textless figures
Method 1 reports as invisible.

That property is now enforced rather than asserted. `tests/test_chunk_freeze.py`
re-chunks the parsed corpus and compares it against
`tests/baselines/method1_chunks.txt`, a frozen list of every Method 1 chunk id
with a digest of its text, so an unintended change fails a test instead of
quietly moving the baseline.

**The baseline is a snapshot, not an invariant.** It currently records the
post-OCR state, 2,968 chunks. It has already been regenerated deliberately
twice: once for the table-degeneracy fix (2,818 → 2,887) and once for the OCR
pass (2,887 → 2,968). When a change *should* move the chunk set again,
regenerate and review the diff rather than weakening the test:

```bash
MMRAG_UPDATE_CHUNK_BASELINE=1 python -m pytest tests/test_chunk_freeze.py
```

The flag was not optional. Flattening drops elements whose `best_text()` is
empty, which at the time was the 147 figures Method 1 then reported as invisible
(the table fix and the OCR pass have since brought that to 37). Those
elements therefore never became chunks — so Method 2's CLIP index was built,
populated, queryable, and contained **none of the figures it exists to
recover**. The image retriever returned only figures that already had captions,
i.e. exactly the ones Method 1 could already find.

That failure was silent in every observable way: the build succeeded, the
collection had 253 points, and queries returned plausible results. It surfaced
only because the build report printed `text_invisible_recoverable: 0` and that
contradicted the 147 measured from the elements. It is the strongest argument in
this repo for making a pipeline report what it *lost*, not just what it did.

---

## Deliberate duplication

`HybridRetriever` (Method 1) and `BM25Retriever`/`DenseRetriever` (Method 2)
both call BM25 and Qdrant, and that overlap is intentional.

Factoring them together would have meant editing `HybridRetriever` — which would
have changed Method 1 after its numbers were already recorded. A few dozen
duplicated lines are cheaper than a benchmark whose baseline moved mid-experiment.
Method 2's retrievers additionally implement the `Retriever` protocol and support
per-modality filtering, which `HybridRetriever` has no use for.

---

## What makes the comparison fair

Both methods:

- read the **same** `parsed.json` sidecars produced by one ingestion run
- chunk with the **same** `Chunker` and the same `ChunkingConfig` defaults
- inherit from the same `configs/default.yaml`
- answer through the **same** `Answerer`, prompt, and provider
- attach **no images** to the model (that is Method 3's defining move)
- record provenance the same way, so citations are directly comparable

They differ in exactly three places: how chunks are partitioned into indexes,
whether a router selects among those indexes, and which retrievers score them.

One asymmetry is worth stating plainly: **Method 2 enables the cross-encoder
reranker and Method 1 does not** (`retrieval.rerank_enabled`). That is a
deliberate part of the Method 2 design rather than an oversight, but it means a
head-to-head result mixes the modality-aware effect with the reranker effect.
Step 6 should report the reranker-off ablation alongside, and
`force_modalities` exists so the routing effect can be isolated the same way.

---

## Enrichment

Enrichment is **shared infrastructure**, not a method's property. It runs in
`ingestion/pipeline.py` between parsing and writing the sidecar, so recovered
text lands in the representation both methods read. Neither can be enriched
without the other, which is what keeps a Method 1 vs Method 2 difference
attributable to retrieval rather than to one of them having been handed better
text. `tests/test_ocr.py` asserts the two configs agree on every OCR setting.

| Pass | Module | Status |
|---|---|---|
| OCR | `ingestion/ocr.py` | Implemented. `rapidocr` (default, pip-only) or `tesseract` behind one `OcrEngine` protocol |
| VLM captions | — | **Not implemented.** `vlm_captions_enabled` is `false` in every config so no run manifest asserts a pass that never ran |

Why this mattered: before OCR existed, a figure's entire textual representation
was its caption, so a figure without one was unreachable by any text query. The
audit found many of those were not visual content at all — register-description
tables, infographic pull-quotes, section-divider banners — i.e. *text* that
belongs to the textified baseline by right. Crediting Method 2's image retriever
for recovering them would have measured an ingestion gap, not an architectural
advantage.

The pass is deliberately conservative about what it keeps: recovered text must
clear `ocr_min_chars` and be at least half alphanumeric, because OCR on a
genuinely textless plot returns axis marks that pass a length check on their
own. A figure that really is image-only stays textless, which is what keeps
Method 2's remaining claim honest.

Provenance is preserved rather than overwritten. `extraction_method` still
records how the element was *found* — error analysis slices on it, and OCR did
not find it. What OCR did is recorded beside it in `figure.ocr_text`,
`figure.ocr_confidence` and `element.metadata["ocr"]`, and the per-document
`stats.ocr` block records the backend and version, so "OCR ran and found
nothing" is distinguishable from "OCR never ran".

Known OCR quality limits, measured on the corpus: recovered text is missing
intra-word spaces in about 1.6% of tokens (`Centralgovernmentsecurities`), which
costs BM25 exact-term matches; chart axis labels arrive unordered, so they add
lexical hooks rather than structured meaning; and the Transformer paper's
attention heatmaps OCR to token soup that clears the alphanumeric gate. Median
per-figure confidence is 0.985.

---

## Known issues

Open defects, deliberately not fixed yet. Both surface as failing tests in
`tests/test_method2.py` rather than as xfails, so they stay visible.

**1. `MetadataResolver` narrows on a single common word.** `resolve()` uses
`min_terms=1`, so "Transformer model architecture diagram" matches
`nvidia_ampere_wp` on the word *architecture* alone — that document's title is
"NVIDIA A100 Tensor Core GPU **Architecture** Whitepaper" — and every other
retriever is then restricted to it. The Transformer paper's own facet terms are
only `{arxiv, attention, all, you, need}`, since neither its `doc_id` nor its
title contains "transformer". Fails
`TestRetrieval::test_finds_the_transformer_architecture_figure` and
`TestAnswering::test_produces_an_answer_with_resolvable_citations`.

This is also an asymmetry: Method 2 gets query→`doc_id` narrowing that Method 1
has no counterpart for, and it is on by default (`use_metadata=True`), so a
head-to-head comparison currently mixes it in. `retrieve(use_metadata=False)`
exists to ablate it.

**2. Figures cannot reach the fused top-k on mixed queries.** A text chunk is
voted for by *two* retrievers (`bm25` and `dense`, weight 1.0 each) while a
figure gets *one* (`image`, weight 0.7), so RRF structurally favours text: two
contributions of `1.0/(k+rank)` beat one of `0.7/(k+rank)` almost regardless of
rank. Measured with the metadata resolver disabled, the top 10 for "Transformer
model architecture diagram" is entirely text with reranking both **on and off**,
while forcing `Modality.IMAGE` puts the correct figure at **rank 1**. So the
figure is retrievable and fusion is what buries it.

This one bears directly on the Method 2 premise: if the image retriever rarely
reaches the final top-k except on explicitly visual queries, the modality-aware
advantage is smaller than the index-build numbers suggest. Worth resolving
before any Method 1 vs Method 2 result is published.
