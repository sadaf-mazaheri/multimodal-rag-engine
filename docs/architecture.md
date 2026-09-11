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

**Both methods now rerank with the same model and the same `rerank_top_n`.**
Method 1 previously did not, which made the reranker a second difference
between them: a head-to-head result mixed the modality-aware effect with the
reranker effect and neither was attributable. Holding it constant costs the
baseline some of its "simplest thing that could work" character and buys a
comparison where retrieval is the only variable. `force_modalities` and
`use_metadata=False` exist so the routing and metadata effects can be isolated
the same way.

The cost is real and worth stating: on CPU the cross-encoder is ~17.6 s per
query over a 25-candidate pool, against ~150–280 ms for retrieval itself. Both
methods pay it equally, so the comparison is unaffected, but neither is
interactive on this hardware without a GPU.

---

## Two-stage fusion

Method 2 fuses in two stages, because RRF is additive across ranked lists and
the modalities did not supply equal numbers of them.

```
bm25 ─┐
      ├─ RRF ─→ text ─┐
dense ┘               │
table ────────────────┼─ weighted RRF ─→ modality-floored pool ─→ rerank
image ────────────────┘
```

`TableRetriever` and `ImageRetriever` already fused their own two signals
internally, but text's `bm25` and `dense` reached the cross-modality stage
separately — so text had two votes and every other modality one. The effect was
a ceiling, not a tendency:

| modality | lists | weight | best achievable score |
|---|---:|---:|---:|
| text | 2 | 1.0 | 0.032787 |
| table | 1 | 1.0 | 0.016393 |
| image | 1 | 0.7 | **0.011475** |

A 2.86× advantage before a single document was scored. Measured on "Transformer
model architecture diagram", the best figure landed at fused rank 44 with a
score of exactly `0.7/(60+1)` — its ceiling — behind 43 text chunks.

`_modality_weights` resolves a weight per modality from a config written
against retriever names, so `bm25: 1.0, dense: 1.0` becomes `text: 1.0` rather
than 2.0. An explicit `text:` key overrides it.

**Equalising the votes is necessary but not sufficient.** RRF ranks by position
and cannot *abstain*: a retriever contributes its rank-1 candidate at full
strength whether or not it holds anything relevant, and the router fans out on
42% of natural queries. Measured over 9 corpus-grounded queries, two-stage
fusion alone surfaced **0/5** figure answers and dropped text purity to 67% by
letting tables in instead. Raising the image weight to 2.0 instead gave 5/5
figures but put 6–7 figures in the top 10 of *"What is positional encoding?"*
and broke the table query — a see-saw, not a fix.

So the second stage is a **modality floor on the rerank pool**
(`retrieval.rerank_pool_per_modality`, default 8). Every fired modality is
guaranteed candidates in the pool the cross-encoder sees; the cross-encoder then
decides the order, and *can* abstain because it reads query and passage
together. The floor governs membership, never position.

| | figure hit | mean rank | table hit | text purity |
|---|---:|---:|---:|---:|
| before | 0/5 | — | 1/1 | 100% |
| **after** | **4/5** | 4.8 | **1/1** | 80% |

Pool size stays at exactly `rerank_top_n`, so reranking costs no more than
before. The floor is well-behaved between 4 and 8; at 12 the pool outgrows
`rerank_top_n` and quality drops.

**Without a reranker there is no floor.** A quota with no arbiter would promote
evidence nothing had vouched for, so `rerank_enabled: false` falls back to plain
cross-modality fusion truncated to `top_k`.

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

**2. Figures cannot reach the fused top-k on mixed queries.** *Resolved* — see
[Two-stage fusion](#two-stage-fusion). Figure answers went from 0/5 to 4/5 on
the sanity set while the table answer held and text purity stayed at 80%.

The one remaining miss is worth recording, because it is a *content* problem
rather than a fusion one: "How much warming is projected under the high
emissions scenario?" puts 8 figures in the pool, including the right IPCC
chart, and the cross-encoder still scores it below the prose. That chart's OCR
text is thin, so there is little for a text-pair model to match on. Nothing in
the retrieval path can fix it; it is an argument for Method 3 rather than
against the fusion change.
