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
| Chunker | `textify/chunker.py` | Elements → page-bounded chunks, per `variant`. One opt-in flag added for Method 2 (see below); Method 1's output is byte-identical |
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
chunk set is byte-identical** — verified by re-chunking the corpus and comparing
all 2,818 chunk ids against the committed index.

The flag was not optional. Flattening drops elements whose `best_text()` is
empty, which is precisely the 147 figures Method 1 reports as invisible. Those
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

## Enrichment gap (open)

`method1.yaml` and `method2.yaml` both set `ocr_enabled: true` and
`vlm_captions_enabled: true`, but neither pass is implemented yet. So the 147
figures with no retrievable text lack it because nothing has tried to extract
it, not because extraction failed.

This matters for interpretation: Method 2's image-retrieval advantage on those
figures is currently measured against a Method 1 that has not been given the
textification its own design allows. Either implement both passes, or turn the
flags off and state that the baseline is caption-only, before publishing a
Method 1 vs Method 2 or Method 3 comparison.
