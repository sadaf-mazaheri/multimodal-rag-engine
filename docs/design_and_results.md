# Design and results

The detailed design notes and evaluation record for the Multimodal RAG Benchmark.
This is the full text of the project README before it was condensed into a
GitHub-facing overview; content is unchanged apart from this preface, link paths
adjusted for the `docs/` directory, and the production-phase pointer in *Status*.

- Overview and headline results: [`../README.md`](../README.md)
- Production evaluation (Phase A offline metrics, Phase B replay load test):
  [`production.md`](production.md)
- Component ownership: [`architecture.md`](architecture.md)

Three retrieval-augmented generation architectures, implemented over **one shared
ingestion layer** and evaluated on **one shared corpus**, so their results are
directly comparable.

The question this repo tries to answer is not "can we chat with a PDF" — it is:

> When a document contains text, tables, charts and diagrams, **how much do you
> actually gain** by preserving each modality instead of flattening everything
> into text? And for *which kinds of question*?

| | Method 1 | Method 2 | Method 3 |
|---|---|---|---|
| **Name** | Textified | Modality-Aware | Hybrid Visual |
| **Idea** | Flatten every modality to text, retrieve once | Keep modalities native, one retriever each, route the query | Add late-interaction retrieval over rendered page images |
| **Text** | BM25 + dense hybrid | BM25 + dense | BM25 + dense |
| **Tables** | → Markdown, then text retrieval | structure-aware retrieval | structure-aware retrieval |
| **Figures** | → caption + OCR text | CLIP image embeddings | CLIP + ColQwen2 page vectors |
| **Fusion** | RRF over 2 retrievers + rerank | best-route within each modality, RRF across, modality-floored rerank | Method 2's fusion + a page signal that always fires, same rerank |
| **Generation** | text LLM | text LLM | text LLM — the same generator, so answers stay comparable |
| **Measures** | the cost of flattening | the value of preserving modality | the value of seeing the page |

---

## Status

This project is being built in stages. Current state:

- [x] **Step 0** — Repo scaffold, configuration system, core data model, Docker infra
- [x] **Step 1** — Pinned corpus manifest/lockfile + verifying downloader
- [x] **Step 2** — Shared ingestion: PDF → elements with rich metadata and provenance
- [x] **Step 3** — Method 1: Textified hybrid RAG
- [x] **Step 4** — Method 2: Modality-aware retrieval + query router
- [x] **Step 5** — Method 3: Hybrid visual RAG (ColQwen2) — page index built on a Colab T4, retrieval and generation evaluated
- [x] **Step 7** — Generation V2.1: evidence pack, answer contract and deterministic validation beside the frozen V1 — see [Generation V1 and V2](#generation-v1-and-v2)
- [x] **Step 6** — Evaluation harness: retrieval metrics, then generation scored by an LLM judge
- [x] **Production Phase A / B** — offline production metrics and a replay load test — see [`production.md`](production.md)

Step 6 was built before Step 5 on purpose: two methods with no numbers were
already one too many, and the evaluation turned up a corpus error and two
retrieval defects that would otherwise have been carried into Method 3.

---

## Design principles

**One ingestion, three indexes.** All three methods read the *same* parsed
`Element` rows out of Postgres. They differ only in how those elements are turned
into retrieval units and how those units are searched. If each method had its own
parser, any difference in the results would be unattributable — you would be
comparing parsers, not retrieval architectures.

**Provenance is a first-class citizen, not a postscript.** Every element carries
`doc_id`, `page_number` and a normalised bounding box. A retrieval hit therefore
resolves to a *region of a page*, which is what makes element-level citations
possible and what lets the evaluation score whether a method retrieved the right
evidence, independently of whether the LLM then wrote a good answer.

**Retrieval is provider-independent.** Embeddings, parsing, indexing, chunking
and evaluation use local open-source components only. No API key is required to
build or evaluate any index. Only the final answer generation talks to a
provider, behind a swappable interface — so retrieval quality can never be
silently confounded with a model vendor's behaviour.

**Environment and experiment are separate configs.** `.env` holds secrets and
endpoints (properties of *your machine*). `configs/*.yaml` holds chunk sizes,
model names and `top_k` (properties of *the experiment*). A run is fully
described by "this YAML, on this corpus" — which is the whole basis of
reproducibility.

---

## System architecture

The three methods are **configurations of one system**, not three systems.
Retrieval signals are pluggable components behind one contract, and a single
engine routes, fuses, reranks and answers over whichever set it is given:

```
                 ┌─────────────────────── RAGEngine ───────────────────────┐
query ─→ metadata resolution ─→ router ─→ retrievers ─→ fusion ─→ rerank ─→ Answerer ─→ cited answer
         (doc_id filters)       (which     │ bm25        within      cross-     context budget,
                                modalities)│ dense       modality,   encoder    prompt, provider,
                                           │ table       then        over a     citation
                                           │ image       weighted    modality-  resolution,
                                           │ visual_page RRF across  floored    refusal
                                                                     pool
```

| Layer | Module | What it owns |
|---|---|---|
| Ingestion | `ingestion/` | PDF → `Document`/`Page`/`Element` with provenance, page renders |
| Index components | `indexing/` | Building indexes from the corpus and opening retrievers over them: `ModalityIndex` (chunk set + text/table/figure sub-indexes), `VisualPageIndexer` (ColQwen2 page index + query cache) |
| Storage adapters | `stores/` | BM25, Qdrant, Postgres, the file-based multi-vector page store |
| Retrievers | `retrieval/` | The `Retriever` protocol and its implementations; router, fusion, reranker |
| Engine | `engine.py` | `RAGEngine`: the query path above, over any registered retrievers |
| Generation | `generation/` | Prompting, providers, citation resolution |
| Methods | `methods/` | Named configurations: which indexes, which retrievers, which always fire |
| Evaluation | `evaluation/` | Consumes `method.retrieve()` / `method.answer()`; nothing in the runtime imports it |

| Method | Configuration |
|---|---|
| Method 1 | Frozen single-index pipeline (`HybridRetriever`), kept exactly as measured |
| Method 2 | `RAGEngine` over `ModalityIndex("method2")`: `bm25`, `dense`, `table`, `image` |
| Method 3 | `RAGEngine` over the same `ModalityIndex` (read-only) + `VisualPageIndexer`: Method 2's four plus `visual_page`, always on |

Adding a retrieval signal means implementing `Retriever.retrieve(query, k,
filters) -> RetrieverOutput` and registering it; routing, fusion, pooling,
reranking, generation and evaluation need no change. Method 3 is the proof: it
is Method 2's configuration plus one retriever, with no subclassing and no
special case downstream.

Method 1 is the one exception. Its fusion is single-stage and it predates the
`Retriever` protocol; porting it onto the engine would risk moving numbers that
are already recorded, so it stays as it was measured. It exposes the same
method surface, so the CLI and evaluation treat all three identically.

---

## Quick start

### 1. Install

```bash
python -m pip install -e ".[openai,dev]"
```

Optional extras: `ocr` (self-contained — `rapidocr` ships its models in the
wheel, so no system package is needed), `visual`
(ColQwen2 — practically needs a GPU), `local` (offline generation).

### 2. Configure

```bash
cp .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`. Nothing before Step 3 needs it.

### 3. Start the infrastructure

```bash
docker compose up -d
```

Postgres (metadata, on port **5433** by default to avoid clashing with a system
Postgres) and Qdrant (vectors, on 6333). The schema in `scripts/sql/` is applied
automatically on first start.

> **Changing the pinned Qdrant version requires wiping its volume.** Qdrant
> storage is not forward-compatible: a newer server panics on segments written
> by an older one and then restart-loops. The vectors are derived data, so
> rebuild rather than migrate — `docker compose stop qdrant && docker volume rm
> rag_project_qdrant_data`, then `mmrag index build`. Only the Qdrant volume;
> Postgres holds the parsed corpus. The client is pinned in lockstep in
> `pyproject.toml`.

### 4. Check your environment

```bash
mmrag doctor
```

Reports what is available and what each missing piece disables — a partial
environment is still usable for the parts it covers.

### 5. Fetch the corpus

```bash
mmrag corpus download
```

Downloads the 14 documents listed in `configs/corpus.yaml` into `data/raw/`
(~45 MB). Each file is streamed to a temp path, hashed, sniffed for a PDF
header, and only then moved into place, so an interrupted or bogus download can
never leave a half-file for the parser to find.

```bash
mmrag corpus lock     # regenerate configs/corpus.lock.yaml (only when sources change)
mmrag corpus verify   # re-check local files against the lockfile
```

---

## The corpus

The PDFs are **not committed** — they are third-party documents of varying
licence. What is committed is a pair of files, split the same way as
`package.json` and `package-lock.json`:

- **`configs/corpus.yaml`** — hand-authored. URLs, licences, and *why* each
  document earns its place in the corpus. Never rewritten by tooling, so its
  comments and structure survive.
- **`configs/corpus.lock.yaml`** — generated by `mmrag corpus lock`. The
  SHA-256, byte size and page count each file must have.

Anyone cloning the repo provably ends up with byte-identical inputs, so any
difference in their numbers comes from the pipeline and not from a silently
re-issued PDF. If an upstream document changes, the download fails loudly with a
hash mismatch rather than quietly indexing different bytes.

**1,620 pages across 14 documents** (951 after the `page_limit` caps).

The lockfile guarantees the bytes match, not that they are the *right* bytes. One
entry paired the TAPAS title with the URL of a different paper (ETC, arXiv
2004.08483), and the hash check passed because it verifies whatever it is told to
fetch. It surfaced during gold-set review, when a TAPAS question had no TAPAS
text to point at. The URL now resolves to arXiv 2004.02349.

Fourteen documents, chosen so each modality is genuinely stressed and results
can be sliced by document character:

| Category | Documents | What they stress |
|---|---|---|
| Finance | Berkshire Hathaway AR, Fed Monetary Policy Report, ECB Annual Report | Multi-level table headers, footnote markers, time-series charts |
| Policy / Energy | IPCC AR6 WGI SPM, EIA Annual Energy Outlook | Composite multi-panel figures, projection charts with scenario bands |
| Health | WHO COVID-19 Situation Report 1 | Short end-to-end smoke test; case-count table plus a map |
| Technical | NASA Systems Engineering Handbook, RP2040 Datasheet, NVIDIA Ampere Whitepaper | Flow diagrams with no extractable text, wide register maps, block schematics |
| Science | Attention Is All You Need, RAG, ColPali, TAPAS, AlphaFold | Two-column layout, equations, small inline figures, rendered 3D structures |

Two documents are truncated (`page_limit`) so a 650-page datasheet cannot
dominate every per-document metric.

---

## The shared representation

All three methods read the *same* parsed records, so any difference in their
results is attributable to retrieval architecture rather than to parsing.
Metadata is structured at three levels and is never folded into embedded text —
`Element.best_text()` returns content, `Element.structured_metadata()` returns
the filterable payload, and the two are deliberately disjoint.

| Level | Carries |
|---|---|
| **Document** | id, title, source + URL, authors, organisation, publication date, version, type, domain, language, licence, file name, SHA-256, page counts, parser version |
| **Page** | number, dimensions, rotation, running section/subsection, header/footer text, rendered image path + size + DPI |
| **Element** | id, type, `parent_id`, page number, normalised bbox, reading order, section, extraction method, extraction confidence |
| **Table** | columns, rows, dimensions, header rows, fill ratio, Markdown, table type (financial / register / comparison / matrix / simple) |
| **Figure** | image path, pixel dimensions, DPI, figure type (chart / diagram / photo / map / screenshot / logo / **unknown**), type confidence + evidence, OCR text, description |

Two fields exist purely to make later error analysis possible.
`extraction_method` records *which* extractor produced an element, so Step 6 can
ask "are the bad retrievals concentrated in one extractor?". `extraction_confidence`
is a single 0–1 scale with the same meaning across every modality (1.0 faithful,
0.5 degraded, 0.0 nothing recovered), so it is comparable rather than a
per-module private score.

Provenance is enforced, not merely recorded: a `Chunk` is rejected at
construction if it mixes documents or pages, and `build_provenance()` resolves a
hit back to a page image and a bounding box.

### What ingestion actually produces

```bash
mmrag ingest run                     # all 14 documents
mmrag ingest run -d arxiv_attention  # one document
mmrag ingest show arxiv_attention -t caption -g   # inspect with geometry
mmrag ingest status                  # per-document counts from Postgres
```

**951 pages → 15,992 elements**, mean extraction confidence **0.94**, 0.3% below 0.5:

| | count | | count |
|---|---:|---|---:|
| text | 12,217 | tables | 437 |
| headings/titles | 1,285 | charts | 163 |
| captions | 694 | figures | 187 |
| headers/footers | 994 | diagrams | 15 |

Table types skew register (171) and financial (148); figure types are chart (163),
unknown (145), photo (42), diagram (15). **`unknown` is a deliberate answer** —
Method 2 routes on `figure_type`, so a confidently wrong label misdirects
retrieval while an honest abstention merely fails to help.

The register-heavy table profile is recent: a degeneracy rule was rejecting any
table whose header had been promoted out of a two-row grid, which is the exact
shape of a register description table. Fixing it moved 83 of these out of the
figure counts and into the table counts, where they belong.

Parsing runs in two passes per document, and the structure is load-bearing: a
single page cannot distinguish a running header from a section heading, nor a
14pt heading from 14pt body text. Both are only answerable once the whole
document's typography has been observed.

Output goes to Postgres *and* to a JSON sidecar per document under
`data/processed/`. The sidecar exists so an index can be built on a machine with
no database — Method 3's page index is built that way on a GPU machine —
and because debugging a parse should not require SQL. Both are written from the
same objects, so they cannot drift.

### Extraction problems this had to solve

Every one of these was found on a real corpus document, and each has a
regression test:

- **Charts are usually vector, not raster.** `get_images()` finds nothing for the
  Fed's line charts or the IPCC's panels; they are built from drawing operators
  and have to be reconstructed by clustering primitives into regions.
- **A filled background panel is not a figure.** The WHO report's blue "Resources"
  box clustered into a convincing chart — and because text inside a figure was
  being suppressed, it *hid the entire section*. Now vector clusters whose area
  is mostly covered by text are rejected as panels, and text inside a genuine
  figure is kept and parented to it rather than dropped.
- **A table's ruling lines cluster into a phantom chart** over the same region,
  double-counting the evidence. Tables are detected and validated first, and
  figure detection excludes their regions.
- **Not every detected table is a table.** The Transformer paper's attention
  visualisations come back as 35-column grids whose cells are single words of a
  sentence — structurally perfect, semantically noise. Rejected on cell content,
  keyed on *single alphabetic words* rather than short cells, since dense numeric
  results grids also have short cells and are exactly what the benchmark cares about.
- **Financial tables are whitespace-aligned, not ruled.** PyMuPDF's `lines`
  strategy returns a 5-column Berkshire table as 8×24 — a column per currency
  symbol and alignment gap. Post-processing merges symbol-only columns into their
  values and drops empty ones. (The `text` strategy was tried and rejected: it
  swallows the whole page, headings included, into one 66×13 pseudo-table.)
- **A heading above a chart is not its caption.** Adopting one poisons figure-type
  classification, which reads the caption. Headings are excluded by relative font
  size, and an unmarked caption must also be set *smaller* than body text.
- **Windows consoles cannot print what PDFs contain.** `∗` from a paper's author
  footnote crashed the CLI under cp1252; output streams are forced to UTF-8.

---

## Method 1 — Textified RAG

Every modality is flattened into text, then retrieved by one hybrid pipeline.
The simplest thing that could work — which is the point. It is the baseline the
other two are measured against, so the number it exists to produce is the *cost
of the flattening*.

```bash
mmrag index build --config method1
mmrag query "What is the Transformer architecture?" --retrieve-only
mmrag query "How many cases were confirmed?" --provider echo   # free, no API key
mmrag query "What was Berkshire's insurance underwriting result?"
```

**Pipeline:** elements → `best_text()` flattening → page-bounded chunking →
BM25 + dense, fused with weighted RRF → prompt with numbered sources → cited
answer whose citations resolve back to a page and bounding box.

### What the baseline can't see

Building the index over the full corpus produces **2,947 chunks** (2,174 text,
445 table, 328 figure) from 951 pages — and reports the headline measurement:

> **37 figures have no retrievable text at all.**

Those are figures where nothing textual could be extracted — no caption, and
nothing OCR could recover. They exist in the corpus and are *structurally
unreachable* for this architecture. That number is the mechanism behind any
deficit Method 3 later makes up, and it is why `FlattenReport` counts and
attributes losses instead of silently dropping them.

**Getting to 37 took two ingestion fixes, and both had to come before any
comparison.** It began at 147. A table-degeneracy bug was emitting register
description tables as textless "charts"; fixing it took the figure to 118, and
reclassified 83 elements from chart to table along the way. Implementing the OCR
pass then recovered text from 81 more. Every one of those 110 figures was text
that the textified baseline should always have had — so had the comparison been
run first, Method 2's image retriever would have been credited for recovering
content that belonged to Method 1 by right. An ingestion gap masquerading as an
architectural result is the failure mode this project most needed to avoid.

### Design decisions

**Rank fusion, not score fusion.** BM25 scores are unbounded and
corpus-dependent; cosine similarities sit in [-1, 1]. Combining them by score
requires a normalisation choice, and every such choice is a hidden tunable. RRF
uses only ordering, so there is nothing to tune away. Per-retriever ranks are
kept on every result, which makes the ablation ("was the hybrid actually better
than either alone?") answerable *after* the run rather than requiring three.

**The two retrievers fail differently, which is why both are there.** Dense
retrieval handles paraphrase and misses rare literals; BM25 does the reverse. A
document corpus asks for `GPIO_OE`, `Figure 12`, and `$22,360` — BM25's
strength — as often as it asks conceptual questions.

**Chunks never cross a page.** A chunk spanning two pages cannot be cited to one
page, so `Chunk` rejects it at construction. Tables and figures are always their
own chunks; a table merged into prose is neither good prose nor a usable table.

**One deliberate exception to "metadata stays structured".** With
`prepend_context_header`, a chunk's text is prefixed with
`Document > Section > Subsection`. A chunk from mid-document is often
unintelligible alone ("It rose to 4.2% in the third quarter" — what did, in
which report?), and the breadcrumb restores the referent the page layout
supplied visually. It is capped at a heading-sized string, recorded separately
in `metadata["context_header"]` so it can be stripped, and switchable off for
the ablation. Identifiers, bounding boxes, extraction methods and confidences
never enter embedded text.

**Refusal is a valid answer.** The model is instructed to emit
`INSUFFICIENT_EVIDENCE` rather than guess. A benchmark that rewards confident
guessing cannot separate "retrieval failed" from "generation hallucinated over
good evidence" — which is exactly the distinction Step 6 needs.

**Citations that don't resolve are counted, not passed through.** A model citing
`[7]` when five sources were supplied is a grounding failure; silently dropping
it would hide the problem behind a clean-looking answer.

### Generation is the only vendor-dependent step

Everything else — parsing, chunking, embedding, BM25, fusion, reranking,
evaluation — is local and open-source, so retrieval quality can never be
confounded with a model vendor's behaviour. Providers: `openai` (default,
gpt-4o-mini), `local` (any OpenAI-compatible server), and `echo`, a deterministic
stub that returns valid citation markers so the whole pipeline can be exercised
and tested for free.

---

## Method 2 — Modality-Aware RAG

Each modality keeps its native representation and gets a retriever suited to it.
A router reads the query and fires only the retrievers that could plausibly hold
the answer; their ranked lists are fused and reranked.

```bash
mmrag index build --config method2
mmrag query "Which table lists confirmed cases by country?" -c method2 --retrieve-only
mmrag query "What does the architecture diagram show?" -c method2 --provider echo
```

### The four retrievers

| Retriever | Signal | Answers |
|---|---|---|
| `bm25` | lexical over text chunks | exact identifiers, figures, names |
| `dense` | bge over text chunks | paraphrase, conceptual questions |
| `table` | BM25 over **cells** + dense over **schema** | "which table contains 22,360" *and* "which table is about revenue by segment" |
| `image` | CLIP over figure crops + BM25 over figure text | figures — including ones with no text at all |

### What Method 2 can do that Method 1 structurally cannot

Method 1 reports **37 of 365 figures with no retrievable text**: no caption, and
nothing OCR could recover. No text index can reach them under any query.

**All 37 have a cropped image on disk.** CLIP puts images and text in one space,
so a text query scores directly against the pixels with no textual intermediary.
That is the concrete mechanism behind any advantage Method 2 shows on figure
questions, and the index build reports it as `text_invisible_recoverable`.

**37 is the honest version of a number that started at 147**, and the difference
is the point. The original 147 mixed three populations: register tables the
parser had misfiled as charts, text baked into images that OCR can read, and
genuinely image-only content. Only the third is a modality gap. Fixing the first
two before measuring anything is what keeps the remaining 37 — mostly protein
structure renders and photographs — an architectural claim rather than an
artefact of incomplete ingestion.

### Two views of a table, not one

Flattened to Markdown, a table's column headers are three tokens among hundreds
of digits — so "which table breaks revenue down by segment" has almost nothing
to match. Method 2 indexes tables twice:

- **schema view** — caption, column headers, table type, shape. Short and
  semantic; dense retrieval handles it.
- **content view** — the cells. Long and literal; BM25 handles it.

Both point at the same `chunk_id`, so provenance is unchanged.

### Fusion happens in two stages

Each modality's own signals are fused first — `bm25 + dense → text`,
`content + schema → table`, `CLIP + figure text → image` — and only then are the
three modalities fused together.

The two stages combine differently, because the lists mean different things.
**Across modalities** they are independent votes, so agreement is evidence and
contributions are summed (standard RRF). **Within table and image** the two
signals are alternative routes to the *same* evidence — a figure found by its
pixels or by its caption is one figure — so the better route decides
(`combine="max"`). Text's `bm25 + dense` still sums.

That split was a fix, not the original design. Summing inside the image
retriever capped any figure only one signal could see at `1/(k+1)`, however
perfect the match, so a figure ranked **first** by figure-text alone lost to
anything mid-table in both lists and was demoted to 23rd, 29th or 39th — below
the rerank floor, so the cross-encoder never judged it. It bit hardest on the 37
figures with no text at all, which are structurally absent from the figure-text
index. Once the demoted figures reached the pool, the cross-encoder ranked three
of them first or second (scores +0.86 to +0.998): it could always identify them,
and had simply never been shown them.

The reason is arithmetic. RRF adds a contribution per ranked list, so a modality
supplying two lists got twice the votes. Table and image already fused
internally; text did not, which handed it a **2.86× score ceiling** over image
before anything was scored. On *"Transformer model architecture diagram"* the
best figure landed at fused rank 44 — at exactly its ceiling of `0.7/(60+1)`,
behind 43 text chunks — despite its own retriever ranking it **first**.

Equalising the votes is necessary but not sufficient: RRF ranks by position and
cannot *abstain*, so an irrelevant modality still contributes its best candidate
at full strength. Two-stage fusion alone surfaced 0/5 figure answers and let
tables crowd in instead; simply raising the image weight gave 5/5 figures but
flooded prose queries and broke the table query.

So the cross-encoder arbitrates, and every fired modality is guaranteed a floor
in the pool it sees (`rerank_pool_per_modality`, default 8). It can abstain,
because it reads query and passage together.

On a 9-query sanity check run while designing it:

| | figure answers found | table | text purity |
|---|---:|---:|---:|
| before | 0/5 | 1/1 | 100% |
| **after** | **4/5** | **1/1** | 80% |

The pool stays at `rerank_top_n`, so this costs no extra reranking. With
reranking disabled there is no floor — a quota with no arbiter would promote
evidence nothing vouched for — and fusion falls back to plain `top_k`.

**Both methods now rerank identically**, so a head-to-head no longer mixes the
modality-aware effect with the reranker effect. On CPU the cross-encoder
dominates latency — median ~18 s per query for either method, against roughly
100–250 ms for retrieval itself — so neither is interactive without a GPU.

### Document resolution

Method 2 can narrow a query to the documents it names ("what does the IPCC
report say…" becomes a `doc_id` filter). It originally judged a word to be a name
if it was unique among the fourteen titles — which is not the same as being
rare. `table` appears in one title and every body, so *"which table lists
confirmed cases"* was filtered onto the table-parsing paper; `architecture` sent
*"Transformer model architecture diagram"* to the NVIDIA whitepaper. Over the
gold set, 8 of 30 narrowings excluded the document holding the answer.

A title word now counts only if it appears in fewer than half the corpus
**bodies**. That removed all 8 wrong narrowings while keeping all 22 correct
ones; the resolver otherwise abstains. It is on by default, and `--no-metadata`
ablates it.

### Routing

Rule-based, not model-based: deterministic, free, and inspectable, so a bad
route traces to the exact phrase that caused it. An LLM router is a later
ablation, not the baseline.

Two commitments: **text always fires** (a false negative is unrecoverable, a
false positive costs only latency), and **the decision is recorded** — matched
signals, scores, and whether it fell back — so Step 6 can ask whether routing
helped rather than treating it as a black box.

**What the router is worth, measured.** A 60-query audit across four strata:

| Query stratum | n | fell back | routed correctly |
|---|---:|---:|---|
| names a table ("which table lists…") | 10 | 0 | 10/10 → exactly `text+table` |
| names a figure ("what does the diagram show") | 10 | 0 | 10/10 → exactly `text+image` |
| prose / conceptual | 15 | 2 | 13/15 → `text` only |
| **natural phrasing, no modality word** | **25** | **23** | fans out to everything |

The router is **exact on explicit cues and silent without them**. "What was the
insurance underwriting result?" is a table lookup with no table word in it, so
it fans out; the only two natural queries that scored did so on incidental
matches ("curve", "look like").

That is a deliberate position, not a defect to patch: this is a **precision
optimisation for explicit modality cues**, and `fallback_to_all` is what carries
natural language. The fallback is load-bearing — switch it off and 23 of those
25 queries route to text alone, making every figure- and table-borne fact asked
in ordinary English unreachable. More regex keywords would widen the case that
already works while making false positives likelier; closing the gap honestly
needs a semantic router, which stays an explicit future option.

One consequence for evaluation: a question written as *"which table shows X"*
hands the router its answer, so a query set must be stratified and the naturally
phrased cases reported separately, or the routing result measures the phrasing
of the questions rather than the retrieval architecture.

### Method 1 is frozen

Method 1's retrieval and generation code is unchanged by Method 2. Method 2 writes
to its own chunk variant and its own Qdrant collections, so both indexes coexist
and either can be rebuilt independently. `HybridRetriever` and Method 2's
`BM25Retriever`/`DenseRetriever` deliberately duplicate a little logic rather
than share a base class, because factoring them together would have meant
editing Method 1 after its numbers were recorded.

Two things did change Method 1, deliberately and on the record. Its
**configuration** enables the same cross-encoder as Method 2, so reranking is not
a second difference between them. And its **chunk set** moved with shared
ingestion — the table-degeneracy fix, OCR, and the corpus correction.
`tests/baselines/method1_chunks.txt` pins the current set, so any further
movement fails a test rather than passing unnoticed. The two Method 2 retrieval
fixes above were checked against this: `method1/rerank` is bit-identical across
every metric before and after them.

**See [`docs/architecture.md`](architecture.md)** for the full component
ownership table — what is shared, what belongs to each method, and the
asymmetries that Step 6 has to control for.

---

## Method 3 — Hybrid Visual RAG

Method 2's retrievers, unchanged, plus **ColQwen2 late-interaction retrieval over
the rendered pages** ingestion already produced. A late-interaction model keeps a
128-d vector per image patch; a query keeps one per token, and a page scores the
sum of each query token's best-matching patch. It can match a query word to an
axis label, a table cell or a diagram box with no text ever extracted from the
page — which is what Method 3 measures: the value of seeing the page.

**Status: built and evaluated.** The page index was built on a Colab Tesla T4 (951
pages); retrieval runs on CPU from the cached query embeddings. Results are in
[Retrieval results](#retrieval-results) and [Generation results](#generation-results).

### How it stays a controlled comparison

- **Same corpus, same pixels.** Pages are the 150-dpi renders under
  `data/processed/<doc>/pages/`, checked against `configs/corpus.lock.yaml`.
- **Method 2's retrieval, read-only.** Method 3 opens Method 2's chunk set, BM25
  and Qdrant indexes, router, resolver and cross-encoder, and never writes them.
  It writes only the visual page index, `data/indexes/visual_pages/`. That
  index belongs to the corpus, not to Method 3: it depends on the page renders,
  the corpus lock and the model, and on no chunk set.
- **A page hit becomes that page's chunks.** The gold set's evidence is
  `(doc_id, page, modality)` and the generator reads text, so a retrieved page
  contributes the Method 2 chunks on that exact page, in document order, each
  carrying the page's score. Provenance stays exact, nothing crosses a page, and
  evaluation, generation and judging need no special case. A page with no chunks
  — 10 of 951 — can be scored but contributes nothing, and is counted.
- **Same generator and judge.** No page images are attached, so answer-quality
  differences stay attributable to retrieval.

Two configuration differences from Method 2, both in `configs/method3.yaml`:

| setting | Method 2 | Method 3 | why |
|---|---|---|---|
| `fusion_weights.visual_page` | — | 1.0 | neutral; tuning it on the gold set would tune the comparison |
| `rerank_pool_per_modality` | 8 | 6 | four modalities at 8 overflow the pool of 25 and would cut the page signal to one candidate; 6 × 4 fits, and the cross-encoder still sees 25 |

The page signal fires on every query, as text does: it sees every modality on a
page at once. The cross-encoder still reads chunk text only, so the page signal
decides which chunks it considers, not their final order.

### Running it

**On Google Colab (recommended)** — no repository checkout needed there. Pack
one self-contained bundle, run the GPU runner on it, and unzip the result at the
repository root. Step by step in [`docs/m3_colab.md`](m3_colab.md), or
open `notebooks/m3_colab_gpu.ipynb`:

```bash
python scripts/pack_m3_colab.py        # -> dist/m3_colab_bundle.zip, upload to Drive
# on Colab: python m3_colab/run_m3_gpu.py all --out /content/m3_out --work-dir <Drive>/work
# back here: extract m3_visual_pages.zip at the repository root
```

The runner ships the repository's own encoder and index-store modules, so it
writes exactly the format `mmrag` reads. It checkpoints every page to Drive, so
a disconnected session resumes, and it runs a smoke test before the full
build. `tests/test_m3_colab_runner.py` packs a synthetic corpus, runs every
stage and opens the result with the main repository's validator.

**On a GPU machine with the full repository**, with `data/processed/` and
`data/indexes/method2/chunks.jsonl` copied over:

```bash
pip install -e ".[visual]"
mmrag index build --config method3 --device cuda --max-pages 4   # smoke test, partial
mmrag index build --config method3 --device cuda                 # full page index
mmrag index embed-queries --config method3 --device cuda         # gold + unanswerable queries
```

Either way, on the machine that ran Methods 1 and 2:

```bash
mmrag index status --config method3
mmrag eval run --config method3 --tag rerank
```

`--device` accepts `auto`, `cpu`, `mps`, `cuda` or `cuda:N`; an explicit device
that is not present is an error, never a silent fallback. A CPU build is refused
unless `--allow-cpu` is given. `mmrag eval run` refuses a partial index built
with `--doc-id` or `--max-pages`.

### What the index records

`data/indexes/visual_pages/index/index.json` records the model name, pinned
revision and resolved commit, device and dtype, library versions, processor
settings, image DPI, the corpus lockfile's SHA-256 and every document's source
hash, the hash of the chunk set pages will expand into, whether the build
covered the whole corpus, and SHA-256 checksums of the embedding, offset and
page files. Loading verifies all of it: an index whose files, model or corpus
have drifted is refused, and a complete index that lacks a page carrying a chunk
is refused when a retriever is opened over that chunk set. The index directory
is written aside and swapped in by rename, so an interrupted build leaves the
previous index intact. The query cache (`visual_pages/query_cache/`) is bound to
one model identity in the same way, and every Method 3 run records the index's
embedding checksum and model in its environment.

### Page → chunk expansion, and its known costs

A page hit contributes the chunks on that page, each carrying the page's score.
That is what keeps provenance, evaluation and generation unchanged, and it is
the design the first Method 3 run will measure. It has costs that are visible
before any number exists, and they are recorded here rather than tuned away on
the gold set:

- **The candidate budget is spent in chunks, not pages.** 50 candidates may be
  only a handful of pages when pages are dense, and the rerank floor of 6 for
  `visual_page` can be one page's chunks.
- **Chunks on one page tie on score but not on rank.** RRF reads rank, so a
  page's first chunk in reading order gets more credit than its last.
- **The cross-encoder reads chunk text only.** A page found by its pixels is
  judged by its extracted text, which is thinnest exactly where the page signal
  should help most.

The alternative, if the first run shows these matter, is to keep ranking at page
level through fusion (all of a page's chunks share its rank, and the floor is
counted in pages), with page images passed to a vision-capable generator as a
separate generation experiment. `VisualPageRetriever.rank_pages()` already
exposes the page ranking on its own, so that change is local to the retriever.

### Requirements

- **Model:** `vidore/colqwen2-v1.0` (Qwen2-VL-2B base plus adapter), roughly
  4.5 GB to download.
- **GPU:** 16 GB of VRAM is comfortable. `dtype: auto` uses bfloat16 when
  `torch.cuda.is_bf16_supported()` says so and float16 otherwise. If half
  precision produces NaNs, the build stops and says so; set
  `visual.dtype: float32`. Out-of-memory errors halve the batch size and retry.
- **The index the results use** was built on a Colab **Tesla T4** (torch
  2.11.0+cu128) in **bfloat16**. The T4 has no native bfloat16 support, but
  this torch build reported it as supported, so `dtype: auto` chose it. The
  build passed every check and was not rebuilt. The encoder rejects non-finite
  embeddings, and the runner verified all 951 pages and 50 cached queries. The
  main repository's loader checks file checksums, model identity, the corpus
  lock and chunk coverage, and `mmrag eval run` enforces it. The dtype is recorded
  in `data/indexes/visual_pages/index/index.json`; vectors are stored as float16
  regardless.
- **Index size:** every page keeps several hundred float16 vectors, on the order
  of 200 MB for the corpus.
- **CPU fallback:** query encoding on CPU works but needs ~9 GB of RAM; the query
  cache avoids it.

---

## Evaluation

```bash
mmrag eval validate                               # gold evidence resolves against the index
mmrag eval run -c method1 --tag rerank
mmrag eval run -c method2 --tag rerank
mmrag eval run -c method2 --no-metadata --tag rerank-nometa
mmrag eval compare data/eval/runs/*.json          # first file is the baseline
```

Retrieval only: no provider is called and nothing is generated, so a run is free,
offline and deterministic. Generation quality is evaluated separately, from these
same saved runs — see [Generation evaluation](#generation-evaluation).

**The gold set** (`data/eval/gold/v1.yaml`) is 42 hand-verified queries over all 14
documents. Evidence is `(doc_id, page, modality)` — not chunk ids, which differ
between the methods by construction, and not element ids, which shift whenever
detection changes. Every entry records how its page was verified.

Queries carry two separate labels: how the question is *phrased*
(text / table / figure / natural) and where the answer *lives*
(text / table / figure). `natural` questions name no modality, and matter most:
an explicitly phrased question hands the router its answer.

### Retrieval results

All methods rerank with the same cross-encoder. Macro-averaged over 42 queries:

| run | Recall@1 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Method 1 | 0.405 | 0.881 | 0.555 | 0.633 |
| **Method 2** | **0.405** | **0.929** | **0.571** | **0.656** |
| Method 2, `--no-metadata` | 0.405 | 0.929 | 0.569 | 0.655 |
| Method 3 | 0.405 | **0.952** | 0.563 | 0.655 |

Recall@10 by where the answer lives:

| run | text (n=15) | table (n=12) | figure (n=15) |
|---|---:|---:|---:|
| Method 1 | 1.000 | 1.000 | 0.667 |
| **Method 2** | 1.000 | 1.000 | **0.800** |
| Method 2, `--no-metadata` | 1.000 | 1.000 | 0.800 |
| Method 3 | 0.933 | 1.000 | **0.933** |

Latency per query, end to end:

| run | median | p90 |
|---|---:|---:|
| Method 1 | 18,368 ms | 24,788 ms |
| Method 2 | 17,887 ms | 18,539 ms |
| Method 3 | 18,130 ms | 19,239 ms |

**Method 2 leads on Recall@10, MRR and nDCG@10. Recall@1 is tied.** The whole
difference is on figures — text and table are both saturated — which is where the
modality-aware design was supposed to earn its keep.

Read these with the sample size in mind. 42 queries is enough to catch a defect,
not to establish significance: the figure slice is 15 queries, so the 0.667 →
0.800 gap is **two queries**. The `--no-metadata` arm now differs from Method 2
only in the third decimal place, which is the point of the resolver fix — it was
worth +0.167 Recall@10 as an ablation before the fix and costs nothing after.
Latency is dominated by the CPU cross-encoder, not by either method's retrieval.

**Method 3 has the highest Recall@10 (0.952) but not the best ranking.** It gained
two figure queries (q021, q037) and lost one text query (q002) against Method 2 — a
net of one query. All three changes sit at the top-10 boundary with identical
cross-encoder scores in both methods; what changed is which chunks reached the
25-candidate rerank pool. On q002, chunks from pages ColQwen2 ranked highly took
extra fusion votes and pushed the gold chunk out of the pool — the cost of spending
the candidate budget in chunks that [Page → chunk expansion](#page--chunk-expansion-and-its-known-costs)
describes. Visual retrieval adds about 440 ms per query (the page scoring itself
about 205 ms).

**These are not the first numbers this harness produced.** The first run had
Method 1 ahead, 0.857 to 0.595. Between that run and this one, three things were
fixed — and none was a change to the modality-aware design:

| fix | Method 2 Recall@10 |
|---|---:|
| first run | 0.595 |
| corpus: `arxiv_tapas` pointed at the wrong paper | 0.619 |
| document resolution: generic title words | 0.786 |
| within-modality fusion: best route instead of sum | **0.929** |

That is the reason the harness was built before Method 3. Pre-fix results are
kept in `data/eval/pre_corpus_fix_baseline.md` and are not comparable, since the
corpus has changed since.

---

## Generation evaluation

```bash
mmrag eval generate --retrieval-run data/eval/runs/<run>.json --dry-run \
    --price-in <usd-per-1M> --price-out <usd-per-1M>          # projected cost; calls nothing
mmrag eval generate --retrieval-run data/eval/runs/<run>.json
mmrag eval judge --generation-run data/eval/generation/<run>_generation.json
mmrag eval compare data/eval/generation/*_judged.json --markdown
```

Retrieval results say whether the evidence was *found*. Generation results say
whether the model then *answered correctly from it*. They are separate runs
with separate records, and neither is folded into the other.

**Answers are generated from saved retrieval runs, not by retrieving again.** The
exact chunks a retrieval run scored are rebuilt by id, so both numbers describe
the same retrieved list. A run whose chunk ids no longer exist in the index is
refused. The only exception is the 8 unanswerable questions, which appear in no
retrieval run: they are retrieved live with the run's own configuration, and
their lists are stored with the answers.

**The generation gold** (`data/eval/gold/generation_v1.yaml`) is a sidecar, so
`v1.yaml` and the retrieval numbers above stay untouched. It holds 105 short
atomic facts for the 42 queries, copied from the indexed corpus text, plus 8
unanswerable questions whose correct response is a refusal. Atomic facts mean
completeness is scored fact by fact and a correct paraphrase is not penalised
for its wording.

**The judge observes; Python scores.** The judge sees the question, the sources
exactly as the generator saw them, the answer and the reference facts — never
the method, retrieval scores or gold pages. It reports which facts are covered,
which claims the sources support, and whether the answer declines. Every score
is arithmetic over those observations:

- **correctness** — 1 if every fact is covered, 0.5 if only some are, 0 if none
  are or any is contradicted; refusals excluded.
- **completeness** — the fraction of facts covered; a refusal covers none.
- **grounded correct** — correct, *and* every claim supported by the sources.

Both generation and judging use `gpt-4o-mini` at temperature 0 with seed 42, and
every call goes through a content-addressed cache, so re-judging is free.

### Generation results

The four authoritative retrieval runs, each generated and judged over 42
answerable and 8 unanswerable questions. All 200 answers were generated and
judged with **zero generation errors, zero judge errors and zero repaired
verdicts**. These are **Generation V1** results; see
[Generation V1 and V2](#generation-v1-and-v2) for V2.1 and for the drift control
that showed these V1 numbers move by about ±3 questions when simply re-run.

From retrieval to answer, over the 42 answerable questions. Rates are quoted as
recorded in the judged run files, to four decimals:

| run | gold evidence in prompt | grounded correct, given evidence | grounded correct, end to end |
|---|---:|---:|---:|
| Method 1 | 37/42 (0.8810) | 17/37 (0.4595) | 17/42 (0.4048) |
| Method 2 | 39/42 (0.9286) | 19/39 (0.4872) | 19/42 (0.4524) |
| **Method 2, `--no-metadata`** | 39/42 (0.9286) | **22/39 (0.5641)** | **22/42 (0.5238)** |
| Method 3 | 40/42 (0.9524) | 20/40 (0.5000) | 20/42 (0.4762) |

"Gold evidence in prompt" equals Recall@10 from the retrieval results: no gold
evidence was lost to the context budget in any run. No answer was judged
grounded correct without its gold evidence in the prompt.

| run | correctness | completeness | correct / partial / incorrect / refused |
|---|---:|---:|---:|
| Method 1 | 0.6282 | 0.5905 | 17 / 15 / 7 / 3 |
| Method 2 | 0.6795 | 0.6341 | 19 / 15 / 5 / 3 |
| **Method 2, `--no-metadata`** | **0.7179** | **0.6778** | 22 / 12 / 5 / 3 |
| Method 3 | 0.6923 | 0.6460 | 21 / 12 / 6 / 3 |

Grounded correct by where the answer lives:

| run | text (n=15) | table (n=12) | figure (n=15) |
|---|---:|---:|---:|
| Method 1 | 8 | 7 | 2 |
| Method 2 | 8 | 7 | 4 |
| Method 2, `--no-metadata` | 11 | 7 | 4 |
| Method 3 | 9 | 7 | 4 |

Refusal behaviour:

| run | unanswerable refused | answerable refused | refused despite sufficient context |
|---|---:|---:|---:|
| Method 1 | 8/8 | 3/42 | 3 |
| Method 2 | 8/8 | 3/42 | 3 |
| Method 2, `--no-metadata` | 8/8 | 3/42 | 2 |
| Method 3 | 8/8 | 3/42 | 3 |

"Sufficient context" is the judge's reading of the sources. No run answered from
context the judge considered insufficient.

**Method 2 improves on Method 1, and in this run Method 2 without the
MetadataResolver does best.** The first gain follows retrieval: Method 2 put gold
evidence in front of the model for two more questions, and doubled grounded
correct answers on figure questions (2 → 4).

The second gain runs opposite to retrieval, where the two Method 2 arms are tied
at 0.929 Recall@10. On three questions the resolver narrowed retrieval to one
document — the IPCC report (q007), TAPAS (q008) and the WHO report (q009). The
gold page still reached the prompt each time, but the rest of the context
changed, and the answers were judged less complete than the unfiltered arm's.
Scoping a query to the document it names can find the right page while starving
the answer of context. Three questions are too few to act on, but they are
exactly the ones worth checking.

**These differences are directional, not significant.** There are 42 answerable
questions, and slices hold 8 to 15: the whole spread between the best and worst
run is five questions end to end.

**Method 3's better retrieval did not produce better V1 answers.** It put gold
evidence in the prompt for one more question than Method 2 and got the most gold
figures in front of the model (14/15), yet grounded-correct figure answers stayed
at 4: the V1 generator reads extracted text only and answered figure questions by
naming the figure rather than describing it.

### Limitations

- **Faithfulness is not measured meaningfully.** The judge is the same model as
  the generator, and it found no unsupported claim in any of the 117 non-refused
  answerable answers of Methods 1 and 2 — faithfulness 1.00 and hallucination 0.
  Method 3's two flagged answers (q020, q040) contain claims the sources do state;
  both look like judge false positives.
  That is self-judge leniency, not evidence of perfect grounding, and those two
  columns are left out above. Every absolute score here is directional; the
  comparison between runs is the sturdier signal, since all arms share one
  generator and one judge.
- **Figure questions are answered from text alone.** Methods 1 and 2 retrieve
  figures but send the generator only their captions and OCR text, never the
  image. Questions phrased as "which figure shows…" score 0 of 9 grounded correct
  in every run, because their reference facts describe what the figure depicts.
  That gap is the one Method 3 exists to measure.
- **Six generation-gold entries are still marked NEEDS REVIEW** — q005, q006,
  q032, q037, q038 and q039. For each, the facts could not be taken cleanly from
  the v1 evidence page — usually because that page does not carry the whole
  answer. They are recorded in the file rather than corrected, because `v1.yaml`
  is frozen, and the same entries apply to every arm.
- **"Correct" is strict.** It needs every reference fact, so most failures are
  partial answers rather than wrong ones: of Method 2's 20 answers that were
  neither correct nor refused, 15 are partial.

**Cost.** Generation and judging for all three runs made 216 provider calls — 108
for each stage. Another 84 were served from the cache, since the Method 2 arms
retrieve identically for most questions. At $0.15 / $0.60 per million
input / output tokens, the recorded usage of the calls actually made comes to
about **$0.14** in total: roughly $0.06 generating and $0.08 judging.

---

## Generation V1 and V2

Every result in [Generation results](#generation-results) is **Generation V1**. Those
runs showed that generation, not retrieval, had become the bottleneck, so further
pipelines sit beside V1 rather than replacing it. V1 is still the default, and its
prompt, prompt version (`1ae772d0aa197ff5`), cache keys and artefacts are unchanged;
`tests/test_generation_v1_golden.py` fails if any of them move.

```
V1     retrieved chunks → prompt ([n] title - page N (type), chunk text) → 1 LLM call → citations
V2.x   retrieved chunks → EvidencePack → 1 LLM call → citations → AnswerValidator → answer + report
```

### What V2 was built to fix

An audit of the V1 judged runs found two dominant failures, neither of them retrieval:

- **Answers that stop at identification.** "Table 3 reports the BLEU scores [6]" is
  a correct sentence and a partial answer: the missing fact was on the page it
  cites. In every arm, most failed answers had *all* their missing facts in the
  sources the model was given. Partial answers had a median of about 19 output
  tokens, against 56–84 for correct ones. V1's instruction to "be concise" and not to
  "describe the sources" works against questions about tables and figures.
- **All-or-nothing refusal.** Questions whose sources held part of the answer were
  refused outright, because V1 has no way to answer partially.

### The pipeline

| stage | module | what it does | model call |
|---|---|---|---|
| EvidencePack | `generation/evidence.py` | Admits sources in retrieval order under V1's 6,000-token budget and tail-drop rule, so source numbers and citation provenance are V1's. Shows them grouped by document page, one header per source — `[n] <title> · page N · <modality> · <section>` — with the chunking breadcrumb removed and table/figure captions labelled. | none |
| AnswererV2 | `generation/answerer_v2.py` | One call with V1's model, temperature, seed and output cap. Answer directly; give the values, units, conditions and scope the sources state; for "which table/figure/section" questions, name it and say what it shows; answer the supported part of a partially covered question; cite every factual sentence; no prior knowledge; exact numbers and names; usually 2–6 sentences. Plain text with `[n]` markers, resolved by V1's own `resolve_citations`. | 1 |
| AnswerValidator | `generation/validation.py` | Sentence-level citation coverage, unresolved citations, whether each number or identifier appears in a source the sentence cites, and mixed refusal. Returns a report; never edits the answer, never retries. | none |

No extra model calls, no synthesis step, no judge in the loop. Across all four arms
V2 admitted exactly V1's sources for all 200 prompts, with nothing dropped for budget.

### V2.0 → V2.1

The first V2 prompt (**V2.0**, `55aea920ebb01b97`) was generated for all four arms
and inspected before judging. It had two defects, and it was not judged:

- **Wrong-entity answers.** Its rule to refuse "only if no source contains
  information relevant to the question" let the model treat a similar entity as
  partial evidence. u003 asks about the RP2350, which the corpus never mentions;
  V1 refused, but V2.0 answered from the RP2040 datasheet in every arm, so
  unanswerable refusals fell to 7/8.
- **Paragraph-level citations.** 88–90% of answers had at least one uncited
  factual sentence; mean sentence citation coverage was 0.56–0.61.

**V2.1** (`c9830e3efb5505f1`) changes only the prompt. It limits answers, and
partial answers, to evidence about the subject the question names, refuses when the
sources describe a different entity, version or year, and says that each factual
sentence needs its own citation. The rules are generic — no product or query is
named. V2.0's prompt and artefacts are kept unchanged as a record.

### Results

V2.1 was generated from the same four frozen retrieval runs and judged with the
unchanged V1 judge protocol. All 400 generation and judge records completed with no
errors and no repaired verdicts.

**V1 re-run as a drift control.** Before comparing, V1 was regenerated and re-judged
the same day for Methods 1, 2 and 3 with the cache bypassed and byte-identical
prompts. Only 32–33 of 50 answers came back word for word: OpenAI's backend for
`gpt-4o-mini` had changed (system fingerprints `fp_d48d865f86` → `fp_b5bcb8c07e`),
and seed 42 does not pin output across backends. Grounded correctness moved from 17
→ 15, 19 → 22 and 20 → 20 questions, with 5–6 per-query flips per arm — 1–2 of them
on answers that had not changed at all, i.e. judge variance alone. **Treat about ±3
questions out of 42 as run-to-run noise.** The V2.1 comparison below is therefore
against the same-day V1 runs, not the 13 September ones.

Over the 42 answerable questions:

| | Method 1: V1 today → V2.1 | Method 2: V1 today → V2.1 | Method 3: V1 today → V2.1 |
|---|---|---|---|
| Gold evidence in prompt | 37/42 → 37/42 | 39/42 → 39/42 | 40/42 → 40/42 |
| Grounded correct, given evidence | 14/37 → **25/37** | 22/39 → 23/39 | 19/40 → **25/40** |
| **Grounded correct, end to end** | 0.3571 → **0.6190** | 0.5238 → 0.5714 | 0.4762 → **0.6190** |
| Correctness | 0.6154 → 0.7439 | 0.7179 → 0.7195 | 0.6625 → 0.7500 |
| Completeness | 0.5786 → 0.7520 | 0.6778 → 0.7274 | 0.6302 → 0.7829 |
| Correct / partial / incorrect / refused | 17/14/8/3 → 27/7/7/1 | 22/12/5/3 → 24/11/6/1 | 20/13/7/2 → 26/11/5/0 |
| Refused despite sufficient context | 3 → 1 | 3 → 1 | 2 → 0 |
| Grounded correct: text / table / figure | 8/4/3 → 11/10/5 | 11/6/5 → 9/8/7 | 8/6/6 → 9/9/8 |
| Answers with an unsupported claim | 2/39 → 1/41 | 0/39 → 0/41 | 0/40 → 0/42 |
| Uncited-claim rate | 0.0075 → 0.0374 | 0.0051 → 0.0577 | 0.0154 → 0.0556 |
| Unanswerable refused | 8/8 → 8/8 | 8/8 → 8/8 | 8/8 → 8/8 |

Method 2 without the metadata resolver has no same-day V1 control. Under V2.1 it
reached 25/42 grounded correct (0.5952), correctness 0.7262, completeness 0.7472,
0 refusals of answerable questions and 8/8 unanswerable refused.

**What holds up:**

- **Methods 1 and 3 improve beyond the noise.** +11 and +6 grounded-correct
  answers against same-day V1, with completeness up 0.17 and 0.15. Partial answers
  became complete ones (Method 1: 14 partial → 7).
- **Method 2 does not.** Its same-day V1 happened to score well, and V2.1's +2 is
  within the measured noise.
- **Refusals behave.** Over-refusals fall to 0–1 in every arm, every unanswerable
  question is still refused, and no arm answered from context the judge rated
  insufficient.
- **Figures improve.** Grounded-correct figure answers rise in every arm, and
  "which figure shows…" questions — 0 of 9 in V1's recorded runs, 1 for Method 3 —
  reach 2–5 of 9.
- **The method ranking changes.** On same-day V1, Method 2 leads (22 vs 20 and 15).
  Under V2.1, Methods 1 and 3 tie at 26 and Method 2 has 24. A weaker generator was
  hiding how usable Method 1's evidence already was.

**Caveats — read these before quoting any number above:**

- **The citation fix only partly worked.** V2.1's uncited-claim rate is 3.6–11×
  same-day V1's: answers make about four claims each and still cite per paragraph.
  Sentence citation coverage rose only from 0.56–0.61 (V2.0) to 0.63–0.65, below the
  0.75 the same validator measures on V1's Method 2 answers. Grounded correctness does not penalise an uncited but supported claim, so
  the headline numbers do not show this cost.
- **Per-query churn is large.** Against the recorded V1 runs, V2.1 gained 10–11
  grounded-correct answers per arm and lost 2–7; q030 and q039 were lost in three or
  four arms, usually by dropping one required fact from a longer answer.
- **The design saw the test set.** V2's rules came from V1's failures on these 42
  questions, and V2.1's fixes from V2.0's outputs on these same 50. Gains here are
  optimistic until measured on questions the design never saw.
- **Self-judging.** The judge is the same model as the generator. The comparison
  between runs is sturdier than any absolute score, and differences of a few
  questions are directional, not significant.
- **One leak in every arm.** q036 is judged grounded correct without its gold page in
  the prompt: the answer is on a page the gold set does not list.
- **Lexical fact coverage fell from V2.0 to V2.1** by 4–8 facts per arm, while staying
  well above V1. Asking the model to omit details it cannot cite probably made some
  answers more conservative.

**Cost.** V2.1 generation and judging for four arms used about 740k generation and
795k judge input tokens — roughly **$0.28** at $0.15 / $0.60 per million tokens, an
upper bound since cached records are included.

### Selecting a pipeline

`generation.pipeline: v1 | v2 | v2.1` in a config, or per command:

```bash
mmrag query "Which table reports BLEU for the model variations?" -c method2 --pipeline v2.1
mmrag eval generate --retrieval-run data/eval/runs/<run>.json --pipeline v2.1
```

Generation reuses a saved retrieval run exactly as V1 does, so every method is
compared under the same generator. Each pipeline is cached under its own prompt
version, so answers can never replay across pipelines, and runs are labelled
`<run>+genv2` or `<run>+genv2.1`. The judge, its prompt and every score are
unchanged; the judged report adds claims per answer, answer length and the
validator's results beside correctness.

---

## Repository layout

```
configs/            default.yaml + method1/2/3.yaml, chained via `extends`
                    corpus.yaml (hand-authored) + corpus.lock.yaml (generated)
scripts/sql/        Postgres schema, applied on first container start
src/mmrag/
  config.py         Settings (env) vs ExperimentConfig (YAML) — kept apart
  schemas.py        Document → Page → Element → Chunk, plus retrieval types
  corpus/           manifest model + verifying downloader
  ingestion/        PDF → Document/Page/Element, with provenance
    parser.py         two-pass PyMuPDF parser
    layout.py         reading order, headers/footers, section tracking
    tables.py         table structuring + Markdown flattening
    figures.py        raster + vector figure detection, caption matching
    classify.py       figure/table type and confidence heuristics
    metadata.py       document metadata merge (manifest > PDF > heuristic)
  textify/          modality → text flattening + chunking
    flatten.py        what to index, and what the flattening lost
    chunker.py        elements → retrieval units, provenance preserved
    tokens.py         token counting + sentence segmentation
  embeddings/       local text / image (CLIP) / visual (ColQwen2) embedders
  stores/           storage adapters: Postgres, Qdrant, BM25,
    multivector.py    file-based multi-vector page store, MaxSim, query cache
  indexing/         index components: build from the corpus, open retrievers
    modality.py       chunk set + text/table/figure sub-indexes
    visual_pages.py   ColQwen2 page index: GPU build, validation, query cache
  retrieval/        the Retriever protocol and everything the engine composes
    base.py           Retriever, RetrieverOutput, MetadataFilter
    modality_retrievers.py  bm25, dense, table, image retrievers
    visual_page.py    visual_page retriever (page ranking -> same-page chunks)
    router.py         query router, with always-on modalities
    modality.py       route -> fan out -> two-stage fusion -> floored rerank pool
    fusion.py, rerank.py, metadata.py, views.py
    hybrid.py         Method 1's frozen BM25 + dense retriever
  engine.py         RAGEngine: resolve -> retrieve -> fuse -> rerank -> answer
  generation/       providers (openai | local | echo) and the two answer pipelines
    answerer.py       Generation V1 (frozen, default)
    evidence.py       V2: deterministic evidence pack
    answerer_v2.py    V2: answer contract, one call
    validation.py     V2: deterministic claim/citation checks
    pipeline.py       generation.pipeline -> answerer, prompt version
  methods/          benchmark configurations of the system
    base.py           RAGMethod contract, EngineMethod
    registry.py       config.method -> method class
    method1_textified.py   (frozen)
    method2_modality.py    engine over ModalityIndex("method2")
    method3_visual.py      same, read-only, plus the visual page index
  evaluation/       gold sets, retrieval metrics and runner, generation runner,
                    LLM judge, response cache, comparison reports
docs/architecture.md  component ownership, and what is shared vs method-specific
data/eval/gold/     versioned, hand-verified gold sets: retrieval + generation (committed)
data/eval/runs/     retrieval run records (gitignored)
data/eval/generation/  generation and judged run records (gitignored)
data/eval/cache/    content-addressed LLM response cache (gitignored)
tests/baselines/    frozen Method 1 chunk set
notebooks/          unused -- Method 3 GPU indexing runs from the CLI
```

---

## Hardware notes

Everything implemented so far runs on CPU. `bge-small-en-v1.5` (384-d) is the
default text embedder precisely because it is usable without a GPU. The one CPU
cost that matters is the cross-encoder, which dominates query latency (see
[Retrieval results](#retrieval-results)).

Method 3 is the exception, and only for encoding. Building its page index runs
ColQwen2 over every page, which is impractical on CPU, so it is done on a GPU
machine and the index is copied back. Scoring pages is plain numpy, and query
embeddings can be precomputed on the same GPU machine, so Method 3's retrieval
benchmark runs on the same CPU, Qdrant and cross-encoder as Methods 1 and 2. See
[Method 3](#method-3--hybrid-visual-rag) for the commands and requirements.

---

## Licence

MIT for the code in this repository. The corpus documents remain under their own
licences, recorded per entry in `configs/corpus.yaml`; none are redistributed here.
