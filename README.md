# Multimodal RAG Benchmark

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
| **Figures** | → caption + OCR + VLM description | CLIP image embeddings | CLIP + ColQwen2 page vectors |
| **Fusion** | weighted RRF over 2 retrievers | weighted RRF + cross-encoder rerank | weighted RRF + rerank |
| **Generation** | text LLM | text LLM | **VLM, with original page images attached** |
| **Measures** | the cost of flattening | the value of preserving modality | the value of seeing the page |

---

## Status

This project is being built in stages. Current state:

- [x] **Step 0** — Repo scaffold, configuration system, core data model, Docker infra
- [x] **Step 1** — Pinned corpus manifest/lockfile + verifying downloader
- [x] **Step 2** — Shared ingestion: PDF → elements with rich metadata and provenance
- [x] **Step 3** — Method 1: Textified hybrid RAG
- [x] **Step 4** — Method 2: Modality-aware retrieval + query router
- [ ] **Step 5** — Method 3: Hybrid visual RAG (ColQwen2)
- [ ] **Step 6** — Evaluation harness and comparison report

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

**1,623 pages across 14 documents** (954 after the `page_limit` caps).

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

**954 pages → 16,390 elements**, mean extraction confidence **0.94**, 0.4% below 0.5:

| | count | | count |
|---|---:|---|---:|
| text | 12,666 | tables | 355 |
| headings/titles | 1,288 | charts | 196 |
| captions | 687 | figures | 189 |
| headers/footers | 994 | diagrams | 15 |

Table types skew financial (148) and register (100); figure types are chart (196),
unknown (147), photo (42), diagram (15). **`unknown` is a deliberate answer** —
Method 2 routes on `figure_type`, so a confidently wrong label misdirects
retrieval while an honest abstention merely fails to help.

Parsing runs in two passes per document, and the structure is load-bearing: a
single page cannot distinguish a running header from a section heading, nor a
14pt heading from 14pt body text. Both are only answerable once the whole
document's typography has been observed.

Output goes to Postgres *and* to a JSON sidecar per document under
`data/processed/`. The sidecar exists because Method 3's visual index is built
on a Colab GPU with no access to a local database, and because debugging a parse
should not require SQL. Both are written from the same objects, so they cannot drift.

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

Building the index over the full corpus produces **2,818 chunks** (2,202 text,
363 table, 253 figure) from 954 pages — and reports the headline measurement:

> **147 figures have no retrievable text at all.**

Those are charts and diagrams where nothing textual was ever extracted — no
caption, no OCR, no VLM description. They exist in the corpus and are
*structurally unreachable* for this architecture. That number is the mechanism
behind any deficit Method 3 later makes up, and it is why `FlattenReport` counts
and attributes losses instead of silently dropping them.

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

Method 1 reports **147 of 400 figures with no retrievable text**: no caption, no
OCR, no description. No text index can reach them under any query.

**All 147 have a cropped image on disk.** CLIP puts images and text in one
space, so a text query scores directly against the pixels with no textual
intermediary. That is the concrete mechanism behind any advantage Method 2 shows
on figure questions, and the index build reports it as
`text_invisible_recoverable`.

### Two views of a table, not one

Flattened to Markdown, a table's column headers are three tokens among hundreds
of digits — so "which table breaks revenue down by segment" has almost nothing
to match. Method 2 indexes tables twice:

- **schema view** — caption, column headers, table type, shape. Short and
  semantic; dense retrieval handles it.
- **content view** — the cells. Long and literal; BM25 handles it.

Both point at the same `chunk_id`, so provenance is unchanged.

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

Method 2 adds no changes to Method 1's retrieval or generation. It writes to its
own chunk variant and its own Qdrant collections, so both indexes coexist and
either can be rebuilt independently. `HybridRetriever` and Method 2's
`BM25Retriever`/`DenseRetriever` deliberately duplicate a little logic rather
than share a base class, because factoring them together would have meant
editing Method 1 after its numbers were recorded.

**See [`docs/architecture.md`](docs/architecture.md)** for the full component
ownership table — what is shared, what belongs to each method, and the
asymmetries that Step 6 has to control for.

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
  embeddings/       local text / image / visual embedders
  stores/           Postgres + Qdrant + BM25 index adapters
  retrieval/        RRF fusion + reranking (shared)
    hybrid.py         Method 1's BM25 + dense retriever
    router.py         Method 2's query router
    modality.py       Method 2's route/fan-out/fuse orchestrator
    views.py          table schema vs content views
    metadata.py       Postgres -> doc_id filters
docs/architecture.md  which components are shared vs method-specific
  generation/       provider interface (openai | local | echo) + answerer
  methods/          the three end-to-end pipelines
    method1_textified.py   (frozen)
    method2_modality.py
  evaluation/       metrics, gold set, comparison report    (Step 6)
notebooks/          Colab GPU notebook for Method 3 visual indexing
```

---

## Hardware notes

Everything except Method 3's visual index runs on CPU. `bge-small-en-v1.5` (384-d)
is the default text embedder precisely because it is usable without a GPU.

ColQwen2 page embedding is the exception — on CPU it is roughly 1–5 s/page, which
is impractical for a corpus of this size. `notebooks/` therefore contains a
Colab/Kaggle notebook that builds the visual index on a free GPU and exports it
for local use. The rest of Method 3 (fusion, reranking, generation) runs locally
against that exported index.

---

## Licence

MIT for the code in this repository. The corpus documents remain under their own
licences, recorded per entry in `configs/corpus.yaml`; none are redistributed here.
