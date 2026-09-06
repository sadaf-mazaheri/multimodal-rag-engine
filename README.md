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
- [ ] **Step 3** — Method 1: Textified hybrid RAG
- [ ] **Step 4** — Method 2: Modality-aware retrieval + query router
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

Optional extras: `ocr` (needs the Tesseract binary on PATH), `visual`
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
  textify/          modality → text flattening              (Step 3)
  embeddings/       local text / image / visual embedders
  stores/           Postgres + Qdrant + BM25 index adapters
  retrieval/        retrievers, RRF fusion, reranking, query router
  generation/       provider interface (openai | local | echo) + answerer
  methods/          the three end-to-end pipelines
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
