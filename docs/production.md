# Production evaluation

How the three methods behave as a service: latency, tokens, cost, reliability,
failure modes, and behaviour under concurrent load. This sits beside the
retrieval and generation *quality* evaluation in
[`design_and_results.md`](design_and_results.md) and never changes it: no
retrieval, generation, prompt, model or evaluation-schema change was made for
either phase.

Two phases, deliberately different in what they claim:

| | Phase A — offline production metrics | Phase B — replay load test |
|---|---|---|
| Command | `mmrag prod report`, `mmrag prod compare` | `mmrag prod load` |
| Code | `src/mmrag/production/` (commit `f993ee6`) | `production/load.py`, `production/resources.py` (commit `f240cd5`) |
| End-to-end latency | **Composed** from components measured separately | **Measured** in one process, around `method.answer(...)` |
| Provider calls | none — reads recorded runs | none in replay mode — recorded answers and latencies |
| Retrieval | not re-run — recorded timings | re-run live, under concurrency |
| What it cannot show | behaviour under load | HTTP overhead, provider-side concurrency or rate limits |

Neither phase is an HTTP benchmark, and neither makes this a production-ready
deployment. They measure the pipeline's own costs and bottlenecks on the
development machine.

---

## Phase A — offline production metrics

### Method

Every number comes from existing artefacts — a retrieval run, the generation run
made from it, and optionally the judged run of that generation run — plus two
deterministic local steps re-executed offline. Each latency component carries a
provenance label:

| component | provenance | source |
|---|---|---|
| retrieval | recorded | retrieval run: routing, query embedding and every retriever |
| fusion | recorded | retrieval run `fusion_ms` |
| reranking | recorded | retrieval run `rerank_ms` |
| retrieval_total | recorded | retrieval run `total_ms` |
| prompt_construction | measured | re-built offline with the frozen answerer (median of 5) |
| generation | recorded | generation run provider latency, **cache misses only** |
| postprocess | measured | citation resolution, refusal checks and `AnswerValidator` over the recorded answer |
| `composed_e2e_latency` | composed | retrieval_total + prompt_construction + generation + postprocess |

**`composed_e2e_latency` was not measured in a single serving process.**
Retrieval and generation were recorded in different runs, and prompt construction
and post-processing were re-measured offline. It excludes one-off model loading,
queueing and any network overhead outside the provider call.

**The re-measured prompts are the evaluated prompts.** A prompt-construction
timing is kept only if the rebuilt prompt's sha256 equals the one the generation
run recorded. All 42 answerable prompts matched in all seven reports.

Other rules:

- **Percentiles** are linearly interpolated and always reported with `n`. Below
  100 samples, p95/p99 sit between the largest few observations and are not
  stable tail estimates.
- **Tokens** are provider-reported usage from the generation run. **Cost** is that
  usage priced at the project's configured evaluation rates in
  `configs/pricing.yaml` (gpt-4o-mini, $0.15 / $0.60 per 1M input / output tokens,
  `as_of` 2026-09-15). It is an estimate, not a billing statement, and not a
  claim about current vendor pricing. Judge spend is reported separately as
  evaluation overhead, never as serving cost.
- **Errors, retries and timeouts** come from each record's status, attempts and
  error type. Retries inside the provider SDK (connection errors, 429, 5xx) are
  **not observable** from offline artefacts; they would appear only as latency.
- **Only answerable queries present in the retrieval run** have retrieval timings
  (n = 42). The 8 unanswerable queries were retrieved live during generation,
  untimed. Cache-hit generation records are excluded from latency, so V2.1
  Method 2 has n = 35 composed samples and the Method 2 no-metadata arm n = 11.
- **Known gaps:** Method 2/3 metadata resolution runs before the retrieval timer
  starts and is in no recorded stage; Method 3's `visual_page_ms` used
  precomputed ColQwen2 query embeddings, so it covers page scoring only.

### Failure decomposition

Each answerable query lands in exactly one bucket, by fixed precedence, computed
from fields already in the judged run (nothing is re-judged):

1. **retrieval failure** — gold evidence not in the top 10;
2. **context failure** — retrieved but dropped by the context budget;
3. **generation failure** — evidence in the prompt, answer not grounded correct;
4. **citation failure** — grounded correct, with an unresolved, missing or
   unsupporting citation;
5. **success**.

Generation or judge errors are a separate `error` bucket, so an outage is never
read as a quality failure. Two non-exclusive counts sit beside the buckets:
citation problems on any answer, and retrieval failures that were nonetheless
grounded correct (q036 in six of seven reports — its answer is on a page the gold
set does not list).

### Results

Retrieval timings are identical for V1 and V2.1 because both were generated from
the same frozen retrieval runs. V1 uses the same-day `--no-cache` control runs,
so every V1 generation latency is a real call made in that run.

| run | pipeline | e2e n | composed e2e p50 / p95 / p99 (s) | retrieval_total p50 (s) | rerank p50 (s) | prompt p50 (ms) | generation p50 (s) | postprocess p50 (ms) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Method 1 | V1 | 42 | 20.440 / 26.826 / 28.722 | 18.31 | 18.23 | 1.5 | 1.66 | 0.4 |
| Method 2 | V1 | 42 | 19.542 / 21.096 / 22.363 | 17.87 | 17.62 | 1.8 | 1.56 | 0.5 |
| Method 3 | V1 | 42 | 19.925 / 29.882 / 42.061 | 18.12 | 17.45 | 1.8 | 1.59 | 0.5 |
| Method 1 | V2.1 | 42 | 20.351 / 27.142 / 29.837 | 18.31 | 18.23 | 2.2 | 1.91 | 0.7 |
| Method 2 | V2.1 | 35 | 19.778 / 21.908 / 23.233 | 17.87 | 17.62 | 2.5 | 1.77 | 0.9 |
| Method 2, no-metadata | V2.1 | 11 | 20.890 / 22.756 / 22.896 | 18.11 | 17.87 | 2.3 | 2.32 | 0.9 |
| Method 3 | V2.1 | 35 | 20.446 / 22.209 / 26.731 | 18.12 | 17.45 | 2.3 | 1.89 | 0.9 |

Method 3's V1 p95/p99 come from a few slow provider calls in that run (up to
23.6 s), not from retrieval. The no-metadata arm has too few uncached samples to
compare.

| run | pipeline | tokens / query, prompt / completion (mean) | $ / 1K queries | errors / retries / timeouts | failures: retrieval / context / generation / citation / success |
|---|---|---|---:|---|---|
| Method 1 | V1 | 3,494 / 56 | 0.558 | 0 / 0 / 0 of 50 | 5 / 0 / 23 / 0 / 14 |
| Method 2 | V1 | 3,545 / 57 | 0.566 | 0 / 0 / 0 of 50 | 3 / 0 / 17 / 0 / 22 |
| Method 3 | V1 | 3,491 / 56 | 0.557 | 0 / 0 / 0 of 50 | 2 / 0 / 21 / 0 / 19 |
| Method 1 | V2.1 | 3,657 / 108 | 0.613 | 0 / 0 / 0 of 50 | 5 / 0 / 12 / 0 / 25 |
| Method 2 | V2.1 | 3,708 / 108 | 0.621 | 0 / 0 / 0 of 50 | 3 / 0 / 16 / 0 / 23 |
| Method 2, no-metadata | V2.1 | 3,775 / 107 | 0.630 | 0 / 0 / 0 of 50 | 3 / 0 / 15 / 0 / 24 |
| Method 3 | V2.1 | 3,654 / 107 | 0.612 | 0 / 0 / 0 of 50 | 2 / 0 / 15 / 0 / 25 |

Judge spend per run was about $0.037, reported as evaluation overhead.

**What Phase A shows:**

- **CPU cross-encoder reranking is about 85–90% of median composed latency**
  (17.5–18.2 s of 19.5–20.9 s). Prompt construction takes about 2 ms and
  post-processing under 1 ms.
- **V2.1 costs roughly 10% more per query than V1** ($0.612–0.621 vs
  $0.557–0.566 per 1K). Answers are longer (about 107–108 vs 56–57 completion
  tokens on average) and prompts about 5% longer; at these rates the two account
  for roughly 56% and 44% of the added cost.
- **No context failures in any run**: gold evidence that was retrieved always
  reached the prompt.
- **No citation failures in any run**, as judged. This bucket depends on the
  judge; the deterministic V2.1 validator does flag uncited sentences, which are
  not counted as citation failures (see the V2.1 caveats in
  [`design_and_results.md`](design_and_results.md#results)).
- **Zero errors, retries and timeouts** — out of only 50 requests per run, with
  SDK-internal retries unobservable.

---

## Phase B — replay load test

### Method

`mmrag prod load` drives each method's own serving path,
`method.answer(query, provider, top_k=10)`, with several requests in flight at
once in one process. Retrieval, fusion, reranking, prompt construction and
post-processing run exactly as everywhere else; no RAG logic is duplicated. Only
the provider is replaced.

- **Replay provider.** It returns the answer a recorded generation run got for
  the same prompt — matched by the sha256 of the messages — after sleeping for
  that call's recorded provider latency. Requests are labelled
  `generation_source: replay`. Defaults replay the same-day V1 `--no-cache`
  generation runs, whose latencies are all real calls from that run. **This
  reproduces recorded per-prompt latency only: provider-side concurrency,
  queueing and rate limits are not measured.**
- **Method rebuilt from the recorded configuration.** The config comes from the
  retrieval run the replay generation run was made from, so the served pipeline
  is the evaluated one.
- **Warm-up.** Each method first serves 2 sequential requests, which absorb lazy
  model loading and are excluded from every level's statistics.
- **Closed loop.** For each concurrency level (1, 5, 10), the same 20 answerable
  queries (q001–q020) are submitted at once to a thread pool of that size sharing
  one method instance, with a 5 s cooldown between levels.
- **Per request:** submitted / started / finished timestamps, status, error and
  error kind, stage timings, tokens, citation count and the replay match kind.
- **Per level:** throughput (successful requests / level wall time),
  p50/p95/p99, error rate, and CPU/RAM sampled every 0.5 s (`psutil`, optional).
  GPU is recorded when available; this machine had none.

Latency definitions:

- **`e2e_latency_ms`** — measured wall-clock time around `method.answer(...)`,
  from a worker starting the request to it returning. It includes metadata
  resolution, routing, retrieval, fusion, reranking, prompt construction, the
  replayed provider call and post-processing. There is no HTTP or serialisation
  layer.
- **`queue_wait_ms`** — submission to start, reported separately and never
  folded into `e2e_latency_ms`.
- **`untimed_overhead_ms`** — `e2e_latency_ms − retrieval_total − generation_ms`,
  a **measured residual**: everything outside the pipeline's own timers
  (metadata resolution, prompt construction, post-processing, answerer
  construction, thread scheduling). It is not a direct measurement of prompt
  construction or post-processing.

Live mode exists (`--mode live`) but refuses to run without both
`--allow-live-calls` and `--max-live-requests`, and prints a projected cost
first. It was not run.

### Run

```bash
mmrag prod load --methods method1,method2,method3 --concurrency 1,5,10 \
    --limit 20 --warmup 2 --cooldown-s 5
```

Artefact: `data/eval/production/load/20260915T202437+0000_method1+method2+method3_replay_load.json`
(gitignored). Windows 11, Python 3.10.11, CPU-only PyTorch, 16 logical / 12
physical cores, 15.7 GB RAM; Qdrant and Postgres in Docker on the same machine.

**Validity checks on the artefact:**

- **180 requests, 180 OK**: 0 errors, hence 0 timeouts. Replay makes no provider
  calls, so there were no provider retries to count.
- **180/180 prompts matched the recorded run by sha256**, with 0 fallbacks:
  retrieval stayed deterministic under concurrency and every prompt was
  byte-identical to the evaluated V1 prompt.
- **Workers really ran concurrently**: the sum of per-request latency over level
  wall time is 1.00 / 4.6–4.8 / 8.7–9.3 at c = 1 / 5 / 10.
- **Wall-clock accounting closes**: the residual p50 is 8–21 ms at every level.
- **Replayed generation stayed at about 1.7–2.1 s** at every level.

### Results

Latency in seconds except the residual (ms). CPU is the process mean normalised
to all logical cores, and the system mean.

| method | c | ok | wall (s) | RPS | e2e p50 | e2e p95 | e2e p99 | queue p50 | retrieval p50 | rerank p50 | generation p50 | residual p50 (ms) | CPU process / system (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Method 1 | 1 | 20/20 | 443.6 | 0.0451 | 18.624 | 31.490 | 51.557 | 252.0 | 0.07 | 17.095 | 1.70 | 8 | 59 / 81 |
| Method 1 | 5 | 20/20 | 300.3 | 0.0666 | 70.204 | 79.201 | 80.771 | 101.1 | 5.13 | 64.211 | 1.72 | 9 | 84 / 100 |
| Method 1 | 10 | 20/20 | 318.5 | 0.0628 | 153.278 | 159.026 | 160.551 | 65.1 | 2.85 | 142.651 | 1.81 | 9 | 85 / 100 |
| Method 2 | 1 | 20/20 | 358.6 | 0.0558 | 18.335 | 20.596 | 22.733 | 174.7 | 0.12 | 16.703 | 1.71 | 11 | 64 / 77 |
| Method 2 | 5 | 20/20 | 289.1 | 0.0692 | 71.147 | 87.160 | 89.411 | 97.1 | 12.58 | 57.760 | 1.72 | 19 | 85 / 100 |
| Method 2 | 10 | 20/20 | 299.1 | 0.0669 | 134.405 | 168.232 | 172.847 | 38.7 | 32.44 | 103.094 | 1.80 | 18 | 86 / 100 |
| Method 3 | 1 | 20/20 | 379.4 | 0.0527 | 18.953 | 22.947 | 24.482 | 180.8 | 0.21 | 16.541 | 2.10 | 10 | 61 / 74 |
| Method 3 | 5 | 20/20 | 296.3 | 0.0675 | 67.691 | 86.833 | 88.824 | 98.8 | 9.69 | 55.440 | 2.12 | 21 | 84 / 99 |
| Method 3 | 10 | 20/20 | 307.0 | 0.0652 | 138.286 | 165.533 | 166.826 | 43.6 | 30.77 | 107.055 | 2.12 | 16 | 85 / 99 |

Queue wait is large at c = 1 by construction: the whole workload is submitted at
once, so later requests wait for earlier ones.

Memory, recorded but **not comparable across methods** (see caveats):

| method | max process RSS (MB), c = 1 / 5 / 10 | max system memory used (%), c = 1 / 5 / 10 |
|---|---|---|
| Method 1 | 1,928 / 4,108 / 6,627 | 99 / 98 / 97 |
| Method 2 | 7,469 / 7,429 / 7,383 | 89 / 94 / 96 |
| Method 3 | 8,365 / 8,355 / 8,288 | 92 / 97 / 100 |

### Interpretation

- **On this CPU-only machine, throughput rose from c = 1 to c = 5 and was
  approximately flat from c = 5 to c = 10 in this single run** (Method 1
  0.0451 → 0.0666 → 0.0628 RPS; Method 2 0.0558 → 0.0692 → 0.0669; Method 3
  0.0527 → 0.0675 → 0.0652). The c = 5 vs c = 10 differences come from one
  20-request sample per level and are not a demonstrated decline.
- **Higher concurrency substantially increased latency.** Median e2e grew from
  about 18–19 s at c = 1 to about 68–71 s at c = 5 and 134–153 s at c = 10, as
  concurrent requests contended for the same cores.
- **Cross-encoder reranking is the dominant bottleneck, and first-stage
  retrieval contends too.** Rerank p50 grew from about 17 s to 103–143 s. For
  Methods 2 and 3, first-stage retrieval (query embedding and retrievers) grew
  from about 0.1–0.2 s to about 31–32 s at c = 10.
- **The system was CPU-saturated at c = 5 and c = 10**: system CPU averaged
  99–100%.
- **Part of the c = 1 → c = 5 throughput gain is not a serving result.** At
  c = 1 about 9–11% of each request is replayed generation, idle time that
  overlaps freely under concurrency, and the process used only about 60% of the
  machine's cores on average.

### Caveats

- **Replay, not live.** Provider calls are replaced by recorded answers and
  recorded latencies. No provider-side concurrency, queueing, rate limiting or
  real API retry behaviour is measured.
- **In-process, not HTTP.** No request parsing, serialisation, network or server
  framework overhead is included.
- **One sample per level.** 20 requests per concurrency level, run once, in the
  fixed order 1 → 5 → 10; later levels benefit from warmer OS caches. p95/p99 over
  20 samples are not stable tail estimates.
- **CPU readings.** Per-sample process CPU normalised to all cores occasionally
  exceeds 100% (a maximum of 106.5% was recorded for Method 1 at c = 10); that is
  a sampling-interval artefact, not a physical value. The system is described as
  CPU-saturated.
- **Memory is not comparable across methods.** All three methods ran
  sequentially in one process, and memory released by one is not necessarily
  returned to the operating system before the next. Method 2 and 3 RSS therefore
  includes what Method 1 had already grown the process to. Method 1's growth with
  concurrency (1.9 → 4.1 → 6.6 GB) is within one method and plausibly reflects
  concurrent cross-encoder activations.
- **System memory pressure is a confound.** System memory reached 89–100% used,
  with Docker services running alongside. Paging may have inflated tail
  latencies; Method 1's c = 1 p95/p99 come from its first two requests after
  warm-up (q001, q002).
- **Method 3 query encoding is excluded.** Method 3 serves the evaluation queries
  from precomputed ColQwen2 query embeddings; encoding a new query on CPU is not
  part of the measured path.
- **Recorded environment gap.** The artefact's `torch_num_threads` and
  `cuda_available` fields are empty: they were read before any method had
  imported PyTorch. No CUDA device was present.
- **Provenance.** The artefact records `git_commit: f993ee6`; the Phase B code
  that produced it was committed afterwards, unchanged, as `f240cd5`.

---

## Reproducing

Both phases read artefacts that are **gitignored** — retrieval runs
(`data/eval/runs/`), generation and judged runs (`data/eval/generation/`), the LLM
cache, the indexes and the Method 3 page index — and write their own outputs to
`data/eval/production/` (also gitignored). The tables above are the committed
record.

**Phase A** (no provider calls, no retrieval; needs the local chunk indexes for
prompt timing):

```bash
mmrag prod report \
    --generation-run data/eval/generation/<run>_generation.json \
    --judged-run data/eval/generation/<run>_judged.json      # optional: failure buckets
mmrag prod compare data/eval/production/*_production.json
```

`--no-prompt-timing` skips the offline re-measurement; `--pricing` selects a
pricing file. A judged run produced from a different generation run is refused.

**The same-day V1 control runs** that Phase A reports and Phase B replays
were generated with the cache bypassed (paid provider calls), for example:

```bash
mmrag eval generate --retrieval-run data/eval/runs/<method1 run>.json \
    --no-cache --tag method1/rerank-v1-nocache
mmrag eval judge --generation-run data/eval/generation/<that run>_generation.json
```

**Phase B** (no provider calls in replay mode; needs Qdrant and Postgres running,
the built indexes, Method 3's page index and query cache, and the V1 no-cache
generation runs):

```bash
docker compose up -d
mmrag prod load --methods method1 --concurrency 1,5 --limit 3 --warmup 1 --cooldown-s 1   # smoke
mmrag prod load --methods method1,method2,method3 --concurrency 1,5,10 \
    --limit 20 --warmup 2 --cooldown-s 5
```

`--replay-run METHOD=PATH` replays a different generation run; `--limit` is
capped at the 42 answerable queries recorded in it. The preflight only probes
Qdrant and Postgres and never starts them; `--skip-preflight` bypasses it.

Tests for both phases are offline and need no services:
`tests/test_production.py` and `tests/test_production_load.py`.
