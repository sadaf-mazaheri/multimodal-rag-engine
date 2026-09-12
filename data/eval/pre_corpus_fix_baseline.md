# Evaluation baseline — PRE-CORPUS-FIX

**These numbers belong to a corpus that no longer exists.** They were produced
while `arxiv_tapas` pointed at arXiv 2004.08483 (*ETC: Encoding Long and
Structured Inputs in Transformers*) rather than the intended arXiv 2004.02349
(*TAPAS: Weakly Supervised Table Parsing via Pre-training*).

Do not compare them directly with post-fix runs. The affected document supplied
75 of 2,968 chunks (2.5%), and every one of those chunks carried the wrong
document title in its embedded context header.

- gold set: v1, 42 queries
- replaced PDF sha256: `a54cb8dc8c903f964457a346c761a3ebcdce69a427a3f5f269454d74783d6e3d`
- raw run files: `data/eval/runs/pre_corpus_fix/` (gitignored)

## Headline (macro-averaged, 42 queries)

| run | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 |
|---|---|---|---|---|---|
| method1/rerank | 0.381 | 0.726 | 0.857 | 0.536 | 0.616 |
| method2/rerank | 0.286 | 0.560 | 0.595 | 0.390 | 0.442 |
| method2/rerank-nometa | 0.333 | 0.655 | 0.762 | 0.471 | 0.544 |
| method1/norerank | 0.286 | 0.655 | 0.833 | 0.434 | 0.530 |
| method2/norerank | 0.143 | 0.393 | 0.429 | 0.237 | 0.284 |
| method2/norerank-nometa | 0.167 | 0.488 | 0.548 | 0.305 | 0.365 |

## recall@10 by answer modality

| run | text (n=15) | table (n=12) | figure (n=15) |
|---|---|---|---|
| method1/rerank | 0.933 | 1.000 | 0.667 |
| method2/rerank | 0.933 | 0.417 | 0.400 |
| method2/rerank-nometa | 0.933 | 0.917 | 0.467 |
| method1/norerank | 0.933 | 0.917 | 0.667 |
| method2/norerank | 0.867 | 0.417 | 0.000 |
| method2/norerank-nometa | 0.800 | 0.917 | 0.000 |

## recall@10 by phrasing

| run | text (n=10) | table (n=8) | figure (n=9) | natural (n=15) |
|---|---|---|---|---|
| method1/rerank | 0.900 | 1.000 | 0.778 | 0.800 |
| method2/rerank | 0.900 | 0.250 | 0.556 | 0.600 |
| method2/rerank-nometa | 0.900 | 1.000 | 0.667 | 0.600 |
| method1/norerank | 0.900 | 0.875 | 0.778 | 0.800 |
| method2/norerank | 0.800 | 0.250 | 0.000 | 0.533 |
| method2/norerank-nometa | 0.800 | 1.000 | 0.000 | 0.467 |

## Latency (ms, median)

| run | total | rerank | retrieval only | wall clock |
|---|---|---|---|---|
| method1/rerank | 17,589 | 17,479 | 110 | 759s |
| method2/rerank | 17,589 | 17,362 | 227 | 734s |
| method2/rerank-nometa | 18,484 | 18,322 | 162 | 800s |
| method1/norerank | 62 | 0 | 62 | 55s |
| method2/norerank | 165 | 0 | 165 | 21s |
| method2/norerank-nometa | 807 | 0 | 807 | 96s |
