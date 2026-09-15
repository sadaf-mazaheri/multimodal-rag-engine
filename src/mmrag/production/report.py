"""Assemble a production report from existing run artefacts.

Inputs are one generation run, the retrieval run it was generated from, an
optional judged run of that generation run, and the pricing table. Prompt
construction and post-processing timings are measured separately by
:mod:`mmrag.production.prompt_timing` and passed in, so this module stays pure
arithmetic over loaded objects and is testable without an index.

Latency components and their provenance:

=======================  ==========  ==============================================
component                provenance  source
=======================  ==========  ==============================================
retrieval                recorded    retrieval run ``latency_ms`` (candidate stage)
fusion                   recorded    retrieval run ``fusion_ms``
reranking                recorded    retrieval run ``rerank_ms``
retrieval_total          recorded    retrieval run ``total_ms``
prompt_construction      measured    re-executed here, sha256-checked
generation               recorded    generation run ``latency_ms``, cache misses only
postprocess              measured    re-executed here over the recorded answer
composed_e2e_latency     composed    retrieval_total + prompt_construction
                                     + generation + postprocess
=======================  ==========  ==============================================
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from mmrag.evaluation.generation_eval import GenerationRun, now_iso
from mmrag.evaluation.judged_eval import JudgedRun
from mmrag.evaluation.retrieval_eval import RetrievalRun
from mmrag.production.failures import bucket_of, decompose
from mmrag.production.pricing import PricingError, PricingTable
from mmrag.production.prompt_timing import PromptTiming
from mmrag.production.reliability import classify_error, reliability
from mmrag.production.stages import canonical_stages
from mmrag.production.stats import summarize

Provenance = Literal["recorded", "measured", "composed"]

COMPONENTS: dict[str, Provenance] = {
    "retrieval": "recorded",
    "fusion": "recorded",
    "reranking": "recorded",
    "retrieval_total": "recorded",
    "prompt_construction": "measured",
    "generation": "recorded",
    "postprocess": "measured",
    "composed_e2e_latency": "composed",
}
COMPOSED_FROM = ("retrieval_total", "prompt_construction", "generation", "postprocess")

COMPOSED_NOTE = (
    "composed_e2e_latency = retrieval_total + prompt_construction + generation + "
    "postprocess. It is a sum of components measured separately and was NOT measured "
    "in a single serving process: retrieval and generation were recorded in different "
    "runs, and prompt construction and post-processing were re-measured offline. It "
    "excludes one-off model loading, queueing and network overhead outside the provider "
    "call. True end-to-end latency will come from the Phase B load test."
)


class ProductionRecord(BaseModel):
    """One query's production view."""

    query_id: str
    answerable: bool
    in_retrieval_run: bool
    status: str
    cache_hit: bool | None = None
    attempts: int = 0
    error_kind: str | None = None

    latency_ms: dict[str, float | None] = Field(default_factory=dict)
    retrieval_keys: list[str] = Field(default_factory=list)
    model_load_ms: float | None = None
    prompt_sha256_match: bool | None = None

    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    failure_bucket: str | None = None
    judged_category: str | None = None


class ProductionReport(BaseModel):
    kind: Literal["production"] = "production"
    schema_version: int = 1
    label: str
    method: str
    pipeline: str
    created_at: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=lambda: dict(COMPONENTS))
    latency: dict[str, Any] = Field(default_factory=dict)
    cold_start: dict[str, Any] = Field(default_factory=dict)
    tokens: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)
    evaluation_overhead: dict[str, Any] = Field(default_factory=dict)
    reliability: dict[str, Any] = Field(default_factory=dict)
    failures: dict[str, Any] | None = None
    prompt_check: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    records: list[ProductionRecord] = Field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> ProductionReport:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def default_filename(self) -> str:
        stamp = self.created_at.replace(":", "").replace("-", "")
        return f"{stamp}_{self.label.replace('/', '_')}_production.json"


def _cost(pricing: PricingTable, model: str | None, fallback: str, usage: dict[str, int]) -> float:
    return pricing.cost(model or fallback, usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0))


def build_report(
    generation: GenerationRun,
    retrieval: RetrievalRun,
    pricing: PricingTable,
    *,
    judged: JudgedRun | None = None,
    prompt_timings: dict[str, PromptTiming] | None = None,
    inputs: dict[str, Any] | None = None,
    extra_warnings: Sequence[str] = (),
) -> ProductionReport:
    """Compute every production metric from loaded artefacts. No I/O, no models."""
    fallback_model = generation.generation.get("model") or ""
    pipeline = generation.generation.get("pipeline") or "v1"
    by_query = {q.query_id: q for q in retrieval.per_query}
    timings = prompt_timings or {}
    warnings: list[str] = list(extra_warnings)

    judged_by_id: dict[str, Any] = {}
    if judged is not None:
        judged_by_id = {r.query_id: r for r in judged.records}
        missing = sorted({r.query_id for r in generation.records} - set(judged_by_id))
        if missing:
            warnings.append(f"judged run lacks {len(missing)} generation record(s): {missing}")

    records: list[ProductionRecord] = []
    inconsistent: list[str] = []
    for gen in generation.records:
        qr = by_query.get(gen.query_id)
        stages = canonical_stages(qr.latency_ms) if qr is not None else None
        if stages is not None and not stages["total_consistent"]:
            inconsistent.append(gen.query_id)
        timing = timings.get(gen.query_id)

        generation_ms = (gen.latency_ms if gen.status == "ok" and not gen.cache_hit
                         and gen.latency_ms is not None else None)
        latency: dict[str, float | None] = {
            "retrieval": stages["retrieval"] if stages else None,
            "fusion": stages["fusion"] if stages else None,
            "reranking": stages["reranking"] if stages else None,
            "retrieval_total": stages["retrieval_total"] if stages else None,
            "prompt_construction": timing.prompt_construction_ms if timing else None,
            "generation": generation_ms,
            "postprocess": timing.postprocess_ms if timing else None,
        }
        parts = [latency[c] for c in COMPOSED_FROM]
        latency["composed_e2e_latency"] = (
            sum(p for p in parts if p is not None) if all(p is not None for p in parts) else None
        )

        rec = ProductionRecord(
            query_id=gen.query_id,
            answerable=gen.answerable,
            in_retrieval_run=qr is not None,
            status=gen.status,
            cache_hit=gen.cache_hit,
            attempts=gen.attempts,
            error_kind=classify_error(gen.error),
            latency_ms={k: (round(v, 3) if v is not None else None) for k, v in latency.items()},
            retrieval_keys=stages["retrieval_keys"] if stages else [],
            model_load_ms=stages["model_load_ms"] if stages else None,
            prompt_sha256_match=timing.sha_match if timing else None,
            model=gen.model,
        )
        if gen.status == "ok" and gen.usage:
            rec.prompt_tokens = gen.usage.get("prompt_tokens", 0)
            rec.completion_tokens = gen.usage.get("completion_tokens", 0)
            rec.total_tokens = gen.usage.get("total_tokens",
                                             rec.prompt_tokens + rec.completion_tokens)
            rec.cost_usd = round(_cost(pricing, gen.model, fallback_model, gen.usage), 8)
        jr = judged_by_id.get(gen.query_id)
        if jr is not None:
            rec.failure_bucket = bucket_of(jr)
            rec.judged_category = jr.category
        records.append(rec)

    if inconsistent:
        warnings.append(f"recorded total_ms differs from the sum of mapped stages for "
                        f"{inconsistent}")

    latency_summary = {
        name: {"provenance": prov, **summarize(r.latency_ms.get(name) for r in records)}
        for name, prov in COMPONENTS.items()
    }
    latency_summary["composed_e2e_latency"]["composed_from"] = list(COMPOSED_FROM)
    latency_summary["composed_e2e_latency"]["composition_note"] = COMPOSED_NOTE
    n_cached = sum(1 for r in records if r.status == "ok" and r.cache_hit)
    latency_summary["generation"]["excluded_cache_hits"] = n_cached

    load_values = [r.model_load_ms for r in records if r.model_load_ms is not None]
    ok = [r for r in records if r.total_tokens is not None]
    costs = [r.cost_usd for r in ok if r.cost_usd is not None]
    price_name, price = pricing.price_for(fallback_model or (ok[0].model if ok else ""))

    evaluation_overhead: dict[str, Any] = {
        "note": "Spend on evaluating quality, not a per-query serving cost.",
    }
    if judged is not None:
        judge_model = judged.judge.get("model") or ""
        tokens_in = int(judged.totals.get("prompt_tokens", 0))
        tokens_out = int(judged.totals.get("completion_tokens", 0))
        try:
            judge_cost: float | None = round(pricing.cost(judge_model, tokens_in, tokens_out), 6)
        except PricingError:
            judge_cost = None
            warnings.append(f"no configured price for judge model {judge_model!r}")
        evaluation_overhead["judge"] = {
            "model": judge_model, "calls_made": judged.totals.get("calls_made"),
            "cache_hits": judged.totals.get("cache_hits"),
            "prompt_tokens": tokens_in, "completion_tokens": tokens_out, "cost_usd": judge_cost,
        }

    timed = [t for t in timings.values()]
    mismatched = sorted(t.query_id for t in timed if not t.sha_match)
    if mismatched:
        warnings.append(f"rebuilt prompt sha256 differs from the generation run for {mismatched}; "
                        "their prompt timings are excluded")

    notes = [
        COMPOSED_NOTE,
        "Only answerable queries present in the retrieval run have retrieval timings; "
        "unanswerable queries were retrieved live during generation, untimed.",
        "Generation latency excludes cache hits, whose latency is replayed from an earlier "
        "call; token usage and cost include them, since serving would pay for the call.",
        "One-off model loading is excluded from every stage and reported under cold_start.",
    ]
    metadata_on = retrieval.overrides.get("use_metadata") is not False
    if retrieval.method in ("method2", "method3") and metadata_on:
        notes.append("Metadata resolution runs before the retrieval timer starts and is not "
                     "included in any recorded retrieval stage.")
    if any("visual_page_ms" in r.retrieval_keys for r in records):
        notes.append("visual_page_ms was recorded with precomputed ColQwen2 query embeddings: "
                     "it covers page scoring only, not query encoding.")

    return ProductionReport(
        label=generation.label,
        method=generation.method,
        pipeline=pipeline,
        created_at=now_iso(),
        inputs=inputs or {},
        latency=latency_summary,
        cold_start={"provenance": "recorded", "model_load_ms": summarize(load_values),
                    "note": "one-off; excluded from all per-query stages"},
        tokens={
            "source": "provider-reported usage in the generation run",
            "prompt_tokens": summarize((r.prompt_tokens for r in ok), digits=1),
            "completion_tokens": summarize((r.completion_tokens for r in ok), digits=1),
            "total_tokens": summarize((r.total_tokens for r in ok), digits=1),
            "sum": {"prompt_tokens": sum(r.prompt_tokens or 0 for r in ok),
                    "completion_tokens": sum(r.completion_tokens or 0 for r in ok)},
        },
        cost={
            "currency": pricing.currency,
            "pricing": {"model": price_name, "input_per_1m": price.input_per_1m,
                        "output_per_1m": price.output_per_1m, "as_of": pricing.as_of,
                        "kind": pricing.kind, "source": pricing.source,
                        "gateway": price.gateway},
            "per_query_usd": summarize(costs, digits=8),
            "total_usd": round(sum(costs), 6),
            "per_1k_queries_usd": round(sum(costs) / len(costs) * 1000, 4) if costs else None,
            "note": "Configured project evaluation rates, not universal or permanent pricing.",
        },
        evaluation_overhead=evaluation_overhead,
        reliability=reliability(generation.records),
        failures=decompose(judged.records) if judged is not None else None,
        prompt_check={"n_timed": len(timed), "n_sha_match": len(timed) - len(mismatched),
                      "mismatched_query_ids": mismatched},
        notes=notes,
        warnings=warnings,
        records=records,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def _pct(block: dict[str, Any] | None) -> str:
    if not block or block.get("rate") is None:
        return "-"
    return f"{block['count']}/{block['n']} ({block['rate']:.1%})"


def render_report(report: ProductionReport, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=f"Production latency (ms) — {report.label} [{report.pipeline}]")
    for col in ("component", "provenance", "n", "p50", "p95", "p99", "max"):
        table.add_column(col, justify="left" if col in ("component", "provenance") else "right")
    for name, block in report.latency.items():
        table.add_row(name, block["provenance"], str(block["n"]), _ms(block["p50"]),
                      _ms(block["p95"]), _ms(block["p99"]), _ms(block["max"]))
    console.print(table)

    t, c, rel = report.tokens, report.cost, report.reliability
    console.print(
        f"tokens/query: prompt p50 {t['prompt_tokens']['p50']}, completion p50 "
        f"{t['completion_tokens']['p50']}, total mean {t['total_tokens']['mean']}  |  "
        f"cost/query mean ${c['per_query_usd']['mean']} (p95 ${c['per_query_usd']['p95']}), "
        f"per 1k ${c['per_1k_queries_usd']}  "
        + escape(f"[{c['pricing']['model']} @ {c['pricing']['input_per_1m']}/"
                 f"{c['pricing']['output_per_1m']} per 1M, as of {c['pricing']['as_of']}]")
    )
    console.print(
        f"errors {_pct(rel['error_rate'])}, retries {_pct(rel['retry_rate'])}, "
        f"timeouts {_pct(rel['timeout_rate'])}  [dim](SDK-internal retries not observable)[/]"
    )
    if report.failures is not None:
        a = report.failures["answerable"]
        console.print("failures (answerable): " + ", ".join(
            f"{k} {_pct(v)}" for k, v in a.items() if v["count"] or k != "error"))
    pc = report.prompt_check
    console.print(f"[dim]prompt sha256 check: {pc['n_sha_match']}/{pc['n_timed']} match[/]")
    for w in report.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    console.print(f"[dim]{COMPOSED_NOTE}[/]")


def compare_table(reports: Sequence[ProductionReport]) -> Table:
    table = Table(title="Production comparison (composed_e2e_latency is composed, not measured)")
    cols = ("label", "pipeline", "e2e n", "e2e p50", "e2e p95", "e2e p99", "rerank p50",
            "prompt p50", "gen p50", "post p50", "tok/q", "$/1k q", "err", "retry", "timeout",
            "retr fail", "ctx fail", "gen fail", "cite fail", "success")
    for col in cols:
        table.add_column(col, justify="left" if col in ("label", "pipeline") else "right")
    for r in reports:
        e2e = r.latency["composed_e2e_latency"]
        f = (r.failures or {}).get("answerable", {})

        def bucket(name: str, f: dict[str, Any] = f) -> str:
            b = f.get(name)
            return "-" if not b else f"{b['count']}/{b['n']}"

        table.add_row(
            r.label, r.pipeline, str(e2e["n"]), _ms(e2e["p50"]), _ms(e2e["p95"]),
            _ms(e2e["p99"]), _ms(r.latency["reranking"]["p50"]),
            _ms(r.latency["prompt_construction"]["p50"]), _ms(r.latency["generation"]["p50"]),
            _ms(r.latency["postprocess"]["p50"]), str(r.tokens["total_tokens"]["mean"]),
            str(r.cost["per_1k_queries_usd"]), _pct(r.reliability["error_rate"]),
            _pct(r.reliability["retry_rate"]), _pct(r.reliability["timeout_rate"]),
            bucket("retrieval_failure"), bucket("context_failure"),
            bucket("generation_failure"), bucket("citation_failure"), bucket("success"),
        )
    return table
