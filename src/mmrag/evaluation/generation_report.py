"""Rendering judged generation runs: retrieval -> generation -> end to end.

Rendering only; every number was computed in ``judged_eval``. Rates are always
printed with their counts, because with 42 queries -- and slices of 12 to 15 --
the difference between 0.667 and 0.733 is a single question.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.table import Table

from mmrag.evaluation.generation_metrics import percentile
from mmrag.evaluation.judged_eval import (
    CATEGORIES_ANSWERABLE,
    CATEGORIES_UNANSWERABLE,
    JudgedRun,
)

MODALITIES = ("text", "table", "figure")
STRATA = ("text", "table", "figure", "natural")


def _rate(cell: dict[str, Any] | None) -> str:
    if not cell or cell.get("rate") is None:
        return "-"
    return f"{cell['rate']:.2f} ({cell['count']}/{cell['n']})"


def _mean(cell: dict[str, Any] | None) -> str:
    if not cell or cell.get("mean") is None:
        return "-"
    return f"{cell['mean']:.2f} (n={cell['n']})"


def decomposition_table(runs: Sequence[JudgedRun]) -> Table:
    table = Table(title="Retrieval -> generation -> end to end (answerable queries)")
    table.add_column("run", style="cyan", no_wrap=True)
    table.add_column("R: evidence in prompt", justify="right")
    table.add_column("G: grounded correct | evidence", justify="right")
    table.add_column("E2E: grounded correct", justify="right")
    table.add_column("leak: correct without evidence", justify="right")
    for run in runs:
        d = run.metrics["decomposition"]
        table.add_row(run.label, _rate(d["R_evidence_in_context"]),
                      _rate(d["G_grounded_correct_given_evidence"]),
                      _rate(d["E2E_grounded_correct"]),
                      _rate(d["leak_grounded_correct_without_evidence"]))
    return table


def quality_table(runs: Sequence[JudgedRun]) -> Table:
    table = Table(title="Answer quality (answerable queries)")
    table.add_column("run", style="cyan", no_wrap=True)
    for col in ("correctness", "completeness", "faithfulness", "hallucination",
                "citation support", "labels c/p/i/r"):
        table.add_column(col, justify="right")
    for run in runs:
        a = run.metrics["answerable"]
        lab = a["correctness_labels"]
        table.add_row(run.label, _mean(a["correctness"]), _mean(a["completeness"]),
                      _mean(a["faithfulness"]), _rate(a["hallucination"]),
                      _mean(a["citation_support"]),
                      f"{lab['correct']}/{lab['partial']}/{lab['incorrect']}/{lab['refused']}")
    return table


def refusal_table(runs: Sequence[JudgedRun]) -> Table:
    table = Table(title="Refusal behaviour")
    table.add_column("run", style="cyan", no_wrap=True)
    table.add_column("answerable: over-refusal\n(sufficient & refused)", justify="right")
    table.add_column("answerable: answered\ninsufficient context", justify="right")
    table.add_column("unanswerable: refused", justify="right")
    table.add_column("unanswerable: hallucinated", justify="right")
    table.add_column("refusal without marker", justify="right")
    for run in runs:
        a = run.metrics["answerable"]
        m = a["refusal_matrix"]
        u = run.metrics["unanswerable"]
        judged = run.metrics["coverage"]["answerable"]["judged"]
        table.add_row(run.label, f"{m['sufficient/refused']}/{judged}",
                      f"{m['insufficient/answered']}/{judged}", _rate(u["refused"]),
                      _rate(u["hallucination"]), _rate(a["refusal_without_marker"]))
    return table


def slice_table(runs: Sequence[JudgedRun], field: str) -> Table:
    keys = MODALITIES if field == "by_requires" else STRATA
    title = "answer modality" if field == "by_requires" else "phrasing"
    table = Table(title=f"grounded correct by {title}")
    table.add_column("run", style="cyan", no_wrap=True)
    for key in keys:
        table.add_column(key, justify="right")
    for run in runs:
        groups = run.metrics["answerable"][field]
        table.add_row(run.label, *[_rate(groups.get(k, {}).get("grounded_correct")) for k in keys])
    return table


def taxonomy_table(runs: Sequence[JudgedRun]) -> Table:
    table = Table(title="Error taxonomy (queries per category)")
    table.add_column("category", style="cyan", no_wrap=True)
    for run in runs:
        table.add_column(run.label, justify="right")
    for group, categories in (("answerable", CATEGORIES_ANSWERABLE),
                              ("unanswerable", CATEGORIES_UNANSWERABLE)):
        for category in categories:
            counts = []
            for run in runs:
                n = sum(1 for r in run.records if r.category == category
                        and r.answerable == (group == "answerable"))
                counts.append(str(n) if n else "[dim]0[/]")
            if any(c != "[dim]0[/]" for c in counts):
                table.add_row(f"{group}: {category}", *counts)
    return table


def reliability_table(runs: Sequence[JudgedRun]) -> Table:
    table = Table(title="Judge reliability, cost and latency")
    table.add_column("run", style="cyan", no_wrap=True)
    for col in ("judged", "repaired", "evidence proxy\nagrees w/ judge",
                "judge agrees w/\nlexical facts", "gen tokens in/out", "judge tokens in/out",
                "gen latency ms\nmedian / p90", "self-judge"):
        table.add_column(col, justify="right")
    for run in runs:
        c = run.metrics["coverage"]
        a = run.metrics["answerable"]
        gen_in = sum(r.generation.usage.get("prompt_tokens", 0) for r in run.records)
        gen_out = sum(r.generation.usage.get("completion_tokens", 0) for r in run.records)
        lat = [r.generation.latency_ms for r in run.records
               if r.generation.latency_ms is not None and not r.generation.cache_hit]
        agree = a["judge_lexical_fact_agreement"]
        table.add_row(
            run.label,
            f"{c['answerable']['judged'] + c['unanswerable']['judged']}"
            f"/{c['answerable']['n'] + c['unanswerable']['n']}",
            str(c["repaired"]),
            _rate(a["evidence_proxy_agrees_with_judge"]),
            f"{agree['rate']:.2f} ({agree['agree']}/{agree['checkable']})"
            if agree.get("rate") is not None else "-",
            f"{gen_in:,} / {gen_out:,}",
            f"{run.totals.get('prompt_tokens', 0):,} / {run.totals.get('completion_tokens', 0):,}",
            f"{percentile(lat, 0.5) or '-'} / {percentile(lat, 0.9) or '-'}",
            "yes" if run.judge.get("self_judge") else "no",
        )
    return table


def pipeline_of(run: JudgedRun) -> str:
    """A run's generation pipeline; runs from before V2 existed were V1."""
    return str(run.generation.get("pipeline") or "v1")


def answer_shape_table(runs: Sequence[JudgedRun]) -> Table:
    """Length, claims and the deterministic validator, beside each other.

    Validator columns show "-" for runs generated before the validator existed.
    """
    table = Table(title="Answer shape and validation (answered answerable queries)")
    table.add_column("run", style="cyan", no_wrap=True)
    for col in ("pipeline", "answer chars", "claims / answer", "sentence\ncitation coverage",
                "validation\npassed", "ungrounded\nidentifier"):
        table.add_column(col, justify="right")
    for run in runs:
        a = run.metrics["answerable"]
        answered = [r.generation for r in run.records
                    if r.answerable and r.scores is not None and not r.scores.refused]
        reports = [g.validation for g in answered if g.validation is not None]
        coverage = [v["sentence_citation_coverage"] for v in reports
                    if v["sentence_citation_coverage"] is not None]
        table.add_row(
            run.label, pipeline_of(run), _mean(a.get("answer_chars")),
            _mean(a.get("claims_per_answer")),
            f"{sum(coverage) / len(coverage):.2f} (n={len(coverage)})" if coverage else "-",
            f"{sum(v['passed'] for v in reports)}/{len(reports)}" if reports else "-",
            f"{sum(bool(v['ungrounded_identifiers']) for v in reports)}/{len(reports)}"
            if reports else "-",
        )
    return table


def disagreement_table(runs: Sequence[JudgedRun], limit: int = 20) -> Table:
    table = Table(title="Queries whose outcome differs between runs")
    table.add_column("query", style="cyan", no_wrap=True)
    for run in runs:
        table.add_column(run.label, no_wrap=True)
    table.add_column("question", overflow="ellipsis", no_wrap=True, max_width=44)
    by_run = [{r.query_id: r for r in run.records} for run in runs]
    ids = list(dict.fromkeys(q for d in by_run for q in d))
    shown = 0
    for qid in ids:
        cats = [d[qid].category if qid in d else "-" for d in by_run]
        if len(set(cats)) > 1:
            question = next(d[qid].generation.query for d in by_run if qid in d)
            table.add_row(qid, *cats, question)
            shown += 1
            if shown >= limit:
                break
    return table


def render_judged(runs: Sequence[JudgedRun], console: Console | None = None) -> None:
    console = console or Console()
    if not runs:
        console.print("[yellow]no judged runs to report[/]")
        return
    for table in (decomposition_table(runs), quality_table(runs), refusal_table(runs),
                  slice_table(runs, "by_requires"), slice_table(runs, "by_stratum"),
                  taxonomy_table(runs), answer_shape_table(runs), reliability_table(runs)):
        console.print()
        console.print(table)
    if len(runs) > 1:
        console.print()
        console.print(disagreement_table(runs))

    n = runs[0].metrics["coverage"]["answerable"]["n"]
    u = runs[0].metrics["coverage"]["unanswerable"]["n"]
    self_judge = any(r.judge.get("self_judge") for r in runs)
    console.print(
        f"\n[dim]{n} answerable and {u} unanswerable queries; slices hold 8 to 15. "
        "Differences of a query or two are noise, and no significance is claimed."
        + (" The judge is the same model as the generator, so absolute scores are "
           "directional; the comparison between runs is the sturdier signal." if self_judge else "")
        + "[/]"
    )


def to_markdown_judged(runs: Sequence[JudgedRun]) -> str:
    lines = ["| run | R | G | E2E | correctness | faithfulness | hallucination | "
             "unanswerable refused |", "|---|---|---|---|---|---|---|---|"]
    for run in runs:
        d = run.metrics["decomposition"]
        a = run.metrics["answerable"]
        u = run.metrics["unanswerable"]
        lines.append(
            f"| {run.label} | {_rate(d['R_evidence_in_context'])} | "
            f"{_rate(d['G_grounded_correct_given_evidence'])} | "
            f"{_rate(d['E2E_grounded_correct'])} | "
            f"{_mean(a['correctness'])} | {_mean(a['faithfulness'])} | "
            f"{_rate(a['hallucination'])} | {_rate(u['refused'])} |"
        )
    return "\n".join(lines)


def run_kind(path: str) -> str:
    """'retrieval', 'generation' or 'judged', without fully validating the file."""
    import json
    from pathlib import Path

    with Path(path).open(encoding="utf-8") as handle:
        head = json.load(handle)
    return head.get("kind", "retrieval")
