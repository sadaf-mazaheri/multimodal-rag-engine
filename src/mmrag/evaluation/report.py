"""Turning runs into a comparison a reader can act on.

Rendering only -- every number here was computed by ``retrieval_eval``. Kept
separate so the report can be re-rendered from saved run files without
re-running retrieval, which on CPU costs about twenty minutes a method.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mmrag.evaluation.retrieval_eval import RetrievalRun

# The metrics worth a reader's attention by default. Precision is computed and
# stored, but with one or two evidence pages per query its ceiling is set by the
# size of the gold set rather than by retrieval quality, so it is not shown
# unless asked for.
HEADLINE = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]

STRATA = ["text", "table", "figure", "natural"]
MODALITIES = ["text", "table", "figure"]


def load_run(path: str | Path) -> RetrievalRun:
    return RetrievalRun.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_run(run: RetrievalRun, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return target


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _delta(new: float, base: float) -> str:
    """Signed difference, coloured by direction rather than by magnitude."""
    diff = new - base
    if abs(diff) < 5e-4:
        return "[dim]  =   [/]"
    colour = "green" if diff > 0 else "red"
    return f"[{colour}]{diff:+.3f}[/]"


def headline_table(runs: list[RetrievalRun], metrics: list[str] | None = None) -> Table:
    metrics = metrics or HEADLINE
    table = Table(title="Retrieval quality (macro-averaged over all queries)")
    table.add_column("run", style="cyan", no_wrap=True)
    table.add_column("n", justify="right")
    for metric in metrics:
        table.add_column(metric, justify="right")

    baseline = runs[0]
    for run in runs:
        row = [run.label(), str(len(run.per_query))]
        for metric in metrics:
            value = run.metrics.get(metric, 0.0)
            cell = _fmt(value)
            if run is not baseline:
                cell += f"  {_delta(value, baseline.metrics.get(metric, 0.0))}"
            row.append(cell)
        table.add_row(*row)
    return table


def slice_table(
    runs: list[RetrievalRun], field: str, keys: list[str], metric: str = "recall@10"
) -> Table:
    """One metric broken out by stratum or by required modality.

    The reason the gold set carries both axes: a pooled number cannot say
    whether a method is better at *finding figures* or merely better at
    *questions that mention figures*.
    """
    attribute = "by_stratum" if field == "stratum" else "by_requires"
    title = "phrasing (stratum)" if field == "stratum" else "answer modality (requires)"
    table = Table(title=f"{metric} by {title}")
    table.add_column("run", style="cyan", no_wrap=True)
    for key in keys:
        first = getattr(runs[0], attribute).get(key, {})
        table.add_column(f"{key}\n(n={int(first.get('n', 0))})", justify="right")

    baseline = runs[0]
    for run in runs:
        row = [run.label()]
        for key in keys:
            value = getattr(run, attribute).get(key, {}).get(metric)
            cell = _fmt(value)
            if run is not baseline and value is not None:
                base = getattr(baseline, attribute).get(key, {}).get(metric, 0.0)
                cell += f"  {_delta(value, base)}"
            row.append(cell)
        table.add_row(*row)
    return table


def latency_table(runs: list[RetrievalRun]) -> Table:
    table = Table(title="Latency (ms, median / p90)")
    table.add_column("run", style="cyan", no_wrap=True)
    table.add_column("total", justify="right")
    table.add_column("rerank", justify="right")
    table.add_column("retrieval only", justify="right")
    table.add_column("wall clock", justify="right")

    for run in runs:
        total_med = run.latency.get("total_ms_median", 0.0)
        total_p90 = run.latency.get("total_ms_p90", 0.0)
        rerank_med = run.latency.get("rerank_ms_median", 0.0)
        table.add_row(
            run.label(),
            f"{total_med:,.0f} / {total_p90:,.0f}",
            f"{rerank_med:,.0f}" if rerank_med else "[dim]off[/]",
            f"{max(total_med - rerank_med, 0.0):,.0f}",
            f"{run.elapsed_s:,.0f}s",
        )
    return table


def disagreements(runs: list[RetrievalRun], metric: str = "recall@10", limit: int = 12) -> Table:
    """Queries where two runs most disagree.

    An aggregate says which method won; this says on what. It is the part of
    the report that tells you where to look next, so it lists the query text
    rather than only its id.
    """
    table = Table(title=f"Largest per-query differences in {metric}")
    table.add_column("query", style="cyan", no_wrap=True, max_width=13)
    table.add_column("stratum", no_wrap=True)
    table.add_column("needs", no_wrap=True)
    for run in runs:
        table.add_column(run.label(), justify="right")
    table.add_column("question", overflow="ellipsis", no_wrap=True, max_width=46)

    if len(runs) < 2:
        return table

    by_id = [{q.query_id: q for q in run.per_query} for run in runs]
    rows = []
    for query_id, first in by_id[0].items():
        values = [d.get(query_id).metrics.get(metric, 0.0) if d.get(query_id) else 0.0
                  for d in by_id]
        rows.append((max(values) - min(values), query_id, first, values))

    for spread, query_id, first, values in sorted(rows, key=lambda r: -r[0])[:limit]:
        if spread < 5e-4:
            break
        table.add_row(
            query_id, first.stratum, first.requires,
            *[_fmt(v) for v in values], first.query,
        )
    return table


# The modalities the router chooses between. Method 3's visual_page signal is
# appended to every decision, so it says nothing about whether the router fanned
# out and is not counted here.
ROUTED_MODALITIES = frozenset({"text", "table", "image"})


def routing_summary(run: RetrievalRun) -> Table | None:
    """How often the router fanned out, for runs that have a router."""
    routed = [q for q in run.per_query if q.routing]
    if not routed:
        return None

    table = Table(title=f"Routing behaviour -- {run.label()}")
    table.add_column("stratum", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("fell back", justify="right")
    table.add_column("fired text+table+image", justify="right")

    for stratum in STRATA:
        group = [q for q in routed if q.stratum == stratum]
        if not group:
            continue
        fell = sum(1 for q in group if q.routing.get("fell_back"))
        fanned = sum(
            1 for q in group if ROUTED_MODALITIES.issubset(q.routing.get("modalities", []))
        )
        table.add_row(stratum, str(len(group)), f"{fell}/{len(group)}", f"{fanned}/{len(group)}")
    return table


def render(runs: list[RetrievalRun], console: Console | None = None) -> None:
    """The full comparison."""
    console = console or Console()
    if not runs:
        console.print("[yellow]no runs to compare[/]")
        return

    console.print()
    console.print(headline_table(runs))
    console.print()
    console.print(slice_table(runs, "requires", MODALITIES))
    console.print()
    console.print(slice_table(runs, "stratum", STRATA))
    console.print()
    console.print(latency_table(runs))

    if len(runs) > 1:
        console.print()
        console.print(disagreements(runs))

    for run in runs:
        summary = routing_summary(run)
        if summary is not None:
            console.print()
            console.print(summary)

    gold = runs[0].gold
    console.print(
        f"\n[dim]gold set v{gold.get('version')} -- {gold.get('n_queries')} queries. "
        "Macro-averaged; `natural` is the slice without modality cues. "
        "Small n: treat differences under a few points as noise.[/]"
    )


def to_markdown(runs: list[RetrievalRun], metrics: list[str] | None = None) -> str:
    """Headline table as Markdown, for pasting into the README."""
    metrics = metrics or HEADLINE
    lines = ["| run | " + " | ".join(metrics) + " |",
             "|---" * (len(metrics) + 1) + "|"]
    for run in runs:
        cells = [f"{run.metrics.get(m, 0.0):.3f}" for m in metrics]
        lines.append(f"| {run.label()} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dump_json(runs: list[RetrievalRun]) -> str:
    """Machine-readable summary, for anything downstream of this report."""
    return json.dumps(
        [
            {
                "run": run.label(),
                "method": run.method,
                "overrides": run.overrides,
                "metrics": run.metrics,
                "by_stratum": run.by_stratum,
                "by_requires": run.by_requires,
                "latency": run.latency,
            }
            for run in runs
        ],
        indent=2,
    )
