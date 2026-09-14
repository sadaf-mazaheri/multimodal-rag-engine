"""``mmrag`` command line interface.

Every stage of the benchmark is reachable from here, so a run is always
reproducible from a shell history rather than from a notebook someone ran once.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from mmrag import __version__
from mmrag.logging_utils import setup_logging

app = typer.Typer(
    name="mmrag",
    help="Multimodal RAG benchmark: three retrieval architectures, one corpus.",
    no_args_is_help=True,
    add_completion=False,
)
corpus_app = typer.Typer(
    help="Acquire and verify the source document corpus.", no_args_is_help=True
)
config_app = typer.Typer(help="Inspect experiment configuration.", no_args_is_help=True)
ingest_app = typer.Typer(
    help="Parse the corpus into the shared Document/Page/Element representation.",
    no_args_is_help=True,
)
index_app = typer.Typer(help="Build and inspect retrieval indexes.", no_args_is_help=True)
eval_app = typer.Typer(
    help="Score retrieval against the gold set and compare methods.", no_args_is_help=True
)
app.add_typer(corpus_app, name="corpus")
app.add_typer(config_app, name="config")
app.add_typer(ingest_app, name="ingest")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")

DEFAULT_GOLD = "data/eval/gold/v1.yaml"
DEFAULT_GENERATION_GOLD = "data/eval/gold/generation_v1.yaml"

console = Console()


@app.callback()
def _root(
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG|INFO|WARNING|ERROR"),
) -> None:
    setup_logging(log_level, force=True)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"mmrag {__version__}")


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


@corpus_app.command("list")
def corpus_list(
    manifest_path: Path | None = typer.Option(None, "--manifest", help="Path to corpus.yaml"),
    show_all: bool = typer.Option(False, "--all", help="Include disabled entries"),
) -> None:
    """Show the manifest and which documents are present locally."""
    from mmrag.corpus import CorpusManifest

    manifest = CorpusManifest.load(manifest_path)
    entries = manifest.entries if show_all else manifest.active

    table = Table(title=f"Corpus manifest ({len(entries)} documents)", show_lines=False)
    table.add_column("doc_id", style="cyan", no_wrap=True)
    table.add_column("category", style="magenta")
    table.add_column("modalities", style="green")
    table.add_column("pages", justify="right")
    table.add_column("pinned", justify="center")
    table.add_column("local", justify="center")

    for entry in entries:
        local = entry.local_path()
        pages = str(entry.n_pages) if entry.n_pages else "-"
        if entry.page_limit:
            pages = f"{pages} (cap {entry.page_limit})"
        table.add_row(
            entry.doc_id,
            entry.category or "-",
            ",".join(entry.modality_profile) or "-",
            pages,
            "[green]yes[/]" if entry.sha256 else "[yellow]no[/]",
            "[green]yes[/]" if local.exists() else "[red]no[/]",
        )
    console.print(table)


@corpus_app.command("download")
def corpus_download(
    doc_id: list[str] = typer.Option(None, "--doc-id", "-d", help="Restrict to these ids"),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached"),
    manifest_path: Path | None = typer.Option(None, "--manifest"),
) -> None:
    """Download the corpus, verifying each file against its pinned hash."""
    from mmrag.corpus import CorpusManifest, download_corpus

    manifest = CorpusManifest.load(manifest_path)
    results = download_corpus(manifest, doc_ids=list(doc_id) if doc_id else None, force=force)

    table = Table(title="Download results")
    table.add_column("doc_id", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("size", justify="right")
    table.add_column("detail", overflow="fold")

    colours = {
        "downloaded": "green",
        "cached": "blue",
        "failed": "red",
        "hash_mismatch": "red",
    }
    for r in results:
        size = f"{r.size_bytes / 1e6:.1f} MB" if r.size_bytes else "-"
        table.add_row(
            r.entry.doc_id,
            f"[{colours.get(r.status, 'white')}]{r.status}[/]",
            size,
            r.message or "",
        )
    console.print(table)

    n_bad = sum(1 for r in results if not r.ok)
    if n_bad:
        console.print(f"[red]{n_bad} of {len(results)} documents failed.[/]")
        raise typer.Exit(code=1)
    console.print(f"[green]All {len(results)} documents present and verified.[/]")


@corpus_app.command("lock")
def corpus_lock(
    lock_path: Path | None = typer.Option(None, "--lock", help="Path to corpus.lock.yaml"),
) -> None:
    """Pin the hash, size, and page count of every downloaded document.

    Writes configs/corpus.lock.yaml. The hand-authored configs/corpus.yaml is
    left untouched. Run after the first download, and again whenever an entry
    is added or a source document legitimately changes.
    """
    from mmrag.corpus import lock_manifest

    lock, updated = lock_manifest(lock_path=lock_path)
    if not updated:
        console.print("[yellow]No downloaded documents found; run 'mmrag corpus download'.[/]")
        raise typer.Exit(code=1)
    total = sum(e.n_pages or 0 for e in lock.entries.values())
    console.print(
        f"[green]Pinned {len(updated)} documents[/] ({total} pages total): {', '.join(updated)}"
    )


@corpus_app.command("verify")
def corpus_verify(
    manifest_path: Path | None = typer.Option(None, "--manifest"),
) -> None:
    """Re-hash local files and check them against the manifest."""
    from mmrag.corpus import CorpusManifest, sha256_file

    manifest = CorpusManifest.load(manifest_path)
    problems = 0

    table = Table(title="Corpus verification")
    table.add_column("doc_id", style="cyan", no_wrap=True)
    table.add_column("result")

    for entry in manifest.active:
        path = entry.local_path()
        if not path.exists():
            table.add_row(entry.doc_id, "[red]missing[/]")
            problems += 1
        elif entry.sha256 is None:
            table.add_row(entry.doc_id, "[yellow]present, not pinned[/]")
        elif sha256_file(path) != entry.sha256:
            table.add_row(entry.doc_id, "[red]HASH MISMATCH[/]")
            problems += 1
        else:
            table.add_row(entry.doc_id, "[green]ok[/]")

    console.print(table)
    if problems:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@config_app.command("show")
def config_show(
    name: str = typer.Argument("default", help="Config name under configs/, or a path"),
) -> None:
    """Resolve a config (following 'extends') and print the effective values."""
    import json

    from mmrag.config import load_experiment_config

    cfg = load_experiment_config(name)
    console.print_json(json.dumps(cfg.model_dump(mode="json")))


@config_app.command("env")
def config_env() -> None:
    """Print environment settings with secrets redacted."""
    import json

    from mmrag.config import get_settings

    console.print_json(json.dumps(get_settings().redacted()))


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@ingest_app.command("run")
def ingest_run(
    doc_id: list[str] = typer.Option(None, "--doc-id", "-d", help="Restrict to these ids"),
    config_name: str = typer.Option("method1", "--config", "-c", help="Experiment config"),
    force: bool = typer.Option(False, "--force", help="Re-parse even if a sidecar exists"),
    max_pages: int | None = typer.Option(
        None, "--max-pages", help="Cap pages per document (for quick smoke runs)"
    ),
    no_postgres: bool = typer.Option(
        False, "--no-postgres", help="Write only the JSON sidecars, skip the database"
    ),
) -> None:
    """Parse PDFs into documents, pages, and elements.

    Ingestion is shared by all three methods, so this runs once regardless of
    which method you intend to evaluate.
    """
    from mmrag.config import load_experiment_config
    from mmrag.corpus import CorpusManifest
    from mmrag.ingestion.pipeline import IngestionPipeline

    cfg = load_experiment_config(config_name)
    if max_pages is not None:
        cfg.ingestion.max_pages_per_doc = max_pages

    manifest = CorpusManifest.load()
    ids = list(doc_id) if doc_id else None

    store = None
    if not no_postgres:
        from mmrag.stores.postgres import PostgresStore

        store = PostgresStore()

    def run(active_store: object | None) -> list[Any]:
        pipeline = IngestionPipeline(cfg, store=active_store)
        return pipeline.ingest_corpus(manifest, doc_ids=ids, force=force)

    if store is not None:
        try:
            with store:
                results = run(store)
        except Exception as exc:
            console.print(
                f"[yellow]Postgres unavailable ({exc}); writing sidecars only. "
                "Start it with 'docker compose up -d'.[/]"
            )
            results = run(None)
    else:
        results = run(None)

    table = Table(title="Ingestion results")
    table.add_column("doc_id", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("pages", justify="right")
    table.add_column("elements", justify="right")
    table.add_column("tables", justify="right")
    table.add_column("figures", justify="right")
    table.add_column("conf", justify="right")
    table.add_column("time", justify="right")
    table.add_column("db", justify="center")

    colours = {"ingested": "green", "skipped": "blue", "failed": "red"}
    for r in results:
        by_type = r.stats.get("by_type", {})
        figures = sum(by_type.get(t, 0) for t in ("figure", "chart", "diagram"))
        table.add_row(
            r.doc_id,
            f"[{colours.get(r.status, 'white')}]{r.status}[/]",
            str(r.n_pages),
            str(r.n_elements),
            str(by_type.get("table", 0)),
            str(figures),
            f"{r.stats.get('mean_confidence', 0):.2f}",
            f"{r.elapsed_s:.1f}s",
            "[green]yes[/]" if r.stored_in_postgres else "[yellow]no[/]",
        )
    console.print(table)

    failed = [r for r in results if not r.ok]
    for r in failed:
        console.print(f"[red]{r.doc_id}:[/] {r.message}")

    # A per-document Postgres failure is not fatal -- the sidecar is still
    # written -- but it must not be reportable only as a quiet "no" in a table
    # column, which is exactly how a NUL-byte write failure went unnoticed.
    if store is not None:
        not_stored = [r.doc_id for r in results if r.ok and not r.stored_in_postgres]
        if not_stored:
            console.print(
                f"[yellow]Warning:[/] {len(not_stored)} document(s) parsed but not written to "
                f"Postgres: {', '.join(not_stored)}\n"
                "  Their JSON sidecars are up to date; re-run once the cause is fixed. "
                "Run with --log-level DEBUG to see the database error."
            )

    if failed:
        raise typer.Exit(code=1)


@ingest_app.command("show")
def ingest_show(
    doc_id: str = typer.Argument(..., help="Document to inspect"),
    page: int | None = typer.Option(None, "--page", "-p", help="Restrict to one page"),
    element_type: str | None = typer.Option(None, "--type", "-t", help="Filter by element type"),
    geometry: bool = typer.Option(
        False, "--geometry", "-g", help="Show bbox, reading order and parent links"
    ),
    limit: int = typer.Option(30, "--limit", "-n"),
) -> None:
    """Inspect parsed elements and their provenance, straight from the sidecar."""
    from mmrag.ingestion.pipeline import read_sidecar, sidecar_path_for

    path = sidecar_path_for(doc_id)
    if not path.exists():
        console.print(f"[red]No parsed output for {doc_id}.[/] Run 'mmrag ingest run -d {doc_id}'.")
        raise typer.Exit(code=1)

    parsed = read_sidecar(path)
    doc = parsed.document
    console.print(
        f"[bold cyan]{doc.title}[/]\n"
        f"  type=[magenta]{doc.doc_type.value}[/] domain={doc.domain} lang={doc.language} "
        f"published={doc.publication_date}\n"
        f"  pages={doc.n_pages_ingested}/{doc.n_pages} sha256={doc.sha256[:12]} "
        f"parser={doc.parser_version}"
    )

    elements = parsed.elements
    if page is not None:
        elements = [e for e in elements if e.page_number == page]
    if element_type:
        elements = [e for e in elements if e.element_type.value == element_type]

    table = Table(
        title=f"Elements ({len(elements)} matching, showing {min(limit, len(elements))})",
        pad_edge=False,
    )
    # Explicit widths on the fixed columns; the content column takes whatever
    # is left. Without this rich starves the small columns to widen content.
    table.add_column("element", style="cyan", no_wrap=True, width=16)
    table.add_column("type", style="magenta", no_wrap=True, width=8)
    table.add_column("pg", justify="right", no_wrap=True, width=3)
    table.add_column("conf", justify="right", no_wrap=True, width=4)
    if geometry:
        # Only shown on request: these three columns squeeze the content column
        # to uselessness in an 80-column terminal.
        table.add_column("ord", justify="right", no_wrap=True, width=3)
        table.add_column("bbox", no_wrap=True, width=19)
        table.add_column("parent", no_wrap=True, width=12)
    table.add_column("content", overflow="ellipsis", no_wrap=True, max_width=60)

    for e in elements[:limit]:
        row = [
            e.element_id.split("#", 1)[-1],
            e.element_type.value,
            str(e.page_number),
            f"{e.extraction_confidence:.2f}",
        ]
        if geometry:
            bbox = (
                f"{e.bbox.x0:.2f},{e.bbox.y0:.2f}-{e.bbox.x1:.2f},{e.bbox.y1:.2f}"
                if e.bbox
                else "-"
            )
            row += [str(e.reading_order), bbox, (e.parent_id or "-").split("#")[-1]]
        row.append(e.best_text().replace("\n", " ")[:200] or "[dim](no text)[/]")
        table.add_row(*row)

    console.print(table)
    if not geometry:
        console.print("[dim]Pass --geometry for bounding boxes, reading order and parent links.[/]")


@ingest_app.command("status")
def ingest_status() -> None:
    """Per-document corpus statistics from Postgres."""
    from mmrag.stores.postgres import PostgresStore

    try:
        with PostgresStore() as store:
            rows = store.corpus_stats()
    except Exception as exc:
        console.print(f"[red]Postgres unavailable:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if not rows:
        console.print("[yellow]No documents ingested yet.[/] Run 'mmrag ingest run'.")
        return

    table = Table(title="Ingested corpus")
    table.add_column("doc_id", style="cyan", no_wrap=True)
    table.add_column("type", style="magenta")
    table.add_column("pages", justify="right")
    table.add_column("elements", justify="right")
    table.add_column("tables", justify="right")
    table.add_column("figures", justify="right")
    table.add_column("captioned", justify="right")
    table.add_column("conf", justify="right")

    for r in rows:
        table.add_row(
            r["doc_id"],
            r["doc_type"],
            f"{r['pages_stored']}/{r['n_pages']}",
            str(r["elements"]),
            str(r["tables"]),
            str(r["figures"]),
            str(r["captioned"]),
            str(r["mean_confidence"] or "-"),
        )
    console.print(table)
    console.print(
        f"[bold]Total:[/] {sum(r['elements'] for r in rows)} elements across {len(rows)} documents"
    )


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------


def _load_method(config_name: str) -> Any:
    """Build the pipeline a config selects.

    Dispatching on ``cfg.method`` rather than on the file name keeps the CLI
    identical across methods, which is what lets the same commands benchmark
    them against each other.
    """
    from mmrag.config import load_experiment_config

    cfg = load_experiment_config(config_name)
    return cfg, _build_method(cfg)


def _build_method(cfg: Any) -> Any:
    from mmrag.methods import build_method

    try:
        return build_method(cfg)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@index_app.command("build")
def index_build(
    config_name: str = typer.Option("method1", "--config", "-c"),
    doc_id: list[str] = typer.Option(None, "--doc-id", "-d", help="Restrict to these documents"),
    device: str | None = typer.Option(
        None, "--device", help="Method 3 only: auto | cpu | mps | cuda | cuda:N"
    ),
    allow_cpu: bool = typer.Option(
        False, "--allow-cpu", help="Method 3 only: permit a (slow) CPU page index build"
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", min=1, help="Method 3 only: pages per forward pass"
    ),
    max_pages: int | None = typer.Option(
        None, "--max-pages", min=1, help="Method 3 only: index just the first N pages (smoke test)"
    ),
) -> None:
    """Chunk the parsed corpus and build this method's indexes.

    For Method 3 this builds only the ColQwen2 page index; Method 2's indexes are
    reused as they are and must already exist wherever retrieval runs.
    """
    cfg, method = _load_method(config_name)
    ids = list(doc_id) if doc_id else None
    visual_options = {"device": device, "allow_cpu": allow_cpu or None,
                      "batch_size": batch_size, "max_pages": max_pages}

    if cfg.method == "method3":
        _build_method3_index(method, ids, visual_options)
        return
    if any(v is not None for v in visual_options.values()):
        raise typer.BadParameter(
            "--device/--allow-cpu/--batch-size/--max-pages apply to method3 only"
        )

    report = method.build_index(doc_ids=ids)

    console.print(
        f"[green]Built '{report.variant}' index[/] in {report.elapsed_s:.1f}s: "
        f"{report.n_chunks} chunks from {report.n_documents} documents"
    )

    table = Table(title="Chunks by type")
    table.add_column("type", style="cyan")
    table.add_column("count", justify="right")
    for chunk_type, count in sorted(report.by_type.items()):
        table.add_row(chunk_type, str(count))
    console.print(table)

    # Each method reports the loss that is characteristic of its own design.
    if hasattr(report, "invisible_figures"):
        console.print(
            f"[yellow]{report.invisible_figures}[/] figures had no retrievable text at all "
            "and are invisible to this method."
        )
        console.print(f"[dim]embedder: {report.embedder}[/]")
    else:
        sub = Table(title="Per-modality indexes")
        sub.add_column("index", style="cyan")
        sub.add_column("chunks", justify="right")
        sub.add_row("text (bm25 + dense)", str(report.n_text_indexed))
        sub.add_row("table (content + schema)", str(report.n_tables_indexed))
        sub.add_row("figure (clip + text)", str(report.n_figures_indexed))
        sub.add_row("  figure images embedded", str(report.n_figure_images_embedded))
        console.print(sub)
        console.print(
            f"[green]{report.text_invisible_recoverable}[/] figures carry no text at all "
            "but do have an image vector: content Method 1 cannot retrieve under any query."
        )
        if report.n_figures_without_image:
            console.print(
                f"[yellow]{report.n_figures_without_image}[/] figures have no image on disk "
                "and are invisible to this method too."
            )
        console.print(f"[dim]embedders: {report.embedders}[/]")


def _build_method3_index(method: Any, doc_ids: list[str] | None, options: dict[str, Any]) -> None:
    from mmrag.embeddings.visual import (
        DeviceUnavailableError,
        VisualEncodingError,
        VisualModelUnavailableError,
    )
    from mmrag.indexing.visual_pages import CpuIndexingRefusedError
    from mmrag.stores.multivector import IndexIntegrityError

    try:
        report = method.build_index(doc_ids=doc_ids, **options)
    except (CpuIndexingRefusedError, DeviceUnavailableError, VisualModelUnavailableError,
            VisualEncodingError,
            IndexIntegrityError, FileNotFoundError) as exc:
        console.print(f"[red]{type(exc).__name__}:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Built Method 3 page index[/] in {report.elapsed_s:.1f}s: {report.n_pages} pages "
        f"from {report.n_documents} documents, {report.n_tokens:,} token vectors "
        f"({report.index_bytes / 1e6:.0f} MB) on {report.device} as {report.dtype}"
    )
    console.print(f"  tokens per page: {report.tokens_per_page}")
    console.print(f"  model: {report.model}")
    if report.pages_without_chunks:
        console.print(
            f"[yellow]{report.pages_without_chunks}[/] indexed pages carry no chunk in the "
            "method2 chunk set, "
            "so retrieving them contributes nothing downstream."
        )
    if not report.complete:
        console.print("[yellow]Partial index (--doc-id/--max-pages): usable for smoke tests, "
                      "refused by 'mmrag eval run'.[/]")
    console.print(f"[dim]{report.index_dir}[/]")


@index_app.command("embed-queries")
def index_embed_queries(
    config_name: str = typer.Option("method3", "--config", "-c"),
    gold_path: str = typer.Option(DEFAULT_GOLD, "--gold"),
    generation_gold_path: str = typer.Option(DEFAULT_GENERATION_GOLD, "--generation-gold"),
    extra: list[str] = typer.Option(None, "--query", "-q", help="Additional query text"),
    device: str | None = typer.Option(None, "--device"),
    batch_size: int | None = typer.Option(None, "--batch-size", min=1),
) -> None:
    """Precompute Method 3 query embeddings for every benchmark query.

    Covers the retrieval gold queries and the generation gold's unanswerable
    questions, so the retrieval benchmark and generation can run on a machine
    without the ColQwen2 model.
    """
    from mmrag.embeddings.visual import DeviceUnavailableError, VisualModelUnavailableError
    from mmrag.evaluation.generation_gold import GenerationGold
    from mmrag.stores.multivector import IndexIntegrityError

    cfg, method = _load_method(config_name)
    if cfg.method != "method3":
        raise typer.BadParameter("embed-queries applies to method3 configs only")

    queries = [q.query for q in _load_gold(gold_path).queries]
    if Path(generation_gold_path).exists():
        queries += [u.query for u in GenerationGold.load(generation_gold_path).unanswerable]
    queries += list(extra or [])

    try:
        stats = method.embed_queries(queries, device=device, batch_size=batch_size)
    except (DeviceUnavailableError, VisualModelUnavailableError, IndexIntegrityError) as exc:
        console.print(f"[red]{type(exc).__name__}:[/] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]query embeddings[/] {stats} -> {method.query_cache_dir}")


@index_app.command("status")
def index_status(
    config_name: str = typer.Option("method1", "--config", "-c"),
) -> None:
    """Show what a built index contains."""
    import json

    from mmrag.methods.method1_textified import MANIFEST_FILE

    cfg, method = _load_method(config_name)
    if cfg.method == "method3":
        try:
            info = method.describe_visual_index()
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}:[/] {escape(str(exc))}")
            raise typer.Exit(code=1) from exc
        for key, value in info.items():
            console.print(f"  {key}: {value}")
        console.print("[dim]Method 3 also reads Method 2's indexes: "
                      "mmrag index status --config method2[/]")
        return

    manifest = method.index_dir / MANIFEST_FILE
    if not manifest.exists():
        console.print(f"[red]No index at {method.index_dir}.[/] Run 'mmrag index build'.")
        raise typer.Exit(code=1)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    report = payload["report"]
    console.print(
        f"[bold]{report['variant']}[/] ({payload['method']}): {report['n_chunks']} chunks from "
        f"{report['n_documents']} documents, built in {report['elapsed_s']}s"
    )
    console.print(f"  by type: {report['by_type']}")

    if "invisible_figures" in report:
        console.print(f"  figures with no text: {report['invisible_figures']}")
        console.print(f"  embedder: {report['embedder']}")
        collections = [report["variant"]]
    else:
        console.print(
            f"  text {report['n_text_indexed']} | tables {report['n_tables_indexed']} | "
            f"figures {report['n_figures_indexed']} "
            f"({report['n_figure_images_embedded']} with image vectors)"
        )
        console.print(
            f"  figures with no text but an image: {report['text_invisible_recoverable']}"
        )
        console.print(f"  embedders: {report['embedders']}")
        collections = [f"{report['variant']}_{n}" for n in ("text", "table_schema", "image")]

    for name in collections:
        try:
            from mmrag.stores.qdrant import QdrantStore

            console.print(f"  qdrant: {QdrantStore(name).describe()}")
        except Exception as exc:
            console.print(f"  [yellow]qdrant '{name}' unavailable: {exc}[/]")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@app.command()
def query(
    text: str = typer.Argument(..., help="The question to ask"),
    config_name: str = typer.Option("method1", "--config", "-c"),
    top_k: int | None = typer.Option(None, "--top-k", "-k"),
    doc_id: list[str] = typer.Option(None, "--doc-id", "-d", help="Restrict to these documents"),
    provider_name: str | None = typer.Option(
        None, "--provider", help="openai | local | echo (overrides .env)"
    ),
    retrieve_only: bool = typer.Option(
        False, "--retrieve-only", help="Show retrieved chunks without calling a model"
    ),
    show_text: bool = typer.Option(False, "--show-text", help="Print each chunk's full text"),
) -> None:
    """Ask a question against a built index."""
    _, method = _load_method(config_name)
    ids = list(doc_id) if doc_id else None

    if retrieve_only:
        result = method.retrieve(text, top_k=top_k, doc_ids=ids)
        _print_retrieval(result, show_text=show_text)
        return

    from mmrag.generation.providers import ProviderError, get_provider

    try:
        provider = get_provider(name=provider_name)
    except ProviderError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    answer = method.answer(text, provider, top_k=top_k, doc_ids=ids)

    console.print(f"\n[bold cyan]{answer.query}[/]\n")
    console.print(answer.text)

    if answer.citations:
        table = Table(title="Citations", show_lines=False)
        table.add_column("#", justify="right", width=3)
        table.add_column("document", style="cyan", no_wrap=True, width=26)
        table.add_column("page", justify="right", width=4)
        table.add_column("section", no_wrap=True, width=18)
        table.add_column("snippet", overflow="ellipsis", no_wrap=True, max_width=46)
        for n, citation in enumerate(answer.citations, start=1):
            table.add_row(
                str(n),
                citation.doc_title,
                str(citation.page_number),
                citation.section or "-",
                citation.snippet or "",
            )
        console.print(table)
    else:
        console.print("[yellow]The answer carried no resolvable citations.[/]")

    console.print(
        f"[dim]provider={answer.metadata.get('provider')} "
        f"model={answer.metadata.get('model')} "
        f"sources={answer.metadata.get('n_sources')} "
        f"tokens={answer.usage.get('total_tokens', 0)} "
        f"retrieval={answer.latency_ms.get('total_ms', 0):.0f}ms "
        f"generation={answer.latency_ms.get('generation_ms', 0):.0f}ms[/]"
    )
    if answer.metadata.get("refused"):
        console.print("[yellow]The model reported insufficient evidence.[/]")


def _print_retrieval(result: Any, *, show_text: bool = False) -> None:
    table = Table(title=f"Retrieved {len(result.results)} chunks")
    table.add_column("#", justify="right", width=3)
    table.add_column("document", style="cyan", no_wrap=True, width=22)
    table.add_column("pg", justify="right", width=4)
    table.add_column("type", style="magenta", width=7)
    table.add_column("score", justify="right", width=6)
    table.add_column("found by", width=12)
    table.add_column("text", overflow="ellipsis", no_wrap=True, max_width=44)

    for hit in result.results:
        chunk = hit.chunk
        body = chunk.text
        header = chunk.metadata.get("context_header")
        if header and body.startswith(header):
            body = body[len(header) :].lstrip()
        table.add_row(
            str(hit.rank),
            str(chunk.metadata.get("doc_title", chunk.doc_id)),
            str(chunk.page_number),
            chunk.chunk_type.value,
            f"{hit.score:.4f}",
            ",".join(f"{k}#{v}" for k, v in sorted(hit.component_ranks.items())),
            " ".join(body.split()),
        )
    console.print(table)

    # Method 2 routes before retrieving; showing the decision makes a bad route
    # diagnosable from the same command that produced the results.
    routing = getattr(result, "routing", None)
    if routing is not None:
        signals = "; ".join(f"{k}: {', '.join(v)}" for k, v in routing.signals.items())
        console.print(
            f"[cyan]routed to[/] {', '.join(m.value for m in routing.modalities)}"
            f"  (strategy={routing.strategy}, confidence={routing.confidence:.2f}"
            f"{', fell back' if routing.fell_back else ''})"
        )
        if signals:
            console.print(f"[dim]  signals -> {signals}[/]")
        if routing.filters.as_dict():
            console.print(f"[dim]  filters -> {routing.filters.as_dict()}[/]")

    console.print(f"[dim]{result.diagnostics}[/]")
    console.print(
        "[dim]" + " ".join(f"{k}={v:.0f}ms" for k, v in result.latency_ms.items()) + "[/]"
    )

    if show_text:
        for hit in result.results:
            console.print(f"\n[bold]#{hit.rank} {hit.chunk.chunk_id}[/]")
            console.print(hit.chunk.text)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Check that everything this project needs is actually available.

    Reports rather than fixes: each row says what is missing and what that
    disables, so a partial environment is still usable for the parts it covers.
    """
    from mmrag.config import get_settings

    settings = get_settings()
    table = Table(title="mmrag environment check")
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail", overflow="fold")

    def row(name: str, ok: bool | None, detail: str) -> None:
        mark = {True: "[green]ok[/]", False: "[red]missing[/]", None: "[yellow]optional[/]"}[ok]
        table.add_row(name, mark, detail)

    row("python", True, sys.version.split()[0])

    # --- torch / device ---
    try:
        import torch

        cuda = torch.cuda.is_available()
        row(
            "torch",
            True,
            f"{torch.__version__}, cuda={cuda}"
            + ("" if cuda else " -- Method 3 visual indexing must run on Colab/Kaggle"),
        )
    except ImportError:
        row("torch", False, "pip install torch")

    # --- parsers ---
    try:
        import pymupdf

        row("pymupdf", True, pymupdf.__doc__ or "installed")
    except ImportError:
        row("pymupdf", False, "pip install pymupdf -- required for ingestion")

    # --- postgres ---
    try:
        import psycopg

        with psycopg.connect(settings.postgres_dsn, connect_timeout=3) as conn:
            ver = conn.execute("SELECT version()").fetchone()
        row("postgres", True, str(ver[0]).split(",")[0] if ver else "connected")
    except Exception as exc:
        row("postgres", False, f"{settings.postgres_host}:{settings.postgres_port} -- {exc}")

    # --- qdrant ---
    try:
        import httpx

        r = httpx.get(f"{settings.qdrant_url}/readyz", timeout=3.0)
        row("qdrant", r.status_code == 200, f"{settings.qdrant_url} -> {r.status_code}")
    except Exception as exc:
        row("qdrant", False, f"{settings.qdrant_url} -- {exc}")

    # --- generation provider ---
    if settings.generation_provider == "openai":
        secret = settings.openai_api_key
        key = secret.get_secret_value().strip() if secret is not None else ""
        valid = bool(key) and not key.startswith("sk-replace")
        row("openai key", valid, "OPENAI_API_KEY set" if valid else "set OPENAI_API_KEY in .env")
    else:
        row("provider", True, f"{settings.generation_provider} (no OpenAI key needed)")

    # --- optional extras ---
    # Probe the backend the experiment config actually selects, not a fixed one:
    # reporting Tesseract as missing while rapidocr is configured and working
    # would send someone installing a binary they do not need.
    try:
        from mmrag.config import load_experiment_config
        from mmrag.ingestion.ocr import get_engine

        enrichment = load_experiment_config("method1").enrichment
        engine = get_engine(enrichment)
        if engine is not None:
            row(f"ocr ({engine.name})", True, engine.version)
        else:
            row(
                f"ocr ({enrichment.ocr_backend})",
                None,
                'not installed -- pip install -e ".[ocr]"',
            )
    except Exception as exc:  # pragma: no cover - defensive; doctor must not crash
        row("ocr", None, f"unavailable -- {type(exc).__name__}")

    try:
        import colpali_engine  # noqa: F401

        row("colpali-engine", True, "installed")
    except ImportError:
        row("colpali-engine", None,
            'Method 3 page encoding -- pip install -e ".[visual]" on a GPU machine')

    console.print(table)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def _validate_generation_gold(path: str, retrieval_gold) -> None:
    """Schema-check the generation sidecar and cross-check it against v1."""
    from mmrag.evaluation.generation_gold import (
        GenerationGold,
        describe,
        validate_against_retrieval_gold,
    )

    try:
        generation = GenerationGold.load(path)
    except Exception as exc:
        console.print(f"[red]{path} is not a valid generation gold file:[/] {exc}")
        raise typer.Exit(code=1) from exc

    info = describe(generation)
    console.print(
        f"[cyan]{path}[/] v{info['version']}: {info['n_answerable']} answerable "
        f"({info['n_required_facts']} required facts), {info['n_unanswerable']} unanswerable"
    )
    problems = validate_against_retrieval_gold(generation, retrieval_gold)
    for problem in problems:
        console.print(f"[red]  {problem.query_id}: {problem.kind}[/] -- {problem.detail}")
    if problems:
        raise typer.Exit(code=1)
    if info["needs_review"]:
        console.print(f"[yellow]  NEEDS REVIEW:[/] {info['needs_review']}")


def _load_gold(path: str):
    from mmrag.evaluation import GoldSet

    try:
        return GoldSet.load(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"no gold set at {path}") from exc
    except Exception as exc:
        console.print(f"[red]{path} is not a valid gold set:[/] {exc}")
        raise typer.Exit(code=1) from exc


@eval_app.command("validate")
def eval_validate(
    gold_path: str = typer.Option(DEFAULT_GOLD, "--gold"),
    config_name: str = typer.Option("method1", "--config", "-c"),
    generation_gold_path: str = typer.Option(
        DEFAULT_GENERATION_GOLD, "--generation-gold",
        help="Also validate the generation sidecar, if the file exists",
    ),
) -> None:
    """Check every gold evidence entry resolves against a built index.

    Run this after any re-ingest. A gold set whose pages have moved otherwise
    reports itself as a retrieval *failure*, and a method looks broken when in
    fact the labels drifted.
    """
    from mmrag.evaluation import validate_against_chunks

    gold = _load_gold(gold_path)
    if Path(generation_gold_path).exists():
        _validate_generation_gold(generation_gold_path, gold)
    _, method = _load_method(config_name)
    chunks = list(method.chunks.values())

    counts = gold.counts()
    console.print(
        f"[cyan]{gold_path}[/] v{gold.version}: {len(gold.queries)} queries, "
        f"{sum(len(q.evidence) for q in gold.queries)} evidence entries"
    )
    console.print(f"[dim]  stratum: {counts['stratum']}[/]")
    console.print(f"[dim]  requires: {counts['requires']}[/]")

    problems = validate_against_chunks(gold, chunks)
    if not problems:
        console.print(
            f"[green]All evidence resolves[/] against '{config_name}' ({len(chunks)} chunks)."
        )
        flagged = [
            q.id for q in gold.queries
            if any("NEEDS REVIEW" in (e.note or "") for e in q.evidence)
        ]
        if flagged:
            console.print(f"[yellow]{len(flagged)} entries marked NEEDS REVIEW:[/] {flagged}")
        return

    console.print(f"[red]{len(problems)} unresolvable evidence entries:[/]")
    table = Table(show_lines=False)
    table.add_column("query", style="cyan")
    table.add_column("problem")
    table.add_column("detail", overflow="fold")
    for problem in problems:
        table.add_row(problem.query_id, problem.kind, problem.detail)
    console.print(table)
    raise typer.Exit(code=1)


@eval_app.command("run")
def eval_run(
    config_name: str = typer.Option("method1", "--config", "-c"),
    gold_path: str = typer.Option(DEFAULT_GOLD, "--gold"),
    tag: str | None = typer.Option(None, "--tag", help="Label for this run in reports"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Ablate the cross-encoder"),
    no_metadata: bool = typer.Option(
        False, "--no-metadata", help="Ablate Method 2's document resolver"
    ),
    out_dir: str = typer.Option("data/eval/runs", "--out"),
) -> None:
    """Score one method over the gold set and save the run.

    Ablations are applied here rather than by editing a config file, so both
    arms of a comparison come from the same committed configuration.
    """
    from mmrag.config import load_experiment_config
    from mmrag.evaluation.report import render, save_run
    from mmrag.evaluation.retrieval_eval import run_evaluation

    gold = _load_gold(gold_path)
    config = load_experiment_config(config_name)

    overrides: dict[str, Any] = {}
    if no_rerank:
        config.retrieval.rerank_enabled = False
        overrides["rerank_enabled"] = False
    if no_metadata:
        overrides["use_metadata"] = False

    method = _build_method(config)

    visual_fingerprint: dict[str, str] = {}
    if config.method == "method3":
        # A benchmark number from a partial or mismatched page index would be a
        # fake result, so it is refused before any query runs.
        try:
            info = method.describe_visual_index()
        except Exception as exc:
            console.print(f"[red]{type(exc).__name__}:[/] {escape(str(exc))}")
            raise typer.Exit(code=1) from exc
        if not info["complete"]:
            console.print("[red]The Method 3 page index is a partial test build.[/] Rebuild it "
                          "over the whole corpus before evaluating.")
            raise typer.Exit(code=1)
        visual_fingerprint = {
            "visual_index_version": str(info["index_version"]),
            "visual_model": str(info["model"]),
            "visual_index_device": f"{info['device']}/{info['dtype']}",
            "visual_embeddings_sha256": str(info["embeddings_sha256"]),
            "visual_query_cache": str(info["cached_queries"]),
        }

    label = tag or ("no-rerank" if no_rerank else None)
    console.print(
        f"Evaluating [cyan]{config.method}[/] over {len(gold.queries)} queries"
        + (f" [yellow](overrides: {overrides})[/]" if overrides else "")
    )
    if config.retrieval.rerank_enabled:
        console.print("[dim]  reranking is on; on CPU this is ~18s per query[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("retrieving", total=len(gold.queries))

        def advance(index: int, total: int, result: Any) -> None:
            progress.update(
                task, completed=index, description=f"[dim]{result.query_id}[/]"
            )

        run = run_evaluation(
            method,
            gold,
            config,
            config_name=config_name,
            tag=label,
            use_metadata=False if no_metadata else None,
            overrides=overrides,
            on_query=advance,
        )

    run.environment.update(visual_fingerprint)
    stamp = run.started_at.replace(":", "").replace("-", "")
    suffix = f"_{label}" if label else ""
    path = save_run(run, Path(out_dir) / f"{stamp}_{run.method}{suffix}.json")

    render([run], console=console)
    console.print(f"\n[green]saved[/] {path}")


@eval_app.command("compare")
def eval_compare(
    runs: list[str] = typer.Argument(..., help="Run JSON files; the first is the baseline"),
    markdown: bool = typer.Option(False, "--markdown", help="Emit the headline table as Markdown"),
) -> None:
    """Compare saved runs. The first is treated as the baseline for deltas.

    Takes either retrieval runs or judged generation runs, not a mix: the two
    report different quantities over different query sets.
    """
    from mmrag.evaluation.generation_report import run_kind
    from mmrag.evaluation.report import load_run, render, to_markdown

    for path in runs:
        if not Path(path).exists():
            raise typer.BadParameter(f"no run file at {path}")
    kinds = {run_kind(path) for path in runs}

    if kinds == {"judged"}:
        from mmrag.evaluation.generation_report import render_judged, to_markdown_judged
        from mmrag.evaluation.judged_eval import JudgedRun

        judged = [JudgedRun.load(path) for path in runs]
        render_judged(judged, console=console)
        if markdown:
            console.print("\n[dim]-- Markdown --[/]")
            print(to_markdown_judged(judged))
        return
    if kinds != {"retrieval"}:
        console.print(
            f"[red]cannot compare run kinds {sorted(kinds)}.[/] Pass only retrieval runs or "
            "only judged runs; a generation run must be judged first (mmrag eval judge)."
        )
        raise typer.Exit(code=1)

    loaded = [load_run(path) for path in runs]
    render(loaded, console=console)
    if markdown:
        console.print("\n[dim]-- Markdown --[/]")
        print(to_markdown(loaded))


def _method_from_run(run: Any):
    """Rebuild the exact method a retrieval run used, including its overrides."""
    from mmrag.config import ExperimentConfig

    config = ExperimentConfig.model_validate(run.config)
    if run.overrides.get("rerank_enabled") is False:
        config.retrieval.rerank_enabled = False
    return config, _build_method(config)


def _resolve_recorded_path(recorded: str | None) -> Path | None:
    """A path recorded in a run file, as given or relative to the repository root.

    Run files record paths as they were typed, usually relative to the root, so
    a command started from another directory must not lose track of them.
    """
    if not recorded:
        return None
    path = Path(recorded)
    if path.exists():
        return path
    if not path.is_absolute():
        from mmrag.config import PROJECT_ROOT

        rooted = PROJECT_ROOT / path
        if rooted.exists():
            return rooted
    return None


def _parse_ids(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


@eval_app.command("generate")
def eval_generate(
    retrieval_run_path: str = typer.Option(..., "--retrieval-run", help="Retrieval run JSON"),
    gold_path: str = typer.Option(DEFAULT_GOLD, "--gold"),
    generation_gold_path: str = typer.Option(DEFAULT_GENERATION_GOLD, "--generation-gold"),
    provider_name: str | None = typer.Option(None, "--provider", help="openai | local | echo"),
    tag: str | None = typer.Option(None, "--tag"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    queries: str | None = typer.Option(None, "--queries", help="Comma-separated ids, e.g. q001,u003"),
    concurrency: int = typer.Option(4, "--concurrency", min=1, max=16),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build prompts and estimate; call nothing"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore cached answers and overwrite them"),
    no_unanswerable: bool = typer.Option(False, "--no-unanswerable"),
    price_in: float | None = typer.Option(None, "--price-in", help="USD per 1M input tokens"),
    price_out: float | None = typer.Option(None, "--price-out", help="USD per 1M output tokens"),
    cache_dir: str = typer.Option("data/eval/cache", "--cache-dir"),
    out_dir: str = typer.Option("data/eval/generation", "--out"),
) -> None:
    """Generate answers from a saved retrieval run.

    Uses the exact chunks the retrieval run scored, so no retrieval is re-run for
    the gold queries. Unanswerable queries are retrieved live with the run's own
    configuration. Nothing is called with --dry-run.
    """
    from mmrag.config import get_settings
    from mmrag.evaluation.generation_eval import (
        PROMPT_VERSION,
        DryRunProvider,
        GenerationRun,
        RetrievalRunMismatch,
        aggregate,
        environment,
        estimate,
        file_fingerprint,
        now_iso,
        run_generation,
        select_work,
        totals,
    )
    from mmrag.evaluation.generation_gold import (
        GenerationGold,
        validate_against_retrieval_gold,
    )
    from mmrag.evaluation.llm_cache import CachingProvider
    from mmrag.evaluation.report import load_run
    from mmrag.generation.answerer import Answerer
    from mmrag.generation.providers import ProviderError, get_provider

    started = time.perf_counter()
    run = load_run(retrieval_run_path)
    gold = _load_gold(gold_path)
    generation_gold = GenerationGold.load(generation_gold_path)
    problems = validate_against_retrieval_gold(generation_gold, gold)
    if problems:
        for p in problems:
            console.print(f"[red]{p.query_id}: {p.kind}[/] -- {p.detail}")
        raise typer.Exit(code=1)

    config, method = _method_from_run(run)
    top_k = max(config.evaluation.k_values)
    use_metadata = run.overrides.get("use_metadata")

    def live_retrieve(query: str):
        kwargs: dict[str, Any] = {"top_k": top_k}
        if use_metadata is False:
            kwargs["use_metadata"] = False
        return method.retrieve(query, **kwargs).results

    try:
        items = select_work(
            run, gold, generation_gold, method.chunks,
            include_unanswerable=not no_unanswerable,
            query_ids=_parse_ids(queries), limit=limit, live_retrieve=live_retrieve,
        )
    except (RetrievalRunMismatch, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    settings = get_settings()
    name = provider_name or settings.generation_provider
    if dry_run:
        inner: Any = DryRunProvider(name)
    else:
        try:
            inner = get_provider(
                name=name,
                timeout=config.generation.request_timeout_s,
                max_retries=config.generation.max_retries,
            )
        except ProviderError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    provider = CachingProvider(
        inner, cache_dir, kind="generation", prompt_version=PROMPT_VERSION,
        seed=config.evaluation.llm_seed, enabled=not no_cache, refresh=refresh,
    )
    answerer = Answerer(config.generation, provider)

    label = tag or run.label()
    console.print(
        f"Generating for [cyan]{run.label()}[/] — {len(items)} queries "
        f"({sum(i.answerable for i in items)} answerable, "
        f"{sum(not i.answerable for i in items)} unanswerable), "
        f"model {config.generation.text_model}, provider {name}"
        + (" [yellow](dry run)[/]" if dry_run else "")
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console) as progress:
        task = progress.add_task("generating", total=len(items))
        records = run_generation(
            items, answerer, provider, dry_run=dry_run, concurrency=concurrency,
            attempts=2, on_record=lambda r: progress.advance(task),
        )

    if dry_run:
        projection = estimate(records, price_in_per_m=price_in, price_out_per_m=price_out)
        table = Table(title="Dry run — projected cost of generating and then judging")
        for col in ("stage", "calls", "input tokens", "output tokens", "cost (USD)"):
            table.add_column(col, justify="right" if col != "stage" else "left")
        for stage in ("generation", "judge", "total"):
            p = projection[stage]
            table.add_row(stage, str(p["calls"]), f"{p['input_tokens']:,}",
                          f"{p['output_tokens']:,}",
                          "-" if p["cost_usd"] is None else f"{p['cost_usd']:.4f}")
        console.print(table)
        console.print(
            f"[dim]generation cache hits: {projection['generation']['cache_hits']}. "
            "Tokens are estimated with the context-budget tokenizer, not OpenAI's. "
            "No provider was called.[/]"
        )
        if price_in is None or price_out is None:
            console.print("[dim]Pass --price-in and --price-out (USD per 1M tokens) for a cost.[/]")
        return

    result = GenerationRun(
        method=run.method, label=label, created_at=now_iso(),
        elapsed_s=round(time.perf_counter() - started, 1),
        retrieval_run={**file_fingerprint(retrieval_run_path), "label": run.label(),
                       "started_at": run.started_at, "overrides": run.overrides},
        gold={**file_fingerprint(gold_path), "version": gold.version},
        generation_gold={**file_fingerprint(generation_gold_path),
                         "version": generation_gold.version},
        generation={"provider": name, "model": config.generation.text_model,
                    "temperature": config.generation.temperature,
                    "max_output_tokens": config.generation.max_output_tokens,
                    "max_context_tokens": config.generation.max_context_tokens,
                    "refuse_without_evidence": config.generation.refuse_without_evidence,
                    "seed": config.evaluation.llm_seed, "prompt_version": PROMPT_VERSION,
                    "top_k": top_k},
        environment=environment(), cache=provider.stats.as_dict(),
        totals=totals(records, provider), records=records, metrics=aggregate(records),
    )
    stamp = result.created_at.replace(":", "").replace("-", "")
    path = result.save(Path(out_dir) / f"{stamp}_{label.replace('/', '_')}_generation.json")

    m = result.metrics
    console.print(
        f"\n[green]{m['n_ok']}[/] ok, [red]{m['n_errors']}[/] errors. "
        f"Calls made {result.totals['calls_made']}, cache hits {result.totals['cache_hits']}, "
        f"tokens {result.totals['prompt_tokens']:,} in / {result.totals['completion_tokens']:,} out."
    )
    a = m["answerable"]
    console.print(
        f"answerable: evidence in context {a['evidence_in_context']['count']}/"
        f"{a['evidence_in_context']['n']}, refused {a['refused']['count']}/{a['refused']['n']}, "
        f"cited {a['citation_presence']['count']}/{a['citation_presence']['n']}"
    )
    u = m["unanswerable"]
    if u["n"]:
        console.print(f"unanswerable: refused {u['refused']['count']}/{u['refused']['n']}")
    console.print(f"[green]saved[/] {path}")


@eval_app.command("judge")
def eval_judge(
    generation_run_path: str = typer.Option(..., "--generation-run", help="Generation run JSON"),
    provider_name: str | None = typer.Option(None, "--provider", help="openai | local | echo"),
    tag: str | None = typer.Option(None, "--tag"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    queries: str | None = typer.Option(None, "--queries"),
    concurrency: int = typer.Option(4, "--concurrency", min=1, max=16),
    dry_run: bool = typer.Option(False, "--dry-run"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    refresh: bool = typer.Option(False, "--refresh"),
    price_in: float | None = typer.Option(None, "--price-in", help="USD per 1M input tokens"),
    price_out: float | None = typer.Option(None, "--price-out", help="USD per 1M output tokens"),
    cache_dir: str = typer.Option("data/eval/cache", "--cache-dir"),
    out_dir: str = typer.Option("data/eval/generation", "--out"),
) -> None:
    """Judge a generation run. Re-runnable without regenerating anything.

    The judge sees only the question, the sources the generator saw, the answer
    and the reference facts, and returns observations; every score is computed
    from those in Python. The judge model is evaluation.llm_judge_model.
    """
    from mmrag.config import ExperimentConfig, get_settings
    from mmrag.evaluation.generation_eval import (
        DryRunProvider,
        GenerationRun,
        environment,
        file_fingerprint,
        now_iso,
    )
    from mmrag.evaluation.judge import JUDGE_PROMPT_VERSION
    from mmrag.evaluation.judged_eval import JudgedRun, aggregate_judged, run_judging
    from mmrag.evaluation.llm_cache import CachingProvider
    from mmrag.evaluation.report import load_run
    from mmrag.generation.providers import ProviderError, get_provider
    from mmrag.textify.tokens import get_token_counter

    started = time.perf_counter()
    if not Path(generation_run_path).exists():
        raise typer.BadParameter(f"no generation run at {generation_run_path}")
    generation = GenerationRun.load(generation_run_path)

    # The retrieval run supplies the experiment config, and with it the judge
    # model, seed and output cap.
    recorded = generation.retrieval_run.get("path")
    retrieval_path = _resolve_recorded_path(recorded)
    if retrieval_path is None:
        console.print(f"[red]the retrieval run this generation came from is missing:[/] {recorded}")
        raise typer.Exit(code=1)
    expected_sha = generation.retrieval_run.get("sha256")
    if expected_sha and file_fingerprint(retrieval_path)["sha256"] != expected_sha:
        console.print(f"[yellow]warning: {retrieval_path} has changed since this generation run "
                      "was produced; its config is used as found.[/]")
    retrieval_run = load_run(retrieval_path)
    config = ExperimentConfig.model_validate(retrieval_run.config)
    model = config.evaluation.llm_judge_model
    max_out = config.evaluation.judge_max_output_tokens

    # Every record is kept: one whose generation failed is not judged, but it
    # stays in the run as an `error` so coverage and the taxonomy account for it.
    records = list(generation.records)
    wanted = _parse_ids(queries)
    if wanted:
        unknown = sorted(set(wanted) - {r.query_id for r in generation.records})
        if unknown:
            console.print(f"[red]unknown query id(s): {unknown}[/]")
            raise typer.Exit(code=1)
        records = [r for r in records if r.query_id in set(wanted)]
    if limit is not None:
        records = records[:limit]

    settings = get_settings()
    name = provider_name or settings.generation_provider
    if dry_run:
        inner: Any = DryRunProvider(name)
    else:
        try:
            inner = get_provider(name=name, timeout=config.generation.request_timeout_s,
                                 max_retries=config.generation.max_retries)
        except ProviderError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    provider = CachingProvider(inner, cache_dir, kind="judge", prompt_version=JUDGE_PROMPT_VERSION,
                               seed=config.evaluation.llm_seed, enabled=not no_cache,
                               refresh=refresh)
    self_judge = model == generation.generation.get("model")
    unjudgeable = sum(1 for r in records if r.status != "ok" or r.answer is None)
    console.print(
        f"Judging [cyan]{generation.label}[/] — {len(records) - unjudgeable} answers with {model}"
        + (f", {unjudgeable} without an answer recorded as errors" if unjudgeable else "")
        + (" [yellow](dry run)[/]" if dry_run else "")
    )
    if self_judge:
        console.print("[dim]  judge model = generation model: a self-judge. Absolute scores are "
                      "directional; compare methods rather than reading them as ground truth.[/]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console) as progress:
        task = progress.add_task("judging", total=len(records))
        judged = run_judging(records, provider, model=model, max_output_tokens=max_out,
                             dry_run=dry_run, concurrency=concurrency, attempts=2,
                             on_record=lambda r: progress.advance(task))

    if dry_run:
        from mmrag.evaluation.judge import build_judge_messages

        counter = get_token_counter()
        # Records without an answer were skipped, not looked up, and cost nothing.
        todo = [j for j in judged if j.judge.status == "dry_run" and not j.judge.cache_hit]
        cached = sum(1 for j in judged if j.judge.status == "dry_run" and j.judge.cache_hit)
        tokens_in = sum(
            sum(counter.count(m.content) for m in build_judge_messages(
                j.generation.query, j.generation.sources_block, j.generation.answer or "",
                j.generation.required_facts))
            for j in todo
        )
        tokens_out = 700 * len(todo)
        cost = (None if price_in is None or price_out is None
                else tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out)
        console.print(
            f"Dry run: {len(todo)} judge calls needed ({cached} cached), "
            f"~{tokens_in:,} input and ~{tokens_out:,} output tokens"
            + (f", ~${cost:.4f}" if cost is not None else "") + ". No provider was called."
        )
        return

    result = JudgedRun(
        method=generation.method, label=tag or generation.label, created_at=now_iso(),
        elapsed_s=round(time.perf_counter() - started, 1),
        generation_run={**file_fingerprint(generation_run_path), "label": generation.label},
        generation=generation.generation,
        judge={"provider": name, "model": model, "temperature": 0.0,
               "max_output_tokens": max_out, "seed": config.evaluation.llm_seed,
               "prompt_version": JUDGE_PROMPT_VERSION, "self_judge": self_judge},
        environment=environment(), cache=provider.stats.as_dict(),
        totals={"calls_made": provider.stats.misses, "cache_hits": provider.stats.hits,
                "prompt_tokens": sum(j.judge.usage.get("prompt_tokens", 0) for j in judged),
                "completion_tokens": sum(
                    j.judge.usage.get("completion_tokens", 0) for j in judged)},
        records=judged, metrics=aggregate_judged(judged),
    )
    stamp = result.created_at.replace(":", "").replace("-", "")
    path = result.save(Path(out_dir) / f"{stamp}_{result.label.replace('/', '_')}_judged.json")

    from mmrag.evaluation.generation_report import render_judged

    render_judged([result], console=console)
    console.print(f"[green]saved[/] {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
