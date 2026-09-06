"""``mmrag`` command line interface.

Every stage of the benchmark is reachable from here, so a run is always
reproducible from a shell history rather than from a notebook someone ran once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
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
app.add_typer(corpus_app, name="corpus")
app.add_typer(config_app, name="config")

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
    try:
        import pytesseract

        row("tesseract", True, str(pytesseract.get_tesseract_version()))
    except Exception as exc:
        row("tesseract", None, f"OCR disabled -- {type(exc).__name__}")

    try:
        import colpali_engine  # noqa: F401

        row("colpali-engine", True, "installed")
    except ImportError:
        row("colpali-engine", None, "Method 3 visual index must be built out-of-band")

    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
