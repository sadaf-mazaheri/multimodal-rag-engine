"""Fetch and verify the corpus described by the manifest.

Deliberately defensive, because the inputs are other people's web servers:
redirects, rate limits, HTML error pages served with a 200, and PDFs that get
silently re-issued are all normal. Every download is streamed to a temporary
file, hashed, sniffed for a PDF header, and only then moved into place -- so an
interrupted or bogus download can never leave a half-file that later ingestion
would happily parse into garbage.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from mmrag.config import RAW_DIR, ensure_data_dirs
from mmrag.corpus.manifest import (
    DEFAULT_LOCK_PATH,
    CorpusEntry,
    CorpusLock,
    CorpusLockEntry,
    CorpusManifest,
)
from mmrag.logging_utils import get_logger

log = get_logger(__name__)

# Some publishers (SEC, a few government hosts) reject requests without a real
# User-Agent. Identify the project honestly rather than impersonating a browser.
USER_AGENT = "mmrag-benchmark/0.1 (+https://github.com/; research corpus fetcher)"

CHUNK_BYTES = 1 << 16  # 64 KiB
PDF_MAGIC = b"%PDF-"


@dataclass
class DownloadResult:
    entry: CorpusEntry
    path: Path | None
    status: str  # downloaded | cached | failed | hash_mismatch
    sha256: str | None = None
    size_bytes: int | None = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"downloaded", "cached"}


def sha256_file(path: Path, *, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _looks_like_pdf(path: Path) -> bool:
    """Guard against HTML error pages saved with a .pdf name.

    The header is not always at byte 0 -- a few generators emit leading
    whitespace or a BOM -- so scan a small prefix rather than testing byte 0.
    """
    with path.open("rb") as fh:
        return PDF_MAGIC in fh.read(1024)


def download_entry(
    entry: CorpusEntry,
    *,
    raw_dir: Path | None = None,
    force: bool = False,
    timeout: float = 60.0,
    max_retries: int = 3,
    client: httpx.Client | None = None,
) -> DownloadResult:
    """Download one manifest entry, verifying integrity before committing it."""
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = entry.local_path(raw_dir)

    # --- fast path: already have a good copy -------------------------------
    if target.exists() and not force:
        actual = sha256_file(target)
        if entry.sha256 is None or actual == entry.sha256:
            return DownloadResult(
                entry, target, "cached", sha256=actual, size_bytes=target.stat().st_size
            )
        log.warning(
            "%s: cached file hash %s does not match pinned %s; re-downloading",
            entry.doc_id,
            actual[:12],
            entry.sha256[:12],
        )

    owns_client = client is None
    client = client or httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"},
    )

    tmp = target.with_suffix(".pdf.part")
    try:
        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                with client.stream("GET", entry.url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length") or 0)
                    written = 0
                    with tmp.open("wb") as fh:
                        for block in response.iter_bytes(CHUNK_BYTES):
                            fh.write(block)
                            written += len(block)
                    if total and written != total:
                        raise OSError(f"truncated: got {written} of {total} bytes")

                if not _looks_like_pdf(tmp):
                    raise OSError("response is not a PDF (probably an error or consent page)")

                actual = sha256_file(tmp)
                size = tmp.stat().st_size

                if entry.sha256 is not None and actual != entry.sha256:
                    tmp.unlink(missing_ok=True)
                    return DownloadResult(
                        entry,
                        None,
                        "hash_mismatch",
                        sha256=actual,
                        size_bytes=size,
                        message=(
                            f"expected {entry.sha256}, got {actual}. The document at this URL "
                            "has changed; re-pin it with 'mmrag corpus lock' after review."
                        ),
                    )

                shutil.move(str(tmp), str(target))
                log.info("%s: downloaded %.1f MB", entry.doc_id, size / 1e6)
                return DownloadResult(entry, target, "downloaded", sha256=actual, size_bytes=size)

            except (httpx.HTTPError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                tmp.unlink(missing_ok=True)
                if attempt < max_retries:
                    backoff = 2.0**attempt
                    log.warning(
                        "%s: attempt %d/%d failed (%s); retrying in %.0fs",
                        entry.doc_id,
                        attempt,
                        max_retries,
                        last_error,
                        backoff,
                    )
                    time.sleep(backoff)

        return DownloadResult(entry, None, "failed", message=last_error)
    finally:
        tmp.unlink(missing_ok=True)
        if owns_client:
            client.close()


def download_corpus(
    manifest: CorpusManifest | None = None,
    *,
    doc_ids: list[str] | None = None,
    raw_dir: Path | None = None,
    force: bool = False,
    include_disabled: bool = False,
) -> list[DownloadResult]:
    """Download every (selected) manifest entry, reusing one HTTP connection."""
    ensure_data_dirs()
    manifest = manifest or CorpusManifest.load()

    entries = manifest.entries if include_disabled else manifest.active
    if doc_ids:
        wanted = set(doc_ids)
        entries = [e for e in manifest.entries if e.doc_id in wanted]
        missing = wanted - {e.doc_id for e in entries}
        if missing:
            raise KeyError(f"doc_ids not in manifest: {sorted(missing)}")

    results: list[DownloadResult] = []
    with httpx.Client(
        follow_redirects=True,
        timeout=60.0,
        headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"},
    ) as client:
        for entry in entries:
            results.append(download_entry(entry, raw_dir=raw_dir, force=force, client=client))
    return results


def lock_manifest(
    manifest: CorpusManifest | None = None,
    *,
    raw_dir: Path | None = None,
    lock_path: Path | None = None,
) -> tuple[CorpusLock, list[str]]:
    """Record the hash, size, and page count of each downloaded file.

    Writes ``configs/corpus.lock.yaml``. The hand-authored ``corpus.yaml`` is
    never touched, so its comments and structure survive re-locking.

    Existing pins for documents that are not present locally are preserved --
    locking a single re-downloaded document must not silently unpin the rest of
    the corpus.
    """
    manifest = manifest or CorpusManifest.load(apply_lock=False)
    raw_dir = raw_dir or RAW_DIR
    lp = lock_path or DEFAULT_LOCK_PATH

    lock = CorpusLock.load(lp) if lp.exists() else CorpusLock()
    updated: list[str] = []

    for entry in manifest.entries:
        local = entry.local_path(raw_dir)
        if not local.exists():
            continue
        previous = lock.entries.get(entry.doc_id)
        lock.entries[entry.doc_id] = CorpusLockEntry(
            sha256=sha256_file(local),
            size_bytes=local.stat().st_size,
            # Re-reading the page count is cheap, but keep the old value if the
            # PDF cannot be opened rather than dropping it to null.
            n_pages=_page_count(local) or (previous.n_pages if previous else None),
        )
        updated.append(entry.doc_id)

    lock.save(lp)
    return lock, updated


def _page_count(path: Path) -> int | None:
    """Page count via PyMuPDF, tolerating an unreadable file."""
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception as exc:  # pragma: no cover - depends on the PDF
        log.warning("could not read page count for %s: %s", path.name, exc)
        return None
