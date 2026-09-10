"""Ingestion orchestration: corpus in, shared representation out.

Writes to two places, deliberately:

* **Postgres** -- the queryable source of truth for all three methods.
* **A JSON sidecar per document** under ``data/processed/`` -- the same records,
  portable. Method 3's visual index is built on a Colab GPU with no access to a
  local database, and debugging a parse should not require SQL. The sidecar is
  written from the same objects, so the two cannot drift.

Postgres is optional at this stage: a parse that cannot reach the database still
produces its sidecar and reports what it did, because a broken container should
not block work on the parser.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mmrag.config import PROCESSED_DIR, RAW_DIR, ExperimentConfig, ensure_data_dirs
from mmrag.corpus.manifest import CorpusEntry, CorpusManifest
from mmrag.ingestion.ocr import OcrEngine, enrich_elements_with_ocr, get_engine
from mmrag.ingestion.parser import ParsedDocument, PdfParser
from mmrag.logging_utils import get_logger
from mmrag.schemas import Document, Element, Page

log = get_logger(__name__)

SIDECAR_VERSION = 1


@dataclass
class IngestionResult:
    """What happened to one document."""

    doc_id: str
    status: str  # ingested | skipped | failed
    n_pages: int = 0
    n_elements: int = 0
    elapsed_s: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)
    sidecar_path: Path | None = None
    stored_in_postgres: bool = False
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ingested", "skipped"}


def sidecar_path_for(doc_id: str, processed_dir: Path | None = None) -> Path:
    return (processed_dir or PROCESSED_DIR) / doc_id / "parsed.json"


def write_sidecar(
    parsed: ParsedDocument, path: Path, *, stats: dict[str, Any] | None = None
) -> Path:
    """Serialise a parsed document to JSON.

    Uses the Pydantic models' own JSON mode, so the sidecar and the database
    hold the same values with the same names -- no second serialisation format
    to keep in step.

    ``stats`` overrides the document's own tally, so a caller that ran an
    enrichment pass can record what it did alongside the parse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sidecar_version": SIDECAR_VERSION,
        "document": parsed.document.model_dump(mode="json"),
        "pages": [p.model_dump(mode="json") for p in parsed.pages],
        "elements": [e.model_dump(mode="json") for e in parsed.elements],
        "stats": stats if stats is not None else parsed.stats,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_sidecar(path: Path) -> ParsedDocument:
    """Load a parsed document back from its sidecar."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("sidecar_version")
    if version != SIDECAR_VERSION:
        raise ValueError(
            f"{path} was written by sidecar version {version}, this build expects "
            f"{SIDECAR_VERSION}; re-run 'mmrag ingest run --force'"
        )
    return ParsedDocument(
        document=Document.model_validate(payload["document"]),
        pages=[Page.model_validate(p) for p in payload["pages"]],
        elements=[Element.model_validate(e) for e in payload["elements"]],
    )


class IngestionPipeline:
    """Parses corpus documents and persists the shared representation."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        raw_dir: Path | None = None,
        processed_dir: Path | None = None,
        store: Any | None = None,
    ):
        self.config = config
        self.raw_dir = raw_dir or RAW_DIR
        self.processed_dir = processed_dir or PROCESSED_DIR
        self.store = store
        self.parser = PdfParser(config.ingestion, output_dir=self.processed_dir)
        # Built once per pipeline rather than per document: loading an OCR model
        # costs about as much as reading a small PDF, and a 14-document run
        # would otherwise pay for it fourteen times.
        self._ocr_engine_cached: OcrEngine | None = None
        self._ocr_engine_resolved = False

    @property
    def _ocr_engine(self) -> OcrEngine | None:
        if not self._ocr_engine_resolved:
            self._ocr_engine_resolved = True
            if self.config.enrichment.ocr_enabled:
                self._ocr_engine_cached = get_engine(self.config.enrichment)
        return self._ocr_engine_cached

    def ingest_entry(self, entry: CorpusEntry, *, force: bool = False) -> IngestionResult:
        """Parse and persist one document."""
        started = time.perf_counter()
        pdf_path = entry.local_path(self.raw_dir)
        sidecar = sidecar_path_for(entry.doc_id, self.processed_dir)

        if not pdf_path.exists():
            return IngestionResult(
                entry.doc_id,
                "failed",
                message=f"{pdf_path.name} is missing; run 'mmrag corpus download'",
            )

        if sidecar.exists() and not force:
            try:
                cached = read_sidecar(sidecar)
                stored = self._persist(cached)
                return IngestionResult(
                    entry.doc_id,
                    "skipped",
                    n_pages=len(cached.pages),
                    n_elements=len(cached.elements),
                    elapsed_s=time.perf_counter() - started,
                    stats=cached.stats,
                    sidecar_path=sidecar,
                    stored_in_postgres=stored,
                    message="already parsed; use --force to re-parse",
                )
            except (ValueError, KeyError) as exc:
                # A stale or corrupt sidecar must not be silently trusted.
                log.warning("%s: unusable sidecar (%s); re-parsing", entry.doc_id, exc)

        try:
            parsed = self.parser.parse(entry, pdf_path)
        except Exception as exc:
            log.exception("%s: parsing failed", entry.doc_id)
            return IngestionResult(
                entry.doc_id,
                "failed",
                elapsed_s=time.perf_counter() - started,
                message=f"{type(exc).__name__}: {exc}",
            )

        # Enrichment runs here, between parsing and persistence, so recovered
        # text becomes part of the shared representation. Both methods read
        # these sidecars, so neither can be enriched without the other -- which
        # is what keeps a Method 1 vs Method 2 difference attributable to
        # retrieval rather than to one of them having been given better text.
        ocr_report = enrich_elements_with_ocr(
            parsed.elements, self.config.enrichment, engine=self._ocr_engine
        )
        stats = {**parsed.stats, "ocr": ocr_report.as_dict()}

        write_sidecar(parsed, sidecar, stats=stats)
        stored = self._persist(parsed)

        return IngestionResult(
            entry.doc_id,
            "ingested",
            n_pages=len(parsed.pages),
            n_elements=len(parsed.elements),
            elapsed_s=time.perf_counter() - started,
            stats=stats,
            sidecar_path=sidecar,
            stored_in_postgres=stored,
        )

    def ingest_corpus(
        self,
        manifest: CorpusManifest | None = None,
        *,
        doc_ids: list[str] | None = None,
        force: bool = False,
    ) -> list[IngestionResult]:
        ensure_data_dirs()
        manifest = manifest or CorpusManifest.load()

        entries = manifest.active
        if doc_ids:
            wanted = set(doc_ids)
            entries = [e for e in manifest.entries if e.doc_id in wanted]
            missing = wanted - {e.doc_id for e in entries}
            if missing:
                raise KeyError(f"doc_ids not in manifest: {sorted(missing)}")

        results = []
        for entry in entries:
            log.info("ingesting %s", entry.doc_id)
            result = self.ingest_entry(entry, force=force)
            log.info(
                "%s: %s (%d pages, %d elements, %.1fs)",
                entry.doc_id,
                result.status,
                result.n_pages,
                result.n_elements,
                result.elapsed_s,
            )
            results.append(result)
        return results

    # -- persistence ---------------------------------------------------------

    def _persist(self, parsed: ParsedDocument) -> bool:
        """Write to Postgres if a store was supplied. Returns whether it happened."""
        if self.store is None:
            return False
        try:
            with self.store.transaction():
                # Replace wholesale: a re-parse must converge, not accumulate.
                self.store.delete_document(parsed.document.doc_id)
                self.store.upsert_document(parsed.document)
                self.store.insert_pages(parsed.pages)
                self.store.insert_elements(parsed.elements)
            return True
        except Exception as exc:
            log.error("%s: Postgres write failed: %s", parsed.document.doc_id, exc)
            return False
