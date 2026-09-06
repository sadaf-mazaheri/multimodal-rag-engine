"""Document-level metadata assembly.

Metadata is merged from three sources, in decreasing order of trust:

1. **The corpus manifest** -- human-curated, so it always wins.
2. **The PDF's own metadata dictionary** -- often present, frequently wrong
   (a title left over from a template, an "author" that is the typesetting tool).
3. **Heuristics over the document text** -- last resort.

Which source won for each field is recorded in ``metadata['field_sources']``.
Without that, a wrong title is untraceable: you cannot tell whether to fix the
manifest, distrust the PDF, or improve the heuristic.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from mmrag.corpus.manifest import CorpusEntry
from mmrag.logging_utils import get_logger
from mmrag.schemas import Document, DocumentType

log = get_logger(__name__)

# Genre inference from the title, checked before the manifest's coarse domain.
_TYPE_PATTERNS: list[tuple[DocumentType, re.Pattern[str]]] = [
    (DocumentType.ANNUAL_REPORT, re.compile(r"\bannual report\b|\b10-K\b|\bshareholder", re.I)),
    (DocumentType.DATASHEET, re.compile(r"\bdatasheet\b|\breference manual\b", re.I)),
    (DocumentType.WHITEPAPER, re.compile(r"\bwhitepaper\b|\bwhite paper\b|\barchitecture\b", re.I)),
    (DocumentType.SITUATION_REPORT, re.compile(r"\bsituation report\b|\bsitrep\b", re.I)),
    (DocumentType.TECHNICAL_MANUAL, re.compile(r"\bhandbook\b|\bmanual\b|\bguide\b", re.I)),
    (
        DocumentType.POLICY_REPORT,
        re.compile(r"\bsummary for policymakers\b|\boutlook\b|\bmonetary policy\b", re.I),
    ),
]

# Domains from the manifest that imply a genre when the title says nothing.
_DOMAIN_DEFAULTS = {
    "science": DocumentType.RESEARCH_PAPER,
    "finance": DocumentType.ANNUAL_REPORT,
    "policy": DocumentType.POLICY_REPORT,
    "technical": DocumentType.TECHNICAL_MANUAL,
    "energy": DocumentType.POLICY_REPORT,
    "health": DocumentType.SITUATION_REPORT,
}

# PDF dates look like D:20240705120000+04'00'.
_PDF_DATE = re.compile(r"D?:?(\d{4})(\d{2})?(\d{2})?")

# Author strings that are really software, not people.
_TOOL_NOISE = re.compile(
    r"acrobat|distiller|latex|pdftex|word|indesign|quark|ghostscript|writer|"
    r"microsoft|adobe|printer|unknown",
    re.I,
)


def parse_pdf_date(value: str | None) -> date | None:
    """Parse a PDF date string, tolerating the many partial forms in the wild."""
    if not value:
        return None
    match = _PDF_DATE.match(value.strip())
    if not match:
        return None
    year = int(match.group(1))
    if not (1900 <= year <= 2100):
        return None
    month = int(match.group(2) or 1)
    day = int(match.group(3) or 1)
    try:
        return date(year, min(max(month, 1), 12), min(max(day, 1), 28 if month == 2 else 31))
    except ValueError:
        return date(year, 1, 1)


def parse_authors(value: str | None) -> list[str]:
    """Split a PDF author string into names, discarding tool names."""
    if not value or _TOOL_NOISE.search(value):
        return []
    parts = re.split(r"\s*(?:;|,|\band\b|&)\s*", value)
    return [p.strip() for p in parts if len(p.strip()) > 2][:20]


def infer_document_type(title: str, domain: str | None) -> tuple[DocumentType, str]:
    """Infer genre from the title, falling back to the manifest domain."""
    for doc_type, pattern in _TYPE_PATTERNS:
        if pattern.search(title):
            return doc_type, f"title matched {pattern.pattern[:30]!r}"
    if domain and domain in _DOMAIN_DEFAULTS:
        return _DOMAIN_DEFAULTS[domain], f"domain default for {domain!r}"
    return DocumentType.OTHER, "no signal"


def build_document(
    entry: CorpusEntry,
    *,
    file_path: Path,
    pdf_metadata: dict[str, Any],
    n_pages: int,
    n_pages_ingested: int,
    parser_version: str,
) -> Document:
    """Assemble the ``Document`` record for one corpus entry."""
    sources: dict[str, str] = {}

    # --- title: the manifest is curated, so it wins outright ---------------
    title = entry.title
    sources["title"] = "manifest"
    pdf_title = (pdf_metadata.get("title") or "").strip()
    if not title and pdf_title:
        title, sources["title"] = pdf_title, "pdf_metadata"

    # --- authors: only the PDF has them; the manifest records a publisher ---
    authors = parse_authors(pdf_metadata.get("author"))
    if authors:
        sources["authors"] = "pdf_metadata"

    # --- publication date ---------------------------------------------------
    publication_date = parse_pdf_date(pdf_metadata.get("creationDate"))
    if publication_date:
        sources["publication_date"] = "pdf_metadata.creationDate"
    else:
        publication_date = parse_pdf_date(pdf_metadata.get("modDate"))
        if publication_date:
            sources["publication_date"] = "pdf_metadata.modDate"
        else:
            year = re.search(r"\b(19|20)\d{2}\b", entry.title)
            if year:
                publication_date = date(int(year.group(0)), 1, 1)
                sources["publication_date"] = "title_year_heuristic"

    doc_type, type_evidence = infer_document_type(entry.title, entry.category)
    sources["doc_type"] = type_evidence

    version = None
    version_match = re.search(r"\b(rev(?:ision)?\.?\s*\d+|v\d+(?:\.\d+)*)\b", entry.title, re.I)
    if version_match:
        version, sources["version"] = version_match.group(0), "title"

    if entry.sha256 is None:
        raise ValueError(
            f"{entry.doc_id} is not pinned in the corpus lockfile; run 'mmrag corpus lock'"
        )

    return Document(
        doc_id=entry.doc_id,
        title=title,
        source=entry.publisher,
        source_url=entry.url,
        authors=authors,
        organization=entry.publisher,
        publication_date=publication_date,
        version=version,
        doc_type=doc_type,
        domain=entry.category,
        language=_detect_language(pdf_metadata),
        license=entry.license,
        file_name=file_path.name,
        file_path=str(file_path),
        sha256=entry.sha256,
        file_size_bytes=entry.size_bytes,
        n_pages=n_pages,
        n_pages_ingested=n_pages_ingested,
        ingested_at=datetime.now(timezone.utc),
        parser_version=parser_version,
        metadata={
            "field_sources": sources,
            "modality_profile": list(entry.modality_profile),
            "page_limit": entry.page_limit,
            "pdf_metadata": {
                k: v for k, v in pdf_metadata.items() if isinstance(v, str) and v.strip()
            },
            "manifest_notes": entry.notes,
        },
    )


def _detect_language(pdf_metadata: dict[str, Any]) -> str:
    """Language code.

    The whole corpus is English, so this reads any declared value and otherwise
    defaults rather than pulling in a detection dependency for no benefit. It
    exists as a real field so that a multilingual corpus is a config change
    rather than a schema change.
    """
    declared = (pdf_metadata.get("language") or "").strip().lower()
    return declared[:5] if declared else "en"
