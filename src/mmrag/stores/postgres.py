"""Postgres repository for documents, pages, and elements.

Plain SQL over psycopg3 rather than an ORM. The schema *is* the contract here
(see ``scripts/sql/001_schema.sql``), and keeping the queries visible means the
provenance joins that matter -- element back to page back to document -- are
readable rather than hidden behind lazy-loading.

Writes are idempotent: re-ingesting a document replaces its rows rather than
duplicating them, so a re-run after a parser fix converges instead of
accumulating.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from mmrag.config import get_settings
from mmrag.logging_utils import get_logger
from mmrag.schemas import (
    BBox,
    Document,
    DocumentType,
    Element,
    ElementType,
    ExtractionMethod,
    FigureData,
    Page,
    TableData,
)

log = get_logger(__name__)

# Batch size for executemany. Large enough to amortise round-trips, small enough
# that a 640-page datasheet does not build a single enormous statement.
_BATCH = 500


class PostgresStore:
    """Metadata store. Open with :meth:`connect` or use as a context manager."""

    def __init__(self, dsn: str | None = None, *, connect_timeout: int = 5):
        self.dsn = dsn or get_settings().postgres_dsn
        # Without an explicit timeout, connecting to a stopped container blocks
        # for the OS default (minutes on Windows). Ingestion is designed to fall
        # back to sidecar-only output when the database is down, and it can only
        # do that if the attempt fails promptly.
        self.connect_timeout = connect_timeout
        self._conn: psycopg.Connection[Any] | None = None

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> PostgresStore:
        self._conn = psycopg.connect(
            self.dsn, row_factory=dict_row, connect_timeout=self.connect_timeout
        )
        return self

    def __exit__(self, *exc: object) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> psycopg.Connection[Any]:
        if self._conn is None:
            raise RuntimeError("PostgresStore is not connected; use it as a context manager")
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection[Any]]:
        with self.conn.transaction():
            yield self.conn

    def ping(self) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            return cur.fetchone() is not None

    # -- writes -------------------------------------------------------------

    def upsert_document(self, document: Document) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (
                    doc_id, title, source, source_url, authors, organization,
                    publication_date, version, doc_type, domain, language, license,
                    file_name, file_path, sha256, file_size_bytes,
                    n_pages, n_pages_ingested, ingested_at, parser_version, metadata
                ) VALUES (
                    %(doc_id)s, %(title)s, %(source)s, %(source_url)s, %(authors)s,
                    %(organization)s, %(publication_date)s, %(version)s, %(doc_type)s,
                    %(domain)s, %(language)s, %(license)s, %(file_name)s, %(file_path)s,
                    %(sha256)s, %(file_size_bytes)s, %(n_pages)s, %(n_pages_ingested)s,
                    COALESCE(%(ingested_at)s, now()), %(parser_version)s, %(metadata)s
                )
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source = EXCLUDED.source,
                    source_url = EXCLUDED.source_url,
                    authors = EXCLUDED.authors,
                    organization = EXCLUDED.organization,
                    publication_date = EXCLUDED.publication_date,
                    version = EXCLUDED.version,
                    doc_type = EXCLUDED.doc_type,
                    domain = EXCLUDED.domain,
                    language = EXCLUDED.language,
                    license = EXCLUDED.license,
                    file_name = EXCLUDED.file_name,
                    file_path = EXCLUDED.file_path,
                    sha256 = EXCLUDED.sha256,
                    file_size_bytes = EXCLUDED.file_size_bytes,
                    n_pages = EXCLUDED.n_pages,
                    n_pages_ingested = EXCLUDED.n_pages_ingested,
                    ingested_at = EXCLUDED.ingested_at,
                    parser_version = EXCLUDED.parser_version,
                    metadata = EXCLUDED.metadata
                """,
                {
                    **document.model_dump(exclude={"metadata", "doc_type"}),
                    "doc_type": document.doc_type.value,
                    "metadata": Jsonb(document.metadata),
                },
            )

    def delete_document(self, doc_id: str) -> None:
        """Remove a document and everything descended from it.

        Called before a re-ingest. Cascades handle pages, elements and chunks,
        which is why the foreign keys are declared ON DELETE CASCADE.
        """
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))

    def insert_pages(self, pages: Iterable[Page]) -> int:
        rows = [
            {
                **page.model_dump(exclude={"metadata"}),
                "metadata": Jsonb(page.metadata),
            }
            for page in pages
        ]
        if not rows:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO pages (
                    page_id, doc_id, page_number, width, height, rotation,
                    image_path, image_width, image_height, image_dpi,
                    section, subsection, header_text, footer_text,
                    raw_text, n_elements, metadata
                ) VALUES (
                    %(page_id)s, %(doc_id)s, %(page_number)s, %(width)s, %(height)s,
                    %(rotation)s, %(image_path)s, %(image_width)s, %(image_height)s,
                    %(image_dpi)s, %(section)s, %(subsection)s, %(header_text)s,
                    %(footer_text)s, %(raw_text)s, %(n_elements)s, %(metadata)s
                )
                ON CONFLICT (page_id) DO NOTHING
                """,
                rows,
            )
        return len(rows)

    def insert_elements(self, elements: Iterable[Element]) -> int:
        """Insert elements, parents before children.

        ``parent_id`` is a self-referential foreign key, so a caption inserted
        before its figure would violate it. Ordering by "has a parent" is
        sufficient here because the hierarchy is only ever one level deep.
        """
        ordered = sorted(elements, key=lambda e: e.parent_id is not None)
        rows = [self._element_row(e) for e in ordered]
        if not rows:
            return 0

        sql = """
            INSERT INTO elements (
                element_id, doc_id, page_id, page_number, element_type,
                parent_id, reading_order,
                bbox_x0, bbox_y0, bbox_x1, bbox_y1,
                section, subsection, extraction_method, extraction_confidence,
                text, caption, table_data, figure_data, metadata
            ) VALUES (
                %(element_id)s, %(doc_id)s, %(page_id)s, %(page_number)s, %(element_type)s,
                %(parent_id)s, %(reading_order)s,
                %(bbox_x0)s, %(bbox_y0)s, %(bbox_x1)s, %(bbox_y1)s,
                %(section)s, %(subsection)s, %(extraction_method)s, %(extraction_confidence)s,
                %(text)s, %(caption)s, %(table_data)s, %(figure_data)s, %(metadata)s
            )
            ON CONFLICT (element_id) DO NOTHING
        """
        with self.conn.cursor() as cur:
            for start in range(0, len(rows), _BATCH):
                cur.executemany(sql, rows[start : start + _BATCH])
        return len(rows)

    @staticmethod
    def _element_row(element: Element) -> dict[str, Any]:
        bbox = element.bbox
        return {
            "element_id": element.element_id,
            "doc_id": element.doc_id,
            "page_id": element.page_id,
            "page_number": element.page_number,
            "element_type": element.element_type.value,
            "parent_id": element.parent_id,
            "reading_order": element.reading_order,
            "bbox_x0": bbox.x0 if bbox else None,
            "bbox_y0": bbox.y0 if bbox else None,
            "bbox_x1": bbox.x1 if bbox else None,
            "bbox_y1": bbox.y1 if bbox else None,
            "section": element.section,
            "subsection": element.subsection,
            "extraction_method": element.extraction_method.value,
            "extraction_confidence": element.extraction_confidence,
            "text": element.text,
            "caption": element.caption,
            "table_data": Jsonb(element.table.model_dump(mode="json")) if element.table else None,
            "figure_data": (
                Jsonb(element.figure.model_dump(mode="json")) if element.figure else None
            ),
            "metadata": Jsonb(element.metadata),
        }

    # -- reads --------------------------------------------------------------

    def get_document(self, doc_id: str) -> Document | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
        return _to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM documents ORDER BY doc_id")
            return [_to_document(r) for r in cur.fetchall()]

    def get_pages(self, doc_id: str) -> list[Page]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM pages WHERE doc_id = %s ORDER BY page_number", (doc_id,))
            return [_to_page(r) for r in cur.fetchall()]

    def get_elements(
        self,
        doc_id: str,
        *,
        page_number: int | None = None,
        element_types: list[ElementType] | None = None,
        min_confidence: float | None = None,
    ) -> list[Element]:
        """Fetch elements with the filters the later methods actually need."""
        clauses = ["doc_id = %(doc_id)s"]
        params: dict[str, Any] = {"doc_id": doc_id}
        if page_number is not None:
            clauses.append("page_number = %(page_number)s")
            params["page_number"] = page_number
        if element_types:
            clauses.append("element_type = ANY(%(types)s)")
            params["types"] = [t.value for t in element_types]
        if min_confidence is not None:
            clauses.append("extraction_confidence >= %(min_conf)s")
            params["min_conf"] = min_confidence

        with self.conn.cursor() as cur:
            # Clause list is built from a fixed vocabulary above; every value
            # is still bound as a parameter, so this is not string-interpolated SQL.
            where = " AND ".join(clauses)
            cur.execute(
                f"SELECT * FROM elements WHERE {where} "
                "ORDER BY page_number, reading_order, element_id",
                params,
            )
            return [_to_element(r) for r in cur.fetchall()]

    def get_children(self, element_id: str) -> list[Element]:
        """Elements parented to this one, e.g. a figure's caption."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM elements WHERE parent_id = %s ORDER BY reading_order", (element_id,)
            )
            return [_to_element(r) for r in cur.fetchall()]

    def corpus_stats(self) -> list[dict[str, Any]]:
        """Per-document counts, for `mmrag ingest status` and the report."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.doc_id, d.title, d.doc_type, d.domain,
                       d.n_pages, d.n_pages_ingested,
                       count(DISTINCT p.page_id) AS pages_stored,
                       count(e.element_id)       AS elements,
                       count(*) FILTER (WHERE e.element_type = 'table')   AS tables,
                       count(*) FILTER (WHERE e.element_type IN
                             ('figure', 'chart', 'diagram'))              AS figures,
                       count(*) FILTER (WHERE e.caption IS NOT NULL)      AS captioned,
                       round(avg(e.extraction_confidence)::numeric, 3)    AS mean_confidence
                FROM documents d
                LEFT JOIN pages p    ON p.doc_id = d.doc_id
                LEFT JOIN elements e ON e.doc_id = d.doc_id
                GROUP BY d.doc_id, d.title, d.doc_type, d.domain, d.n_pages, d.n_pages_ingested
                ORDER BY d.doc_id
                """
            )
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Row -> model
# ---------------------------------------------------------------------------


def _to_document(row: dict[str, Any]) -> Document:
    return Document(
        doc_id=row["doc_id"],
        title=row["title"],
        source=row["source"],
        source_url=row["source_url"],
        authors=list(row["authors"] or []),
        organization=row["organization"],
        publication_date=row["publication_date"],
        version=row["version"],
        doc_type=DocumentType(row["doc_type"]),
        domain=row["domain"],
        language=row["language"],
        license=row["license"],
        file_name=row["file_name"],
        file_path=row["file_path"],
        sha256=row["sha256"],
        file_size_bytes=row["file_size_bytes"],
        n_pages=row["n_pages"],
        n_pages_ingested=row["n_pages_ingested"],
        ingested_at=row["ingested_at"],
        parser_version=row["parser_version"],
        metadata=_as_dict(row["metadata"]),
    )


def _to_page(row: dict[str, Any]) -> Page:
    return Page(
        page_id=row["page_id"],
        doc_id=row["doc_id"],
        page_number=row["page_number"],
        width=row["width"],
        height=row["height"],
        rotation=row["rotation"],
        image_path=row["image_path"],
        image_width=row["image_width"],
        image_height=row["image_height"],
        image_dpi=row["image_dpi"],
        section=row["section"],
        subsection=row["subsection"],
        header_text=row["header_text"],
        footer_text=row["footer_text"],
        raw_text=row["raw_text"],
        n_elements=row["n_elements"],
        metadata=_as_dict(row["metadata"]),
    )


def _to_element(row: dict[str, Any]) -> Element:
    bbox = None
    if row["bbox_x0"] is not None:
        bbox = BBox(x0=row["bbox_x0"], y0=row["bbox_y0"], x1=row["bbox_x1"], y1=row["bbox_y1"])
    table = TableData.model_validate(_as_dict(row["table_data"])) if row["table_data"] else None
    figure = FigureData.model_validate(_as_dict(row["figure_data"])) if row["figure_data"] else None
    return Element(
        element_id=row["element_id"],
        doc_id=row["doc_id"],
        page_id=row["page_id"],
        page_number=row["page_number"],
        element_type=ElementType(row["element_type"]),
        parent_id=row["parent_id"],
        reading_order=row["reading_order"],
        bbox=bbox,
        section=row["section"],
        subsection=row["subsection"],
        extraction_method=ExtractionMethod(row["extraction_method"]),
        extraction_confidence=row["extraction_confidence"],
        text=row["text"],
        caption=row["caption"],
        table=table,
        figure=figure,
        metadata=_as_dict(row["metadata"]),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    """psycopg returns JSONB as a dict already, but tolerate a raw string."""
    if value is None:
        return {}
    if isinstance(value, str):
        return dict(json.loads(value))
    return dict(value)
