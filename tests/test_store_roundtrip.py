"""Postgres round-trip integration tests.

Skipped automatically when Postgres is unreachable, so the suite still passes on
a machine with the services stopped. Run them with::

    docker compose up -d
    pytest -m integration

They use a throwaway ``doc_id`` and delete it afterwards, so they are safe to run
against a database that already holds a real ingested corpus.

``test_store_schema.py`` catches schema/repository drift statically; this file
checks what only a live database can: that constraints actually fire, that JSONB
payloads survive a round trip, and that a re-ingest converges rather than
accumulating.
"""

from __future__ import annotations

import uuid

import pytest

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
    make_element_id,
    make_page_id,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def store():
    from mmrag.stores.postgres import PostgresStore

    instance = PostgresStore()
    try:
        with instance:
            instance.ping()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"Postgres unavailable ({exc}); run 'docker compose up -d'")

    with PostgresStore() as connected:
        yield connected


TEST_DOC_PREFIX = "test-"


@pytest.fixture
def doc_id() -> str:
    """A unique id so a test run cannot collide with the real corpus."""
    return f"{TEST_DOC_PREFIX}{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="module", autouse=True)
def _sweep_test_documents(store):
    """Remove every test document at the end of the module.

    Per-test teardown already deletes on the happy path, but a test that fails
    part-way through an insert leaves its rows behind -- and those rows then show
    up in `mmrag ingest status` alongside the real corpus and skew the Step 6
    report. A prefix sweep makes pollution self-correcting rather than something
    to notice later and clean up by hand.
    """
    yield
    with store.transaction():
        store.conn.execute("DELETE FROM documents WHERE doc_id LIKE %s", (f"{TEST_DOC_PREFIX}%",))


@pytest.fixture
def records(doc_id):
    document = Document(
        doc_id=doc_id,
        title="Round-trip Fixture",
        source="Test Publisher",
        source_url="https://example.invalid/x.pdf",
        authors=["A. Author", "B. Author"],
        organization="Test Publisher",
        doc_type=DocumentType.RESEARCH_PAPER,
        domain="science",
        language="en",
        license="CC BY 4.0",
        file_name=f"{doc_id}.pdf",
        file_path=f"/tmp/{doc_id}.pdf",
        sha256="d" * 64,
        file_size_bytes=1234,
        n_pages=2,
        n_pages_ingested=2,
        parser_version="test-1",
        metadata={"field_sources": {"title": "manifest"}, "nested": {"a": [1, 2, 3]}},
    )
    pages = [
        Page(
            page_id=make_page_id(doc_id, n),
            doc_id=doc_id,
            page_number=n,
            width=595.0,
            height=842.0,
            rotation=0,
            image_path=f"/tmp/{doc_id}/p{n}.png",
            image_width=1240,
            image_height=1754,
            image_dpi=150,
            section="Results",
            subsection="Ablations" if n == 2 else None,
            header_text="Running Header",
            footer_text=f"Page {n}",
            raw_text=f"page {n} text",
            n_elements=2 if n == 1 else 1,
            metadata={"n_columns": 2},
        )
        for n in (1, 2)
    ]
    figure = Element(
        element_id=make_element_id(doc_id, 1, "chart", 1),
        doc_id=doc_id,
        page_id=make_page_id(doc_id, 1),
        page_number=1,
        element_type=ElementType.CHART,
        reading_order=0,
        bbox=BBox(x0=0.1, y0=0.2, x1=0.6, y1=0.5),
        section="Results",
        extraction_method=ExtractionMethod.PYMUPDF_DRAWING,
        extraction_confidence=0.72,
        caption="Figure 1: Revenue by region.",
        figure=FigureData(
            image_path="/tmp/fig.png",
            width_px=800,
            height_px=600,
            dpi=200,
            figure_type="chart",
            figure_type_confidence=0.8,
            figure_type_evidence="caption matched 'chart'",
            n_colors=512,
            mean_saturation=0.31,
        ),
        metadata={"is_vector": True, "n_primitives": 42},
    )
    caption = Element(
        element_id=make_element_id(doc_id, 1, "caption", 1),
        doc_id=doc_id,
        page_id=make_page_id(doc_id, 1),
        page_number=1,
        element_type=ElementType.CAPTION,
        parent_id=figure.element_id,
        reading_order=1,
        bbox=BBox(x0=0.1, y0=0.52, x1=0.6, y1=0.55),
        text="Figure 1: Revenue by region.",
        section="Results",
        extraction_method=ExtractionMethod.PYMUPDF_TEXT,
        extraction_confidence=1.0,
    )
    table = Element(
        element_id=make_element_id(doc_id, 2, "table", 1),
        doc_id=doc_id,
        page_id=make_page_id(doc_id, 2),
        page_number=2,
        element_type=ElementType.TABLE,
        reading_order=0,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
        section="Results",
        subsection="Ablations",
        extraction_method=ExtractionMethod.PYMUPDF_TABLE,
        extraction_confidence=0.88,
        caption="Table 1: Headcount.",
        table=TableData(
            n_rows=2,
            n_cols=2,
            header_rows=1,
            columns=["Department", "Headcount"],
            rows=[["Engineering", "412"], ["Sales", None]],
            markdown="| Department | Headcount |\n| --- | --- |\n| Engineering | 412 |",
            table_type="financial",
            fill_ratio=0.75,
        ),
    )
    return document, pages, [figure, caption, table]


@pytest.fixture
def stored(store, records, doc_id):
    document, pages, elements = records
    with store.transaction():
        store.delete_document(doc_id)
        store.upsert_document(document)
        store.insert_pages(pages)
        store.insert_elements(elements)
    yield document, pages, elements
    with store.transaction():
        store.delete_document(doc_id)


class TestDocumentRoundTrip:
    def test_scalar_fields_survive(self, store, stored, doc_id):
        original, _, _ = stored
        loaded = store.get_document(doc_id)
        assert loaded is not None
        for field in (
            "doc_id",
            "title",
            "source",
            "source_url",
            "organization",
            "version",
            "domain",
            "language",
            "license",
            "file_name",
            "sha256",
            "n_pages",
            "n_pages_ingested",
            "parser_version",
        ):
            assert getattr(loaded, field) == getattr(original, field), field

    def test_enum_and_array_fields_survive(self, store, stored, doc_id):
        original, _, _ = stored
        loaded = store.get_document(doc_id)
        assert loaded.doc_type is original.doc_type
        assert loaded.authors == original.authors

    def test_nested_jsonb_metadata_survives(self, store, stored, doc_id):
        loaded = store.get_document(doc_id)
        assert loaded.metadata["field_sources"]["title"] == "manifest"
        assert loaded.metadata["nested"] == {"a": [1, 2, 3]}


class TestPageRoundTrip:
    def test_all_page_metadata_survives(self, store, stored, doc_id):
        _, original, _ = stored
        loaded = store.get_pages(doc_id)
        assert len(loaded) == len(original)
        for got, want in zip(loaded, original, strict=True):
            assert got.page_number == want.page_number
            assert got.width == pytest.approx(want.width)
            assert got.rotation == want.rotation
            assert got.image_path == want.image_path
            assert (got.image_width, got.image_height, got.image_dpi) == (
                want.image_width,
                want.image_height,
                want.image_dpi,
            )
            assert (got.section, got.subsection) == (want.section, want.subsection)
            assert (got.header_text, got.footer_text) == (want.header_text, want.footer_text)


class TestElementRoundTrip:
    def test_bbox_survives_the_four_column_split(self, store, stored, doc_id):
        _, _, original = stored
        loaded = {e.element_id: e for e in store.get_elements(doc_id)}
        want = original[0]
        got = loaded[want.element_id]
        assert got.bbox is not None
        for axis in ("x0", "y0", "x1", "y1"):
            assert getattr(got.bbox, axis) == pytest.approx(getattr(want.bbox, axis), abs=1e-6)

    def test_structured_table_payload_survives(self, store, stored, doc_id):
        _, _, original = stored
        table = next(e for e in original if e.table)
        loaded = {e.element_id: e for e in store.get_elements(doc_id)}[table.element_id]
        assert loaded.table is not None
        assert loaded.table.columns == table.table.columns
        assert loaded.table.rows == table.table.rows, "a None cell must not become ''"
        assert loaded.table.table_type is table.table.table_type
        assert loaded.table.markdown == table.table.markdown

    def test_structured_figure_payload_survives(self, store, stored, doc_id):
        _, _, original = stored
        figure = next(e for e in original if e.figure)
        loaded = {e.element_id: e for e in store.get_elements(doc_id)}[figure.element_id]
        assert loaded.figure is not None
        assert loaded.figure.figure_type is figure.figure.figure_type
        assert loaded.figure.figure_type_evidence == figure.figure.figure_type_evidence
        assert loaded.figure.width_px == figure.figure.width_px

    def test_extraction_provenance_survives(self, store, stored, doc_id):
        _, _, original = stored
        loaded = {e.element_id: e for e in store.get_elements(doc_id)}
        for want in original:
            got = loaded[want.element_id]
            assert got.extraction_method is want.extraction_method
            assert got.extraction_confidence == pytest.approx(want.extraction_confidence, abs=1e-6)

    def test_parent_link_survives(self, store, stored, doc_id):
        _, _, original = stored
        caption = next(e for e in original if e.parent_id)
        loaded = {e.element_id: e for e in store.get_elements(doc_id)}[caption.element_id]
        assert loaded.parent_id == caption.parent_id

    def test_children_are_queryable_from_the_parent(self, store, stored, doc_id):
        _, _, original = stored
        figure = next(e for e in original if e.element_type is ElementType.CHART)
        children = store.get_children(figure.element_id)
        assert [c.element_type for c in children] == [ElementType.CAPTION]


class TestQueryFilters:
    def test_filter_by_page(self, store, stored, doc_id):
        assert all(e.page_number == 2 for e in store.get_elements(doc_id, page_number=2))

    def test_filter_by_element_type(self, store, stored, doc_id):
        found = store.get_elements(doc_id, element_types=[ElementType.TABLE])
        assert [e.element_type for e in found] == [ElementType.TABLE]

    def test_filter_by_confidence(self, store, stored, doc_id):
        found = store.get_elements(doc_id, min_confidence=0.8)
        assert found and all(e.extraction_confidence >= 0.8 for e in found)

    def test_results_are_ordered_by_page_then_reading_order(self, store, stored, doc_id):
        found = store.get_elements(doc_id)
        keys = [(e.page_number, e.reading_order) for e in found]
        assert keys == sorted(keys)


class TestConstraintsAreEnforced:
    def test_payload_type_mismatch_is_rejected_by_the_database(self, store, stored, doc_id):
        """Pydantic blocks this in-process; the database must block it too."""
        import psycopg

        with pytest.raises(psycopg.errors.CheckViolation), store.conn.transaction():
            store.conn.execute(
                "INSERT INTO elements (element_id, doc_id, page_id, page_number, "
                "element_type, figure_data) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    f"{doc_id}#bad",
                    doc_id,
                    make_page_id(doc_id, 1),
                    1,
                    "table",
                    '{"figure_type": "chart"}',
                ),
            )

    def test_out_of_range_bbox_is_rejected(self, store, stored, doc_id):
        import psycopg

        with pytest.raises(psycopg.errors.CheckViolation), store.conn.transaction():
            store.conn.execute(
                "INSERT INTO elements (element_id, doc_id, page_id, page_number, "
                "element_type, bbox_x0) VALUES (%s, %s, %s, %s, %s, %s)",
                (f"{doc_id}#bad2", doc_id, make_page_id(doc_id, 1), 1, "text", 1.5),
            )

    def test_deleting_a_document_cascades_to_pages_and_elements(self, store, records, doc_id):
        document, pages, elements = records
        with store.transaction():
            store.upsert_document(document)
            store.insert_pages(pages)
            store.insert_elements(elements)
        assert store.get_elements(doc_id)

        with store.transaction():
            store.delete_document(doc_id)
        assert store.get_document(doc_id) is None
        assert store.get_pages(doc_id) == []
        assert store.get_elements(doc_id) == []


class TestDurability:
    """Writes must survive the connection that made them.

    Regression test for a silent data-loss bug: with psycopg's default
    autocommit=False, reading before writing opened an implicit transaction, so
    `with store.transaction()` nested as a savepoint and never committed. The
    writing connection still saw its own rows, so everything looked fine until a
    *different* connection looked.
    """

    def test_write_after_read_is_visible_to_another_connection(self, store, records, doc_id):
        from mmrag.stores.postgres import PostgresStore

        document, pages, elements = records

        # The read is the whole point: it is what used to open the implicit
        # transaction that swallowed the subsequent write.
        store.get_document(doc_id)

        with store.transaction():
            store.upsert_document(document)
            store.insert_pages(pages)
            store.insert_elements(elements)

        with PostgresStore() as other:
            assert other.get_document(doc_id) is not None, "write was rolled back on close"
            assert len(other.get_elements(doc_id)) == len(elements)

        with store.transaction():
            store.delete_document(doc_id)
        with PostgresStore() as other:
            assert other.get_document(doc_id) is None, "delete was rolled back on close"

    def test_a_failed_transaction_rolls_back_fully(self, store, records, doc_id):
        """A partial write must not survive: no half-ingested documents."""
        import psycopg

        from mmrag.stores.postgres import PostgresStore

        document, pages, _ = records
        with pytest.raises(psycopg.Error), store.transaction():
            store.upsert_document(document)
            store.insert_pages(pages)
            # Violates the page_number >= 1 CHECK, aborting the transaction.
            store.conn.execute(
                "INSERT INTO elements (element_id, doc_id, page_id, page_number, element_type) "
                "VALUES (%s, %s, %s, %s, %s)",
                (f"{doc_id}#bad", doc_id, make_page_id(doc_id, 1), 0, "text"),
            )

        with PostgresStore() as other:
            assert other.get_document(doc_id) is None, "partial write survived a failed transaction"


class TestReingestionConverges:
    def test_reingesting_replaces_rather_than_accumulates(self, store, records, doc_id):
        """A re-run after a parser fix must converge, not duplicate."""
        document, pages, elements = records
        for _ in range(2):
            with store.transaction():
                store.delete_document(doc_id)
                store.upsert_document(document)
                store.insert_pages(pages)
                store.insert_elements(elements)

        assert len(store.get_pages(doc_id)) == len(pages)
        assert len(store.get_elements(doc_id)) == len(elements)

        with store.transaction():
            store.delete_document(doc_id)


class TestCorpusStats:
    """Counts must be per-document row counts, never a joined product.

    The fixture deliberately stores more than one page *and* more than one
    element. With a single page of either, a cartesian fan-out between the two
    one-to-many tables multiplies by 1 and is invisible.
    """

    def test_fixture_can_actually_detect_fan_out(self, stored):
        _, pages, elements = stored
        assert len(pages) > 1 and len(elements) > 1, (
            "with one page or one element, a fan-out bug multiplies by 1 and hides"
        )

    def test_stats_report_the_stored_document(self, store, stored, doc_id):
        _, pages, elements = stored
        rows = {r["doc_id"]: r for r in store.corpus_stats()}
        assert doc_id in rows
        row = rows[doc_id]

        expected = {
            "pages_stored": len(pages),
            "elements": len(elements),
            "tables": sum(1 for e in elements if e.element_type is ElementType.TABLE),
            "figures": sum(1 for e in elements if e.element_type.is_visual),
            "captioned": sum(1 for e in elements if e.caption),
        }
        # Compared as a whole rather than one assert at a time: the fan-out bug
        # inflated *every* count, but a sequence of asserts stops at the first
        # and makes a systematic error look like a single off-by-N.
        actual = {key: row[key] for key in expected}
        assert actual == expected

    def test_counts_are_not_multiplied_by_the_page_count(self, store, stored, doc_id):
        """Names the failure mode directly, so a regression says what broke."""
        _, pages, elements = stored
        row = {r["doc_id"]: r for r in store.corpus_stats()}[doc_id]
        assert row["elements"] != len(elements) * len(pages), (
            "elements count equals elements x pages: the aggregate has fanned out again"
        )

    def test_mean_confidence_reflects_the_stored_elements(self, store, stored, doc_id):
        _, _, elements = stored
        row = {r["doc_id"]: r for r in store.corpus_stats()}[doc_id]
        expected = sum(e.extraction_confidence for e in elements) / len(elements)
        assert float(row["mean_confidence"]) == pytest.approx(expected, abs=1e-3)

    def test_a_document_with_no_elements_reports_zero_not_null(self, store, records, doc_id):
        """LEFT JOIN + COALESCE: an un-parsed document must not render as None."""
        document, _, _ = records
        with store.transaction():
            store.delete_document(doc_id)
            store.upsert_document(document)

        row = {r["doc_id"]: r for r in store.corpus_stats()}[doc_id]
        assert (row["pages_stored"], row["elements"], row["tables"]) == (0, 0, 0)
        assert row["figures"] == 0 and row["captioned"] == 0

        with store.transaction():
            store.delete_document(doc_id)
