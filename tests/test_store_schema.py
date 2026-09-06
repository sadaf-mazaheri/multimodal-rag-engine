"""Consistency between the SQL schema, the repository, and the Pydantic models.

Three definitions of the same records have to agree:
``scripts/sql/001_schema.sql``, the INSERT statements in
``mmrag.stores.postgres``, and the models in ``mmrag.schemas``. Drift between
them fails at runtime with an opaque psycopg error, and only once a database is
actually reachable -- which on a CPU-only laptop with Docker stopped may be a
long time after the mistake was made.

These tests parse all three and compare them, so the mismatch is caught in the
normal test run with no services required. They are not a substitute for the
round-trip integration test (see ``test_store_roundtrip.py``), but they catch
the failure that is by far the most likely: a field added to a model and to the
SQL, and forgotten in the INSERT.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mmrag.config import PROJECT_ROOT
from mmrag.schemas import Document, Element, Page

SCHEMA_PATH = PROJECT_ROOT / "scripts" / "sql" / "001_schema.sql"
STORE_PATH = PROJECT_ROOT / "src" / "mmrag" / "stores" / "postgres.py"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE = re.compile(
    r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\((.*?)\n\);", re.DOTALL | re.IGNORECASE
)
_INSERT = re.compile(r"INSERT INTO\s+(\w+)\s*\((.*?)\)\s*VALUES", re.DOTALL | re.IGNORECASE)
_PLACEHOLDER = re.compile(r"%\((\w+)\)s")

# Lines inside a CREATE TABLE that declare a constraint rather than a column.
_CONSTRAINT_PREFIX = ("constraint", "primary", "unique", "check", "foreign", "--")


def _strip_comments(sql: str) -> str:
    return "\n".join(line.split("--")[0] for line in sql.splitlines())


def parse_schema() -> dict[str, dict[str, str]]:
    """``{table: {column: definition}}`` from the committed schema."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    tables: dict[str, dict[str, str]] = {}

    for name, body in _CREATE_TABLE.findall(sql):
        columns: dict[str, str] = {}
        for raw in _strip_comments(body).split("\n"):
            line = raw.strip().rstrip(",").strip()
            if not line or line.lower().startswith(_CONSTRAINT_PREFIX):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                columns[parts[0]] = parts[1]
        tables[name] = columns
    return tables


def parse_inserts() -> dict[str, list[set[str]]]:
    """``{table: [set of columns per INSERT statement]}`` from the repository."""
    source = STORE_PATH.read_text(encoding="utf-8")
    out: dict[str, list[set[str]]] = {}
    for table, columns in _INSERT.findall(source):
        names = {c.strip() for c in columns.replace("\n", " ").split(",") if c.strip()}
        out.setdefault(table, []).append(names)
    return out


@pytest.fixture(scope="module")
def schema() -> dict[str, dict[str, str]]:
    return parse_schema()


@pytest.fixture(scope="module")
def inserts() -> dict[str, list[set[str]]]:
    return parse_inserts()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchemaParsing:
    def test_schema_file_exists_and_defines_the_four_tables(self, schema):
        assert set(schema) == {"documents", "pages", "elements", "chunks"}

    def test_every_table_has_columns(self, schema):
        for table, columns in schema.items():
            assert columns, f"{table} parsed with no columns"

    def test_store_issues_inserts_for_the_write_paths(self, inserts):
        assert set(inserts) >= {"documents", "pages", "elements"}


class TestInsertsMatchSchema:
    @pytest.mark.parametrize("table", ["documents", "pages", "elements"])
    def test_inserted_columns_all_exist(self, schema, inserts, table):
        known = set(schema[table])
        for statement in inserts[table]:
            unknown = statement - known
            assert not unknown, f"INSERT INTO {table} names non-existent columns: {unknown}"

    @pytest.mark.parametrize("table", ["documents", "pages", "elements"])
    def test_every_required_column_is_supplied(self, schema, inserts, table):
        """A NOT NULL column with no DEFAULT must appear in the INSERT."""
        required = {
            name
            for name, definition in schema[table].items()
            if "NOT NULL" in definition.upper() and "DEFAULT" not in definition.upper()
        }
        for statement in inserts[table]:
            missing = required - statement
            assert not missing, f"INSERT INTO {table} omits required columns: {missing}"

    def test_element_insert_covers_the_provenance_columns(self, inserts):
        """The columns that make a hit traceable are not optional."""
        statement = inserts["elements"][0]
        for column in (
            "element_id",
            "doc_id",
            "page_id",
            "page_number",
            "parent_id",
            "reading_order",
            "bbox_x0",
            "bbox_y0",
            "bbox_x1",
            "bbox_y1",
            "extraction_method",
            "extraction_confidence",
        ):
            assert column in statement, f"elements INSERT is missing {column}"


class TestPlaceholdersMatchColumns:
    def test_every_insert_binds_a_placeholder_per_column(self):
        """A column list longer than its VALUES list fails only at execution.

        The VALUES section cannot be matched with a naive ``\\(...\\)`` group,
        because psycopg's own ``%(name)s`` placeholders contain parentheses.
        It is delimited by the following ON CONFLICT / end of statement instead.
        """
        source = STORE_PATH.read_text(encoding="utf-8")
        statements = 0

        for match in _INSERT.finditer(source):
            columns = [c.strip() for c in match.group(2).replace("\n", " ").split(",") if c.strip()]

            tail = source[match.end() :]
            end = min(
                (pos for pos in (tail.find("ON CONFLICT"), tail.find('"""')) if pos != -1),
                default=len(tail),
            )
            placeholders = _PLACEHOLDER.findall(tail[:end])

            assert len(columns) == len(placeholders), (
                f"INSERT INTO {match.group(1)}: {len(columns)} columns but "
                f"{len(placeholders)} placeholders"
            )
            # Order matters as much as count: a transposed pair type-checks.
            assert columns == placeholders, (
                f"INSERT INTO {match.group(1)}: column order does not match placeholder order\n"
                f"  columns:      {columns}\n  placeholders: {placeholders}"
            )
            statements += 1

        assert statements >= 3, "expected INSERTs for documents, pages and elements"


class TestModelsMatchSchema:
    """The Pydantic models and the SQL must describe the same records."""

    @pytest.mark.parametrize(
        ("model", "table", "renames", "model_only"),
        [
            (Document, "documents", {}, set()),
            (Page, "pages", {}, set()),
            (
                Element,
                "elements",
                # The model nests structured payloads; SQL stores them as JSONB.
                {"table": "table_data", "figure": "figure_data"},
                {"bbox"},  # exploded into four bbox_* columns
            ),
        ],
    )
    def test_every_model_field_has_a_column(self, schema, model, table, renames, model_only):
        columns = set(schema[table])
        for field in model.model_fields:
            if field in model_only:
                continue
            assert renames.get(field, field) in columns, (
                f"{model.__name__}.{field} has no column in {table}"
            )

    def test_bbox_is_stored_as_four_columns(self, schema):
        for axis in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"):
            assert axis in schema["elements"]

    def test_normalised_bbox_range_is_constrained_in_sql(self):
        """The [0,1] guarantee is enforced by the database, not only by Pydantic."""
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        assert sql.count("BETWEEN 0 AND 1") >= 4

    def test_payload_type_agreement_is_enforced_in_sql(self):
        """A figure payload on a table row must be impossible at the storage layer."""
        sql = SCHEMA_PATH.read_text(encoding="utf-8").lower()
        assert "elements_payload_matches_type" in sql

    def test_cascade_deletes_keep_reingestion_convergent(self):
        """Re-ingesting replaces a document; children must not survive it."""
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        assert sql.count("ON DELETE CASCADE") >= 4


class TestSchemaSupportsTheAccessPatterns:
    """Indexes for the queries the later steps will actually issue."""

    @pytest.mark.parametrize(
        "index",
        [
            "elements_reading_idx",  # context expansion around a hit
            "elements_parent_idx",  # a figure's caption
            "elements_type_idx",  # modality-filtered retrieval (Method 2)
            "elements_section_idx",  # section filtering
            "chunks_fts_idx",  # Postgres full-text cross-check
            "chunks_elements_idx",  # chunk -> element provenance lookup
        ],
    )
    def test_index_exists(self, index):
        assert index in SCHEMA_PATH.read_text(encoding="utf-8")

    def test_provenance_view_joins_the_whole_chain(self):
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        view = sql[sql.index("CREATE OR REPLACE VIEW element_provenance") :]
        for table in ("elements", "pages", "documents"):
            assert table in view


def test_schema_file_is_valid_utf8_and_non_trivial():
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    assert len(text) > 1000
    assert text.count("CREATE TABLE") == 4


def test_store_module_has_no_unparameterised_interpolation():
    """Guard against SQL built by formatting user-controlled values in."""
    source = Path(STORE_PATH).read_text(encoding="utf-8")
    # The one f-string in a query joins a fixed clause vocabulary; every value
    # is still bound. Anything else interpolating into SQL is a red flag.
    risky = [
        line
        for line in source.splitlines()
        if 'f"' in line and any(kw in line.upper() for kw in ("SELECT", "INSERT", "DELETE"))
    ]
    assert len(risky) <= 1, f"unexpected SQL interpolation: {risky}"
