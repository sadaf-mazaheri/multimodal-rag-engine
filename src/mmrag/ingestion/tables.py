"""Table extraction and structuring.

PyMuPDF's ruling-line strategy is the primary detector. Its raw output is not
directly usable: on the Berkshire annual report a five-column financial table
comes back as 8x24, because every currency symbol, every alignment gap and
every rule gets its own column. Post-processing is therefore not optional
polish -- without it the Markdown flattening that Method 1 depends on produces
tables with twenty empty columns, and the structured view Method 2 depends on
is meaningless.

Both representations are kept: ``rows``/``columns`` for structure-aware
retrieval, and ``markdown`` for the textified baseline. Keeping both is what
makes the comparison between the two methods possible.
"""

from __future__ import annotations

from typing import Any

from mmrag.ingestion.classify import classify_table, table_confidence
from mmrag.ingestion.text_utils import collapse_whitespace, normalize_text
from mmrag.schemas import TableData, TableType

# A column made only of these is an artefact of alignment, not real data: it
# belongs merged into the value column beside it.
_SYMBOL_ONLY = {"$", "€", "£", "¥", "%", "(", ")", "-", "—", ""}


def _clean_cell(value: Any) -> str | None:
    """Normalise one cell, mapping blank to None so fill ratio is meaningful."""
    if value is None:
        return None
    text = collapse_whitespace(normalize_text(str(value)))
    return text or None


def _is_symbol_column(column: list[str | None]) -> bool:
    populated = [c for c in column if c]
    if not populated:
        return False
    return all(c.strip() in _SYMBOL_ONLY for c in populated)


def _drop_empty_and_merge_symbols(
    grid: list[list[str | None]],
) -> tuple[list[list[str | None]], dict[str, int]]:
    """Remove empty rows/columns and fold symbol-only columns into their neighbour.

    Returns the compacted grid and a record of what was removed, which is kept
    on the element so an odd-looking table can be traced back to this step
    rather than blamed on the PDF.
    """
    stats = {"dropped_rows": 0, "dropped_cols": 0, "merged_symbol_cols": 0}
    if not grid:
        return grid, stats

    n_cols = max(len(r) for r in grid)
    # Pad ragged rows so column indexing is safe.
    grid = [row + [None] * (n_cols - len(row)) for row in grid]

    # --- merge symbol-only columns rightwards into the next column ---------
    keep_cols: list[int] = []
    col_index = 0
    while col_index < n_cols:
        column = [row[col_index] for row in grid]
        if _is_symbol_column(column) and col_index + 1 < n_cols:
            for row in grid:
                symbol, value = row[col_index], row[col_index + 1]
                if symbol and value:
                    row[col_index + 1] = f"{symbol}{value}"
                elif symbol and not value:
                    row[col_index + 1] = symbol
            stats["merged_symbol_cols"] += 1
        else:
            keep_cols.append(col_index)
        col_index += 1

    grid = [[row[i] for i in keep_cols] for row in grid]
    stats["dropped_cols"] += n_cols - len(keep_cols) - stats["merged_symbol_cols"]

    # --- drop wholly empty columns ------------------------------------------
    if grid:
        n_cols = len(grid[0])
        non_empty = [i for i in range(n_cols) if any(row[i] for row in grid)]
        stats["dropped_cols"] += n_cols - len(non_empty)
        grid = [[row[i] for i in non_empty] for row in grid]

    # --- drop wholly empty rows ---------------------------------------------
    before = len(grid)
    grid = [row for row in grid if any(cell for cell in row)]
    stats["dropped_rows"] = before - len(grid)

    return grid, stats


def _fill_ratio(grid: list[list[str | None]]) -> float:
    total = sum(len(r) for r in grid)
    if total == 0:
        return 0.0
    return sum(1 for r in grid for c in r if c) / total


def _looks_like_header(row: list[str | None]) -> bool:
    """Whether a row reads as column names rather than data.

    Header cells are short, mostly non-numeric, and mostly populated.
    """
    populated = [c for c in row if c]
    if len(populated) < max(2, len(row) * 0.5):
        return False
    non_numeric = sum(1 for c in populated if not c.replace(",", "").replace(".", "").isdigit())
    return non_numeric / len(populated) >= 0.6 and all(len(c) <= 40 for c in populated)


def to_markdown(columns: list[str], rows: list[list[str | None]]) -> str:
    """GitHub-flavoured Markdown for a table.

    This string *is* Method 1's entire view of the table, so it is built to be
    read by a language model: pipes escaped so a cell containing one cannot
    corrupt the grid, and empty cells left visibly blank rather than filled.
    """

    def cell(value: str | None) -> str:
        return (value or "").replace("|", "\\|").replace("\n", " ")

    if not columns and not rows:
        return ""

    header = columns or [f"col{i + 1}" for i in range(len(rows[0]) if rows else 0)]
    lines = [
        "| " + " | ".join(cell(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(cell(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def build_table_data(
    raw_grid: list[list[Any]],
    *,
    header_names: list[str] | None = None,
) -> tuple[TableData, dict[str, Any]]:
    """Turn a raw extracted grid into a structured :class:`TableData`.

    ``header_names`` is the extractor's own header guess, used when the
    post-processed grid does not obviously start with a header row.

    Returns the table plus diagnostics for the element's metadata.
    """
    cleaned = [[_clean_cell(c) for c in row] for row in raw_grid]
    grid, stats = _drop_empty_and_merge_symbols(cleaned)

    if not grid:
        empty = TableData(n_rows=0, n_cols=0, header_rows=0, fill_ratio=0.0)
        return empty, {**stats, "reason": "no populated cells after compaction"}

    n_cols = len(grid[0])

    # --- decide the header --------------------------------------------------
    columns: list[str]
    body: list[list[str | None]]
    if _looks_like_header(grid[0]):
        columns = [c or f"col{i + 1}" for i, c in enumerate(grid[0])]
        body = grid[1:]
        header_rows = 1
    elif header_names and len([h for h in header_names if h]) >= 2:
        cleaned_names = [_clean_cell(h) for h in header_names][:n_cols]
        cleaned_names += [None] * (n_cols - len(cleaned_names))
        columns = [c or f"col{i + 1}" for i, c in enumerate(cleaned_names)]
        body = grid
        header_rows = 0  # the header came from the extractor, not from a grid row
    else:
        columns = [f"col{i + 1}" for i in range(n_cols)]
        body = grid
        header_rows = 0

    ragged = sum(1 for row in body if len(row) != n_cols)
    body = [row[:n_cols] + [None] * max(0, n_cols - len(row)) for row in body]

    classification = classify_table(columns, body)
    fill = _fill_ratio(body) if body else 0.0

    table = TableData(
        n_rows=len(body),
        n_cols=n_cols,
        header_rows=header_rows,
        columns=columns,
        rows=body,
        markdown=to_markdown(columns, body),
        table_type=TableType(classification.label),
        fill_ratio=fill,
    )
    diagnostics = {
        **stats,
        "ragged_rows": ragged,
        "table_type_confidence": classification.confidence,
        "table_type_evidence": classification.evidence,
        "raw_shape": [len(raw_grid), max((len(r) for r in raw_grid), default=0)],
    }
    return table, diagnostics


def confidence_for(table: TableData, diagnostics: dict[str, Any]) -> float:
    """Extraction confidence for a structured table."""
    return table_confidence(
        fill_ratio=table.fill_ratio,
        n_rows=table.n_rows,
        n_cols=table.n_cols,
        has_header=bool(table.columns) and table.header_rows > 0,
        ragged_rows=int(diagnostics.get("ragged_rows", 0)),
    )
