"""Modality-specific views of a chunk.

**Ownership: unique to Method 2.** Method 1 has exactly one view of every chunk
-- ``Element.best_text()``, the flattened string -- and that single-view design
*is* the baseline.

Method 2's central claim is that one textual view per chunk throws away
structure a specialised retriever could have used. A table demonstrates it
plainly. Flattened to Markdown it becomes a wall of numbers in which the column
headers are three tokens among hundreds, so "which table breaks revenue down by
segment" has almost nothing to match against. Split into two views it becomes
tractable:

* the **schema view** answers *which table is this?* -- caption, column headers,
  table type. Short and semantic: what a dense retriever is good at.
* the **content view** answers *which table contains this value?* -- the cells.
  Long and literal: what BM25 is good at.

Neither view replaces the chunk. Both point at the same ``chunk_id``, so
provenance is unchanged and a hit still resolves to one region of one page.

These builders take the source :class:`Element` rather than reading fields off
``Chunk.metadata``, specifically so that Method 2 needs no change to the shared
chunker and Method 1's indexes stay byte-identical.
"""

from __future__ import annotations

from mmrag.schemas import Chunk, Element

# A schema view is meant to be short and semantic. Past this many columns the
# list has stopped describing the table and started being data.
MAX_SCHEMA_COLUMNS = 40


def table_schema_view(chunk: Chunk, element: Element | None) -> str:
    """What this table *is*: caption, columns, type, shape.

    Deliberately excludes cell values. Mixing them in would swamp the handful of
    tokens that actually identify the table -- which is exactly the failure mode
    of Method 1's flattening.
    """
    parts: list[str] = []
    table = element.table if element else None

    caption = (element.caption if element else None) or _leading_caption(chunk)
    if caption:
        parts.append(caption)

    if table:
        columns = [c for c in table.columns if c and c.strip()]
        if columns:
            parts.append("Columns: " + ", ".join(columns[:MAX_SCHEMA_COLUMNS]))
        if table.table_type.value != "unknown":
            parts.append(f"Table type: {table.table_type.value}")
        parts.append(f"{table.n_rows} rows by {table.n_cols} columns")

    # Section context is genuinely disambiguating for a table: "Table 3" means
    # little, "Table 3, under Segment Results" means a lot.
    if chunk.section:
        parts.append(f"Section: {chunk.section}")

    return "\n".join(p for p in parts if p.strip())


def table_content_view(chunk: Chunk) -> str:
    """The cell values, for literal lookup.

    Close to what Method 1 indexes, and intentionally so: the comparison is not
    "Method 2 sees more cells", it is "Method 2 can tell cells and schema apart".
    """
    return chunk.text


def figure_text_view(chunk: Chunk) -> str:
    """Whatever text a figure carries: caption, OCR, VLM description."""
    return _strip_header(chunk)


def has_figure_text(chunk: Chunk) -> bool:
    """Whether a figure chunk carries text beyond its breadcrumb header.

    The header was prepended for the embedder's benefit. A figure whose only
    "text" is its document title is textually invisible, and counting it as
    visible would understate exactly the gap Method 2 exists to close.
    """
    return bool(_strip_header(chunk).strip())


def _strip_header(chunk: Chunk) -> str:
    header = chunk.metadata.get("context_header") or ""
    body = chunk.text
    if header and body.startswith(header):
        body = body[len(header) :]
    return body.lstrip()


def _leading_caption(chunk: Chunk) -> str:
    """First line of a table chunk, when it looks like a caption.

    Table chunks are assembled as caption-then-Markdown, so the first line is
    the caption unless the table had none and that line is already a table row.
    """
    first = _strip_header(chunk).split("\n", 1)[0].strip()
    return "" if first.startswith("|") else first
