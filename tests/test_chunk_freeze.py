"""Regression guard on Method 1's chunk set.

**This is a snapshot, not an invariant.** It records what Method 1 chunked the
corpus into at one point in time, so that an *unintended* change shows up as a
failing test rather than as a silently different baseline. It is not a claim
that the chunk set may never change.

Why it exists: Method 1 is the baseline the other methods are measured against,
and rebuilding it takes several minutes, so a drift introduced by an unrelated
edit to ``flatten.py``, ``chunker.py`` or the parser is easy to miss and
expensive to discover late. ``docs/architecture.md`` claims Method 1's chunk set
is unchanged by Method 2's work; before this file that claim rested on a manual
check performed once.

What a chunk id actually commits to
-----------------------------------
``Chunker`` builds the id from ``variant | doc_id | page_number | element_ids |
body``, so it covers both the grouping *and* the body text. A chunk whose text
changes therefore gets a new id rather than keeping its own: the OCR pass showed
this plainly, where 232 figures that gained recovered text appeared as 232
removals paired with 232 additions.

The frozen digest is taken over ``Chunk.text``, which is *not* quite ``body``:
with ``prepend_context_header`` the stored text carries a
``Document > Section > Subsection`` breadcrumb that the id does not hash. So the
digest covers the one class of drift the id cannot see -- a change to the
breadcrumb, from section tracking, document metadata, or the flag itself,
altering what gets embedded while every id stays put.

The three categories the test reports:

* **added**   -- chunks that did not exist before (new elements, new grouping,
  or changed body text)
* **removed** -- chunks that no longer exist
* **changed** -- same id, different stored text: a breadcrumb change, since the
  body is already covered by the id

Updating the baseline deliberately
----------------------------------
When a change *should* move the chunk set -- enabling OCR is the expected next
one -- regenerate rather than edit by hand::

    MMRAG_UPDATE_CHUNK_BASELINE=1 python -m pytest tests/test_chunk_freeze.py

Then review the diff. Because the file is one sorted line per chunk, the diff
names precisely which chunks moved, which is the point: an intentional update
should be reviewable, and a surprise in it should be visible.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest

BASELINE = Path(__file__).parent / "baselines" / "method1_chunks.txt"
UPDATE_ENV = "MMRAG_UPDATE_CHUNK_BASELINE"

# "method1#" followed by 16 hex characters -- see schemas.make_chunk_id.
CHUNK_ID_RE = re.compile(r"^method1#[0-9a-f]{16}$")

# How many differing chunks to name before truncating the failure message.
MAX_LISTED = 15


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _current_chunks() -> dict[str, str]:
    """Re-chunk the parsed corpus exactly as ``Method1Textified`` does.

    Reads the sidecars rather than the built index: the index is a derived
    artefact that may be stale, while re-chunking exercises the real path from
    parsed elements to chunks. A parser change reaches this test as soon as the
    corpus is re-ingested.
    """
    from mmrag.config import PROCESSED_DIR, load_experiment_config
    from mmrag.ingestion.pipeline import read_sidecar
    from mmrag.textify import Chunker

    config = load_experiment_config("method1")
    chunker = Chunker(
        config.chunking,
        variant=config.chunk_variant,
        embedding_model=config.embedding.text_model,
    )

    out: dict[str, str] = {}
    for sidecar in sorted(PROCESSED_DIR.glob("*/parsed.json")):
        parsed = read_sidecar(sidecar)
        chunks, _ = chunker.chunk_document(parsed.document, parsed.elements)
        for chunk in chunks:
            if chunk.chunk_id in out:
                raise AssertionError(
                    f"duplicate chunk id {chunk.chunk_id} while re-chunking; "
                    "ids must be unique or a vector would be overwritten"
                )
            out[chunk.chunk_id] = _digest(chunk.text)
    return out


def _read_baseline() -> dict[str, str]:
    rows: dict[str, str] = {}
    for number, line in enumerate(BASELINE.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise AssertionError(f"{BASELINE.name}:{number}: expected '<chunk_id> <digest>'")
        rows[parts[0]] = parts[1]
    return rows


def _write_baseline(chunks: dict[str, str]) -> None:
    header = [
        "# Method 1 chunk baseline -- A SNAPSHOT, NOT AN INVARIANT.",
        "#",
        "# One line per chunk: <chunk_id> <sha256(chunk.text)[:12]>, sorted by id.",
        "# The digest is here because a chunk id hashes element *grouping*, not",
        "# text, so an enrichment pass can change a chunk's content without",
        "# changing its id.",
        "#",
        "# Regenerate deliberately, then review the diff:",
        "#   MMRAG_UPDATE_CHUNK_BASELINE=1 python -m pytest tests/test_chunk_freeze.py",
        "#",
        f"# chunks: {len(chunks)}",
        "",
    ]
    body = [f"{chunk_id}  {digest}" for chunk_id, digest in sorted(chunks.items())]
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text("\n".join(header + body) + "\n", encoding="utf-8")


def _summarize(added: list[str], removed: list[str], changed: list[str]) -> str:
    lines = [
        "Method 1's chunk set no longer matches the frozen baseline.",
        f"  added:   {len(added)}",
        f"  removed: {len(removed)}",
        f"  changed: {len(changed)}  (same id, different text)",
        "",
    ]
    for label, ids in (("added", added), ("removed", removed), ("changed", changed)):
        if not ids:
            continue
        shown = ids[:MAX_LISTED]
        lines.append(f"  {label}:")
        lines.extend(f"    {i}" for i in shown)
        if len(ids) > len(shown):
            lines.append(f"    ... and {len(ids) - len(shown)} more")
    lines += [
        "",
        "If this change was intended -- enabling OCR, a chunking change, a",
        "re-ingest after a parser fix -- regenerate the baseline and review the",
        "diff before committing:",
        "",
        f"  {UPDATE_ENV}=1 python -m pytest tests/test_chunk_freeze.py",
        "",
        "If it was not intended, something upstream of retrieval moved and",
        "Method 1's recorded numbers no longer describe the current code.",
    ]
    return "\n".join(lines)


class TestBaselineFileIsWellFormed:
    """Cheap structural checks. These need no corpus, so they run anywhere."""

    def test_baseline_exists(self):
        assert BASELINE.exists(), f"missing {BASELINE}; regenerate with {UPDATE_ENV}=1"

    def test_every_line_is_an_id_and_a_digest(self):
        rows = _read_baseline()
        assert rows, "baseline is empty"
        for chunk_id, digest in rows.items():
            assert CHUNK_ID_RE.match(chunk_id), f"malformed chunk id: {chunk_id!r}"
            assert re.fullmatch(r"[0-9a-f]{12}", digest), f"malformed digest: {digest!r}"

    def test_ids_are_unique(self):
        text = BASELINE.read_text(encoding="utf-8")
        ids = [
            line.split()[0]
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert len(ids) == len(set(ids)), "duplicate chunk ids in the baseline"

    def test_recorded_count_matches_the_rows(self):
        """The header's count is documentation; keep it honest."""
        text = BASELINE.read_text(encoding="utf-8")
        declared = re.search(r"^# chunks: (\d+)$", text, re.MULTILINE)
        assert declared, "baseline header does not record a chunk count"
        assert int(declared.group(1)) == len(_read_baseline())


@pytest.mark.slow
class TestMethod1ChunkFreeze:
    """The freeze itself. Needs the parsed corpus, so it skips without one."""

    def test_chunk_set_matches_the_frozen_baseline(self):
        from mmrag.config import PROCESSED_DIR

        if not sorted(PROCESSED_DIR.glob("*/parsed.json")):
            pytest.skip("no parsed corpus; run 'mmrag ingest run' to check the freeze")

        current = _current_chunks()

        if os.environ.get(UPDATE_ENV) == "1":
            _write_baseline(current)
            pytest.skip(f"baseline rewritten with {len(current)} chunks; review the diff")

        baseline = _read_baseline()
        added = sorted(set(current) - set(baseline))
        removed = sorted(set(baseline) - set(current))
        changed = sorted(c for c in set(current) & set(baseline) if current[c] != baseline[c])

        assert not (added or removed or changed), _summarize(added, removed, changed)
