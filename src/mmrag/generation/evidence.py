"""Deterministic evidence organisation for Generation V2.

Turns the retrieved chunks into the numbered source text the generator reads,
without calling a model and without changing what can be cited:

* **Numbering and provenance are V1's.** Sources are admitted in retrieval
  order under the same token budget, with the same rule for skipping a source
  that would overflow it, and source ``n`` is always ``sources[n - 1]``. A
  citation therefore resolves to exactly the chunk it would have resolved to
  in V1, through the same ``resolve_citations``.
* **Display order groups each document page.** Admitted sources are shown
  grouped by (document, page), groups ordered by their best-ranked source and
  sources within a group by number, so a page's caption, table and prose are
  read together. Only the order on screen changes; the numbers do not.
* **Headers carry the metadata once.** ``[n] <title> · page N · <modality> ·
  <section>``, and the breadcrumb that chunking prepended to every body for the
  embedder's benefit is removed, since the header already says it.
* **Captions are labelled.** A table or figure caption is moved to the top of
  its body, where ingestion normally already put it, and marked ``Caption:``.

No text other than the duplicated breadcrumb is ever removed from a body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mmrag.schemas import ChunkType, ScoredChunk
from mmrag.textify.tokens import TokenCounter

# Bump when anything below changes what the generator reads; it is part of the
# V2 prompt version, and so of every V2 cache key.
EVIDENCE_FORMAT_VERSION = "evidence-pack-1"

GROUP_SEPARATOR = "\n\n---\n\n"
NO_SOURCES = "(no sources retrieved)"

# "Table 3:", "Figure SPM.6", "Fig. 2 |", "Table 1." -- a label and a number at
# the start of a line.
_CAPTION_LINE = re.compile(
    r"^\s*(?:table|fig(?:ure)?\.?)\s*[a-z]{0,4}\.?\s*\d+[a-z]?\b", re.IGNORECASE
)


@dataclass(frozen=True)
class EvidenceSource:
    """One admitted source: its citation number, chunk and rendered block."""

    number: int
    scored: ScoredChunk
    block: str
    tokens: int
    caption: str | None


@dataclass
class EvidencePack:
    """The sources a prompt carries, their citation order and their display order."""

    sources: list[EvidenceSource] = field(default_factory=list)
    dropped_for_budget: int = 0
    context_tokens: int = 0

    @property
    def scored(self) -> list[ScoredChunk]:
        """Chunks indexed by citation number minus one, as ``resolve_citations`` expects."""
        return [s.scored for s in self.sources]

    @property
    def groups(self) -> list[list[EvidenceSource]]:
        """Sources grouped by (document, page), in display order."""
        grouped: dict[tuple[str, int], list[EvidenceSource]] = {}
        for source in self.sources:  # already in number order, i.e. best rank first
            chunk = source.scored.chunk
            grouped.setdefault((chunk.doc_id, chunk.page_number), []).append(source)
        return list(grouped.values())

    def render(self) -> str:
        if not self.sources:
            return NO_SOURCES
        return GROUP_SEPARATOR.join(
            "\n\n".join(s.block for s in group) for group in self.groups
        )


def build_evidence_pack(
    retrieved: list[ScoredChunk], *, budget: int, tokens: TokenCounter
) -> EvidencePack:
    """Admit sources in retrieval order under the V1 budget rule, then render them.

    The rule is V1's exactly: a source that would take the total over budget is
    skipped (a later, smaller one may still fit), and the first source is always
    admitted, so a prompt is never empty while evidence exists.
    """
    pack = EvidencePack()
    for candidate in retrieved:
        number = len(pack.sources) + 1
        block, caption = format_source(number, candidate)
        cost = tokens.count(block)
        if pack.sources and pack.context_tokens + cost > budget:
            continue
        pack.sources.append(EvidenceSource(number, candidate, block, cost, caption))
        pack.context_tokens += cost
    pack.dropped_for_budget = len(retrieved) - len(pack.sources)
    return pack


def format_source(number: int, candidate: ScoredChunk) -> tuple[str, str | None]:
    """``[n] <title> · page N · <modality> · <section>``, then the cleaned body."""
    chunk = candidate.chunk
    title = str(chunk.metadata.get("doc_title", chunk.doc_id))
    parts = [f"[{number}] {_one_line(title)}", f"page {chunk.page_number}", chunk.chunk_type.value]
    if chunk.section and chunk.section.strip():
        parts.append(_one_line(chunk.section))
    body, caption = _body(candidate)
    return " · ".join(parts) + "\n" + body, caption


def strip_breadcrumb(text: str, header: str | None) -> str:
    """The chunk body without the context header chunking prepended to it."""
    if header and text.startswith(header):
        return text[len(header):].lstrip()
    return text


def _body(candidate: ScoredChunk) -> tuple[str, str | None]:
    chunk = candidate.chunk
    body = strip_breadcrumb(chunk.text, chunk.metadata.get("context_header"))
    if chunk.chunk_type not in (ChunkType.TABLE, ChunkType.FIGURE):
        return body, None

    lines = body.split("\n")
    index = _caption_index(lines, has_caption=bool(chunk.metadata.get("has_caption")))
    if index is None:
        return body, None
    caption = lines[index].strip()
    rest = [line for i, line in enumerate(lines) if i != index]
    rendered = f"Caption: {caption}"
    remainder = "\n".join(rest).strip("\n")
    return (f"{rendered}\n{remainder}" if remainder else rendered), caption


def _caption_index(lines: list[str], *, has_caption: bool) -> int | None:
    """Where the caption line is, if there is one.

    Ingestion puts a caption first (``Element.best_text``), so the first line
    is the caption whenever ingestion recorded one for a figure, or when it
    looks like one. Otherwise the first caption-shaped line, if any.
    """
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    if not non_empty:
        return None
    first = non_empty[0]
    if has_caption or _CAPTION_LINE.match(lines[first]):
        return first
    return next((i for i in non_empty[1:] if _CAPTION_LINE.match(lines[i])), None)


def _one_line(text: str) -> str:
    return " ".join(text.split())
