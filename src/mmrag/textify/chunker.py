"""Chunking: elements in, retrieval units out.

Three rules shape every decision here.

**Never cross a page boundary.** A chunk spanning two pages cannot be cited to a
single page, which breaks the provenance contract the whole benchmark rests on.
``Chunk`` enforces this at construction; the chunker respects it by grouping
strictly within a page.

**Tables and figures are their own chunks.** Merging a table into surrounding
prose produces a unit that is neither good prose nor a usable table, and makes
``chunk_type`` -- which Method 2 routes on -- meaningless.

**Overlap is carried as whole sentences.** Splitting mid-sentence to hit an exact
token count produces chunks that begin mid-clause, which hurts both the embedder
and the reader of a citation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mmrag.config import ChunkingConfig
from mmrag.logging_utils import get_logger
from mmrag.schemas import (
    BBox,
    Chunk,
    ChunkType,
    Document,
    Element,
    ElementType,
    make_chunk_id,
)
from mmrag.textify.flatten import FlattenReport, context_header, flatten_elements
from mmrag.textify.tokens import TokenCounter, get_token_counter, split_sentences

log = get_logger(__name__)


def chunk_type_for(element: Element) -> ChunkType:
    if element.element_type is ElementType.TABLE:
        return ChunkType.TABLE
    if element.element_type.is_visual:
        return ChunkType.FIGURE
    return ChunkType.TEXT


@dataclass
class ChunkingReport:
    """What chunking produced, alongside what flattening dropped."""

    flatten: FlattenReport
    n_chunks: int = 0
    by_type: dict[str, int] | None = None
    oversized: int = 0
    token_counter: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_chunks": self.n_chunks,
            "by_type": self.by_type or {},
            "oversized": self.oversized,
            "token_counter": self.token_counter,
            "flatten": self.flatten.as_dict(),
        }


class Chunker:
    """Builds Method 1's retrieval units from parsed elements."""

    def __init__(
        self,
        config: ChunkingConfig,
        *,
        variant: str = "method1",
        token_counter: TokenCounter | None = None,
        embedding_model: str | None = None,
        keep_textless_visuals: bool = False,
    ):
        self.config = config
        self.variant = variant
        self.tokens = token_counter or get_token_counter(embedding_model)
        # Off by default, so Method 1's chunk set is unchanged. See
        # flatten_elements for why Method 2 turns it on.
        self.keep_textless_visuals = keep_textless_visuals

    # -- public API ---------------------------------------------------------

    def chunk_document(
        self, document: Document, elements: list[Element]
    ) -> tuple[list[Chunk], ChunkingReport]:
        """Chunk one document's elements, page by page."""
        kept, flatten_report = flatten_elements(
            elements, keep_textless_visuals=self.keep_textless_visuals
        )
        report = ChunkingReport(flatten=flatten_report, token_counter=self.tokens.name)

        chunks: list[Chunk] = []
        for page_number in sorted({e.page_number for e in kept}):
            page_elements = [e for e in kept if e.page_number == page_number]
            page_elements.sort(key=lambda e: (e.reading_order, e.element_id))
            chunks.extend(self._chunk_page(document, page_elements))

        by_type: dict[str, int] = {}
        for chunk in chunks:
            by_type[chunk.chunk_type.value] = by_type.get(chunk.chunk_type.value, 0) + 1

        report.n_chunks = len(chunks)
        report.by_type = dict(sorted(by_type.items()))
        report.oversized = sum(
            1 for c in chunks if (c.token_count or 0) > self.config.max_table_tokens
        )
        return chunks, report

    # -- page-level ----------------------------------------------------------

    def _chunk_page(self, document: Document, elements: list[Element]) -> list[Chunk]:
        chunks: list[Chunk] = []
        prose_run: list[Element] = []

        for element in elements:
            if element.element_type is ElementType.TABLE:
                chunks.extend(self._flush_prose(document, prose_run))
                prose_run = []
                chunks.extend(self._chunk_table(document, element))
            elif element.element_type.is_visual:
                chunks.extend(self._flush_prose(document, prose_run))
                prose_run = []
                chunks.append(self._chunk_figure(document, element))
            else:
                prose_run.append(element)

        chunks.extend(self._flush_prose(document, prose_run))
        return chunks

    # -- prose ---------------------------------------------------------------

    def _flush_prose(self, document: Document, elements: list[Element]) -> list[Chunk]:
        """Pack a run of consecutive text elements into overlapping chunks."""
        if not elements:
            return []

        # Sentences tagged with the element they came from, so a chunk can still
        # name every element it draws on.
        pieces: list[tuple[str, Element]] = []
        for element in elements:
            text = element.best_text().strip()
            if not text:
                continue
            sentences = split_sentences(text) or [text]
            pieces.extend((s, element) for s in sentences)

        if not pieces:
            return []

        chunks: list[Chunk] = []
        current: list[tuple[str, Element]] = []
        current_tokens = 0

        for sentence, element in pieces:
            sentence_tokens = self.tokens.count(sentence)

            if current and current_tokens + sentence_tokens > self.config.target_tokens:
                chunks.append(self._build_prose_chunk(document, current))
                current = self._overlap_tail(current)
                current_tokens = sum(self.tokens.count(s) for s, _ in current)

            current.append((sentence, element))
            current_tokens += sentence_tokens

            # A single sentence longer than the target (a run-on caption, a
            # pasted URL list) would otherwise sit in a chunk that never closes.
            if current_tokens >= self.config.target_tokens and len(current) == 1:
                chunks.append(self._build_prose_chunk(document, current))
                current, current_tokens = [], 0

        if current:
            chunks.append(self._build_prose_chunk(document, current))
        return chunks

    def _overlap_tail(self, pieces: list[tuple[str, Element]]) -> list[tuple[str, Element]]:
        """The trailing sentences to repeat at the start of the next chunk.

        Overlap exists so a fact stated across a chunk boundary is still fully
        present in at least one chunk.
        """
        if self.config.overlap_tokens <= 0:
            return []
        tail: list[tuple[str, Element]] = []
        total = 0
        for sentence, element in reversed(pieces):
            count = self.tokens.count(sentence)
            if total + count > self.config.overlap_tokens and tail:
                break
            tail.insert(0, (sentence, element))
            total += count
        # Never repeat the whole chunk: that would make the next chunk a
        # duplicate and stall progress through the page.
        return tail if len(tail) < len(pieces) else tail[-1:]

    def _build_prose_chunk(self, document: Document, pieces: list[tuple[str, Element]]) -> Chunk:
        body = " ".join(s for s, _ in pieces)
        elements = list(dict.fromkeys(e.element_id for _, e in pieces))
        first = pieces[0][1]
        boxes = [e.bbox for _, e in pieces if e.bbox is not None]
        return self._assemble(
            document=document,
            body=body,
            element_ids=elements,
            page_number=first.page_number,
            chunk_type=ChunkType.TEXT,
            section=first.section,
            subsection=first.subsection,
            bbox=BBox.union_all(boxes),
            header_source=first,
            extra={"n_sentences": len(pieces)},
        )

    # -- tables --------------------------------------------------------------

    def _chunk_table(self, document: Document, element: Element) -> list[Chunk]:
        """One chunk per table, split by rows only if it is genuinely too large.

        Splits repeat the header row in every part. A table fragment without its
        header is unreadable: the columns lose their meaning entirely, which is
        precisely the failure mode Method 2 is meant to avoid.
        """
        table = element.table
        text = element.best_text()
        if table is None or not text.strip():
            return []

        if self.tokens.count(text) <= self.config.max_table_tokens:
            return [self._table_chunk(document, element, text, part=None)]

        from mmrag.ingestion.tables import to_markdown

        header = [element.caption] if element.caption else []
        chunks: list[Chunk] = []
        batch: list[list[str | None]] = []
        part = 1

        for row in table.rows:
            batch.append(row)
            candidate = "\n".join([*header, to_markdown(table.columns, batch)])
            if self.tokens.count(candidate) > self.config.max_table_tokens and len(batch) > 1:
                batch.pop()
                chunks.append(
                    self._table_chunk(
                        document,
                        element,
                        "\n".join([*header, to_markdown(table.columns, batch)]),
                        part=part,
                    )
                )
                part += 1
                batch = [row]

        if batch:
            chunks.append(
                self._table_chunk(
                    document,
                    element,
                    "\n".join([*header, to_markdown(table.columns, batch)]),
                    part=part if part > 1 else None,
                )
            )
        return chunks

    def _table_chunk(
        self, document: Document, element: Element, text: str, *, part: int | None
    ) -> Chunk:
        extra: dict[str, Any] = {
            "table_type": element.table.table_type.value if element.table else None,
            "table_shape": list(element.table.shape) if element.table else None,
        }
        if part is not None:
            extra["table_part"] = part
        return self._assemble(
            document=document,
            body=text,
            element_ids=[element.element_id],
            page_number=element.page_number,
            chunk_type=ChunkType.TABLE,
            section=element.section,
            subsection=element.subsection,
            bbox=element.bbox,
            header_source=element,
            extra=extra,
            # Parts of one table must not collide on a content hash.
            id_salt=f"part{part}" if part else "",
        )

    # -- figures -------------------------------------------------------------

    def _chunk_figure(self, document: Document, element: Element) -> Chunk:
        figure = element.figure
        return self._assemble(
            document=document,
            body=element.best_text(),
            element_ids=[element.element_id],
            page_number=element.page_number,
            chunk_type=ChunkType.FIGURE,
            section=element.section,
            subsection=element.subsection,
            bbox=element.bbox,
            header_source=element,
            extra={
                "figure_type": figure.figure_type.value if figure else None,
                # Records which derived representations the retrievable text came
                # from, so a failed figure retrieval can be attributed to a
                # missing OCR pass rather than to the retriever.
                "has_caption": bool(element.caption),
                "has_ocr": bool(figure and figure.ocr_text),
                "has_description": bool(figure and figure.description),
                "image_path": figure.image_path if figure else None,
            },
        )

    # -- shared --------------------------------------------------------------

    def _assemble(
        self,
        *,
        document: Document,
        body: str,
        element_ids: list[str],
        page_number: int,
        chunk_type: ChunkType,
        section: str | None,
        subsection: str | None,
        bbox: BBox | None,
        header_source: Element,
        extra: dict[str, Any],
        id_salt: str = "",
    ) -> Chunk:
        header = ""
        text = body
        if self.config.prepend_context_header:
            header = context_header(document, header_source)
            if header:
                text = f"{header}\n\n{body}"

        return Chunk(
            # The element ids are part of the identity, not just the content.
            # Two distinct charts on one page routinely share a caption -- "Source:"
            # or an identical NOTE line -- so hashing content alone collides, and
            # a collision silently maps a retrieval hit onto the wrong figure.
            # Element ids are unique and stable, so including them keeps ids
            # deterministic across re-runs while guaranteeing they are distinct.
            chunk_id=make_chunk_id(
                self.variant,
                document.doc_id,
                str(page_number),
                ",".join(element_ids),
                body,
                id_salt,
            ),
            doc_id=document.doc_id,
            page_number=page_number,
            chunk_type=chunk_type,
            text=text,
            element_ids=element_ids,
            bbox=bbox,
            section=section,
            subsection=subsection,
            token_count=self.tokens.count(text),
            variant=self.variant,
            metadata={
                # Kept so the header can be stripped for an ablation without
                # re-deriving it, and so evaluation can tell whether a match
                # landed on content or on the breadcrumb.
                "context_header": header,
                "doc_title": document.title,
                "doc_type": document.doc_type.value,
                "domain": document.domain,
                **{k: v for k, v in extra.items() if v is not None},
            },
        )
