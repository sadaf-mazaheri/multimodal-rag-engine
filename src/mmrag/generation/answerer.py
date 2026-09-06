"""Turning retrieved chunks into a cited answer.

Two things matter here beyond producing fluent text.

**Citations must resolve.** The model is asked to cite numbered sources, and the
numbers are then mapped back to real chunks -- which carry doc id, page number
and bounding box. A citation that does not resolve is dropped and counted, not
passed through, because an unverifiable citation is worse than none: it looks
like grounding while providing none.

**Refusal is a valid answer.** With ``refuse_without_evidence`` the model is told
to say so when the context does not contain the answer. A benchmark that
rewards confident guessing measures the wrong thing -- Step 6 needs to separate
"retrieval failed" from "generation hallucinated over good evidence".
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from mmrag.config import GenerationConfig
from mmrag.generation.providers.base import ImageInput, LLMProvider, Message
from mmrag.logging_utils import get_logger
from mmrag.schemas import Answer, Chunk, Citation, ScoredChunk
from mmrag.textify.tokens import TokenCounter, get_token_counter

log = get_logger(__name__)

# The model is asked for [1], [2] markers; accept [1, 2] and [1][2] too, since
# models drift between these and a stricter parser would silently discard
# perfectly good citations.
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

REFUSAL_MARKER = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.

Rules:
- Use only information present in the sources. Do not use prior knowledge.
- Cite every factual claim with the source number in square brackets, like [2].
- A claim drawn from several sources cites each one: [1][3].
- Quote figures, dates and names exactly as they appear in the sources.
- If the sources do not contain the answer, reply with exactly {refusal} and \
one sentence saying what is missing. Do not guess.
- Be concise. Do not restate the question or describe the sources."""

SYSTEM_PROMPT_NO_REFUSAL = """You answer questions using only the numbered sources provided.

Rules:
- Use only information present in the sources. Do not use prior knowledge.
- Cite every factual claim with the source number in square brackets, like [2].
- Quote figures, dates and names exactly as they appear in the sources.
- Be concise. Do not restate the question or describe the sources."""


@dataclass
class PromptBuild:
    """An assembled prompt and the source numbering it used."""

    messages: list[Message]
    sources: list[ScoredChunk]
    prompt_tokens: int = 0
    dropped_for_budget: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Answerer:
    """Assembles prompts, calls a provider, and resolves citations."""

    def __init__(
        self,
        config: GenerationConfig,
        provider: LLMProvider,
        *,
        token_counter: TokenCounter | None = None,
    ):
        self.config = config
        self.provider = provider
        self.tokens = token_counter or get_token_counter()

    # -- prompt assembly -----------------------------------------------------

    def build_prompt(
        self,
        query: str,
        retrieved: list[ScoredChunk],
        *,
        images: list[ImageInput] | None = None,
    ) -> PromptBuild:
        """Assemble the prompt, dropping the weakest sources if over budget.

        Sources are numbered in retrieval order and truncated from the *tail*,
        so the best-ranked evidence is never the evidence that gets cut.
        """
        system = SYSTEM_PROMPT if self.config.refuse_without_evidence else SYSTEM_PROMPT_NO_REFUSAL
        system = system.format(refusal=REFUSAL_MARKER)

        blocks: list[str] = []
        used: list[ScoredChunk] = []
        budget = self.config.max_context_tokens
        spent = 0

        for candidate in retrieved:
            block = self._format_source(len(used) + 1, candidate)
            cost = self.tokens.count(block)
            if used and spent + cost > budget:
                continue
            blocks.append(block)
            used.append(candidate)
            spent += cost

        body = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
        user = f"Sources:\n\n{body}\n\nQuestion: {query}\n\nAnswer:"

        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user, images=list(images or [])),
        ]
        return PromptBuild(
            messages=messages,
            sources=used,
            prompt_tokens=self.tokens.count(system) + self.tokens.count(user),
            dropped_for_budget=len(retrieved) - len(used),
            metadata={"n_images": len(images or []), "context_tokens": spent},
        )

    def _format_source(self, number: int, candidate: ScoredChunk) -> str:
        """One numbered source block.

        The header carries document, page and modality because they are what the
        model needs to attribute a claim correctly -- and because a table read as
        prose is far more likely to be misread if the model does not know it is
        looking at a table.
        """
        chunk = candidate.chunk
        title = chunk.metadata.get("doc_title", chunk.doc_id)
        kind = chunk.chunk_type.value
        return f"[{number}] {title} - page {chunk.page_number} ({kind})\n{chunk.text}"

    # -- generation ----------------------------------------------------------

    def answer(
        self,
        query: str,
        retrieved: list[ScoredChunk],
        *,
        images: list[ImageInput] | None = None,
        method: str = "unknown",
        model: str | None = None,
    ) -> Answer:
        prompt = self.build_prompt(query, retrieved, images=images)

        started = time.perf_counter()
        completion = self.provider.complete(
            prompt.messages,
            model=model or self.config.text_model,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        elapsed = (time.perf_counter() - started) * 1000

        citations, unresolved = resolve_citations(completion.text, prompt.sources)

        return Answer(
            query=query,
            text=completion.text,
            citations=citations,
            retrieved=retrieved,
            method=method,
            latency_ms={"generation_ms": elapsed},
            usage=completion.usage.as_dict(),
            metadata={
                "provider": getattr(self.provider, "name", "unknown"),
                "model": completion.model,
                "n_sources": len(prompt.sources),
                "prompt_tokens_estimated": prompt.prompt_tokens,
                "dropped_for_budget": prompt.dropped_for_budget,
                "n_images": len(images or []),
                "unresolved_citations": unresolved,
                "refused": is_refusal(completion.text),
                **completion.metadata,
            },
        )


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def parse_citation_numbers(text: str) -> list[int]:
    """Source numbers cited in an answer, in order of first appearance."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number not in seen:
                seen.append(number)
    return seen


def resolve_citations(text: str, sources: list[ScoredChunk]) -> tuple[list[Citation], list[int]]:
    """Map cited numbers back to real chunks.

    Returns the resolved citations and the numbers that pointed at nothing.
    Out-of-range numbers are a real and common failure -- a model citing [7] when
    five sources were supplied -- and silently dropping them would hide a
    grounding problem that Step 6 needs to measure.
    """
    resolved: list[Citation] = []
    unresolved: list[int] = []

    for number in parse_citation_numbers(text):
        index = number - 1
        if not (0 <= index < len(sources)):
            unresolved.append(number)
            continue
        chunk = sources[index].chunk
        resolved.append(
            Citation(
                doc_id=chunk.doc_id,
                doc_title=str(chunk.metadata.get("doc_title", chunk.doc_id)),
                page_number=chunk.page_number,
                element_ids=list(chunk.element_ids),
                chunk_id=chunk.chunk_id,
                bbox=chunk.bbox,
                section=chunk.section,
                snippet=_snippet(chunk),
            )
        )

    if unresolved:
        log.warning("answer cited %s but only %d sources were supplied", unresolved, len(sources))
    return resolved, unresolved


def _snippet(chunk: Chunk, limit: int = 240) -> str:
    """A short quotable extract, with the breadcrumb header stripped.

    The header was added for the embedder's benefit; showing it back to a reader
    as if it were the source text would be misleading.
    """
    text = chunk.text
    header = chunk.metadata.get("context_header")
    if header and text.startswith(header):
        text = text[len(header) :].lstrip()
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def is_refusal(text: str) -> bool:
    return REFUSAL_MARKER in text
