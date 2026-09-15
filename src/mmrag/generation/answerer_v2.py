"""Generation V2: evidence pack, one answer call, deterministic validation.

    retrieved chunks -> EvidencePack -> one LLM call -> AnswerValidator -> Answer

What changed from V1, and why, in one place:

* **The answer contract.** V1 asked the model to be concise and not to describe
  the sources. Measured on V1's judged runs, the dominant failure was answers
  that stopped at identification ("Table 3 reports the BLEU scores [6]") while
  the missing facts sat in the source they cited, and the second was refusing
  outright when the sources covered only part of the question. V2 asks for a
  direct answer with the specifics the sources give, a description of what a
  named table or figure shows, and a partial answer instead of a refusal when
  some evidence is relevant.
* **The source text.** Grouped by page, with one clean header per source and
  labelled captions (see ``evidence``).
* **A validation report** on every answer (see ``validation``).

What did not change: the provider, model, temperature and output cap; one call
per query; plain-text answers with ``[n]`` markers resolved by V1's own
``resolve_citations``; the ``INSUFFICIENT_EVIDENCE`` marker; the context budget.
V1 itself (``answerer.py``) is untouched and remains the default.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from mmrag.config import GenerationConfig
from mmrag.generation.answerer import (
    REFUSAL_MARKER,
    PromptBuild,
    is_refusal,
    resolve_citations,
)
from mmrag.generation.evidence import EVIDENCE_FORMAT_VERSION, EvidencePack, build_evidence_pack
from mmrag.generation.providers.base import ImageInput, LLMProvider, Message
from mmrag.generation.validation import AnswerValidator, ValidationReport, answer_sentences
from mmrag.schemas import Answer, ScoredChunk
from mmrag.textify.tokens import TokenCounter, get_token_counter

PIPELINE = "v2"

_ANSWER_RULES = (
    "How to answer:\n"
    "- Start with a direct answer to the question.\n"
    "- Then give the details the sources state that support it: values with their units, "
    "dates, conditions, scope, and the names of the things involved.\n"
    "- If the question asks which table, figure or section contains something, name it "
    "exactly as the sources do and say what it shows or reports, using its caption and the "
    "surrounding text.{partial}\n"
    "- Usually write 2 to 6 sentences."
)

_PARTIAL = (
    "\n- If the sources answer only part of the question, answer that part and say briefly "
    "what the sources do not cover."
)
_PARTIAL_WITH_REFUSAL = _PARTIAL + (
    "\n- Only if no source contains information relevant to the question, reply with exactly "
    "{refusal} followed by one sentence saying what is missing."
)
_PARTIAL_NO_REFUSAL = _PARTIAL

_SOURCE_RULES = (
    "Rules:\n"
    "- Use only information stated in the sources. Do not use prior knowledge.\n"
    "- End every sentence that states a fact with the number of each source it comes from, "
    "like [2] or [1][3].\n"
    "- Copy numbers, units, names and identifiers exactly as they appear in the sources.\n"
    "- Do not restate the question."
)

_PREAMBLE = (
    "You answer questions using only the numbered sources provided. Sources from the same "
    "document page are grouped together, separated from other pages by a line of dashes."
)

SYSTEM_PROMPT_V2 = "\n\n".join(
    [_PREAMBLE, _ANSWER_RULES.format(partial=_PARTIAL_WITH_REFUSAL), _SOURCE_RULES]
)
SYSTEM_PROMPT_V2_NO_REFUSAL = "\n\n".join(
    [_PREAMBLE, _ANSWER_RULES.format(partial=_PARTIAL_NO_REFUSAL), _SOURCE_RULES]
)

# Covers both system prompts, the user-message template and the evidence format,
# so any change to what the V2 generator reads changes every V2 cache key.
USER_TEMPLATE_V2 = "Sources:\n\n{sources}\n\nQuestion: {query}\n\nAnswer:"
PROMPT_VERSION_V2 = hashlib.sha256(
    "\x00".join([PIPELINE, SYSTEM_PROMPT_V2, SYSTEM_PROMPT_V2_NO_REFUSAL, USER_TEMPLATE_V2,
                 EVIDENCE_FORMAT_VERSION]).encode("utf-8")
).hexdigest()[:16]

# ---------------------------------------------------------------------------
# V2.1: the V2.0 contract with two fixes, and nothing else changed.
#
# * Wrong-entity answers. V2.0 refused "only if no source contains information
#   relevant to the question", so a question about one product answered from a
#   page about a similar product counted as a partial answer. V2.1 limits every
#   answer, and the partial-answer rule, to evidence about the subject the
#   question names, and refuses when the sources describe a different one.
# * Sentence-level citations. V2.0 answers routinely cited only the last
#   sentence of a paragraph. V2.1 states that each sentence needs its own
#   citation and that a later citation does not cover an earlier sentence.
#
# The evidence pack, the user template and the single call are V2.0's.
# ---------------------------------------------------------------------------

PIPELINE_V2_1 = "v2.1"

_ANSWER_RULES_V2_1 = (
    "How to answer:\n"
    "- Start with a direct answer to the question.\n"
    "- Then give the details the sources state that support it: values with their units, "
    "dates, conditions, scope, and the names of the things involved.\n"
    "- If the question asks which table, figure or section contains something, name it "
    "exactly as the sources do and say what it shows or reports, using its caption and the "
    "surrounding text.\n"
    "- Answer only about the subject the question asks about. If the question names a specific "
    "entity, item, product, model, version, document, year or other subject, use only evidence "
    "about that same subject. Never substitute a similar, related, newer, older or otherwise "
    "different one.\n"
    "- If the sources answer only part of the question about that same subject, answer that "
    "part and say briefly what the sources do not cover.{refusal_rule}\n"
    "- Usually write 2 to 6 sentences."
)
_REFUSAL_RULE_V2_1 = (
    "\n- If the sources contain no information about the subject the question asks about, "
    "including when they only describe a different entity, version or year, reply with exactly "
    "{refusal} followed by one sentence saying what is missing."
)
_NO_REFUSAL_RULE_V2_1 = (
    "\n- If the sources only describe a different entity, version or year, say so instead of "
    "answering about it."
)

_CITATION_RULES_V2_1 = (
    "Citations:\n"
    "- End every sentence that states a fact with at least one source number, like [2] or "
    "[1][3], naming the sources that support that sentence.\n"
    "- Each sentence needs its own citation. A citation at the end of a paragraph or of a later "
    "sentence does not cover earlier sentences.\n"
    "- If no source supports a detail, leave the detail out rather than stating it without a "
    "citation.\n"
    "- A sentence that only says what the sources do not cover{refusal_note} needs no citation."
)

_SOURCE_RULES_V2_1 = (
    "Rules:\n"
    "- Use only information stated in the sources. Do not use prior knowledge.\n"
    "- Copy numbers, units, names and identifiers exactly as they appear in the sources.\n"
    "- Do not restate the question."
)

SYSTEM_PROMPT_V2_1 = "\n\n".join([
    _PREAMBLE,
    _ANSWER_RULES_V2_1.format(refusal_rule=_REFUSAL_RULE_V2_1, refusal="{refusal}"),
    _CITATION_RULES_V2_1.format(refusal_note=", or the {refusal} reply,"),
    _SOURCE_RULES_V2_1,
])
SYSTEM_PROMPT_V2_1_NO_REFUSAL = "\n\n".join([
    _PREAMBLE,
    _ANSWER_RULES_V2_1.format(refusal_rule=_NO_REFUSAL_RULE_V2_1),
    _CITATION_RULES_V2_1.format(refusal_note=""),
    _SOURCE_RULES_V2_1,
])
PROMPT_VERSION_V2_1 = hashlib.sha256(
    "\x00".join([PIPELINE_V2_1, SYSTEM_PROMPT_V2_1, SYSTEM_PROMPT_V2_1_NO_REFUSAL,
                 USER_TEMPLATE_V2, EVIDENCE_FORMAT_VERSION]).encode("utf-8")
).hexdigest()[:16]

# Prompt pair and version per V2 variant; the evidence pack and template are shared.
VARIANTS: dict[str, tuple[str, str, str]] = {
    PIPELINE: (SYSTEM_PROMPT_V2, SYSTEM_PROMPT_V2_NO_REFUSAL, PROMPT_VERSION_V2),
    PIPELINE_V2_1: (SYSTEM_PROMPT_V2_1, SYSTEM_PROMPT_V2_1_NO_REFUSAL, PROMPT_VERSION_V2_1),
}


@dataclass
class PromptBuildV2(PromptBuild):
    """A V1 ``PromptBuild`` plus the evidence pack and the exact source text sent."""

    pack: EvidencePack | None = None
    sources_block: str = ""
    pipeline: str = PIPELINE


class AnswererV2:
    """Assembles an evidence-pack prompt, makes one provider call, validates the answer."""

    pipeline = PIPELINE

    def __init__(
        self,
        config: GenerationConfig,
        provider: LLMProvider,
        *,
        token_counter: TokenCounter | None = None,
        validator: AnswerValidator | None = None,
        variant: str = PIPELINE,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unknown V2 variant {variant!r}; expected one of {tuple(VARIANTS)}")
        self.config = config
        self.provider = provider
        self.tokens = token_counter or get_token_counter()
        self.validator = validator or AnswerValidator()
        self.pipeline = variant
        self._system, self._system_no_refusal, self.prompt_version = VARIANTS[variant]

    def build_prompt(
        self, query: str, retrieved: list[ScoredChunk], *, images: list[ImageInput] | None = None
    ) -> PromptBuildV2:
        system = (self._system if self.config.refuse_without_evidence
                  else self._system_no_refusal).format(refusal=REFUSAL_MARKER)
        pack = build_evidence_pack(retrieved, budget=self.config.max_context_tokens,
                                   tokens=self.tokens)
        body = pack.render()
        user = USER_TEMPLATE_V2.format(sources=body, query=query)
        return PromptBuildV2(
            messages=[
                Message(role="system", content=system),
                Message(role="user", content=user, images=list(images or [])),
            ],
            sources=pack.scored,
            prompt_tokens=self.tokens.count(system) + self.tokens.count(user),
            dropped_for_budget=pack.dropped_for_budget,
            metadata={"n_images": len(images or []), "context_tokens": pack.context_tokens,
                      "page_groups": len(pack.groups)},
            pack=pack,
            sources_block=body,
            pipeline=self.pipeline,
        )

    def validate(self, text: str, prompt: PromptBuild) -> ValidationReport:
        return self.validator.validate(text, prompt.sources)

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
        report = self.validate(completion.text, prompt)

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
                "pipeline": self.pipeline,
                "n_sources": len(prompt.sources),
                "prompt_tokens_estimated": prompt.prompt_tokens,
                "dropped_for_budget": prompt.dropped_for_budget,
                "n_images": len(images or []),
                "unresolved_citations": unresolved,
                "refused": is_refusal(completion.text),
                "answer_chars": len(completion.text),
                "answer_sentences": len(answer_sentences(completion.text)),
                "validation": {k: v for k, v in report.as_dict().items() if k != "sentences"},
                **completion.metadata,
            },
        )
