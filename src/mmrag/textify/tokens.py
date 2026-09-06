"""Token counting for chunk sizing.

Chunk boundaries are defined in tokens, so the counter has to agree with the
model that will eventually embed the text -- otherwise a chunk sized to 384
"tokens" can silently exceed the embedder's 512-token window and get truncated,
losing the tail of every long chunk with no error anywhere.

Two implementations behind one interface:

* :class:`HFTokenCounter` -- the embedding model's own tokenizer. Exact, and the
  default, because "does this chunk fit the encoder" is the question that matters.
* :class:`HeuristicTokenCounter` -- a dependency-free approximation, used when the
  tokenizer cannot be loaded (no network on a fresh clone, say). Deterministic,
  so chunking stays reproducible; just less precise.

Which one ran is recorded on every chunk, because a corpus chunked with one and
compared against a corpus chunked with the other is not a controlled experiment.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol

from mmrag.logging_utils import get_logger

log = get_logger(__name__)

# Words, numbers, and standalone punctuation. Subword tokenizers split on
# roughly these boundaries, then further inside long or rare words.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

# Empirical ratio of BPE tokens to whitespace-ish tokens for English prose with
# the bge/BERT vocabulary. Numbers, code, and Markdown tables run higher, which
# is why this is the fallback rather than the default.
_SUBWORD_FACTOR = 1.25


class TokenCounter(Protocol):
    """Anything that can count tokens in a string."""

    name: str

    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Approximate counter with no model dependency."""

    name = "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return int(len(_TOKEN_PATTERN.findall(text)) * _SUBWORD_FACTOR)


class HFTokenCounter:
    """Exact counter using the embedding model's tokenizer."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.name = f"hf:{model_name}"
        # add_special_tokens=False when counting: the [CLS]/[SEP] pair is added
        # by the encoder itself and must not be charged against chunk content.
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))


@lru_cache(maxsize=4)
def get_token_counter(model_name: str | None = None) -> TokenCounter:
    """Best available counter for ``model_name``, cached per model.

    Falls back to the heuristic rather than failing: a machine without network
    access should still be able to chunk and evaluate a corpus.
    """
    if model_name:
        try:
            return HFTokenCounter(model_name)
        except Exception as exc:
            log.warning(
                "could not load tokenizer for %s (%s); falling back to the heuristic "
                "counter -- chunk sizes will be approximate",
                model_name,
                exc,
            )
    return HeuristicTokenCounter()


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

# Split after . ! or ? when followed by whitespace and the start of something
# that looks like a new sentence. Python's re only allows fixed-width lookbehind,
# so abbreviations cannot be excluded in the pattern itself; they are repaired by
# a merge pass below.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

# Tokens that end with a period without ending a sentence. Splitting after these
# fragments prose mid-clause and produces chunks that begin with a dangling
# phrase -- bad for the embedder and worse for a human reading a citation.
_ABBREVIATIONS = frozenset(
    {
        "e.g.",
        "i.e.",
        "cf.",
        "vs.",
        "etc.",
        "al.",
        "fig.",
        "eq.",
        "no.",
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "st.",
        "approx.",
        "vol.",
        "pp.",
        "ref.",
        "sec.",
    }
)


def _ends_with_abbreviation(text: str) -> bool:
    tail = text.rstrip().rsplit(None, 1)[-1].lower() if text.strip() else ""
    if tail in _ABBREVIATIONS:
        return True
    # A single initial, as in "J. Smith" or an enumerated list item "1."
    return len(tail) == 2 and tail.endswith(".") and tail[0].isalnum()


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping their trailing punctuation.

    Deliberately simple and dependency-free. Chunk boundaries only need to be
    *reasonable* -- a sentence split in the wrong place costs a little retrieval
    quality, it does not corrupt provenance, because the chunk still records the
    element it came from.
    """
    if not text or not text.strip():
        return []

    out: list[str] = []
    # Paragraph breaks are stronger boundaries than sentence ends, so split on
    # them first and never merge across one.
    for paragraph in text.split("\n\n"):
        stripped = paragraph.strip()
        if not stripped:
            continue
        merged: list[str] = []
        for fragment in (f.strip() for f in _SENTENCE_SPLIT.split(stripped)):
            if not fragment:
                continue
            if merged and _ends_with_abbreviation(merged[-1]):
                merged[-1] = f"{merged[-1]} {fragment}"
            else:
                merged.append(fragment)
        out.extend(merged)
    return out
