"""Text normalisation for extracted PDF content.

PDF text extraction produces characteristic damage: ligatures as single
codepoints, words hyphenated across line breaks, non-breaking spaces, and
unmappable glyphs that arrive as U+FFFD. Left alone, all of these hurt both BM25
(token mismatch) and dense retrieval (subword fragmentation).

Normalisation is deliberately conservative. Anything that cannot be repaired
reliably is *measured* instead -- see :func:`text_quality` -- so a badly
extracted document shows up as a low-confidence element rather than as silently
corrupted evidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Typographic characters that hurt retrieval. Written as explicit codepoints
# because most of them are invisible or indistinguishable in a source editor --
# which is exactly why they cause silent token mismatches in the first place.
#
# NFKC already folds the ligatures and most of the spaces; they are listed anyway
# so the mapping does not depend on Python's Unicode version. Applying a fix
# twice is a no-op, so the overlap is harmless.
_CHAR_FIXES: dict[str, str] = {
    # -- ligatures: one codepoint that must become several tokens -----------
    chr(0xFB00): "ff",  # LATIN SMALL LIGATURE FF
    chr(0xFB01): "fi",  # LATIN SMALL LIGATURE FI
    chr(0xFB02): "fl",  # LATIN SMALL LIGATURE FL
    chr(0xFB03): "ffi",  # LATIN SMALL LIGATURE FFI
    chr(0xFB04): "ffl",  # LATIN SMALL LIGATURE FFL
    chr(0xFB05): "st",  # LATIN SMALL LIGATURE LONG S T
    chr(0xFB06): "st",  # LATIN SMALL LIGATURE ST
    # -- quotation marks. NFKC leaves these alone, and a curly apostrophe makes
    #    a possessive tokenise differently from a straight one, so BM25 misses.
    chr(0x2018): "'",  # LEFT SINGLE QUOTATION MARK
    chr(0x2019): "'",  # RIGHT SINGLE QUOTATION MARK (also used as apostrophe)
    chr(0x201A): "'",  # SINGLE LOW-9 QUOTATION MARK
    chr(0x201C): '"',  # LEFT DOUBLE QUOTATION MARK
    chr(0x201D): '"',  # RIGHT DOUBLE QUOTATION MARK
    chr(0x201E): '"',  # DOUBLE LOW-9 QUOTATION MARK
    # -- dashes ---------------------------------------------------------------
    chr(0x2013): "-",  # EN DASH
    chr(0x2014): "-",  # EM DASH
    chr(0x2212): "-",  # MINUS SIGN
    # -- spaces that are not U+0020 -------------------------------------------
    chr(0x00A0): " ",  # NO-BREAK SPACE
    chr(0x2007): " ",  # FIGURE SPACE
    chr(0x202F): " ",  # NARROW NO-BREAK SPACE
    # -- zero-width and formatting characters, deleted outright ---------------
    chr(0x200B): "",  # ZERO WIDTH SPACE
    chr(0x200C): "",  # ZERO WIDTH NON-JOINER
    chr(0x200D): "",  # ZERO WIDTH JOINER
    chr(0xFEFF): "",  # ZERO WIDTH NO-BREAK SPACE (byte-order mark)
    chr(0x00AD): "",  # SOFT HYPHEN
}

_TRANSLATION = str.maketrans(_CHAR_FIXES)

# "hyphen-\nated" -> "hyphenated". Requires a lowercase letter either side so
# that genuine compounds ("state-\nof-the-art") and numbers are left alone.
_HYPHEN_BREAK = re.compile(r"([a-z])-\s*\n\s*([a-z])")

_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")

# C0 control characters and DEL, excluding tab and newline which are real
# layout. PDFs with a broken font map emit these as the extraction of an
# unmapped glyph -- U+0000 and U+0001 both appear in the corpus.
#
# Stripping them is not cosmetic: a PostgreSQL `text` column cannot hold a NUL
# byte at all, so a single one anywhere in a document fails the entire write
# with "PostgreSQL text fields cannot contain NUL (0x00) bytes". Doing it here
# rather than in the store keeps every consumer safe -- Qdrant payloads and the
# JSON sidecars included -- instead of fixing one backend at a time.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Normalise line endings before anything else measures or joins lines.
_CRLF = re.compile(r"\r\n?")

REPLACEMENT_CHAR = chr(0xFFFD)  # U+FFFD REPLACEMENT CHARACTER: an unmappable glyph


def normalize_text(text: str | None, *, join_hyphens: bool = True) -> str:
    """Clean extracted text without changing what it says.

    Applied to every text element at parse time. Idempotent: running it twice
    produces the same result as running it once, which matters because element
    ids are content-derived downstream.
    """
    if not text:
        return ""

    # NFKC first: folds full-width forms and many compatibility characters.
    out = unicodedata.normalize("NFKC", text)
    out = out.translate(_TRANSLATION)

    # Before any line-based work, so \r\n does not survive as a stray \r and
    # then get stripped as a control character, silently joining two lines.
    out = _CRLF.sub("\n", out)
    out = _CONTROL_CHARS.sub("", out)

    if join_hyphens:
        out = _HYPHEN_BREAK.sub(r"\1\2", out)

    out = _TRAILING_WS.sub("\n", out)
    out = _MULTI_SPACE.sub(" ", out)
    out = _MULTI_NEWLINE.sub("\n\n", out)
    return out.strip()


@dataclass(frozen=True)
class TextQuality:
    """Measured extraction damage for one piece of text."""

    n_chars: int
    n_replacement: int
    n_alpha: int

    @property
    def replacement_ratio(self) -> float:
        """Fraction of characters the PDF's font map could not resolve."""
        return self.n_replacement / self.n_chars if self.n_chars else 0.0

    @property
    def alpha_ratio(self) -> float:
        """Fraction of alphabetic characters.

        Very low values indicate either a numeric table dumped as text or a
        failed extraction that produced punctuation soup.
        """
        return self.n_alpha / self.n_chars if self.n_chars else 0.0

    @property
    def is_usable(self) -> bool:
        return self.n_chars > 0 and self.replacement_ratio < 0.1


def text_quality(text: str) -> TextQuality:
    return TextQuality(
        n_chars=len(text),
        n_replacement=text.count(REPLACEMENT_CHAR),
        n_alpha=sum(1 for c in text if c.isalpha()),
    )


def collapse_whitespace(text: str) -> str:
    """Single-line form, for comparing header/footer candidates across pages."""
    return " ".join(text.split())


_DIGITS = re.compile(r"\d+")


def mask_digits(text: str, placeholder: str = "#") -> str:
    """Replace digit runs so that 'Page 3' and 'Page 4' compare equal.

    Header/footer detection depends on this: the whole point of a running
    footer is that it is identical on every page *except* for the page number.
    """
    return _DIGITS.sub(placeholder, text)


def is_probably_numeric(text: str, *, threshold: float = 0.4) -> bool:
    """Whether a string is mostly numeric, e.g. a table cell or an axis label."""
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return False
    numeric = sum(1 for c in stripped if c.isdigit() or c in ".,%$()-+")
    return numeric / len(stripped) >= threshold
