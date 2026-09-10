"""OCR enrichment: recovering the text baked into figure images.

A figure's ``best_text()`` is ``caption + description + ocr_text``. Until this
pass existed the last two were always ``None``, so a figure's entire textual
representation was its caption -- and a figure without one was unreachable by
any text query. The audit that motivated this found the corpus full of such
figures that are not visual content at all: register-description tables, ECB
infographic pull-quotes, section-divider banners. Those are *text*, and they
belong to the textified baseline by right.

Where this runs, and why that makes it symmetric
------------------------------------------------
Between parsing and writing the sidecar, so the recovered text is part of the
shared representation rather than of any one method. Method 1 and Method 2 read
the same sidecars and both call ``best_text()``, so neither can be enriched
without the other -- which is what keeps the comparison attributable to
retrieval. Enriching inside a method would have handed one of them an advantage
that has nothing to do with its architecture.

Engines
-------
``rapidocr`` is the default because it installs with ``pip install -e ".[ocr]"``
and nothing else: the ONNX models ship inside the wheel, so a fresh clone can
reproduce an ingestion run without a system package. ``tesseract`` remains
available for anyone who already has the binary. Both satisfy :class:`OcrEngine`,
so adding a third is a new class rather than an edit to the pass.

Failure is expected, not exceptional
------------------------------------
An engine may be absent, a crop may be unreadable, a model may fall over on one
image. None of that should cost a 30-minute ingestion run, so every failure is
counted and stepped over: a missing engine skips the pass entirely, and a single
bad image degrades that one figure. What is *not* silent is the tally -- an
:class:`OcrReport` records how many images were read, skipped, failed and
rejected, so "OCR ran and found nothing" is distinguishable from "OCR never ran".
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mmrag.config import EnrichmentConfig
from mmrag.ingestion.classify import figure_confidence
from mmrag.ingestion.text_utils import collapse_whitespace, normalize_text
from mmrag.logging_utils import get_logger
from mmrag.schemas import Element

log = get_logger(__name__)

# Below this, a "word" is punctuation noise rather than recovered text. OCR on a
# textless plot reliably returns a handful of stray marks, and letting those
# through would make every figure look enriched.
_MIN_ALNUM_RATIO = 0.5


class _Unset:
    """Sentinel: distinguishes "no engine given" from "explicitly no engine"."""


_UNSET = _Unset()


@dataclass(frozen=True)
class OcrResult:
    """What an engine recovered from one image."""

    text: str
    confidence: float
    n_regions: int


class OcrEngine(Protocol):
    """Anything that turns an image path into text.

    ``name`` and ``version`` are recorded per element: an OCR run is only
    reproducible if you know which engine produced it.
    """

    name: str
    version: str

    def read(self, path: Path) -> OcrResult | None: ...


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


class RapidOcrEngine:
    """ONNX-based OCR that installs from PyPI with no system dependency."""

    name = "rapidocr"

    def __init__(self, *, languages: str = "eng"):
        from rapidocr_onnxruntime import RapidOCR

        # Languages are accepted for interface parity with Tesseract; the
        # bundled model is multilingual and takes no language hint.
        self.languages = languages
        self._engine = RapidOCR()
        self.version = self._package_version()

    @staticmethod
    def _package_version() -> str:
        try:
            from importlib.metadata import version

            return version("rapidocr-onnxruntime")
        except Exception:  # pragma: no cover - metadata should always be present
            return "unknown"

    def read(self, path: Path) -> OcrResult | None:
        raw, _elapsed = self._engine(str(path))
        if not raw:
            return None
        # Each entry is (box, text, score).
        texts = [str(item[1]) for item in raw if len(item) >= 2 and item[1]]
        scores = [float(item[2]) for item in raw if len(item) >= 3]
        if not texts:
            return None
        return OcrResult(
            text="\n".join(texts),
            confidence=(sum(scores) / len(scores)) if scores else 0.0,
            n_regions=len(texts),
        )


class TesseractEngine:
    """Tesseract via pytesseract. Needs the binary on PATH."""

    name = "tesseract"

    def __init__(self, *, languages: str = "eng"):
        import pytesseract

        self._pytesseract = pytesseract
        self.languages = languages
        self.version = str(pytesseract.get_tesseract_version())

    def read(self, path: Path) -> OcrResult | None:
        from PIL import Image

        with Image.open(path) as image:
            data = self._pytesseract.image_to_data(
                image,
                lang=self.languages,
                output_type=self._pytesseract.Output.DICT,
            )

        words: list[str] = []
        scores: list[float] = []
        for word, raw_conf in zip(data.get("text", []), data.get("conf", [])):
            if not str(word).strip():
                continue
            try:
                conf = float(raw_conf)
            except (TypeError, ValueError):
                continue
            # Tesseract reports -1 for regions it declined to score.
            if conf < 0:
                continue
            words.append(str(word))
            scores.append(conf / 100.0)

        if not words:
            return None
        return OcrResult(
            text=" ".join(words),
            confidence=(sum(scores) / len(scores)) if scores else 0.0,
            n_regions=len(words),
        )


_ENGINES: dict[str, type] = {
    "rapidocr": RapidOcrEngine,
    "tesseract": TesseractEngine,
}


def get_engine(config: EnrichmentConfig) -> OcrEngine | None:
    """Build the configured engine, or ``None`` if it is unavailable.

    Returning ``None`` rather than raising is deliberate: OCR is an optional
    enrichment, and a machine without the backend installed should still be able
    to ingest the corpus. The caller reports that it was skipped.
    """
    factory = _ENGINES.get(config.ocr_backend)
    if factory is None:  # pragma: no cover - guarded by the config Literal
        log.warning("unknown ocr backend %r; skipping OCR", config.ocr_backend)
        return None
    try:
        engine = factory(languages=config.ocr_languages)
    except ImportError as exc:
        log.warning(
            "OCR backend %r is not installed (%s); skipping OCR. "
            'Install it with: pip install -e ".[ocr]"',
            config.ocr_backend,
            exc,
        )
        return None
    except Exception as exc:
        log.warning("OCR backend %r could not start (%s); skipping OCR", config.ocr_backend, exc)
        return None
    log.info("OCR enabled: %s %s", engine.name, engine.version)
    return engine


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


@dataclass
class OcrReport:
    """What an OCR pass did, so that "found nothing" is not mistaken for "did not run"."""

    ran: bool = False
    backend: str | None = None
    version: str | None = None
    n_candidates: int = 0
    n_read: int = 0
    n_no_text: int = 0
    n_rejected: int = 0
    n_failed: int = 0
    n_missing_image: int = 0
    elapsed_s: float = 0.0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "backend": self.backend,
            "version": self.version,
            "n_candidates": self.n_candidates,
            "n_read": self.n_read,
            "n_no_text": self.n_no_text,
            "n_rejected": self.n_rejected,
            "n_failed": self.n_failed,
            "n_missing_image": self.n_missing_image,
            "elapsed_s": round(self.elapsed_s, 1),
            # Bounded: a systemic failure would otherwise write thousands of
            # identical lines into every sidecar.
            "failures": self.failures[:5],
        }


def clean_ocr_text(raw: str) -> str:
    """Normalise recovered text, collapsing the whitespace OCR invents."""
    return collapse_whitespace(normalize_text(raw))


def is_meaningful(text: str, *, min_chars: int) -> bool:
    """Whether recovered text is content rather than speckle.

    Two gates. Length, because a genuine label is more than a stray glyph; and
    the share of alphanumeric characters, because OCR on a textless plot returns
    punctuation and box-drawing marks that pass a length check on their own.
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    alnum = sum(1 for c in stripped if c.isalnum())
    visible = len(re.sub(r"\s", "", stripped))
    if visible == 0:
        return False
    return (alnum / visible) >= _MIN_ALNUM_RATIO


def enrich_elements_with_ocr(
    elements: list[Element],
    config: EnrichmentConfig,
    *,
    engine: OcrEngine | None | _Unset = _UNSET,
) -> OcrReport:
    """Fill in ``figure.ocr_text`` for visual elements, in place.

    ``engine`` is injectable so the pass can be tested without an OCR backend
    installed -- the tests drive it with a stub rather than shelling out to a
    model.

    Provenance is preserved rather than overwritten: ``extraction_method`` still
    records how the element was *found*, because that is what error analysis
    slices on, and OCR did not find it. What OCR did is recorded beside it, in
    ``figure.ocr_text`` / ``ocr_confidence`` and ``element.metadata["ocr"]``.
    """
    report = OcrReport()
    if not config.ocr_enabled:
        return report

    # A caller that resolved the engine itself passes it explicitly -- including
    # an explicit ``None`` meaning "the backend is unavailable, I already said
    # so". Only an omitted argument triggers a lookup here, so a pipeline that
    # caches one engine across a corpus does not re-resolve (and re-warn) once
    # per document.
    if isinstance(engine, _Unset):
        engine = get_engine(config)
    if engine is None:
        return report

    report.ran = True
    report.backend = engine.name
    report.version = engine.version
    started = time.perf_counter()

    for element in elements:
        if not element.element_type.is_visual or element.figure is None:
            continue
        report.n_candidates += 1

        image_path = element.figure.image_path
        if not image_path or not Path(image_path).exists():
            report.n_missing_image += 1
            continue

        try:
            result = engine.read(Path(image_path))
        except Exception as exc:
            report.n_failed += 1
            report.failures.append(f"{Path(image_path).name}: {type(exc).__name__}: {exc}")
            log.debug("OCR failed on %s: %s", image_path, exc)
            continue

        if result is None or not result.text.strip():
            report.n_no_text += 1
            continue

        text = clean_ocr_text(result.text)
        if not is_meaningful(text, min_chars=config.ocr_min_chars):
            report.n_rejected += 1
            continue

        _apply(element, text, result)
        report.n_read += 1

    report.elapsed_s = time.perf_counter() - started
    log.info(
        "OCR: %d/%d figures enriched (%d no text, %d rejected, %d failed) in %.1fs",
        report.n_read,
        report.n_candidates,
        report.n_no_text,
        report.n_rejected,
        report.n_failed,
        report.elapsed_s,
    )
    return report


def _apply(element: Element, text: str, result: OcrResult) -> None:
    """Attach recovered text to an element and refresh what it implies."""
    figure = element.figure
    assert figure is not None  # guarded by the caller

    figure.ocr_text = text
    figure.ocr_confidence = round(result.confidence, 4)

    element.metadata["ocr"] = {
        "n_regions": result.n_regions,
        "n_chars": len(text),
        "confidence": round(result.confidence, 4),
    }

    # The element is now better understood than when it was detected, and
    # extraction_confidence is documented as comparable across modalities, so it
    # has to reflect that rather than stay frozen at its detection-time value.
    element.extraction_confidence = figure_confidence(
        area_ratio=element.bbox.area if element.bbox else 0.0,
        has_caption=bool(element.caption),
        has_derived_text=figure.has_derived_text,
        is_vector=element.extraction_method.value == "pymupdf_drawing",
    )
