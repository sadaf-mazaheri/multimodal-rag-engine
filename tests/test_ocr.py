"""OCR enrichment.

Driven by a stub engine rather than a real backend. The pass's job is to decide
*which* elements get read, *whether* the result is worth keeping, and *what* it
records -- none of which depends on a particular OCR model, and all of which
would become untestable on a machine without one installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmrag.config import EnrichmentConfig
from mmrag.ingestion.ocr import (
    OcrReport,
    OcrResult,
    clean_ocr_text,
    enrich_elements_with_ocr,
    get_engine,
    is_meaningful,
)
from mmrag.schemas import (
    BBox,
    Element,
    ElementType,
    ExtractionMethod,
    FigureData,
    make_element_id,
)


class StubEngine:
    """An OCR engine with scripted answers, keyed by image filename."""

    name = "stub"
    version = "1.0"

    def __init__(self, answers: dict[str, OcrResult | None] | None = None, *, raises: bool = False):
        self.answers = answers or {}
        self.raises = raises
        self.seen: list[str] = []

    def read(self, path: Path) -> OcrResult | None:
        self.seen.append(path.name)
        if self.raises:
            raise RuntimeError("engine exploded")
        return self.answers.get(path.name, OcrResult("Bits Description Type Reset", 0.95, 4))


@pytest.fixture
def enabled() -> EnrichmentConfig:
    return EnrichmentConfig(ocr_enabled=True, ocr_backend="rapidocr", ocr_min_chars=8)


def _image(tmp_path: Path, name: str = "fig.png") -> Path:
    """A real file on disk -- the pass checks existence before reading."""
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (40, 20), "white").save(path)
    return path


def _figure_element(image_path: Path | None, *, caption: str | None = None) -> Element:
    return Element(
        element_id=make_element_id("doc", 1, "chart", 0),
        doc_id="doc",
        page_id="doc#p0001",
        page_number=1,
        element_type=ElementType.CHART,
        bbox=BBox(x0=0.1, y0=0.1, x1=0.6, y1=0.5),
        caption=caption,
        extraction_method=ExtractionMethod.PYMUPDF_DRAWING,
        extraction_confidence=0.5,
        figure=FigureData(image_path=str(image_path) if image_path else None),
    )


def _text_element() -> Element:
    return Element(
        element_id=make_element_id("doc", 1, "text", 1),
        doc_id="doc",
        page_id="doc#p0001",
        page_number=1,
        element_type=ElementType.TEXT,
        text="ordinary prose",
        extraction_method=ExtractionMethod.PYMUPDF_TEXT,
        extraction_confidence=0.9,
    )


class TestTextCleaning:
    def test_collapses_the_whitespace_ocr_invents(self):
        assert clean_ocr_text("Bits   \n\n  Description") == "Bits Description"

    @pytest.mark.parametrize("text", ["Bits Description Type", "0x13 ROM 31:0 RW"])
    def test_real_labels_are_meaningful(self, text):
        assert is_meaningful(text, min_chars=8)

    @pytest.mark.parametrize("text", ["", "  ", "abc", "|-|"])
    def test_speckle_and_short_strings_are_not(self, text):
        assert not is_meaningful(text, min_chars=8)

    def test_punctuation_heavy_output_is_rejected_despite_being_long(self):
        """OCR on a textless plot returns axis marks that pass a length check."""
        assert not is_meaningful("--|--|--|--|--+--+", min_chars=8)

    def test_min_chars_is_honoured(self):
        assert is_meaningful("abcdefgh", min_chars=8)
        assert not is_meaningful("abcdefgh", min_chars=9)


class TestPassSelection:
    def test_disabled_config_does_nothing(self, tmp_path):
        element = _figure_element(_image(tmp_path))
        engine = StubEngine()
        report = enrich_elements_with_ocr(
            [element], EnrichmentConfig(ocr_enabled=False), engine=engine
        )
        assert not report.ran
        assert engine.seen == []
        assert element.figure.ocr_text is None

    def test_only_visual_elements_are_read(self, enabled, tmp_path):
        figure = _figure_element(_image(tmp_path))
        engine = StubEngine()
        report = enrich_elements_with_ocr([figure, _text_element()], enabled, engine=engine)
        assert report.n_candidates == 1
        assert len(engine.seen) == 1

    def test_a_figure_with_no_image_on_disk_is_counted_not_read(self, enabled, tmp_path):
        element = _figure_element(tmp_path / "absent.png")
        engine = StubEngine()
        report = enrich_elements_with_ocr([element], enabled, engine=engine)
        assert report.n_missing_image == 1
        assert engine.seen == []

    def test_a_figure_with_no_image_path_at_all_is_skipped(self, enabled):
        report = enrich_elements_with_ocr([_figure_element(None)], enabled, engine=StubEngine())
        assert report.n_missing_image == 1

    def test_captioned_figures_are_still_read(self, enabled, tmp_path):
        """A caption does not mean the axis labels are already captured."""
        element = _figure_element(_image(tmp_path), caption="Figure 3: Throughput.")
        enrich_elements_with_ocr([element], enabled, engine=StubEngine())
        assert element.figure.ocr_text is not None


class TestEnrichment:
    def test_recovered_text_lands_on_the_figure(self, enabled, tmp_path):
        element = _figure_element(_image(tmp_path))
        report = enrich_elements_with_ocr([element], enabled, engine=StubEngine())
        assert element.figure.ocr_text == "Bits Description Type Reset"
        assert element.figure.ocr_confidence == pytest.approx(0.95)
        assert report.n_read == 1

    def test_ocr_text_reaches_best_text(self, enabled, tmp_path):
        """The whole point: a captionless figure becomes retrievable."""
        element = _figure_element(_image(tmp_path))
        assert element.best_text() == ""
        enrich_elements_with_ocr([element], enabled, engine=StubEngine())
        assert "Bits" in element.best_text()

    def test_provenance_records_what_ocr_did(self, enabled, tmp_path):
        element = _figure_element(_image(tmp_path))
        enrich_elements_with_ocr([element], enabled, engine=StubEngine())
        assert element.metadata["ocr"]["n_regions"] == 4
        assert element.metadata["ocr"]["n_chars"] == len("Bits Description Type Reset")

    def test_extraction_method_is_not_overwritten(self, enabled, tmp_path):
        """It records how the element was *found*; OCR did not find it.

        Error analysis slices on this field, so repurposing it would destroy the
        ability to ask which extractor bad retrievals concentrate in.
        """
        element = _figure_element(_image(tmp_path))
        enrich_elements_with_ocr([element], enabled, engine=StubEngine())
        assert element.extraction_method is ExtractionMethod.PYMUPDF_DRAWING

    def test_extraction_confidence_rises_once_text_is_recovered(self, enabled, tmp_path):
        element = _figure_element(_image(tmp_path))
        before = element.extraction_confidence
        enrich_elements_with_ocr([element], enabled, engine=StubEngine())
        assert element.extraction_confidence > before

    def test_report_names_the_engine_for_reproducibility(self, enabled, tmp_path):
        report = enrich_elements_with_ocr(
            [_figure_element(_image(tmp_path))], enabled, engine=StubEngine()
        )
        assert (report.backend, report.version) == ("stub", "1.0")


class TestRejection:
    def test_empty_result_is_counted_as_no_text(self, enabled, tmp_path):
        image = _image(tmp_path)
        element = _figure_element(image)
        engine = StubEngine({image.name: None})
        report = enrich_elements_with_ocr([element], enabled, engine=engine)
        assert (report.n_no_text, report.n_read) == (1, 0)
        assert element.figure.ocr_text is None

    def test_speckle_is_rejected_rather_than_stored(self, enabled, tmp_path):
        image = _image(tmp_path)
        element = _figure_element(image)
        engine = StubEngine({image.name: OcrResult("|- .", 0.3, 2)})
        report = enrich_elements_with_ocr([element], enabled, engine=engine)
        assert (report.n_rejected, report.n_read) == (1, 0)
        assert element.figure.ocr_text is None

    def test_a_textless_figure_stays_textless(self, enabled, tmp_path):
        """Method 2's headline claim depends on this staying honest."""
        image = _image(tmp_path)
        element = _figure_element(image)
        engine = StubEngine({image.name: None})
        enrich_elements_with_ocr([element], enabled, engine=engine)
        assert element.best_text() == ""


class TestFailureHandling:
    def test_one_bad_image_does_not_stop_the_pass(self, enabled, tmp_path):
        good = _figure_element(_image(tmp_path, "good.png"))
        bad = _figure_element(_image(tmp_path, "bad.png"))
        bad.element_id = make_element_id("doc", 1, "chart", 9)

        class Flaky(StubEngine):
            def read(self, path: Path) -> OcrResult | None:
                if path.name == "bad.png":
                    raise RuntimeError("decode error")
                return OcrResult("Recovered label text", 0.9, 3)

        report = enrich_elements_with_ocr([bad, good], enabled, engine=Flaky())
        assert report.n_failed == 1
        assert report.n_read == 1
        assert good.figure.ocr_text == "Recovered label text"

    def test_failures_are_recorded_but_bounded(self, enabled, tmp_path):
        elements = []
        for i in range(8):
            element = _figure_element(_image(tmp_path, f"f{i}.png"))
            element.element_id = make_element_id("doc", 1, "chart", i)
            elements.append(element)
        report = enrich_elements_with_ocr(elements, enabled, engine=StubEngine(raises=True))
        assert report.n_failed == 8
        assert len(report.as_dict()["failures"]) == 5

    def test_a_missing_backend_skips_the_pass_rather_than_raising(self):
        """A machine without the OCR extra must still be able to ingest."""
        config = EnrichmentConfig(ocr_enabled=True, ocr_backend="tesseract")
        report = OcrReport()
        try:
            report = enrich_elements_with_ocr([], config)
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"missing backend should not raise, got {exc!r}")
        assert isinstance(report, OcrReport)

    def test_get_engine_returns_none_for_an_uninstalled_backend(self):
        engine = get_engine(EnrichmentConfig(ocr_enabled=True, ocr_backend="tesseract"))
        assert engine is None or hasattr(engine, "read")


class TestSymmetry:
    def test_both_methods_configure_ocr_identically(self):
        """Enrichment is part of the shared representation, not a method's edge.

        Method 1 and Method 2 read the same sidecars, so a difference here would
        not be a retrieval difference -- it would mean the two methods were
        measured on different corpora.
        """
        from mmrag.config import load_experiment_config

        one = load_experiment_config("method1").enrichment
        two = load_experiment_config("method2").enrichment
        assert one.ocr_enabled == two.ocr_enabled
        assert one.ocr_backend == two.ocr_backend
        assert one.ocr_languages == two.ocr_languages
        assert one.ocr_min_chars == two.ocr_min_chars

    def test_configs_do_not_claim_passes_that_do_not_exist(self):
        """A run manifest records the config verbatim, so a true flag for an
        unimplemented pass would make every index assert an enrichment that
        never ran."""
        from mmrag.config import load_experiment_config

        for name in ("method1", "method2"):
            assert load_experiment_config(name).enrichment.vlm_captions_enabled is False
