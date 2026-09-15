"""Load artefacts from disk, check they belong together, and produce a report.

The consistency checks are strict where a mismatch would silently mix runs:

* the retrieval run defaults to the path the generation run recorded; if its
  sha256 no longer matches the recorded one, the report carries a warning;
* a judged run must have been produced from *this* generation run file (by
  sha256), or it is refused -- a failure decomposition over another run's
  answers would be meaningless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmrag.config import PROJECT_ROOT, ExperimentConfig
from mmrag.evaluation.generation_eval import GenerationRun, file_fingerprint
from mmrag.evaluation.judged_eval import JudgedRun
from mmrag.evaluation.report import load_run
from mmrag.logging_utils import get_logger
from mmrag.production.pricing import DEFAULT_PRICING, PricingTable
from mmrag.production.prompt_timing import offline_answerer, time_prompts
from mmrag.production.report import ProductionReport, build_report

log = get_logger(__name__)

DEFAULT_OUT = "data/eval/production"


class ArtifactMismatchError(ValueError):
    """Artefacts passed together do not describe the same run."""


def resolve_path(recorded: str | None) -> Path | None:
    """A recorded path as given, or relative to the repository root."""
    if not recorded:
        return None
    path = Path(recorded)
    if path.exists():
        return path
    if not path.is_absolute() and (PROJECT_ROOT / path).exists():
        return PROJECT_ROOT / path
    return None


def produce(
    generation_path: str | Path,
    *,
    retrieval_path: str | Path | None = None,
    judged_path: str | Path | None = None,
    pricing_path: str | Path = DEFAULT_PRICING,
    repeats: int = 5,
    measure_prompts: bool = True,
    chunks: dict[str, Any] | None = None,
) -> ProductionReport:
    """Build a production report from files. No provider calls, no retrieval."""
    generation = GenerationRun.load(generation_path)
    gen_fp = file_fingerprint(generation_path)
    warnings: list[str] = []

    recorded_retrieval = generation.retrieval_run.get("path")
    rpath = Path(retrieval_path) if retrieval_path else resolve_path(recorded_retrieval)
    if rpath is None or not rpath.exists():
        raise FileNotFoundError(f"retrieval run not found: {retrieval_path or recorded_retrieval}")
    retrieval = load_run(rpath)
    ret_fp = file_fingerprint(rpath)
    expected = generation.retrieval_run.get("sha256")
    if expected and ret_fp["sha256"] != expected:
        warnings.append(f"{rpath} does not match the retrieval run sha256 the generation run "
                        "recorded; retrieval timings may not describe the generated prompts")

    judged = None
    judged_fp = None
    if judged_path is not None:
        judged = JudgedRun.load(judged_path)
        judged_fp = file_fingerprint(judged_path)
        if judged.generation_run.get("sha256") != gen_fp["sha256"]:
            raise ArtifactMismatchError(
                f"{judged_path} was judged from a different generation run "
                f"({judged.generation_run.get('path')}), not {generation_path}"
            )

    pricing = PricingTable.load(pricing_path)

    timings = None
    if measure_prompts:
        config = ExperimentConfig.model_validate(retrieval.config)
        config.generation.pipeline = generation.generation.get("pipeline") or "v1"
        if chunks is None:
            from mmrag.methods import build_method

            chunks = build_method(config).chunks
        answerer = offline_answerer(config.generation)
        timings = time_prompts(list(generation.records), retrieval, chunks, answerer,
                               repeats=repeats)

    inputs = {
        "generation_run": {**gen_fp, "label": generation.label},
        "retrieval_run": {**ret_fp, "label": retrieval.label()},
        "judged_run": {**judged_fp, "label": judged.label} if judged is not None else None,
        "pricing": file_fingerprint(pricing_path),
        "prompt_timing": {"measured": measure_prompts, "repeats": repeats,
                          "statistic": "median of repeats"},
    }
    return build_report(generation, retrieval, pricing, judged=judged, prompt_timings=timings,
                        inputs=inputs, extra_warnings=warnings)
