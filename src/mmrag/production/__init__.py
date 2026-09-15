"""Production evaluation: latency, tokens, cost, reliability and failure decomposition.

Kept apart from retrieval and generation *quality* evaluation on purpose. Nothing
here changes how a method retrieves or answers, and nothing here calls a model
provider or re-runs retrieval: every number is read from existing run artefacts
or measured by re-executing a deterministic, local step (prompt construction and
answer post-processing) whose output is checked byte for byte against the run.

Every latency component carries a provenance label:

* ``recorded`` -- copied from a saved artefact, measured when that run happened;
* ``measured`` -- measured in this process by re-executing a local step;
* ``composed`` -- a sum of components measured separately.

``composed_e2e_latency`` is composed. It was **not** measured in a single serving
process: retrieval and generation were recorded in different runs, possibly on
different days, and prompt construction and post-processing are re-measured
here. A true end-to-end measurement is the job of the (separate, later) load test.
"""

from mmrag.production.report import (
    ProductionReport,
    build_report,
    compare_table,
    render_report,
)

__all__ = ["ProductionReport", "build_report", "compare_table", "render_report"]
