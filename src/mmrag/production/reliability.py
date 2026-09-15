"""Error, retry and timeout rates from generation records.

What an offline artefact can and cannot show:

* ``status == "error"`` and the redacted error string are recorded per query;
* ``attempts`` counts the evaluation runner's own bounded retry loop;
* retries performed *inside* the provider SDK (connection errors, 429, 5xx) are
  **not observable** from offline artefacts: they surface only as extra latency.

The error string keeps the original exception's type name (the provider wraps it
as ``"<Type>: <message>"``), which is what classification keys on.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from mmrag.evaluation.generation_eval import GenerationRecord
from mmrag.production.stats import rate

_TIMEOUT = re.compile(r"\b(?:APITimeoutError|\w*Timeout\w*)\b|timed out", re.IGNORECASE)
_RATE_LIMIT = re.compile(r"\bRateLimitError\b|\b429\b", re.IGNORECASE)
_CONNECTION = re.compile(r"\b(?:APIConnectionError|ConnectError|ConnectionError)\b")

SDK_RETRIES_NOTE = (
    "Retries inside the provider SDK (connection errors, 429, 5xx) are not observable "
    "from offline artifacts; they appear only as added latency. retry_rate counts the "
    "evaluation runner's own attempts."
)


def classify_error(error: str | None) -> str | None:
    """``timeout`` | ``rate_limit`` | ``connection`` | ``other``, or None for no error."""
    if not error:
        return None
    if _TIMEOUT.search(error):
        return "timeout"
    if _RATE_LIMIT.search(error):
        return "rate_limit"
    if _CONNECTION.search(error):
        return "connection"
    return "other"


def reliability(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    """Error, retry and timeout rates over records that attempted a provider call."""
    attempted = [r for r in records if r.status in ("ok", "error")]
    n = len(attempted)
    errors = [r for r in attempted if r.status == "error"]
    kinds = Counter(classify_error(r.error) or "other" for r in errors)
    return {
        "n": n,
        "error_rate": rate(len(errors), n),
        "retry_rate": rate(sum(1 for r in attempted if r.attempts > 1), n),
        "timeout_rate": rate(kinds.get("timeout", 0), n),
        "errors_by_kind": dict(sorted(kinds.items())),
        "error_query_ids": [r.query_id for r in errors],
        "sdk_internal_retries_observable": False,
        "note": SDK_RETRIES_NOTE,
    }
