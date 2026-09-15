"""Latency and distribution summaries for production metrics.

Percentiles here use linear interpolation between closest ranks (NumPy's default
``linear`` method). The quality reports keep their own nearest-rank helpers, so
no recorded number changes meaning; production numbers are computed only here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

# Below this many samples a p99 sits between the two largest observations and is
# not a stable tail estimate. Stated beside the number rather than hidden.
SMALL_N = 100


def percentile(values: Iterable[float], fraction: float) -> float | None:
    """Linearly interpolated percentile; ``fraction`` in [0, 1]. None when empty."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be within [0, 1], got {fraction}")
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def summarize(values: Iterable[float | None], *, digits: int = 1) -> dict[str, Any]:
    """n, mean, p50/p95/p99 and max over the non-None values, with a small-n note."""
    present = [float(v) for v in values if v is not None]
    n = len(present)
    out: dict[str, Any] = {
        "n": n,
        "mean": _round(sum(present) / n, digits) if n else None,
        "p50": _round(percentile(present, 0.50), digits),
        "p95": _round(percentile(present, 0.95), digits),
        "p99": _round(percentile(present, 0.99), digits),
        "max": _round(max(present), digits) if n else None,
    }
    if 0 < n < SMALL_N:
        out["note"] = (
            f"n={n} < {SMALL_N}: p95/p99 are interpolated between the largest few "
            "samples and are not stable tail estimates"
        )
    return out


def rate(count: int, n: int) -> dict[str, Any]:
    """A rate with its count and denominator, never a bare fraction."""
    return {"count": count, "n": n, "rate": round(count / n, 4) if n else None}
