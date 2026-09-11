"""Run a method over the gold set and score its retrieval.

Retrieval only. No provider is called, nothing is generated, and the whole
thing runs offline -- which is what makes it free to re-run after every change
and impossible to confound with a vendor's behaviour.

The runner talks to methods through the two attributes they already share,
``.retrieve(query, top_k=...)`` and a result carrying ``.results`` and
``.latency_ms``. Method 2 additionally accepts ``use_metadata``; that is
detected rather than assumed, so neither method needed changing to be
evaluated.
"""

from __future__ import annotations

import inspect
import platform
import time
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field

from mmrag.config import ExperimentConfig
from mmrag.evaluation.gold import GoldQuery, GoldSet, describe
from mmrag.evaluation.metrics import aggregate, summarize
from mmrag.logging_utils import get_logger
from mmrag.schemas import Chunk

log = get_logger(__name__)


class Method(Protocol):
    """What the evaluator needs from a method. Both already satisfy it."""

    name: str

    def retrieve(self, query: str, *, top_k: int | None = None) -> Any: ...


class RetrievedChunk(BaseModel):
    """One hit, kept compact so a run file stays readable."""

    rank: int
    chunk_id: str
    doc_id: str
    page: int
    chunk_type: str
    score: float
    retriever: str
    # Index of the gold evidence entry this satisfies, or None.
    matched: int | None = None


class QueryResult(BaseModel):
    """One gold query's outcome."""

    query_id: str
    query: str
    stratum: str
    requires: str
    n_gold: int
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    # Method 2 only: what the router decided.
    routing: dict[str, Any] | None = None

    @property
    def matched(self) -> list[int | None]:
        return [hit.matched for hit in self.retrieved]


class RetrievalRun(BaseModel):
    """A complete, reproducible record of one evaluation run.

    Carries the full experiment config and the gold-set description, so a run
    file answers "what produced this number" without reference to the working
    tree it came from.
    """

    method: str
    config_name: str
    tag: str | None = None
    started_at: str
    elapsed_s: float
    # Deviations from the config file, applied at run time so both arms of an
    # ablation come from one committed config.
    overrides: dict[str, Any] = Field(default_factory=dict)
    gold: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    per_query: list[QueryResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    by_stratum: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_requires: dict[str, dict[str, float]] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)

    def label(self) -> str:
        return f"{self.method}{f'/{self.tag}' if self.tag else ''}"


def _accepts(method: Method, parameter: str) -> bool:
    try:
        return parameter in inspect.signature(method.retrieve).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
        return False


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def evaluate_query(
    method: Method,
    query: GoldQuery,
    *,
    top_k: int,
    k_values: list[int],
    use_metadata: bool | None = None,
) -> QueryResult:
    """Retrieve for one gold query and score the result."""
    kwargs: dict[str, Any] = {"top_k": top_k}
    if use_metadata is not None and _accepts(method, "use_metadata"):
        kwargs["use_metadata"] = use_metadata

    result = method.retrieve(query.query, **kwargs)
    chunks: list[Chunk] = [hit.chunk for hit in result.results]
    matched = query.matched(chunks)

    retrieved = [
        RetrievedChunk(
            rank=hit.rank,
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            page=hit.chunk.page_number,
            chunk_type=hit.chunk.chunk_type.value,
            score=hit.score,
            retriever=hit.retriever,
            matched=match,
        )
        for hit, match in zip(result.results, matched, strict=True)
    ]

    routing = getattr(result, "routing", None)
    return QueryResult(
        query_id=query.id,
        query=query.query,
        stratum=query.stratum,
        requires=query.requires,
        n_gold=len(query.evidence),
        retrieved=retrieved,
        metrics=summarize(matched, len(query.evidence), k_values),
        latency_ms=dict(result.latency_ms),
        routing=routing.as_dict() if routing is not None else None,
    )


def run_evaluation(
    method: Method,
    gold: GoldSet,
    config: ExperimentConfig,
    *,
    config_name: str,
    tag: str | None = None,
    use_metadata: bool | None = None,
    overrides: dict[str, Any] | None = None,
    on_query: Any = None,
) -> RetrievalRun:
    """Score one method over the whole gold set.

    ``top_k`` is the largest k being measured: a metric at k=10 needs ten
    results, and retrieving more would only inflate latency.
    """
    k_values = sorted(config.evaluation.k_values)
    top_k = max(k_values)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    per_query: list[QueryResult] = []
    for index, query in enumerate(gold.queries, start=1):
        outcome = evaluate_query(
            method, query, top_k=top_k, k_values=k_values, use_metadata=use_metadata
        )
        per_query.append(outcome)
        if on_query is not None:
            on_query(index, len(gold.queries), outcome)

    run = RetrievalRun(
        method=getattr(method, "name", "unknown"),
        config_name=config_name,
        tag=tag,
        started_at=started_at,
        elapsed_s=round(time.perf_counter() - started, 1),
        overrides=overrides or {},
        gold=describe(gold),
        config=config.model_dump(mode="json"),
        environment={"python": platform.python_version(), "platform": platform.platform()},
        per_query=per_query,
        metrics=aggregate([q.metrics for q in per_query]),
    )
    run.by_stratum = _slice(per_query, "stratum")
    run.by_requires = _slice(per_query, "requires")
    run.latency = _latency(per_query)
    return run


def _slice(results: list[QueryResult], field: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[QueryResult]] = {}
    for result in results:
        groups.setdefault(getattr(result, field), []).append(result)
    return {
        key: {**aggregate([r.metrics for r in group]), "n": float(len(group))}
        for key, group in sorted(groups.items())
    }


def _latency(results: list[QueryResult]) -> dict[str, float]:
    """Median and p90 per stage.

    Reported per stage because reranking dominates by two orders of magnitude
    on CPU, and a single total would hide which part of the pipeline a change
    actually moved.
    """
    stages: dict[str, list[float]] = {}
    for result in results:
        for stage, value in result.latency_ms.items():
            stages.setdefault(stage, []).append(value)

    out: dict[str, float] = {}
    for stage, values in sorted(stages.items()):
        out[f"{stage}_median"] = round(_percentile(values, 0.5), 1)
        out[f"{stage}_p90"] = round(_percentile(values, 0.9), 1)
    return out
