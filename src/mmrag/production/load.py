"""Production load testing: real in-process end-to-end latency under concurrency.

Phase A's ``composed_e2e_latency`` sums components measured in different runs.
This module measures the real thing: the wall-clock time of one call to the
method's own serving path, ``method.answer(query, provider, top_k=...)``, with
several requests in flight at once in one process. No RAG logic is duplicated:
retrieval, fusion, reranking, prompt construction, the provider call and
post-processing all run exactly as they do everywhere else. Only the provider
is chosen here.

What each request records
-------------------------
``e2e_latency_ms``
    Measured wall-clock time around ``method.answer(...)``, from the moment a
    worker starts the request to the moment it returns. It includes metadata
    resolution, routing, retrieval, fusion, reranking, prompt construction, the
    provider call and post-processing. It excludes queue wait, process start-up
    and one-off model loading (absorbed by the untimed warm-up).
``queue_wait_ms``
    Time between submission and a worker starting the request. Reported
    separately, never folded into ``e2e_latency_ms``.
``untimed_overhead_ms``
    A measured **residual**: ``e2e_latency_ms - retrieval_total - generation_ms``.
    It is whatever the pipeline did outside its own timers -- metadata
    resolution, prompt construction, post-processing, answerer construction and
    thread scheduling together. It is NOT a direct measurement of prompt
    construction or post-processing latency.

Modes
-----
``replay`` (default)
    A :class:`ReplayProvider` returns the answer a recorded generation run got
    for the same prompt (matched by sha256), after sleeping for that call's
    recorded provider latency. ``generation_source`` is ``replay``. It reproduces
    recorded per-prompt latency only: provider-side concurrency, queueing and
    rate limiting are NOT measured.
``live``
    The configured provider, unchanged and uncached, making real calls. Refused
    unless explicitly enabled and capped (:func:`check_live_allowed`).
"""

from __future__ import annotations

import gc
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from mmrag.config import ExperimentConfig
from mmrag.evaluation.generation_eval import (
    GenerationRecord,
    GenerationRun,
    _prompt_sha256,
    environment,
    file_fingerprint,
    now_iso,
)
from mmrag.evaluation.report import load_run
from mmrag.generation.providers.base import (
    Completion,
    Message,
    ProviderError,
    Usage,
    redact_secrets,
)
from mmrag.logging_utils import get_logger
from mmrag.production.pricing import PricingTable
from mmrag.production.reliability import SDK_RETRIES_NOTE, classify_error
from mmrag.production.resources import ResourceSampler, environment_resources
from mmrag.production.runner import resolve_path
from mmrag.production.stages import canonical_stages
from mmrag.production.stats import rate, summarize

log = get_logger(__name__)

METHODS = ("method1", "method2", "method3")
DEFAULT_LEVELS = (1, 5, 10)
DEFAULT_LIMIT = 20
DEFAULT_WARMUP = 2
DEFAULT_COOLDOWN_S = 5.0
DEFAULT_SAMPLE_INTERVAL_S = 0.5
DEFAULT_OUT = "data/eval/production/load"

# Same-day, uncached V1 generation runs: every record is a real provider call
# made in that run, so replayed latencies are genuine per-prompt measurements.
_GENERATION_DIR = "data/eval/generation"
DEFAULT_REPLAY_RUNS = {
    "method1": f"{_GENERATION_DIR}/20260915T140529+0000_method1_rerank-v1-nocache_generation.json",
    "method2": f"{_GENERATION_DIR}/20260915T135511+0000_method2_rerank-v1-nocache_generation.json",
    "method3": f"{_GENERATION_DIR}/20260915T141046+0000_method3_rerank-v1-nocache_generation.json",
}

Mode = Literal["replay", "live"]

E2E_NOTE = (
    "e2e_latency_ms is measured in-process wall-clock time around method.answer(...): "
    "metadata resolution, routing, retrieval, fusion, reranking, prompt construction, the "
    "provider call and post-processing. It excludes queue wait (queue_wait_ms, reported "
    "separately), process start-up and one-off model loading (absorbed by the warm-up). "
    "There is no HTTP or serialisation layer."
)
RESIDUAL_NOTE = (
    "untimed_overhead_ms = e2e_latency_ms - retrieval_total - generation_ms is a measured "
    "residual covering everything outside the pipeline's own timers (metadata resolution, "
    "prompt construction, post-processing, answerer construction, thread scheduling). It is "
    "NOT a direct measurement of prompt construction or post-processing latency."
)
REPLAY_NOTE = (
    "generation_source=replay: the provider call is replaced by the recorded answer for the "
    "same prompt, returned after sleeping for that call's recorded latency. Provider-side "
    "concurrency, queueing and rate-limit behaviour are NOT measured."
)
CPU_NOTE = (
    "Runs on this machine as configured: no thread tuning, batching or model locking. On a "
    "CPU-only machine concurrent cross-encoder calls contend for the same cores."
)


class LoadConfigError(ValueError):
    """Invalid load-test settings or replay artefacts."""


class LiveCallsRefusedError(LoadConfigError):
    """Live mode was requested without being explicitly enabled and capped."""


# ---------------------------------------------------------------------------
# Replay provider
# ---------------------------------------------------------------------------


class ReplayProvider:
    """An ``LLMProvider`` that replays a recorded generation run.

    A prompt is matched by the sha256 of its messages, exactly as generation
    evaluation hashed it. If the live pipeline produced a different prompt (for
    example because retrieval drifted), the recorded answer for the query being
    served is used instead and the request is flagged ``query_id_fallback``.
    """

    name = "replay"

    def __init__(self, records: Sequence[GenerationRecord], *,
                 sleep: Callable[[float], None] = time.sleep):
        usable = [r for r in records if r.status == "ok" and r.answer is not None]
        self._by_sha = {r.prompt_sha256: r for r in usable if r.prompt_sha256}
        self._by_query = {r.query_id: r for r in usable}
        self._sleep = sleep
        self._local = threading.local()

    def supports_images(self) -> bool:
        return False

    @contextmanager
    def serving(self, query_id: str) -> Iterator[None]:
        """Bind the query this thread is serving, for fallback matching."""
        self._local.query_id = query_id
        self._local.match = None
        try:
            yield
        finally:
            self._local.query_id = None

    @property
    def last_match(self) -> str | None:
        return getattr(self._local, "match", None)

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        record = self._by_sha.get(_prompt_sha256(messages))
        match = "prompt_sha"
        if record is None:
            record = self._by_query.get(getattr(self._local, "query_id", None) or "")
            match = "query_id_fallback"
        if record is None:
            raise ProviderError("replay: no recorded completion for this prompt or query")
        self._local.match = match
        latency_ms = float(record.latency_ms or 0.0)
        self._sleep(latency_ms / 1000)
        return Completion(
            text=record.answer or "",
            model=record.model or model,
            usage=Usage(prompt_tokens=int(record.usage.get("prompt_tokens", 0)),
                        completion_tokens=int(record.usage.get("completion_tokens", 0))),
            latency_ms=latency_ms,
            metadata={"finish_reason": record.finish_reason, "generation_source": "replay",
                      "replay_match": match},
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadItem:
    query_id: str
    query: str


class LoadRequest(BaseModel):
    request_id: str
    method: str
    phase: Literal["warmup", "level"]
    concurrency: int
    index: int
    query_id: str
    worker: str

    submitted_at: str
    started_at: str
    finished_at: str
    queue_wait_ms: float
    e2e_latency_ms: float

    status: Literal["ok", "error"]
    error: str | None = None
    error_kind: str | None = None

    generation_source: Literal["replay", "live"]
    replay_match: str | None = None

    latency_ms_raw: dict[str, float] = Field(default_factory=dict)
    stages_ms: dict[str, float | None] = Field(default_factory=dict)
    generation_ms: float | None = None
    untimed_overhead_ms: float | None = None
    model_load_ms: float | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    n_citations: int | None = None
    answer_chars: int | None = None


class LoadLevel(BaseModel):
    concurrency: int
    n_requests: int
    n_ok: int
    n_errors: int
    started_at: str
    finished_at: str
    wall_s: float
    throughput_rps: float | None
    error_rate: dict[str, Any]
    errors_by_kind: dict[str, int] = Field(default_factory=dict)
    replay_matches: dict[str, int] = Field(default_factory=dict)
    e2e_latency_ms: dict[str, Any] = Field(default_factory=dict)
    queue_wait_ms: dict[str, Any] = Field(default_factory=dict)
    stages_ms: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    requests: list[LoadRequest] = Field(default_factory=list)


class LoadMethodResult(BaseModel):
    method: str
    label: str
    pipeline: str
    config_name: str
    top_k: int
    inputs: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    workload_query_ids: list[str] = Field(default_factory=list)
    warmup: dict[str, Any] = Field(default_factory=dict)
    levels: list[LoadLevel] = Field(default_factory=list)


class LoadRun(BaseModel):
    kind: Literal["load"] = "load"
    schema_version: int = 1
    created_at: str
    mode: Mode
    generation_source: Literal["replay", "live"]
    settings: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    methods: list[LoadMethodResult] = Field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> LoadRun:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def default_filename(self) -> str:
        stamp = self.created_at.replace(":", "").replace("-", "")
        names = "+".join(m.method for m in self.methods) or "none"
        return f"{stamp}_{names}_{self.mode}_load.json"


# ---------------------------------------------------------------------------
# Executing requests
# ---------------------------------------------------------------------------


def _iso(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Clocks:
    """Injectable time sources: ``perf`` for durations, ``wall`` for timestamps."""

    perf: Callable[[], float] = time.perf_counter
    wall: Callable[[], float] = time.time
    pause: Callable[[float], None] = time.sleep


def execute_request(
    method: Any,
    provider: Any,
    item: LoadItem,
    *,
    method_name: str,
    phase: Literal["warmup", "level"],
    concurrency: int,
    index: int,
    top_k: int,
    generation_source: Literal["replay", "live"],
    submitted_perf: float,
    submitted_wall: float,
    clocks: Clocks,
) -> LoadRequest:
    """Serve one request through ``method.answer`` and record what happened."""
    serving = getattr(provider, "serving", None)
    started_wall = clocks.wall()
    started = clocks.perf()
    answer = None
    error: str | None = None
    try:
        if serving is not None:
            with serving(item.query_id):
                answer = method.answer(item.query, provider, top_k=top_k)
                match = provider.last_match
        else:
            answer = method.answer(item.query, provider, top_k=top_k)
            match = None
    except Exception as exc:  # recorded, never raised: one failure must not stop a level
        error = redact_secrets(f"{type(exc).__name__}: {exc}")
        match = getattr(provider, "last_match", None)
    finished = clocks.perf()
    finished_wall = clocks.wall()

    request = LoadRequest(
        request_id=f"{method_name}-{phase}-c{concurrency}-{index:03d}",
        method=method_name, phase=phase, concurrency=concurrency, index=index,
        query_id=item.query_id, worker=threading.current_thread().name,
        submitted_at=_iso(submitted_wall), started_at=_iso(started_wall),
        finished_at=_iso(finished_wall),
        queue_wait_ms=max(0.0, (started - submitted_perf) * 1000),
        e2e_latency_ms=(finished - started) * 1000,
        status="error" if error else "ok", error=error, error_kind=classify_error(error),
        generation_source=generation_source, replay_match=match,
    )
    if answer is None:
        return request

    raw = dict(answer.latency_ms)
    request.latency_ms_raw = raw
    generation_ms = raw.get("generation_ms")
    stage_keys = {k: v for k, v in raw.items() if k != "generation_ms"}
    if stage_keys:
        stages = canonical_stages(stage_keys)
        request.stages_ms = {k: stages[k] for k in
                             ("retrieval", "fusion", "reranking", "retrieval_total")}
        request.model_load_ms = stages["model_load_ms"]
    request.generation_ms = generation_ms
    retrieval_total = request.stages_ms.get("retrieval_total")
    if generation_ms is not None and retrieval_total is not None:
        request.untimed_overhead_ms = request.e2e_latency_ms - retrieval_total - generation_ms
    usage = answer.usage or {}
    if usage:
        request.prompt_tokens = usage.get("prompt_tokens")
        request.completion_tokens = usage.get("completion_tokens")
        request.total_tokens = usage.get("total_tokens")
    request.n_citations = len(answer.citations)
    request.answer_chars = len(answer.text)
    return request


def summarize_level(
    concurrency: int,
    requests: Sequence[LoadRequest],
    *,
    started_wall: float,
    finished_wall: float,
    wall_s: float,
    resources: dict[str, Any],
) -> LoadLevel:
    """Aggregate one concurrency level. Pure arithmetic over its requests."""
    ok = [r for r in requests if r.status == "ok"]
    errors = [r for r in requests if r.status == "error"]
    stage_names = ("retrieval", "fusion", "reranking", "retrieval_total")
    stages = {name: summarize(r.stages_ms.get(name) for r in ok) for name in stage_names}
    stages["generation"] = summarize(r.generation_ms for r in ok)
    stages["untimed_overhead"] = {**summarize(r.untimed_overhead_ms for r in ok),
                                  "kind": "measured residual"}
    return LoadLevel(
        concurrency=concurrency,
        n_requests=len(requests), n_ok=len(ok), n_errors=len(errors),
        started_at=_iso(started_wall), finished_at=_iso(finished_wall),
        wall_s=round(wall_s, 3),
        throughput_rps=round(len(ok) / wall_s, 4) if wall_s > 0 else None,
        error_rate=rate(len(errors), len(requests)),
        errors_by_kind=dict(sorted(Counter(r.error_kind or "other" for r in errors).items())),
        replay_matches=dict(sorted(Counter(r.replay_match for r in requests
                                           if r.replay_match).items())),
        e2e_latency_ms=summarize(r.e2e_latency_ms for r in ok),
        queue_wait_ms=summarize(r.queue_wait_ms for r in requests),
        stages_ms=stages,
        resources=resources,
        requests=list(requests),
    )


def run_level(
    method: Any,
    provider: Any,
    workload: Sequence[LoadItem],
    concurrency: int,
    *,
    method_name: str,
    top_k: int,
    generation_source: Literal["replay", "live"],
    sampler_factory: Callable[[], Any],
    clocks: Clocks,
) -> LoadLevel:
    """A closed loop: submit the whole workload to ``concurrency`` workers at once."""
    if concurrency < 1:
        raise LoadConfigError("concurrency must be >= 1")
    sampler = sampler_factory().start()
    started_wall = clocks.wall()
    started = clocks.perf()
    with ThreadPoolExecutor(max_workers=concurrency,
                            thread_name_prefix=f"{method_name}-c{concurrency}") as pool:
        futures = []
        for index, item in enumerate(workload):
            futures.append(pool.submit(
                execute_request, method, provider, item, method_name=method_name,
                phase="level", concurrency=concurrency, index=index, top_k=top_k,
                generation_source=generation_source, submitted_perf=clocks.perf(),
                submitted_wall=clocks.wall(), clocks=clocks,
            ))
        requests = [f.result() for f in futures]
    wall_s = clocks.perf() - started
    finished_wall = clocks.wall()
    resources = sampler.stop()
    return summarize_level(concurrency, requests, started_wall=started_wall,
                           finished_wall=finished_wall, wall_s=wall_s, resources=resources)


def run_warmup(
    method: Any,
    provider: Any,
    workload: Sequence[LoadItem],
    n: int,
    *,
    method_name: str,
    top_k: int,
    generation_source: Literal["replay", "live"],
    clocks: Clocks,
) -> dict[str, Any]:
    """Sequential, untimed-for-results requests that load models and build the engine.

    Model loading is lazy and unguarded, so it must happen before any request
    runs concurrently; the warm-up is also where one-off load time lands.
    """
    started = clocks.perf()
    requests = []
    for index, item in enumerate(list(workload)[:n]):
        requests.append(execute_request(
            method, provider, item, method_name=method_name, phase="warmup", concurrency=1,
            index=index, top_k=top_k, generation_source=generation_source,
            submitted_perf=clocks.perf(), submitted_wall=clocks.wall(), clocks=clocks,
        ))
    return {
        "n": len(requests),
        "wall_s": round(clocks.perf() - started, 3),
        "n_errors": sum(1 for r in requests if r.status == "error"),
        "note": "sequential; excluded from every level's statistics",
        "requests": [r.model_dump(mode="json") for r in requests],
    }


# ---------------------------------------------------------------------------
# Sources, gating and orchestration
# ---------------------------------------------------------------------------


@dataclass
class ReplaySource:
    method: str
    generation: GenerationRun
    generation_path: Path
    retrieval_path: Path
    config: ExperimentConfig
    top_k: int
    warnings: list[str]

    def inputs(self) -> dict[str, Any]:
        return {"replay_generation_run": {**file_fingerprint(self.generation_path),
                                          "label": self.generation.label},
                "retrieval_run": file_fingerprint(self.retrieval_path)}


def load_source(method_name: str, generation_path: str | Path) -> ReplaySource:
    """The generation run to replay (or take queries from), its retrieval run and config."""
    gpath = resolve_path(str(generation_path))
    if gpath is None:
        raise LoadConfigError(f"{method_name}: generation run not found: {generation_path}")
    generation = GenerationRun.load(gpath)
    if generation.method != method_name:
        raise LoadConfigError(f"{gpath} is a {generation.method} run, not {method_name}")
    recorded = generation.retrieval_run.get("path")
    rpath = resolve_path(recorded)
    if rpath is None:
        raise LoadConfigError(f"{method_name}: retrieval run not found: {recorded}")
    retrieval = load_run(rpath)
    if retrieval.overrides.get("use_metadata") is False:
        raise LoadConfigError(
            f"{gpath} comes from a no-metadata retrieval arm, which method.answer cannot serve")
    warnings = []
    expected = generation.retrieval_run.get("sha256")
    if expected and file_fingerprint(rpath)["sha256"] != expected:
        warnings.append(f"{rpath} no longer matches the sha256 its generation run recorded")
    config = ExperimentConfig.model_validate(retrieval.config)
    if retrieval.overrides.get("rerank_enabled") is False:
        config.retrieval.rerank_enabled = False
    config.generation.pipeline = generation.generation.get("pipeline") or "v1"
    return ReplaySource(method=method_name, generation=generation, generation_path=gpath,
                        retrieval_path=rpath, config=config,
                        top_k=max(config.evaluation.k_values), warnings=warnings)


def workload_from(generation: GenerationRun, limit: int) -> list[LoadItem]:
    """The first ``limit`` answerable queries with a recorded answer, in gold order.

    Restricting to queries the generation run answered keeps replay complete and,
    for Method 3, keeps every query inside the precomputed query-embedding cache.
    """
    if limit < 1:
        raise LoadConfigError("limit must be >= 1")
    items = [LoadItem(r.query_id, r.query) for r in generation.records
             if r.answerable and r.status == "ok" and r.answer is not None]
    if limit > len(items):
        raise LoadConfigError(f"limit {limit} exceeds the {len(items)} answerable queries "
                              f"recorded in {generation.label}")
    return items[:limit]


def requests_per_method(levels: Sequence[int], limit: int, warmup: int) -> int:
    return min(warmup, limit) + limit * len(levels)


def check_live_allowed(mode: Mode, *, allow_live_calls: bool, max_live_requests: int | None,
                       planned_requests: int) -> None:
    """Refuse live provider calls unless explicitly enabled and capped."""
    if mode != "live":
        return
    if not allow_live_calls:
        raise LiveCallsRefusedError("live mode makes real provider calls; pass "
                                    "--allow-live-calls to enable it")
    if max_live_requests is None:
        raise LiveCallsRefusedError("live mode needs --max-live-requests as a hard cap")
    if planned_requests > max_live_requests:
        raise LiveCallsRefusedError(f"planned {planned_requests} live requests exceed "
                                    f"--max-live-requests {max_live_requests}")


def estimate_live_cost(sources: Sequence[ReplaySource], *, levels: Sequence[int], limit: int,
                       warmup: int, pricing: PricingTable) -> dict[str, Any]:
    """Projected live spend from the recorded mean tokens of each source run."""
    per_method: dict[str, Any] = {}
    total = 0.0
    for src in sources:
        ok = [r for r in src.generation.records if r.status == "ok" and r.usage]
        n = requests_per_method(levels, limit, warmup)
        mean_in = sum(r.usage.get("prompt_tokens", 0) for r in ok) / len(ok) if ok else 0.0
        mean_out = sum(r.usage.get("completion_tokens", 0) for r in ok) / len(ok) if ok else 0.0
        model = src.generation.generation.get("model") or ""
        cost = pricing.cost(model, int(mean_in * n), int(mean_out * n))
        per_method[src.method] = {"requests": n, "mean_prompt_tokens": round(mean_in, 1),
                                  "mean_completion_tokens": round(mean_out, 1),
                                  "cost_usd": round(cost, 6)}
        total += cost
    return {"per_method": per_method, "total_usd": round(total, 6),
            "pricing_as_of": pricing.as_of, "kind": pricing.kind}


def _live_provider(config: ExperimentConfig) -> Any:  # pragma: no cover - real network
    from mmrag.config import get_settings
    from mmrag.generation.providers import get_provider

    return get_provider(name=get_settings().generation_provider,
                        timeout=config.generation.request_timeout_s,
                        max_retries=config.generation.max_retries)


def _default_method_factory(config: ExperimentConfig) -> Any:  # pragma: no cover - needs indexes
    from mmrag.methods import build_method

    return build_method(config)


def run_load(
    method_names: Sequence[str],
    *,
    mode: Mode = "replay",
    replay_runs: dict[str, str] | None = None,
    levels: Sequence[int] = DEFAULT_LEVELS,
    limit: int = DEFAULT_LIMIT,
    warmup: int = DEFAULT_WARMUP,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_S,
    allow_live_calls: bool = False,
    max_live_requests: int | None = None,
    method_factory: Callable[[ExperimentConfig], Any] | None = None,
    provider_factory: Callable[[ReplaySource], Any] | None = None,
    sampler_factory: Callable[[], Any] | None = None,
    replay_sleep: Callable[[float], None] = time.sleep,
    clocks: Clocks | None = None,
    on_event: Callable[[str], None] | None = None,
) -> LoadRun:
    """Warm up, then run every concurrency level for each method in turn."""
    clocks = clocks or Clocks()
    notify = on_event or (lambda message: log.info("%s", message))
    names = list(dict.fromkeys(method_names))
    unknown = [m for m in names if m not in METHODS]
    if not names or unknown:
        raise LoadConfigError(f"methods must be from {METHODS}; got {list(method_names)}")
    levels = list(levels)
    if not levels or any(c < 1 for c in levels) or len(set(levels)) != len(levels):
        raise LoadConfigError(f"concurrency levels must be distinct positive integers: {levels}")
    if warmup < 0 or cooldown_s < 0:
        raise LoadConfigError("warmup and cooldown must be non-negative")
    if mode not in ("replay", "live"):
        raise LoadConfigError(f"mode must be replay or live, got {mode!r}")

    runs = {**DEFAULT_REPLAY_RUNS, **(replay_runs or {})}
    sources = [load_source(name, runs[name]) for name in names]
    workloads = {s.method: workload_from(s.generation, limit) for s in sources}
    check_live_allowed(mode, allow_live_calls=allow_live_calls,
                       max_live_requests=max_live_requests,
                       planned_requests=requests_per_method(levels, limit, warmup) * len(names))

    method_factory = method_factory or _default_method_factory
    sampler_factory = sampler_factory or (lambda: ResourceSampler(sample_interval_s))
    source_label: Literal["replay", "live"] = "replay" if mode == "replay" else "live"
    if provider_factory is None:
        def provider_factory(src: ReplaySource) -> Any:
            if mode == "replay":
                return ReplayProvider(src.generation.records, sleep=replay_sleep)
            return _live_provider(src.config)  # pragma: no cover - real network

    run = LoadRun(
        created_at=now_iso(), mode=mode, generation_source=source_label,
        settings={"methods": names, "levels": levels, "limit": limit, "warmup": warmup,
                  "cooldown_s": cooldown_s, "sample_interval_s": sample_interval_s,
                  "max_live_requests": max_live_requests,
                  "workload": "closed loop: the whole workload is submitted at once to "
                              "`concurrency` worker threads sharing one method instance"},
        environment={**environment(), **environment_resources()},
        notes=[E2E_NOTE, RESIDUAL_NOTE, CPU_NOTE, SDK_RETRIES_NOTE]
        + ([REPLAY_NOTE] if mode == "replay" else []),
    )

    for src in sources:
        notify(f"{src.method}: building method from {src.retrieval_path.name}")
        method = method_factory(src.config)
        provider = provider_factory(src)
        workload = workloads[src.method]
        result = LoadMethodResult(
            method=src.method, label=src.generation.label, pipeline=src.config.generation.pipeline,
            config_name=src.config.name, top_k=src.top_k, inputs=src.inputs(),
            warnings=list(src.warnings), workload_query_ids=[i.query_id for i in workload],
        )
        notify(f"{src.method}: warm-up ({min(warmup, len(workload))} sequential requests)")
        result.warmup = run_warmup(method, provider, workload, warmup, method_name=src.method,
                                   top_k=src.top_k, generation_source=source_label,
                                   clocks=clocks)
        if result.warmup["n_errors"]:
            result.warnings.append(f"{result.warmup['n_errors']} warm-up request(s) failed")
        for position, concurrency in enumerate(levels):
            if position and cooldown_s:
                clocks.pause(cooldown_s)
            notify(f"{src.method}: concurrency {concurrency} ({len(workload)} requests)")
            level = run_level(method, provider, workload, concurrency, method_name=src.method,
                              top_k=src.top_k, generation_source=source_label,
                              sampler_factory=sampler_factory, clocks=clocks)
            result.levels.append(level)
            fallbacks = level.replay_matches.get("query_id_fallback", 0)
            if fallbacks:
                result.warnings.append(
                    f"concurrency {concurrency}: {fallbacks} prompt(s) differed from the recorded "
                    "run and were replayed by query id")
        run.methods.append(result)
        del method, provider
        gc.collect()
    return run


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def load_table(run: LoadRun) -> Table:
    table = Table(title=f"Load test ({run.mode}; generation_source={run.generation_source}) — "
                        "e2e is measured in-process wall-clock")
    cols = ("method", "conc", "n", "ok", "err", "wall s", "rps", "e2e p50", "e2e p95", "e2e p99",
            "queue p50", "rerank p50", "gen p50", "resid p50", "cpu% mean/max", "rss MB max",
            "gpu")
    for col in cols:
        table.add_column(col, justify="left" if col == "method" else "right")
    for m in run.methods:
        for lv in m.levels:
            res = lv.resources
            cpu = (f"{res['process_cpu_percent']['mean']}/{res['process_cpu_percent']['max']}"
                   if res.get("available") else "-")
            rss = str(res["process_rss_mb"]["max"]) if res.get("available") else "-"
            gpu = res.get("gpu", {})
            table.add_row(
                m.method, str(lv.concurrency), str(lv.n_requests), str(lv.n_ok),
                str(lv.n_errors), f"{lv.wall_s:.1f}",
                "-" if lv.throughput_rps is None else f"{lv.throughput_rps:.4f}",
                _ms(lv.e2e_latency_ms["p50"]), _ms(lv.e2e_latency_ms["p95"]),
                _ms(lv.e2e_latency_ms["p99"]), _ms(lv.queue_wait_ms["p50"]),
                _ms(lv.stages_ms["reranking"]["p50"]), _ms(lv.stages_ms["generation"]["p50"]),
                _ms(lv.stages_ms["untimed_overhead"]["p50"]), cpu, rss,
                "n/a" if not gpu.get("available") else str(gpu.get("max_memory_allocated_mb")),
            )
    return table


def render_load(run: LoadRun, console: Console | None = None) -> None:
    console = console or Console()
    console.print(load_table(run))
    for m in run.methods:
        for warning in m.warnings:
            console.print(f"[yellow]{m.method}: {warning}[/]")
    for note in run.notes:
        console.print(f"[dim]{note}[/]")
