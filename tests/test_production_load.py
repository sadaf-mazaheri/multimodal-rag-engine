"""Phase B load testing, offline: fake method, provider, sampler and clocks.

No network, no models, no services. Real threads are used where concurrency
itself is under test; everything else runs on injected clocks and sleeps.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import mmrag.cli as cli
import mmrag.production.load as load
from mmrag.evaluation.generation_eval import (
    GenerationRecord,
    GenerationRun,
    _prompt_sha256,
    file_fingerprint,
)
from mmrag.evaluation.retrieval_eval import RetrievalRun
from mmrag.generation.providers.base import Message, ProviderError
from mmrag.production.load import (
    Clocks,
    LiveCallsRefusedError,
    LoadConfigError,
    LoadItem,
    LoadRun,
    ReplayProvider,
    check_live_allowed,
    estimate_live_cost,
    execute_request,
    load_source,
    render_load,
    requests_per_method,
    run_level,
    run_load,
    summarize_level,
    workload_from,
)
from mmrag.production.pricing import PricingTable
from mmrag.production.resources import ResourceSampler
from mmrag.schemas import Answer, Citation

REPO = Path(__file__).resolve().parents[1]

STAGE_LATENCY = {"bm25_ms": 10.0, "dense_ms": 40.0, "fusion_ms": 1.0, "rerank_ms": 249.0,
                 "total_ms": 300.0}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def messages_for(query: str) -> list[Message]:
    return [Message(role="user", content=f"Q: {query}")]


class FakeMethod:
    """Stands in for method.answer: one provider call, fixed stage timings."""

    def __init__(self, *, sleep_s: float = 0.0, drift: bool = False, fail_on: str = "boom"):
        self.sleep_s = sleep_s
        self.drift = drift
        self.fail_on = fail_on
        self.calls: list[tuple[str, int | None]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def answer(self, query, provider, *, top_k=None):
        with self._lock:
            self.calls.append((query, top_k))
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.sleep_s:
                time.sleep(self.sleep_s)
            if self.fail_on in query:
                raise ProviderError("APITimeoutError: Request timed out.")
            msgs = messages_for(query + (" drifted" if self.drift else ""))
            completion = provider.complete(msgs, model="gpt-4o-mini")
            return Answer(
                query=query, text=completion.text,
                citations=[Citation(doc_id="doc", doc_title="Doc", page_number=1, chunk_id="c1")],
                latency_ms={**STAGE_LATENCY, "generation_ms": 200.0},
                usage=completion.usage.as_dict(),
            )
        finally:
            with self._lock:
                self.in_flight -= 1


class StepClock:
    """A thread-safe clock that advances one second per call."""

    def __init__(self, start: float = 1_000_000.0):
        self.now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            self.now += 1.0
            return self.now


class FakeSampler:
    started = 0

    def start(self):
        FakeSampler.started += 1
        return self

    def stop(self):
        return {"available": False, "reason": "fake", "n_samples": 0,
                "gpu": {"available": False, "reason": "fake"}}


def record(qid, query, *, answerable=True, status="ok", latency=1500.0):
    return GenerationRecord(
        query_id=qid, query=query, answerable=answerable, status=status,
        answer=f"answer for {qid} [1]" if status == "ok" else None,
        prompt_sha256=_prompt_sha256(messages_for(query)),
        usage={"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050},
        latency_ms=latency, model="openai/gpt-4o-mini", finish_reason="stop",
    )


QUERIES = [(f"q{i:03d}", f"question {i}?") for i in range(1, 7)]


def generation_run(method="method1", retrieval_path=None, retrieval_sha=None, records=None):
    records = records if records is not None else (
        [record(q, text) for q, text in QUERIES]
        + [record("u001", "unanswerable?", answerable=False)]
    )
    return GenerationRun(method=method, label=f"{method}/rerank-v1-nocache",
                         created_at="2026-09-15T14:05:29+00:00", elapsed_s=1.0,
                         retrieval_run={"path": retrieval_path, "sha256": retrieval_sha},
                         generation={"model": "gpt-4o-mini", "pipeline": "v1"}, records=records)


def write_artifacts(tmp_path, method="method1", overrides=None):
    rpath = tmp_path / f"{method}_retrieval.json"
    rpath.write_text(RetrievalRun(method=method, config_name=method, started_at="t",
                                  elapsed_s=1.0, overrides=overrides or {}).model_dump_json(),
                     encoding="utf-8")
    gen = generation_run(method, str(rpath), file_fingerprint(rpath)["sha256"])
    gpath = gen.save(tmp_path / f"{method}_generation.json")
    return rpath, gpath


def item(i=1):
    qid, text = QUERIES[i - 1]
    return LoadItem(qid, text)


# ---------------------------------------------------------------------------
# Replay provider
# ---------------------------------------------------------------------------


class TestReplayProvider:
    def test_prompt_sha_match_returns_the_recording_after_its_latency(self):
        slept = []
        provider = ReplayProvider([record("q001", "question 1?", latency=1234.0)],
                                  sleep=slept.append)
        with provider.serving("q001"):
            completion = provider.complete(messages_for("question 1?"), model="x")
            assert provider.last_match == "prompt_sha"
        assert slept == [pytest.approx(1.234)]
        assert completion.text == "answer for q001 [1]"
        assert completion.usage.prompt_tokens == 1000 and completion.usage.completion_tokens == 50
        assert completion.metadata["generation_source"] == "replay"

    def test_drifted_prompt_falls_back_to_the_served_query(self):
        provider = ReplayProvider([record("q001", "question 1?")], sleep=lambda s: None)
        with provider.serving("q001"):
            completion = provider.complete(messages_for("something else"), model="x")
            assert provider.last_match == "query_id_fallback"
        assert completion.text == "answer for q001 [1]"

    def test_unknown_prompt_and_query_is_a_provider_error(self):
        provider = ReplayProvider([record("q001", "question 1?")], sleep=lambda s: None)
        with provider.serving("q999"), pytest.raises(ProviderError, match="replay"):
            provider.complete(messages_for("nope"), model="x")

    def test_failed_records_are_never_replayed(self):
        provider = ReplayProvider([record("q001", "question 1?", status="error")],
                                  sleep=lambda s: None)
        with provider.serving("q001"), pytest.raises(ProviderError):
            provider.complete(messages_for("question 1?"), model="x")

    def test_bound_query_is_per_thread(self):
        provider = ReplayProvider([record("q001", "a?"), record("q002", "b?")],
                                  sleep=lambda s: None)
        seen = {}

        def serve(qid):
            with provider.serving(qid):
                seen[qid] = provider.complete(messages_for("drift"), model="x").text

        threads = [threading.Thread(target=serve, args=(q,)) for q in ("q001", "q002")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen == {"q001": "answer for q001 [1]", "q002": "answer for q002 [1]"}


# ---------------------------------------------------------------------------
# One request
# ---------------------------------------------------------------------------


def run_one(method, provider, *, it=None, clocks=None):
    clocks = clocks or Clocks(perf=StepClock(), wall=StepClock(), pause=lambda s: None)
    return execute_request(method, provider, it or item(), method_name="method1", phase="level",
                           concurrency=1, index=0, top_k=10, generation_source="replay",
                           submitted_perf=clocks.perf(), submitted_wall=clocks.wall(),
                           clocks=clocks)


class TestExecuteRequest:
    def provider(self):
        return ReplayProvider([record(q, t) for q, t in QUERIES], sleep=lambda s: None)

    def test_ok_request_on_a_deterministic_clock(self):
        method = FakeMethod()
        req = run_one(method, self.provider())
        # StepClock: submit=1, start=2, finish=3 -> 1 s queued, 1 s served.
        assert req.status == "ok" and req.error is None
        assert req.e2e_latency_ms == 1000.0
        assert req.queue_wait_ms == 1000.0
        assert req.generation_source == "replay" and req.replay_match == "prompt_sha"
        assert req.stages_ms == {"retrieval": 50.0, "fusion": 1.0, "reranking": 249.0,
                                 "retrieval_total": 300.0}
        assert req.generation_ms == 200.0
        assert req.untimed_overhead_ms == 1000.0 - 300.0 - 200.0
        assert req.total_tokens == 1050 and req.n_citations == 1
        assert method.calls == [("question 1?", 10)]
        assert req.started_at.endswith("+00:00") and req.submitted_at < req.started_at

    def test_error_is_recorded_and_classified(self):
        method = FakeMethod()
        req = run_one(method, self.provider(), it=LoadItem("q009", "boom question"))
        assert req.status == "error"
        assert req.error_kind == "timeout"
        assert "APITimeoutError" in req.error
        assert req.e2e_latency_ms == 1000.0
        assert req.untimed_overhead_ms is None and req.stages_ms == {}

    def test_secrets_are_redacted(self):
        class Leaky:
            def answer(self, query, provider, *, top_k=None):
                raise RuntimeError("bad key sk-proj-ABCDEFGHIJKLMNOP")

        assert "[REDACTED]" in run_one(Leaky(), self.provider()).error

    def test_drift_is_flagged(self):
        assert run_one(FakeMethod(drift=True), self.provider()).replay_match == \
            "query_id_fallback"

    def test_provider_without_serving_hook(self):
        class Plain:
            name = "fake"

            def complete(self, messages, **kw):
                return ReplayProvider([record("q001", "question 1?")],
                                      sleep=lambda s: None).complete(messages_for("question 1?"),
                                                                     model="x")

        req = run_one(FakeMethod(), Plain())
        assert req.status == "ok" and req.replay_match is None


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


class TestLevels:
    def test_summary_arithmetic(self):
        provider = ReplayProvider([record(q, t) for q, t in QUERIES], sleep=lambda s: None)
        reqs = [run_one(FakeMethod(), provider, it=item(i)) for i in (1, 2, 3)]
        reqs.append(run_one(FakeMethod(), provider, it=LoadItem("q9", "boom")))
        level = summarize_level(5, reqs, started_wall=0.0, finished_wall=8.0, wall_s=8.0,
                                resources={"available": False})
        assert level.n_requests == 4 and level.n_ok == 3 and level.n_errors == 1
        assert level.throughput_rps == pytest.approx(3 / 8)
        assert level.error_rate == {"count": 1, "n": 4, "rate": 0.25}
        assert level.errors_by_kind == {"timeout": 1}
        assert level.e2e_latency_ms["n"] == 3 and level.e2e_latency_ms["p50"] == 1000.0
        assert level.stages_ms["untimed_overhead"]["kind"] == "measured residual"
        assert level.stages_ms["reranking"]["p50"] == 249.0
        assert level.replay_matches == {"prompt_sha": 3}

    def test_zero_wall_time_has_no_throughput(self):
        level = summarize_level(1, [], started_wall=0.0, finished_wall=0.0, wall_s=0.0,
                                resources={})
        assert level.throughput_rps is None and level.e2e_latency_ms["n"] == 0

    @pytest.mark.parametrize("concurrency", [1, 5, 10])
    def test_concurrency_is_real_and_bounded(self, concurrency):
        method = FakeMethod(sleep_s=0.1)
        provider = ReplayProvider([record(q, t) for q, t in QUERIES], sleep=lambda s: None)
        workload = [item(1 + i % 6) for i in range(10)]
        FakeSampler.started = 0
        level = run_level(method, provider, workload, concurrency, method_name="method1",
                          top_k=10, generation_source="replay", sampler_factory=FakeSampler,
                          clocks=Clocks())
        assert method.max_in_flight == concurrency
        assert level.n_requests == 10 and level.n_ok == 10
        assert [r.index for r in level.requests] == list(range(10))
        assert [r.query_id for r in level.requests] == [w.query_id for w in workload]
        assert all(r.concurrency == concurrency for r in level.requests)
        assert FakeSampler.started == 1
        if concurrency == 1:
            waits = [r.queue_wait_ms for r in level.requests]
            assert waits == sorted(waits) and waits[-1] > 0
            assert level.wall_s >= 10 * 0.1

    def test_invalid_concurrency(self):
        with pytest.raises(LoadConfigError):
            run_level(FakeMethod(), None, [], 0, method_name="m", top_k=10,
                      generation_source="replay", sampler_factory=FakeSampler, clocks=Clocks())


# ---------------------------------------------------------------------------
# Sources, workload and gating
# ---------------------------------------------------------------------------


class TestSources:
    def test_workload_is_answerable_ok_in_order(self):
        gen = generation_run(records=[record("q001", "a?"), record("q002", "b?", status="error"),
                                      record("u001", "c?", answerable=False),
                                      record("q003", "d?")])
        assert [w.query_id for w in workload_from(gen, 2)] == ["q001", "q003"]
        with pytest.raises(LoadConfigError, match="exceeds"):
            workload_from(gen, 3)
        with pytest.raises(LoadConfigError):
            workload_from(gen, 0)

    def test_load_source(self, tmp_path):
        _, gpath = write_artifacts(tmp_path)
        src = load_source("method1", gpath)
        assert src.top_k == 10 and src.config.generation.pipeline == "v1"
        assert src.warnings == []
        assert set(src.inputs()) == {"replay_generation_run", "retrieval_run"}

    def test_wrong_method_is_refused(self, tmp_path):
        _, gpath = write_artifacts(tmp_path)
        with pytest.raises(LoadConfigError, match="not method2"):
            load_source("method2", gpath)

    def test_no_metadata_arm_is_refused(self, tmp_path):
        _, gpath = write_artifacts(tmp_path, method="method2", overrides={"use_metadata": False})
        with pytest.raises(LoadConfigError, match="no-metadata"):
            load_source("method2", gpath)

    def test_changed_retrieval_run_warns(self, tmp_path):
        rpath, gpath = write_artifacts(tmp_path)
        rpath.write_text(rpath.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        assert load_source("method1", gpath).warnings

    def test_missing_files(self, tmp_path):
        with pytest.raises(LoadConfigError, match="not found"):
            load_source("method1", tmp_path / "absent.json")
        rpath, gpath = write_artifacts(tmp_path)
        rpath.unlink()
        with pytest.raises(LoadConfigError, match="retrieval run not found"):
            load_source("method1", gpath)

    def test_default_replay_runs_are_the_v1_nocache_runs(self):
        assert set(load.DEFAULT_REPLAY_RUNS) == {"method1", "method2", "method3"}
        assert all("v1-nocache_generation.json" in p for p in load.DEFAULT_REPLAY_RUNS.values())
        assert (load.DEFAULT_LEVELS, load.DEFAULT_LIMIT, load.DEFAULT_WARMUP,
                load.DEFAULT_COOLDOWN_S) == ((1, 5, 10), 20, 2, 5.0)


class TestLiveGate:
    def test_replay_needs_no_permission(self):
        check_live_allowed("replay", allow_live_calls=False, max_live_requests=None,
                           planned_requests=10_000)

    def test_live_requires_explicit_enable(self):
        with pytest.raises(LiveCallsRefusedError, match="allow-live-calls"):
            check_live_allowed("live", allow_live_calls=False, max_live_requests=100,
                               planned_requests=1)

    def test_live_requires_a_cap(self):
        with pytest.raises(LiveCallsRefusedError, match="max-live-requests"):
            check_live_allowed("live", allow_live_calls=True, max_live_requests=None,
                               planned_requests=1)

    def test_live_cap_is_enforced(self):
        with pytest.raises(LiveCallsRefusedError, match="exceed"):
            check_live_allowed("live", allow_live_calls=True, max_live_requests=5,
                               planned_requests=6)
        check_live_allowed("live", allow_live_calls=True, max_live_requests=6,
                           planned_requests=6)

    def test_request_count(self):
        assert requests_per_method([1, 5, 10], 20, 2) == 62
        assert requests_per_method([1], 3, 5) == 6  # warm-up is bounded by the workload

    def test_live_cost_projection(self, tmp_path):
        _, gpath = write_artifacts(tmp_path)
        pricing = PricingTable.load(REPO / "configs" / "pricing.yaml")
        est = estimate_live_cost([load_source("method1", gpath)], levels=[1, 5], limit=2,
                                 warmup=1, pricing=pricing)
        n = 1 + 2 * 2
        assert est["per_method"]["method1"]["requests"] == n
        assert est["total_usd"] == pytest.approx((1000 * n * 0.15 + 50 * n * 0.60) / 1e6)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def fake_run(tmp_path, **kw):
    _, gpath = write_artifacts(tmp_path)
    pauses: list[float] = []
    built: list = []

    def factory(config):
        method = FakeMethod()
        built.append((config, method))
        return method

    defaults = dict(mode="replay", replay_runs={"method1": str(gpath)}, levels=[1, 2, 3],
                    limit=3, warmup=2, cooldown_s=5.0, method_factory=factory,
                    sampler_factory=FakeSampler, replay_sleep=lambda s: None,
                    clocks=Clocks(pause=pauses.append))
    defaults.update(kw)
    return run_load(["method1"], **defaults), pauses, built


class TestRunLoad:
    def test_end_to_end_replay(self, tmp_path):
        run, pauses, built = fake_run(tmp_path)
        assert run.kind == "load" and run.mode == "replay"
        assert run.generation_source == "replay"
        assert pauses == [5.0, 5.0]  # between levels, not before the first
        assert len(built) == 1
        m = run.methods[0]
        assert m.pipeline == "v1" and m.top_k == 10
        assert m.workload_query_ids == ["q001", "q002", "q003"]
        assert m.warmup["n"] == 2 and m.warmup["n_errors"] == 0
        assert [lv.concurrency for lv in m.levels] == [1, 2, 3]
        assert all(lv.n_ok == 3 for lv in m.levels)
        assert all(r.phase == "level" for lv in m.levels for r in lv.requests)
        method = built[0][1]
        # 2 warm-up + 3 per level, all through method.answer.
        assert len(method.calls) == 2 + 3 * 3
        notes = " ".join(run.notes)
        assert "NOT a direct measurement" in notes
        assert "rate-limit behaviour are NOT measured" in notes
        assert "not observable" in notes
        assert run.environment["cpu_logical_cores"]

    def test_save_load_and_render(self, tmp_path):
        from rich.console import Console

        run, _, _ = fake_run(tmp_path)
        path = run.save(tmp_path / "out" / run.default_filename())
        assert path.name.endswith("_method1_replay_load.json")
        loaded = LoadRun.load(path)
        assert loaded.model_dump() == run.model_dump()
        console = Console(record=True, width=250)
        render_load(loaded, console=console)
        assert "generation_source=replay" in console.export_text()

    def test_validation(self, tmp_path):
        for kw in ({"levels": []}, {"levels": [1, 1]}, {"levels": [0]}, {"warmup": -1},
                   {"mode": "bogus"}):
            with pytest.raises(LoadConfigError):
                fake_run(tmp_path, **kw)
        with pytest.raises(LoadConfigError):
            run_load(["method9"])

    def test_live_is_refused_before_anything_is_built(self, tmp_path):
        with pytest.raises(LiveCallsRefusedError):
            fake_run(tmp_path, mode="live")
        with pytest.raises(LiveCallsRefusedError):
            fake_run(tmp_path, mode="live", allow_live_calls=True, max_live_requests=3)

    def test_live_with_an_injected_provider(self, tmp_path):
        calls = []

        class FakeLive:
            name = "fake-live"

            def complete(self, messages, **kw):
                calls.append(messages)
                return ReplayProvider([record(q, t) for q, t in QUERIES],
                                      sleep=lambda s: None).complete(messages, model="x")

        run, _, _ = fake_run(tmp_path, mode="live", allow_live_calls=True,
                             max_live_requests=100, provider_factory=lambda src: FakeLive())
        assert run.generation_source == "live"
        assert "rate-limit" not in " ".join(run.notes)
        assert len(calls) == 2 + 3 * 3

    def test_prompt_drift_is_warned(self, tmp_path):
        def factory(config):
            return FakeMethod(drift=True)

        run, _, _ = fake_run(tmp_path, method_factory=factory)
        assert any("replayed by query id" in w for w in run.methods[0].warnings)


# ---------------------------------------------------------------------------
# Resource sampler
# ---------------------------------------------------------------------------


class TestSampler:
    def fake_psutil(self):
        proc = SimpleNamespace(cpu_percent=lambda _: 320.0,
                               memory_info=lambda: SimpleNamespace(rss=512 * 2**20))
        return SimpleNamespace(Process=lambda: proc, cpu_percent=lambda _: 40.0,
                               virtual_memory=lambda: SimpleNamespace(percent=55.0))

    def test_summary_normalises_cpu(self):
        sampler = ResourceSampler(60.0, psutil_module=self.fake_psutil(),
                                  gpu_probe=lambda: (None, "no CUDA"))
        summary = sampler.start().stop()
        cores = sampler._cores
        assert summary["available"] and summary["n_samples"] == 1
        assert summary["process_cpu_percent_raw"]["max"] == 320.0
        assert summary["process_cpu_percent"]["max"] == pytest.approx(320.0 / cores, abs=0.05)
        assert summary["process_rss_mb"]["max"] == 512.0
        assert summary["system_memory_percent"]["mean"] == 55.0
        assert summary["gpu"] == {"available": False, "reason": "no CUDA"}

    def test_without_psutil(self):
        summary = ResourceSampler(60.0, psutil_module=None, gpu_probe=None).start().stop()
        assert summary["available"] is False and "psutil" in summary["reason"]
        assert summary["gpu"]["available"] is False

    def test_gpu_samples(self):
        snap = {"device": "FakeGPU", "memory_allocated_mb": 100.0,
                "max_memory_allocated_mb": 900.0, "utilization_percent": 70.0}
        summary = ResourceSampler(60.0, psutil_module=None,
                                  gpu_probe=lambda: (snap, None)).start().stop()
        assert summary["gpu"]["available"] and summary["gpu"]["max_memory_allocated_mb"] == 900.0

    def test_samples_on_its_own_thread(self):
        sampler = ResourceSampler(0.01, psutil_module=self.fake_psutil(), gpu_probe=None).start()
        time.sleep(0.1)
        assert sampler.stop()["n_samples"] >= 3

    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError):
            ResourceSampler(0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def invoke(self, *args):
        return CliRunner().invoke(cli.app, ["prod", "load", *args])

    def test_success_saves_a_run(self, tmp_path, monkeypatch):
        run, _, _ = fake_run(tmp_path)
        seen = {}

        def fake_run_load(names, **kw):
            seen.update(kw, names=names)
            return run

        monkeypatch.setattr(load, "run_load", fake_run_load)
        monkeypatch.setattr(cli, "_preflight_services", lambda methods: [])
        result = self.invoke("--methods", "method1", "--concurrency", "1,5,10", "--limit", "3",
                             "--replay-run", "method1=some/path.json", "--out", str(tmp_path / "o"))
        assert result.exit_code == 0, result.output
        assert seen["names"] == ["method1"] and seen["levels"] == [1, 5, 10]
        assert seen["replay_runs"] == {"method1": "some/path.json"}
        assert seen["mode"] == "replay" and seen["warmup"] == 2 and seen["cooldown_s"] == 5.0
        assert len(list((tmp_path / "o").glob("*_load.json"))) == 1

    def test_preflight_failure_exits_without_running(self, monkeypatch):
        monkeypatch.setattr(load, "run_load", lambda *a, **k: pytest.fail("ran"))
        monkeypatch.setattr(cli, "_preflight_services", lambda methods: ["Qdrant unreachable"])
        result = self.invoke("--methods", "method1")
        assert result.exit_code == 1 and "Qdrant unreachable" in result.output

    def test_live_without_permission_is_refused(self, monkeypatch):
        monkeypatch.setattr(load, "run_load", lambda *a, **k: pytest.fail("ran"))
        monkeypatch.setattr(cli, "_preflight_services", lambda methods: pytest.fail("probed"))
        result = self.invoke("--methods", "method1", "--mode", "live")
        assert result.exit_code == 1 and "allow-live-calls" in result.output

    @pytest.mark.parametrize("args", [
        ("--methods", "method9"),
        ("--concurrency", "1,x"),
        ("--replay-run", "method1"),
        ("--mode", "stream"),
    ])
    def test_bad_arguments(self, args, monkeypatch):
        monkeypatch.setattr(load, "run_load", lambda *a, **k: pytest.fail("ran"))
        assert self.invoke(*args).exit_code == 2
