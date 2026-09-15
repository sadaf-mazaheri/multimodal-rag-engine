"""Production evaluation: latency, tokens, cost, reliability, failure buckets.

No network, no models, no index: artefacts are built in memory or in tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import mmrag.cli as cli
from mmrag.config import GenerationConfig
from mmrag.evaluation.generation_eval import (
    GenerationRecord,
    GenerationRun,
    _prompt_sha256,
    rebuild_retrieved,
)
from mmrag.evaluation.judge import JudgeOutcome
from mmrag.evaluation.judged_eval import JudgedRecord, JudgedRun, Scores
from mmrag.evaluation.retrieval_eval import QueryResult, RetrievalRun, RetrievedChunk
from mmrag.generation.answerer import Answerer, resolve_citations
from mmrag.production.failures import bucket_of, decompose
from mmrag.production.pricing import PricingError, PricingTable
from mmrag.production.prompt_timing import (
    PromptTiming,
    offline_answerer,
    postprocess,
    time_prompts,
)
from mmrag.production.reliability import classify_error, reliability
from mmrag.production.report import (
    COMPONENTS,
    ProductionReport,
    build_report,
    compare_table,
    render_report,
)
from mmrag.production.runner import ArtifactMismatchError, produce
from mmrag.production.stages import canonical_stages
from mmrag.production.stats import percentile, rate, summarize
from mmrag.schemas import BBox, Chunk, ChunkType
from mmrag.textify.tokens import HeuristicTokenCounter

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def chunk(cid, doc, page, kind=ChunkType.TEXT, text="some text"):
    return Chunk(chunk_id=cid, doc_id=doc, page_number=page, chunk_type=kind, text=text,
                 element_ids=[f"{doc}#p{page}#x000"], bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
                 variant="method1", metadata={"doc_title": doc.upper()})


CHUNKS = {
    "c_a": chunk("c_a", "doc", 1, text="Alpha is 42 percent in 2023."),
    "c_b": chunk("c_b", "doc", 2, text="Beta grew to 7 units."),
}


def hit(cid, rank):
    c = CHUNKS[cid]
    return RetrievedChunk(rank=rank, chunk_id=cid, doc_id=c.doc_id, page=c.page_number,
                          chunk_type=c.chunk_type.value, score=1.0 / rank, retriever="stub")


M1_LATENCY = {"bm25_ms": 4.0, "embed_ms": 50.0, "dense_ms": 30.0, "fusion_ms": 1.0,
              "rerank_ms": 1000.0, "total_ms": 1085.0}


def retrieval_run(**latencies):
    per_query = [
        QueryResult(query_id=qid, query=f"question {qid}?", stratum="text", requires="text",
                    n_gold=1, retrieved=[hit("c_a", 1), hit("c_b", 2)], latency_ms=lat)
        for qid, lat in latencies.items()
    ]
    return RetrievalRun(method="method1", config_name="method1", tag="rerank",
                        started_at="2026-01-01T00:00:00+00:00", elapsed_s=1.0,
                        per_query=per_query)


def record(qid, *, answerable=True, status="ok", latency=2000.0, cache_hit=False, attempts=1,
           error=None, usage=(1000, 100), model="openai/gpt-4o-mini", **kw):
    fields = dict(
        query_id=qid, query=f"question {qid}?", answerable=answerable, status=status,
        error=error, attempts=attempts, cache_hit=cache_hit if status == "ok" else None,
        latency_ms=latency if status == "ok" else None,
        usage=({"prompt_tokens": usage[0], "completion_tokens": usage[1],
                "total_tokens": sum(usage)} if status == "ok" else {}),
        model=model if status == "ok" else None,
        answer="Alpha is 42 percent [1]." if status == "ok" else None,
    )
    fields.update(kw)
    return GenerationRecord(**fields)


def generation_run(records, pipeline="v2.1", retrieval_path=None, retrieval_sha=None):
    return GenerationRun(method="method1", label="method1/rerank+genv2.1",
                         created_at="2026-01-01T00:00:00+00:00", elapsed_s=1.0,
                         retrieval_run={"path": retrieval_path, "sha256": retrieval_sha},
                         generation={"model": "gpt-4o-mini", "pipeline": pipeline},
                         records=records)


PRICING_YAML = """
version: 1
as_of: "2026-09-15"
source: test rates
models:
  gpt-4o-mini:
    aliases: ["openai/gpt-4o-mini"]
    input_per_1m: 0.15
    output_per_1m: 0.60
    gateway: test
"""


@pytest.fixture
def pricing(tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(PRICING_YAML, encoding="utf-8")
    return PricingTable.load(path)


def scores(*, grounded_correct=True, citation_problem=False, refused=False):
    return Scores(
        n_claims=2, n_supported=2, n_cited=2, n_citation_supported=2, faithfulness=1.0,
        has_unsupported=False, uncited_claim_rate=0.0, citation_support=1.0, n_facts=1,
        n_covered=1, completeness=1.0, correctness="refused" if refused else "correct",
        correctness_score=None if refused else 1.0, refused=refused,
        refusal_without_marker=False, context_sufficient=True,
        grounded_correct=grounded_correct, citation_problem=citation_problem,
    )


def judged(qid, *, answerable=True, retrieved=True, in_context=True, status="ok",
           judge_status="ok", category="fully_correct", **score_kw):
    gen = record(qid, answerable=answerable, status=status,
                 evidence_retrieved=retrieved if answerable else None,
                 evidence_in_context=in_context if answerable else None)
    has_scores = status == "ok" and judge_status == "ok"
    return JudgedRecord(query_id=qid, answerable=answerable, generation=gen,
                        judge=JudgeOutcome(status=judge_status),
                        scores=scores(**score_kw) if has_scores else None, category=category)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_linear_interpolation(self):
        values = [1, 2, 3, 4]
        assert percentile(values, 0.5) == 2.5
        assert percentile(values, 0.0) == 1
        assert percentile(values, 1.0) == 4
        assert percentile(values, 0.99) == pytest.approx(3.97)

    def test_order_does_not_matter(self):
        assert percentile([10, 1, 5], 0.5) == 5

    def test_empty_and_single(self):
        assert percentile([], 0.5) is None
        assert percentile([7], 0.99) == 7

    def test_fraction_is_validated(self):
        with pytest.raises(ValueError):
            percentile([1], 1.5)

    def test_summary_ignores_none_and_notes_small_n(self):
        s = summarize([1.0, None, 3.0])
        assert s["n"] == 2 and s["mean"] == 2.0 and s["p50"] == 2.0 and s["max"] == 3.0
        assert "not stable" in s["note"]

    def test_large_n_has_no_note(self):
        assert "note" not in summarize(range(100))

    def test_empty_summary(self):
        assert summarize([]) == {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None,
                                 "max": None}

    def test_rate_keeps_its_denominator(self):
        assert rate(1, 4) == {"count": 1, "n": 4, "rate": 0.25}
        assert rate(0, 0)["rate"] is None


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


class TestStages:
    def test_method1_keys(self):
        s = canonical_stages({**M1_LATENCY, "model_load_ms": 29000.0})
        assert s["retrieval"] == 84.0
        assert s["fusion"] == 1.0 and s["reranking"] == 1000.0
        assert s["retrieval_total"] == 1085.0
        assert s["model_load_ms"] == 29000.0
        assert s["retrieval_keys"] == ["bm25_ms", "dense_ms", "embed_ms"]
        assert s["total_consistent"]

    def test_method3_keys_include_routing_and_every_retriever(self):
        lat = {"routing_ms": 0.5, "bm25_ms": 4.0, "dense_ms": 90.0, "table_ms": 80.0,
               "image_ms": 85.0, "visual_page_ms": 200.0, "fusion_ms": 1.0,
               "rerank_ms": 17000.0}
        lat["total_ms"] = sum(lat.values())
        s = canonical_stages(lat)
        assert s["retrieval"] == pytest.approx(459.5)
        assert "visual_page_ms" in s["retrieval_keys"] and "routing_ms" in s["retrieval_keys"]
        assert s["total_consistent"]

    def test_unmapped_total_is_flagged(self):
        assert not canonical_stages({**M1_LATENCY, "total_ms": 5000.0})["total_consistent"]

    def test_no_reranker(self):
        s = canonical_stages({"bm25_ms": 4.0, "fusion_ms": 1.0, "total_ms": 5.0})
        assert s["reranking"] is None and s["total_consistent"]


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------


class TestPricing:
    def test_alias_and_arithmetic(self, pricing):
        assert pricing.price_for("openai/gpt-4o-mini")[0] == "gpt-4o-mini"
        assert pricing.cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
        assert pricing.cost("openai/gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)
        assert pricing.cost("gpt-4o-mini", 3551, 218) == pytest.approx(
            (3551 * 0.15 + 218 * 0.60) / 1e6)

    def test_unknown_model_is_an_error(self, pricing):
        with pytest.raises(PricingError, match="no configured price"):
            pricing.cost("gpt-5", 1, 1)

    def test_missing_or_malformed_file(self, tmp_path):
        with pytest.raises(PricingError):
            PricingTable.load(tmp_path / "absent.yaml")
        bad = tmp_path / "bad.yaml"
        bad.write_text("models: {x: {input_per_1m: -1}}", encoding="utf-8")
        with pytest.raises(PricingError):
            PricingTable.load(bad)

    def test_committed_pricing_file(self):
        table = PricingTable.load(REPO / "configs" / "pricing.yaml")
        name, price = table.price_for("openai/gpt-4o-mini")
        assert name == "gpt-4o-mini"
        assert (price.input_per_1m, price.output_per_1m) == (0.15, 0.60)
        assert table.kind == "configured_evaluation_rates"
        assert table.as_of and table.source and price.gateway


# ---------------------------------------------------------------------------
# reliability
# ---------------------------------------------------------------------------


class TestReliability:
    @pytest.mark.parametrize("error,kind", [
        ("ProviderError: APITimeoutError: Request timed out.", "timeout"),
        ("ProviderError: ReadTimeout", "timeout"),
        ("ProviderError: RateLimitError: Error code: 429", "rate_limit"),
        ("ProviderError: APIConnectionError: Connection error.", "connection"),
        ("ProviderError: BadRequestError: invalid", "other"),
        (None, None),
    ])
    def test_classification(self, error, kind):
        assert classify_error(error) == kind

    def test_rates(self):
        records = [
            record("q1"),
            record("q2", attempts=2),
            record("q3", status="error", attempts=2, error="ProviderError: APITimeoutError: x"),
            record("q4", status="error", attempts=2, error="ProviderError: boom"),
            record("q5", status="dry_run"),
        ]
        r = reliability(records)
        assert r["n"] == 4
        assert r["error_rate"] == {"count": 2, "n": 4, "rate": 0.5}
        assert r["retry_rate"]["count"] == 3
        assert r["timeout_rate"]["count"] == 1
        assert r["errors_by_kind"] == {"other": 1, "timeout": 1}
        assert r["error_query_ids"] == ["q3", "q4"]
        assert r["sdk_internal_retries_observable"] is False
        assert "not observable" in r["note"]


# ---------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------


class TestFailures:
    def test_one_record_per_bucket(self):
        cases = {
            "retrieval_failure": judged("q1", retrieved=False, in_context=False),
            "context_failure": judged("q2", retrieved=True, in_context=False),
            "generation_failure": judged("q3", grounded_correct=False),
            "citation_failure": judged("q4", citation_problem=True),
            "success": judged("q5"),
            "error": judged("q6", status="error", category="error"),
        }
        for expected, rec in cases.items():
            assert bucket_of(rec) == expected

    def test_precedence_retrieval_before_everything(self):
        rec = judged("q1", retrieved=False, in_context=False, grounded_correct=False,
                     citation_problem=True)
        assert bucket_of(rec) == "retrieval_failure"

    def test_judge_error_is_not_a_quality_failure(self):
        assert bucket_of(judged("q1", judge_status="error", category="error")) == "error"

    def test_unanswerable(self):
        assert bucket_of(judged("u1", answerable=False, category="correct_refusal")) == \
            "correct_refusal"
        assert bucket_of(judged("u2", answerable=False, category="answered_unanswerable")) == \
            "answered"

    def test_decomposition(self):
        records = [
            judged("q1", retrieved=False, in_context=False),  # leak: grounded correct anyway
            judged("q2", grounded_correct=False, citation_problem=True),
            judged("q3"),
            judged("u1", answerable=False, category="correct_refusal"),
        ]
        d = decompose(records)
        a = d["answerable"]
        assert a["retrieval_failure"]["query_ids"] == ["q1"]
        assert a["generation_failure"]["count"] == 1 and a["generation_failure"]["n"] == 3
        assert a["success"]["query_ids"] == ["q3"]
        assert sum(v["count"] for v in a.values()) == 3
        assert d["unanswerable"]["correct_refusal"]["count"] == 1
        assert d["citation_problem_any"]["query_ids"] == ["q2"]
        assert d["retrieval_failure_but_grounded_correct"]["query_ids"] == ["q1"]


# ---------------------------------------------------------------------------
# prompt timing
# ---------------------------------------------------------------------------


def v1_answerer():
    return offline_answerer(GenerationConfig())


class TestPromptTiming:
    def _records(self, answerer, run):
        out = []
        for q in run.per_query:
            prompt = answerer.build_prompt(q.query, rebuild_retrieved(q, CHUNKS))
            out.append(record(q.query_id, prompt_sha256=_prompt_sha256(prompt.messages)))
        return out

    def test_matching_prompts_are_timed(self):
        run = retrieval_run(q1=M1_LATENCY, q2=M1_LATENCY)
        answerer = v1_answerer()
        timings = time_prompts(self._records(answerer, run), run, CHUNKS, answerer, repeats=2)
        assert set(timings) == {"q1", "q2"}
        for t in timings.values():
            assert t.sha_match
            assert t.prompt_construction_ms is not None and t.prompt_construction_ms >= 0
            assert t.postprocess_ms is not None and t.postprocess_ms >= 0

    def test_mismatched_prompt_is_not_timed(self):
        run = retrieval_run(q1=M1_LATENCY)
        answerer = v1_answerer()
        rec = self._records(answerer, run)[0].model_copy(update={"prompt_sha256": "0" * 64})
        t = time_prompts([rec], run, CHUNKS, answerer, repeats=1)["q1"]
        assert not t.sha_match
        assert t.prompt_construction_ms is None and t.postprocess_ms is None

    def test_queries_outside_the_retrieval_run_are_skipped(self):
        run = retrieval_run(q1=M1_LATENCY)
        answerer = v1_answerer()
        recs = [*self._records(answerer, run), record("u1", answerable=False)]
        assert set(time_prompts(recs, run, CHUNKS, answerer, repeats=1)) == {"q1"}

    def test_failed_generation_has_no_postprocess_time(self):
        run = retrieval_run(q1=M1_LATENCY)
        answerer = v1_answerer()
        ok = self._records(answerer, run)[0]
        failed = record("q1", status="error", error="x", prompt_sha256=ok.prompt_sha256)
        t = time_prompts([failed], run, CHUNKS, answerer, repeats=1)["q1"]
        assert t.sha_match and t.postprocess_ms is None

    def test_offline_answerer_never_calls_a_provider(self):
        answerer = v1_answerer()
        with pytest.raises(AssertionError, match="dry run"):
            answerer.answer("q?", [])

    def test_offline_answerer_follows_the_pipeline(self):
        assert offline_answerer(GenerationConfig(pipeline="v2.1")).pipeline == "v2.1"

    def test_postprocess_matches_generation_citations(self):
        run = retrieval_run(q1=M1_LATENCY)
        sources = Answerer(GenerationConfig(), None,
                           token_counter=HeuristicTokenCounter()).build_prompt(
            "q?", rebuild_retrieved(run.per_query[0], CHUNKS)).sources
        text = "Alpha is 42 percent [1]. Beta grew [2]."
        out = postprocess(text, sources)
        assert out["citations"] == resolve_citations(text, sources)[0]
        assert out["numbers"] == [1, 2] and not out["refused"]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def timing(qid, prompt=2.0, post=3.0, match=True):
    return PromptTiming(query_id=qid, sha_match=match, prompt_construction_ms=prompt,
                        postprocess_ms=post)


class TestReport:
    def _report(self, pricing, **kw):
        run = retrieval_run(q1={**M1_LATENCY, "model_load_ms": 29000.0}, q2=M1_LATENCY,
                            q3=M1_LATENCY)
        gen = generation_run([
            record("q1", latency=2000.0),
            record("q2", latency=1500.0, cache_hit=True),
            record("q3", status="error", attempts=2, error="ProviderError: APITimeoutError: x"),
            record("u1", answerable=False, latency=900.0),
        ])
        timings = {q: timing(q) for q in ("q1", "q2", "q3")}
        return build_report(gen, run, pricing, prompt_timings=timings, **kw)

    def test_composed_latency_is_the_sum_of_its_components(self, pricing):
        report = self._report(pricing)
        q1 = next(r for r in report.records if r.query_id == "q1")
        assert q1.latency_ms["composed_e2e_latency"] == pytest.approx(1085.0 + 2.0 + 2000.0 + 3.0)
        assert q1.model_load_ms == 29000.0
        e2e = report.latency["composed_e2e_latency"]
        assert e2e["n"] == 1 and e2e["provenance"] == "composed"
        assert "NOT measured in a single serving process" in e2e["composition_note"]

    def test_cache_hits_errors_and_unanswerable_are_not_composed(self, pricing):
        by_id = {r.query_id: r for r in self._report(pricing).records}
        assert by_id["q2"].latency_ms["generation"] is None
        assert by_id["q2"].latency_ms["composed_e2e_latency"] is None
        assert by_id["q3"].latency_ms["composed_e2e_latency"] is None
        assert by_id["u1"].latency_ms["retrieval_total"] is None
        assert by_id["u1"].latency_ms["generation"] == 900.0
        assert not by_id["u1"].in_retrieval_run

    def test_provenance_labels(self, pricing):
        report = self._report(pricing)
        assert report.provenance == COMPONENTS
        assert {k: v["provenance"] for k, v in report.latency.items()} == COMPONENTS
        assert report.latency["prompt_construction"]["provenance"] == "measured"
        assert report.latency["generation"]["provenance"] == "recorded"
        assert report.latency["generation"]["excluded_cache_hits"] == 1
        assert report.cold_start["model_load_ms"]["n"] == 1

    def test_tokens_and_cost_include_cache_hits(self, pricing):
        report = self._report(pricing)
        assert report.tokens["total_tokens"]["n"] == 3  # q1, q2 (cached), u1
        per_query = (1000 * 0.15 + 100 * 0.60) / 1e6
        assert report.cost["total_usd"] == pytest.approx(3 * per_query, abs=1e-6)
        assert report.cost["per_1k_queries_usd"] == pytest.approx(per_query * 1000, abs=1e-4)
        assert report.cost["pricing"]["as_of"] == "2026-09-15"
        assert "not universal" in report.cost["note"]

    def test_reliability_and_prompt_check(self, pricing):
        report = self._report(pricing)
        assert report.reliability["error_rate"]["count"] == 1
        assert report.reliability["timeout_rate"]["count"] == 1
        assert report.prompt_check == {"n_timed": 3, "n_sha_match": 3, "mismatched_query_ids": []}
        assert report.failures is None

    def test_judge_cost_is_evaluation_overhead(self, pricing):
        jrun = JudgedRun(method="method1", label="x", created_at="t", elapsed_s=1.0,
                         judge={"model": "gpt-4o-mini"},
                         totals={"prompt_tokens": 1_000_000, "completion_tokens": 0},
                         records=[judged("q1"), judged("q2"), judged("q3"),
                                  judged("u1", answerable=False, category="correct_refusal")])
        report = self._report(pricing, judged=jrun)
        assert report.evaluation_overhead["judge"]["cost_usd"] == pytest.approx(0.15)
        assert report.cost["total_usd"] < 0.01  # judge spend is not in serving cost
        assert report.failures["answerable"]["success"]["count"] == 3
        assert {r.failure_bucket for r in report.records} == {"success", "correct_refusal"}

    def test_mismatched_prompts_and_totals_warn(self, pricing):
        run = retrieval_run(q1={**M1_LATENCY, "total_ms": 9999.0})
        gen = generation_run([record("q1")])
        report = build_report(gen, run, pricing, prompt_timings={"q1": timing("q1", match=False,
                                                                              prompt=None,
                                                                              post=None)})
        assert report.prompt_check["mismatched_query_ids"] == ["q1"]
        assert any("sha256" in w for w in report.warnings)
        assert any("total_ms" in w for w in report.warnings)

    def test_save_load_and_render(self, pricing, tmp_path):
        report = self._report(pricing)
        path = report.save(tmp_path / report.default_filename())
        loaded = ProductionReport.load(path)
        assert loaded.model_dump() == report.model_dump()
        console = Console(record=True, width=250)
        render_report(loaded, console=console)
        console.print(compare_table([loaded, loaded]))
        text = console.export_text()
        assert "composed_e2e_latency" in text and "not measured" in text


# ---------------------------------------------------------------------------
# runner and CLI
# ---------------------------------------------------------------------------


def write_artifacts(tmp_path):
    from mmrag.evaluation.generation_eval import file_fingerprint

    run = retrieval_run(q1=M1_LATENCY, q2=M1_LATENCY)
    rpath = tmp_path / "retrieval.json"
    rpath.write_text(run.model_dump_json(), encoding="utf-8")
    answerer = offline_answerer(GenerationConfig(pipeline="v1"))
    recs = []
    for q in run.per_query:
        prompt = answerer.build_prompt(q.query, rebuild_retrieved(q, CHUNKS))
        recs.append(record(q.query_id, prompt_sha256=_prompt_sha256(prompt.messages)))
    gen = generation_run(recs, pipeline="v1", retrieval_path=str(rpath),
                         retrieval_sha=file_fingerprint(rpath)["sha256"])
    gpath = gen.save(tmp_path / "generation.json")
    ppath = tmp_path / "pricing.yaml"
    ppath.write_text(PRICING_YAML, encoding="utf-8")
    return rpath, gpath, ppath


class TestRunner:
    def test_produce_measures_prompts_offline(self, tmp_path):
        _, gpath, ppath = write_artifacts(tmp_path)
        report = produce(gpath, pricing_path=ppath, repeats=1, chunks=CHUNKS)
        assert report.prompt_check["n_sha_match"] == 2
        assert report.latency["composed_e2e_latency"]["n"] == 2
        assert report.inputs["prompt_timing"]["measured"] is True
        assert report.warnings == []

    def test_changed_retrieval_run_warns(self, tmp_path):
        rpath, gpath, ppath = write_artifacts(tmp_path)
        rpath.write_text(rpath.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        report = produce(gpath, pricing_path=ppath, measure_prompts=False)
        assert any("does not match" in w for w in report.warnings)

    def test_judged_run_of_another_generation_is_refused(self, tmp_path):
        _, gpath, ppath = write_artifacts(tmp_path)
        jpath = JudgedRun(method="method1", label="x", created_at="t", elapsed_s=1.0,
                          generation_run={"path": "other.json", "sha256": "f" * 64}).save(
            tmp_path / "judged.json")
        with pytest.raises(ArtifactMismatchError):
            produce(gpath, judged_path=jpath, pricing_path=ppath, measure_prompts=False)

    def test_missing_retrieval_run(self, tmp_path):
        rpath, gpath, ppath = write_artifacts(tmp_path)
        rpath.unlink()
        with pytest.raises(FileNotFoundError):
            produce(gpath, pricing_path=ppath, measure_prompts=False)


class TestCli:
    def test_report_and_compare(self, tmp_path):
        runner = CliRunner()
        _, gpath, ppath = write_artifacts(tmp_path)
        out = tmp_path / "prod"
        result = runner.invoke(cli.app, ["prod", "report", "--generation-run", str(gpath),
                                         "--pricing", str(ppath), "--no-prompt-timing",
                                         "--out", str(out)])
        assert result.exit_code == 0, result.output
        saved = list(out.glob("*_production.json"))
        assert len(saved) == 1
        data = json.loads(saved[0].read_text(encoding="utf-8"))
        assert data["kind"] == "production"
        assert data["latency"]["composed_e2e_latency"]["n"] == 0  # prompts not timed

        result = runner.invoke(cli.app, ["prod", "compare", str(saved[0])])
        assert result.exit_code == 0, result.output

    def test_bad_pricing_exits_cleanly(self, tmp_path):
        _, gpath, _ = write_artifacts(tmp_path)
        result = CliRunner().invoke(cli.app, ["prod", "report", "--generation-run", str(gpath),
                                              "--pricing", str(tmp_path / "none.yaml"),
                                              "--no-prompt-timing", "--out", str(tmp_path)])
        assert result.exit_code == 1
