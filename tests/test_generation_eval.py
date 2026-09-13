"""The generation runner and its deterministic metrics. No network, no models."""

from __future__ import annotations

import json
import threading
import time

import pytest

from mmrag.config import GenerationConfig
from mmrag.evaluation.generation_eval import (
    DryRunProvider,
    GenerationRun,
    RetrievalRunMismatch,
    aggregate,
    estimate,
    generate_one,
    rebuild_retrieved,
    run_generation,
    select_work,
    sources_block_of,
)
from mmrag.evaluation.generation_gold import GenerationGold
from mmrag.evaluation.generation_metrics import (
    fact_lexical_coverage,
    is_mixed_refusal,
    key_tokens,
    rate,
)
from mmrag.evaluation.gold import GoldSet
from mmrag.evaluation.llm_cache import CachingProvider
from mmrag.evaluation.retrieval_eval import QueryResult, RetrievalRun, RetrievedChunk
from mmrag.generation.answerer import Answerer
from mmrag.generation.providers.base import Completion, ProviderError, Usage
from mmrag.schemas import BBox, Chunk, ChunkType
from mmrag.textify.tokens import HeuristicTokenCounter

FAKE_KEY = "sk-proj-NOTAREALKEYxxxxxxxxxxxxxxxxxxxx"


def chunk(cid, doc, page, kind=ChunkType.TEXT, text="some text"):
    return Chunk(chunk_id=cid, doc_id=doc, page_number=page, chunk_type=kind, text=text,
                 element_ids=[f"{doc}#p{page}#x000"], bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
                 variant="method1", metadata={"doc_title": doc.upper()})


CHUNKS = {
    "c_other": chunk("c_other", "doc", 9, text="Unrelated prose."),
    "c_gold": chunk("c_gold", "doc", 3, text="The answer is 42 percent in 2023."),
    "c_table": chunk("c_table", "doc2", 1, ChunkType.TABLE, text="| a | b |"),
}

GOLD = GoldSet.model_validate({"version": 1, "queries": [
    {"id": "q001", "query": "What is the answer?", "stratum": "natural", "requires": "text",
     "evidence": [{"doc_id": "doc", "page": 3, "modality": "text"}]},
    {"id": "q002", "query": "Which table?", "stratum": "table", "requires": "table",
     "evidence": [{"doc_id": "doc2", "page": 5, "modality": "table"}]},
]})

GEN_GOLD = GenerationGold.model_validate({
    "version": 1, "retrieval_gold": "v1.yaml",
    "queries": {"q001": {"required_facts": ["the answer is 42 percent"]},
                "q002": {"required_facts": ["Table 7 lists it"]}},
    "unanswerable": [{"id": "u001", "query": "Something absent?", "note": "checked"}],
})


def hit(cid, rank):
    c = CHUNKS[cid]
    return RetrievedChunk(rank=rank, chunk_id=cid, doc_id=c.doc_id, page=c.page_number,
                          chunk_type=c.chunk_type.value, score=1.0 / rank, retriever="stub")


RUN = RetrievalRun(method="method1", config_name="method1", tag="rerank",
                   started_at="2026-01-01T00:00:00+00:00", elapsed_s=1.0, per_query=[
    QueryResult(query_id="q001", query="What is the answer?", stratum="natural",
                requires="text", n_gold=1, retrieved=[hit("c_other", 1), hit("c_gold", 2),
                                                      hit("c_table", 3)]),
    QueryResult(query_id="q002", query="Which table?", stratum="table", requires="table",
                n_gold=1, retrieved=[hit("c_other", 1)]),
])


class ScriptedProvider:
    """Answers by query text; optionally fails a number of times first."""

    name = "openai"

    def __init__(self, answers=None, *, fail_times=0, fail_message="boom", delay=None):
        self.answers = answers or {}
        self.fail_times = fail_times
        self.fail_message = fail_message
        self.delay = delay or {}
        self.calls = []
        self._lock = threading.Lock()

    def supports_images(self):
        return False

    def complete(self, messages, *, model, temperature=0.0, max_output_tokens=1024,
                 seed=None, response_format=None):
        user = messages[-1].content
        with self._lock:
            self.calls.append({"messages": [(m.role, m.content) for m in messages],
                               "model": model, "temperature": temperature,
                               "max_output_tokens": max_output_tokens, "seed": seed})
            if self.fail_times > 0:
                self.fail_times -= 1
                raise ProviderError(self.fail_message)
        for needle, seconds in self.delay.items():
            if needle in user:
                time.sleep(seconds)
        text = next((a for q, a in self.answers.items() if q in user), "fallback [1]")
        return Completion(text=text, model="gpt-4o-mini-2024-07-18",
                          usage=Usage(prompt_tokens=100, completion_tokens=20), latency_ms=5.0,
                          metadata={"finish_reason": "stop", "system_fingerprint": "fp_x"})


def answerer(provider, **config):
    return Answerer(GenerationConfig(**config), provider, token_counter=HeuristicTokenCounter())


def caching(provider, tmp_path, **kwargs):
    return CachingProvider(provider, tmp_path, kind="generation", prompt_version="t", **kwargs)


def work(**kwargs):
    kwargs.setdefault("live_retrieve", lambda q: [])
    return select_work(RUN, GOLD, GEN_GOLD, CHUNKS, **kwargs)


# ---------------------------------------------------------------------------


class TestRebuildingSources:
    def test_sources_come_back_in_rank_order(self):
        rebuilt = rebuild_retrieved(RUN.per_query[0], CHUNKS)
        assert [s.chunk.chunk_id for s in rebuilt] == ["c_other", "c_gold", "c_table"]
        assert [s.rank for s in rebuilt] == [1, 2, 3]

    def test_a_missing_chunk_id_refuses_the_whole_run(self):
        """An id absent from the index means the text may have changed."""
        partial = {k: v for k, v in CHUNKS.items() if k != "c_gold"}
        with pytest.raises(RetrievalRunMismatch, match="re-run retrieval"):
            rebuild_retrieved(RUN.per_query[0], partial)


class TestSelectingWork:
    def test_answerable_in_gold_order_then_unanswerable(self):
        assert [i.query_id for i in work()] == ["q001", "q002", "u001"]

    def test_query_ids_filter(self):
        assert [i.query_id for i in work(query_ids=["q002", "u001"])] == ["q002", "u001"]

    def test_limit_applies_after_ordering(self):
        assert [i.query_id for i in work(limit=1)] == ["q001"]

    def test_unknown_id_is_an_error(self):
        with pytest.raises(ValueError, match="unknown"):
            work(query_ids=["q999"])

    def test_unanswerable_can_be_excluded(self):
        assert [i.query_id for i in work(include_unanswerable=False)] == ["q001", "q002"]

    def test_live_retrieval_only_runs_for_selected_unanswerable_queries(self):
        called = []
        select_work(RUN, GOLD, GEN_GOLD, CHUNKS, query_ids=["q001"],
                    live_retrieve=lambda q: called.append(q) or [])
        assert called == []

    def test_unanswerable_without_a_retriever_is_an_error(self):
        with pytest.raises(RetrievalRunMismatch, match="live retrieval"):
            select_work(RUN, GOLD, GEN_GOLD, CHUNKS, query_ids=["u001"], live_retrieve=None)

    def test_facts_and_gold_are_attached(self):
        item = work(query_ids=["q001"])[0]
        assert item.facts == ["the answer is 42 percent"] and item.gold.id == "q001"


class TestParityWithAnswerer:
    def test_the_provider_sees_exactly_what_answerer_answer_sends(self, tmp_path):
        """The evaluation must not answer a different prompt than the method would."""
        direct = ScriptedProvider()
        answerer(direct).answer("What is the answer?", work(query_ids=["q001"])[0].retrieved)

        via_eval = ScriptedProvider()
        run_generation(work(query_ids=["q001"]), answerer(via_eval),
                       caching(via_eval, tmp_path, enabled=False), concurrency=1)

        keys = ("messages", "model", "temperature", "max_output_tokens")
        assert {k: direct.calls[0][k] for k in keys} == {k: via_eval.calls[0][k] for k in keys}

    def test_sources_block_is_the_exact_source_text(self):
        item = work(query_ids=["q001"])[0]
        prompt = answerer(ScriptedProvider()).build_prompt(item.query, item.retrieved)
        block = sources_block_of(prompt, item.query)
        assert block.startswith("[1] DOC - page 9 (text)")
        assert "[2] DOC - page 3 (text)\nThe answer is 42 percent in 2023." in block
        assert "Question:" not in block

    def test_a_changed_template_fails_loudly(self):
        item = work(query_ids=["q001"])[0]
        prompt = answerer(ScriptedProvider()).build_prompt(item.query, item.retrieved)
        prompt.messages[-1].content = "Context:\n" + prompt.messages[-1].content
        with pytest.raises(ValueError, match="template has changed"):
            sources_block_of(prompt, item.query)


class TestPerQueryRecord:
    def _run(self, tmp_path, answer, query="q001", **config):
        provider = ScriptedProvider({"What is the answer?": answer, "Which table?": answer})
        records = run_generation(work(query_ids=[query]), answerer(provider, **config),
                                 caching(provider, tmp_path), concurrency=1, backoff_s=0)
        return records[0]

    def test_citations_resolve_and_unresolved_are_kept(self, tmp_path):
        record = self._run(tmp_path, "It is 42 percent [2]. Also [9].")
        assert [(c.number, c.chunk_id, c.is_gold) for c in record.citations] == [(2, "c_gold", True)]
        assert record.unresolved_citations == [9]
        assert record.gold_page_cited is True

    def test_evidence_retrieved_and_in_context(self, tmp_path):
        record = self._run(tmp_path, "x [1]")
        assert record.evidence_retrieved is True and record.evidence_in_context is True
        assert [s.is_gold for s in record.sources] == [False, True, False]

    def test_gold_lost_to_the_context_budget(self, tmp_path):
        """Retrieved at rank 2, but only the first source fits: retrieval is not the culprit."""
        from mmrag.evaluation.generation_eval import WorkItem
        from mmrag.schemas import Modality, ScoredChunk

        long_first = chunk("c_long", "doc", 9, text="filler " * 400)
        retrieved = [
            ScoredChunk(chunk=c, score=1.0, rank=r, retriever="stub", modality=Modality.TEXT)
            for r, c in enumerate([long_first, CHUNKS["c_gold"]], start=1)
        ]
        item = WorkItem(query_id="q001", query="What is the answer?", answerable=True,
                        stratum="natural", requires="text", facts=["42 percent"],
                        gold=GOLD.queries[0], retrieved=retrieved)
        provider = ScriptedProvider()
        record = run_generation([item], answerer(provider, max_context_tokens=256),
                                caching(provider, tmp_path), concurrency=1)[0]
        assert record.evidence_retrieved is True
        assert record.evidence_in_context is False
        assert record.dropped_for_budget == 1

    def test_a_retrieval_miss(self, tmp_path):
        record = self._run(tmp_path, "x [1]", query="q002")
        assert record.evidence_retrieved is False and record.evidence_in_context is False

    def test_refusal(self, tmp_path):
        record = self._run(tmp_path, "INSUFFICIENT_EVIDENCE: the sources do not say.")
        assert record.refused is True and record.mixed_refusal is False
        assert record.gold_page_cited is None

    def test_mixed_refusal(self, tmp_path):
        record = self._run(tmp_path, "INSUFFICIENT_EVIDENCE, though [2] mentions 42 percent.")
        assert record.refused is True and record.mixed_refusal is True

    def test_provenance_fields(self, tmp_path):
        record = self._run(tmp_path, "x [2]")
        assert record.status == "ok" and record.model == "gpt-4o-mini-2024-07-18"
        assert record.system_fingerprint == "fp_x" and record.cache_hit is False
        assert record.fact_lexical == {"checkable": 1, "found": 0}
        assert len(record.prompt_sha256) == 64


class TestFailuresAndRetries:
    def test_a_failure_is_recorded_and_the_run_continues(self, tmp_path):
        provider = ScriptedProvider(fail_times=10)
        records = run_generation(work(query_ids=["q001", "q002"]), answerer(provider),
                                 caching(provider, tmp_path), concurrency=1, attempts=2,
                                 backoff_s=0)
        assert [r.status for r in records] == ["error", "error"]
        assert records[0].attempts == 2 and records[0].answer is None

    def test_one_failure_among_successes(self, tmp_path):
        provider = ScriptedProvider(fail_times=2)  # exhausts q001's two attempts only
        records = run_generation(work(query_ids=["q001", "q002"]), answerer(provider),
                                 caching(provider, tmp_path), concurrency=1, attempts=2,
                                 backoff_s=0)
        assert [r.status for r in records] == ["error", "ok"]

    def test_a_transient_failure_is_retried(self, tmp_path):
        provider = ScriptedProvider(fail_times=1)
        record = run_generation(work(query_ids=["q001"]), answerer(provider),
                                caching(provider, tmp_path), concurrency=1, attempts=2,
                                backoff_s=0)[0]
        assert record.status == "ok" and record.attempts == 2

    def test_an_error_message_never_carries_a_key(self, tmp_path):
        provider = ScriptedProvider(fail_times=10, fail_message=f"401 bad key {FAKE_KEY}")
        record = run_generation(work(query_ids=["q001"]), answerer(provider),
                                caching(provider, tmp_path), concurrency=1, attempts=1,
                                backoff_s=0)[0]
        assert FAKE_KEY not in (record.error or "") and "[REDACTED]" in record.error


class TestDryRun:
    def test_nothing_is_called(self, tmp_path):
        provider = caching(DryRunProvider("openai"), tmp_path)
        records = run_generation(work(), answerer(ScriptedProvider()), provider, dry_run=True)
        assert all(r.status == "dry_run" and r.answer is None for r in records)
        assert provider.stats.misses == 0

    def test_it_reports_what_is_already_cached(self, tmp_path):
        real = ScriptedProvider()
        run_generation(work(query_ids=["q001"]), answerer(real), caching(real, tmp_path),
                       concurrency=1)
        records = run_generation(work(query_ids=["q001", "q002"]), answerer(real),
                                 caching(DryRunProvider("openai"), tmp_path), dry_run=True)
        assert [r.cache_hit for r in records] == [True, False]

    def test_the_dry_run_provider_refuses_to_be_called(self):
        with pytest.raises(AssertionError, match="dry run"):
            DryRunProvider("openai").complete([], model="m")


class TestConcurrency:
    def test_records_keep_input_order(self, tmp_path):
        provider = ScriptedProvider(delay={"What is the answer?": 0.2})
        records = run_generation(work(), answerer(provider), caching(provider, tmp_path),
                                 concurrency=4)
        assert [r.query_id for r in records] == ["q001", "q002", "u001"]


class TestAggregation:
    def test_counts_beside_rates(self, tmp_path):
        provider = ScriptedProvider({"What is the answer?": "42 percent [2]",
                                     "Which table?": "INSUFFICIENT_EVIDENCE",
                                     "Something absent?": "INSUFFICIENT_EVIDENCE"})
        records = run_generation(work(), answerer(provider), caching(provider, tmp_path),
                                 concurrency=1)
        m = aggregate(records)
        assert m["answerable"]["evidence_in_context"] == {"n": 2, "count": 1, "rate": 0.5}
        assert m["answerable"]["refused"] == {"n": 2, "count": 1, "rate": 0.5}
        assert m["answerable"]["citation_presence"] == {"n": 1, "count": 1, "rate": 1.0}
        assert m["unanswerable"]["refused"] == {"n": 1, "count": 1, "rate": 1.0}
        # q002's fact "Table 7" is checkable (it has a digit); a refusal covers none.
        assert m["answerable"]["fact_lexical"] == {"found": 1, "checkable": 2, "rate": 0.5}

    def test_rate_excludes_none(self):
        assert rate([True, None, False]) == {"n": 2, "count": 1, "rate": 0.5}
        assert rate([None]) == {"n": 0, "count": 0, "rate": None}


class TestEstimate:
    def test_calls_tokens_and_cost(self, tmp_path):
        records = run_generation(work(), answerer(ScriptedProvider()),
                                 caching(DryRunProvider("openai"), tmp_path), dry_run=True)
        e = estimate(records, output_tokens_per_answer=100, judge_prompt_overhead_tokens=50,
                     judge_output_tokens=10, price_in_per_m=1.0, price_out_per_m=2.0)
        prompt = sum(r.prompt_tokens_estimated for r in records)
        assert e["generation"]["calls"] == 3 and e["judge"]["calls"] == 3
        assert e["generation"]["input_tokens"] == prompt
        assert e["judge"]["input_tokens"] == prompt + 3 * (100 + 50)
        assert e["total"]["cost_usd"] == pytest.approx(
            (2 * prompt + 450) / 1e6 * 1.0 + (300 + 30) / 1e6 * 2.0, abs=1e-4)

    def test_no_prices_means_no_cost(self, tmp_path):
        records = run_generation(work(query_ids=["q001"]), answerer(ScriptedProvider()),
                                 caching(DryRunProvider("openai"), tmp_path), dry_run=True)
        assert estimate(records)["total"]["cost_usd"] is None


class TestLexicalFactCheck:
    def test_key_tokens_are_numbers_and_identifiers(self):
        assert key_tokens("inflation was 2.9% in December 2023") == ["2.9%", "2023"]
        assert key_tokens("Table SPM.2 gives budgets") == ["spm.2"]
        assert key_tokens("no digits here") == []

    def test_coverage_only_counts_checkable_facts(self):
        facts = ["it was 2.9% in 2023", "down from 9.2%", "prose with no numbers"]
        assert fact_lexical_coverage(facts, "Headline inflation hit 2.9% in December 2023.") == {
            "checkable": 2, "found": 1}

    def test_mixed_refusal_needs_citations(self):
        assert not is_mixed_refusal("INSUFFICIENT_EVIDENCE", 0)
        assert is_mixed_refusal("INSUFFICIENT_EVIDENCE but [1]", 1)


class TestArtefact:
    def test_roundtrips_and_holds_no_secret(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        provider = ScriptedProvider(fail_times=1, fail_message=f"bad {FAKE_KEY}")
        records = run_generation(work(), answerer(provider), caching(provider, tmp_path),
                                 concurrency=1, attempts=1, backoff_s=0)
        run = GenerationRun(method="method1", label="method1/rerank", created_at="t",
                            elapsed_s=0.0, records=records, metrics=aggregate(records))
        path = run.save(tmp_path / "run.json")
        text = path.read_text(encoding="utf-8")
        assert FAKE_KEY not in text
        assert json.loads(text)["kind"] == "generation"
        assert GenerationRun.load(path).records[1].query_id == "q002"
