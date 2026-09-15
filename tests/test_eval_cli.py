"""The generate -> judge -> compare CLI pipeline, end to end. No network, no models.

The method and the provider are replaced with fakes, so these tests exercise the
command wiring -- which records are judged, what is saved, what is refused --
rather than retrieval or answer quality.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

import mmrag.cli as cli
import mmrag.generation.providers as providers
from mmrag.config import ExperimentConfig
from mmrag.evaluation.generation_eval import GenerationRun
from mmrag.evaluation.generation_report import render_judged, run_kind, to_markdown_judged
from mmrag.evaluation.judged_eval import JudgedRun
from mmrag.evaluation.retrieval_eval import QueryResult, RetrievalRun, RetrievedChunk
from mmrag.generation.providers.base import Completion, ProviderError, Usage
from mmrag.schemas import BBox, Chunk, ChunkType, Modality, ScoredChunk

runner = CliRunner()


def chunk(cid, doc, page, kind=ChunkType.TEXT, text="some text"):
    return Chunk(chunk_id=cid, doc_id=doc, page_number=page, chunk_type=kind, text=text,
                 element_ids=[f"{doc}#p{page}#x000"], bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
                 variant="method1", metadata={"doc_title": doc.upper()})


CHUNKS = {
    "c_other": chunk("c_other", "doc", 9, text="Unrelated prose."),
    "c_gold": chunk("c_gold", "doc", 3, text="The answer is 42 percent in 2023."),
    "c_table": chunk("c_table", "doc2", 1, ChunkType.TABLE, text="| a | b |"),
}

GOLD = {"version": 1, "queries": [
    {"id": "q001", "query": "What is the answer?", "stratum": "natural", "requires": "text",
     "evidence": [{"doc_id": "doc", "page": 3, "modality": "text"}]},
    {"id": "q002", "query": "Which table?", "stratum": "table", "requires": "table",
     "evidence": [{"doc_id": "doc2", "page": 5, "modality": "table"}]},
]}

GENERATION_GOLD = {
    "version": 1, "retrieval_gold": "v1.yaml",
    "queries": {"q001": {"required_facts": ["the answer is 42 percent", "it is for 2023"]},
                "q002": {"required_facts": ["Table 7 lists it"]}},
    "unanswerable": [{"id": "u001", "query": "Something absent?", "note": "checked"}],
}

RETRIEVED = {"q001": ["c_other", "c_gold", "c_table"], "q002": ["c_other"]}


def hit(cid, rank):
    c = CHUNKS[cid]
    return RetrievedChunk(rank=rank, chunk_id=cid, doc_id=c.doc_id, page=c.page_number,
                          chunk_type=c.chunk_type.value, score=1.0 / rank, retriever="stub")


class FakeMethod:
    """Stands in for a built method: its chunks, and a retriever that records calls."""

    def __init__(self):
        self.chunks = dict(CHUNKS)
        self.retrieve_calls: list[str] = []

    def retrieve(self, query, **kwargs):
        self.retrieve_calls.append(query)
        live = [ScoredChunk(chunk=CHUNKS["c_other"], score=1.0, rank=1, retriever="live",
                            modality=Modality.TEXT)]
        return SimpleNamespace(results=live)


class FakeProvider:
    """Answers generation prompts from a script and judge prompts with a valid verdict."""

    name = "openai"

    def __init__(self, *, fail_generation_on: str | None = None):
        self.fail_generation_on = fail_generation_on
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def supports_images(self):
        return False

    def complete(self, messages, *, model, temperature=0.0, max_output_tokens=1024,
                 seed=None, response_format=None):
        user = messages[-1].content
        kind = "judge" if response_format is not None else "generation"
        with self._lock:
            self.calls.append({"kind": kind, "model": model, "seed": seed, "user": user})
        if kind == "generation":
            if self.fail_generation_on and self.fail_generation_on in user:
                raise ProviderError("503 upstream unavailable")
            if "What is the answer?" in user:
                text = "The answer is 42 percent in 2023 [2]."
            else:
                text = "INSUFFICIENT_EVIDENCE: the sources do not say."
        else:
            text = json.dumps(_verdict_for(user))
        return Completion(text=text, model="gpt-4o-mini-2024-07-18",
                          usage=Usage(prompt_tokens=100, completion_tokens=20), latency_ms=5.0,
                          metadata={"finish_reason": "stop", "system_fingerprint": "fp_t"})


def _verdict_for(user: str) -> dict:
    facts_block = user.split("<reference_facts>\n", 1)[1].split("\n</reference_facts>", 1)[0]
    facts = [re.sub(r"^\d+\.\s*", "", line) for line in facts_block.splitlines()
             if re.match(r"^\d+\.", line)]
    answer = user.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]
    refused = "INSUFFICIENT_EVIDENCE" in answer
    claims = [] if refused else [{"text": "the answer is 42 percent", "cited_sources": [2],
                                  "supported_by_sources": True, "citation_supports": True}]
    return {"claims": claims,
            "required_facts": [{"fact": f, "covered": not refused} for f in facts],
            "contradicts_reference": False, "declines_to_answer": refused,
            "context_sufficient": not refused, "rationale": "scripted"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Gold files, a saved retrieval run, and the fakes wired into the CLI."""
    gold = tmp_path / "gold.yaml"
    gold.write_text(yaml.safe_dump(GOLD), encoding="utf-8")
    generation_gold = tmp_path / "generation_gold.yaml"
    generation_gold.write_text(yaml.safe_dump(GENERATION_GOLD), encoding="utf-8")

    config = ExperimentConfig(method="method1")
    run = RetrievalRun(
        method="method1", config_name="method1", tag="rerank",
        started_at="2026-01-01T00:00:00+00:00", elapsed_s=1.0,
        config=config.model_dump(mode="json"),
        per_query=[
            QueryResult(query_id=q["id"], query=q["query"], stratum=q["stratum"],
                        requires=q["requires"], n_gold=1,
                        retrieved=[hit(cid, r) for r, cid in enumerate(RETRIEVED[q["id"]], 1)])
            for q in GOLD["queries"]
        ],
    )
    retrieval = tmp_path / "retrieval.json"
    retrieval.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    method = FakeMethod()
    monkeypatch.setattr(
        cli, "_method_from_run",
        lambda r: (ExperimentConfig.model_validate(r.config), method),
    )
    provider = FakeProvider()
    monkeypatch.setattr(providers, "get_provider", lambda **kw: provider)
    # Retries back off with real sleeps; nothing here needs the wait.
    monkeypatch.setattr("mmrag.evaluation.generation_eval.time.sleep", lambda s: None)
    monkeypatch.setattr("mmrag.evaluation.judge.time.sleep", lambda s: None)

    class Env:
        pass

    e = Env()
    e.tmp, e.gold, e.generation_gold, e.retrieval = tmp_path, gold, generation_gold, retrieval
    e.method, e.provider, e.monkeypatch = method, provider, monkeypatch
    e.cache, e.out = tmp_path / "cache", tmp_path / "out"
    return e


def generate(env, *extra):
    result = runner.invoke(cli.app, [
        "eval", "generate", "--retrieval-run", str(env.retrieval), "--gold", str(env.gold),
        "--generation-gold", str(env.generation_gold), "--provider", "openai",
        "--cache-dir", str(env.cache), "--out", str(env.out), "--concurrency", "1", *extra])
    assert result.exit_code == 0, result.output
    return result


def judge(env, generation_path, *extra, expect=0):
    result = runner.invoke(cli.app, [
        "eval", "judge", "--generation-run", str(generation_path), "--provider", "openai",
        "--cache-dir", str(env.cache), "--out", str(env.out), "--concurrency", "1", *extra])
    assert result.exit_code == expect, result.output
    return result


def only(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    assert len(found) == 1, found
    return found[0]


# ---------------------------------------------------------------------------


class TestGenerateUsesTheSavedRetrievalRun:
    def test_answerable_queries_never_rerun_retrieval(self, env):
        generate(env, "--no-unanswerable")
        assert env.method.retrieve_calls == []

        run = GenerationRun.load(only(env.out, "*_generation.json"))
        for record in run.records:
            assert [s.chunk_id for s in record.retrieved] == RETRIEVED[record.query_id]
        assert run.retrieval_run["sha256"] and run.retrieval_run["label"] == "method1/rerank"

    def test_only_unanswerable_queries_are_retrieved_live(self, env):
        generate(env)
        assert env.method.retrieve_calls == ["Something absent?"]
        run = GenerationRun.load(only(env.out, "*_generation.json"))
        assert [r.query_id for r in run.records] == ["q001", "q002", "u001"]

    def test_generation_uses_gpt_4o_mini_with_the_pinned_seed(self, env):
        generate(env, "--no-unanswerable")
        calls = [c for c in env.provider.calls if c["kind"] == "generation"]
        assert calls and {c["model"] for c in calls} == {"gpt-4o-mini"}
        assert {c["seed"] for c in calls} == {42}
        run = GenerationRun.load(only(env.out, "*_generation.json"))
        assert run.generation["model"] == "gpt-4o-mini"

    def test_dry_run_calls_nothing_and_saves_nothing(self, env):
        def refuse(**kw):
            raise AssertionError("a dry run built a real provider")

        env.monkeypatch.setattr(providers, "get_provider", refuse)
        result = generate(env, "--dry-run", "--no-unanswerable")
        assert "No provider was called" in result.output
        assert not env.out.exists()


class TestJudge:
    def _generated(self, env, *extra) -> Path:
        generate(env, *extra)
        return only(env.out, "*_generation.json")

    def test_end_to_end_is_a_self_judge_with_gpt_4o_mini(self, env):
        result = judge(env, self._generated(env))
        run = JudgedRun.load(only(env.out, "*_judged.json"))

        assert run.kind == "judged" and run_kind(str(only(env.out, "*_judged.json"))) == "judged"
        assert run.judge["model"] == "gpt-4o-mini" and run.judge["self_judge"] is True
        assert run.generation["model"] == "gpt-4o-mini"
        judge_calls = [c for c in env.provider.calls if c["kind"] == "judge"]
        assert {c["model"] for c in judge_calls} == {"gpt-4o-mini"} and len(judge_calls) == 3
        assert "self-judge" in result.output

        categories = {r.query_id: r.category for r in run.records}
        assert categories == {"q001": "fully_correct", "q002": "retrieval_miss_refused",
                              "u001": "correct_refusal"}
        assert run.metrics["decomposition"]["E2E_grounded_correct"] == {
            "n": 2, "count": 1, "rate": 0.5}

    def test_a_failed_generation_is_kept_as_an_error_not_dropped(self, env):
        env.provider.fail_generation_on = "Which table?"
        generation = self._generated(env, "--no-unanswerable")
        assert [r.status for r in GenerationRun.load(generation).records] == ["ok", "error"]

        judge(env, generation)
        run = JudgedRun.load(only(env.out, "*_judged.json"))
        assert {r.query_id: r.category for r in run.records} == {
            "q001": "fully_correct", "q002": "error"}
        assert run.metrics["coverage"]["generation_errors"] == 1
        judged_queries = [c["user"] for c in env.provider.calls if c["kind"] == "judge"]
        assert len(judged_queries) == 1 and "What is the answer?" in judged_queries[0]

    def test_dry_run_counts_only_answers_that_need_a_call(self, env):
        env.provider.fail_generation_on = "Which table?"
        generation = self._generated(env, "--no-unanswerable")

        def refuse(**kw):
            raise AssertionError("a dry run built a real provider")

        env.monkeypatch.setattr(providers, "get_provider", refuse)
        result = judge(env, generation, "--dry-run")
        assert "1 judge calls needed (0 cached)" in result.output
        assert not list(env.out.glob("*_judged.json"))

    def test_rejudging_is_served_from_the_cache(self, env):
        generation = self._generated(env)
        judge(env, generation, "--tag", "first")
        judge(env, generation, "--tag", "second")
        second = JudgedRun.load(only(env.out, "*_second_judged.json"))
        assert second.totals["calls_made"] == 0 and second.totals["cache_hits"] == 3

    def test_query_filter_and_unknown_ids(self, env):
        generation = self._generated(env)
        judge(env, generation, "--queries", "u001")
        run = JudgedRun.load(only(env.out, "*_judged.json"))
        assert [r.query_id for r in run.records] == ["u001"]
        judge(env, generation, "--queries", "q999", expect=1)

    def test_a_relative_retrieval_path_resolves_against_the_repo_root(self, env, monkeypatch):
        generation = self._generated(env, "--no-unanswerable")
        run = GenerationRun.load(generation)
        run.retrieval_run["path"] = "retrieval.json"
        run.save(generation)

        monkeypatch.setattr("mmrag.config.PROJECT_ROOT", env.tmp)
        monkeypatch.chdir(env.tmp.parent)
        judge(env, generation)

    def test_a_missing_retrieval_run_is_a_clean_error(self, env):
        generation = self._generated(env, "--no-unanswerable")
        env.retrieval.unlink()
        result = judge(env, generation, expect=1)
        assert "missing" in result.output

    def test_a_changed_retrieval_run_is_flagged(self, env):
        generation = self._generated(env, "--no-unanswerable")
        env.retrieval.write_text(env.retrieval.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        result = judge(env, generation)
        assert "has changed" in result.output


class TestCompare:
    def _judged_pair(self, env) -> list[Path]:
        generate(env)
        generation = only(env.out, "*_generation.json")
        judge(env, generation, "--tag", "arm-a")
        env.provider.fail_generation_on = None
        judge(env, generation, "--tag", "arm-b", "--queries", "q001,q002")
        return [only(env.out, "*_arm-a_judged.json"), only(env.out, "*_arm-b_judged.json")]

    def test_judged_runs_are_reported(self, env):
        paths = self._judged_pair(env)
        result = runner.invoke(cli.app, ["eval", "compare", *map(str, paths), "--markdown"])
        assert result.exit_code == 0, result.output
        assert "| arm-a |" in result.output and "| arm-b |" in result.output

    def test_retrieval_runs_still_compare(self, env):
        result = runner.invoke(cli.app, ["eval", "compare", str(env.retrieval)])
        assert result.exit_code == 0, result.output

    def test_mixed_or_unjudged_runs_are_refused(self, env):
        paths = self._judged_pair(env)
        generation = only(env.out, "*_generation.json")
        for mix in ([env.retrieval, paths[0]], [generation]):
            result = runner.invoke(cli.app, ["eval", "compare", *map(str, mix)])
            assert result.exit_code == 1, result.output
            assert "cannot compare" in result.output

    def test_a_missing_file_is_a_usage_error(self, env):
        result = runner.invoke(cli.app, ["eval", "compare", str(env.tmp / "nope.json")])
        assert result.exit_code == 2


class TestReportRendering:
    def test_all_tables_render_for_several_runs(self, env):
        paths = TestCompare()._judged_pair(env)
        runs = [JudgedRun.load(p) for p in paths]
        console = Console(record=True, width=240)
        render_judged(runs, console=console)
        text = console.export_text()
        for title in ("Retrieval -> generation -> end to end", "Answer quality",
                      "Refusal behaviour", "grounded correct by answer modality",
                      "grounded correct by phrasing", "Error taxonomy",
                      "Judge reliability", "Queries whose outcome differs between runs"):
            assert title in text
        assert "directional" in text  # the self-judge caveat

    def test_markdown_has_one_row_per_run(self, env):
        paths = TestCompare()._judged_pair(env)
        lines = to_markdown_judged([JudgedRun.load(p) for p in paths]).splitlines()
        assert len(lines) == 4 and lines[2].startswith("| arm-a |")

    def test_no_runs(self):
        console = Console(record=True)
        render_judged([], console=console)
        assert "no judged runs" in console.export_text()

    def test_run_kind_defaults_to_retrieval(self, env):
        assert run_kind(str(env.retrieval)) == "retrieval"


class TestGenerationPipelines:
    """--pipeline selects V1 or V2 over the same saved retrieval run, in isolation."""

    def test_default_is_v1_with_the_published_prompt_version(self, env):
        from mmrag.evaluation.generation_eval import PROMPT_VERSION

        generate(env, "--no-unanswerable")
        run = GenerationRun.load(only(env.out, "*_generation.json"))
        assert run.label == "method1/rerank"
        assert run.generation["pipeline"] == "v1"
        assert run.generation["prompt_version"] == PROMPT_VERSION == "1ae772d0aa197ff5"
        assert {r.pipeline for r in run.records} == {"v1"}

    def test_v2_is_labelled_versioned_and_never_served_from_the_v1_cache(self, env):
        from mmrag.generation.answerer_v2 import PROMPT_VERSION_V2

        generate(env, "--no-unanswerable")
        result = generate(env, "--no-unanswerable", "--pipeline", "v2")
        assert "cache hits 0" in result.output and "pipeline v2" in result.output

        v2_path = only(env.out, "*+genv2_generation.json")
        run = GenerationRun.load(v2_path)
        assert run.label == "method1/rerank+genv2"
        assert run.generation["pipeline"] == "v2"
        assert run.generation["prompt_version"] == PROMPT_VERSION_V2
        assert run.cache["hits"] == 0 and run.cache["misses"] == 2
        assert {r.pipeline for r in run.records} == {"v2"}
        assert all(r.validation is not None for r in run.records if r.status == "ok")
        # Retrieval was not re-run for either pipeline.
        assert env.method.retrieve_calls == []

    def test_an_unknown_pipeline_is_a_usage_error(self, env):
        result = runner.invoke(cli.app, [
            "eval", "generate", "--retrieval-run", str(env.retrieval), "--gold", str(env.gold),
            "--generation-gold", str(env.generation_gold), "--pipeline", "v3"])
        assert result.exit_code == 2

    def test_a_v2_run_judges_and_reports_beside_v1(self, env):
        generate(env, "--no-unanswerable")
        generate(env, "--no-unanswerable", "--pipeline", "v2")
        judge(env, only(env.out, "*rerank_generation.json"))
        judge(env, only(env.out, "*+genv2_generation.json"))

        v1_run = JudgedRun.load(only(env.out, "*rerank_judged.json"))
        v2_run = JudgedRun.load(only(env.out, "*+genv2_judged.json"))
        assert v2_run.label == "method1/rerank+genv2" and v2_run.generation["pipeline"] == "v2"
        assert v1_run.judge["prompt_version"] == v2_run.judge["prompt_version"]
        assert v2_run.metrics["answerable"]["claims_per_answer"]["n"] == 1

        console = Console(record=True, width=240)
        render_judged([v1_run, v2_run], console=console)
        text = console.export_text()
        assert "Answer shape and validation" in text and "method1/rerank+genv2" in text


class TestGenerationV21:
    def test_v2_1_has_its_own_label_version_and_cache(self, env):
        from mmrag.generation.answerer_v2 import PROMPT_VERSION_V2, PROMPT_VERSION_V2_1

        generate(env, "--no-unanswerable", "--pipeline", "v2")
        result = generate(env, "--no-unanswerable", "--pipeline", "v2.1")
        assert "cache hits 0" in result.output and "pipeline v2.1" in result.output

        v2_run = GenerationRun.load(only(env.out, "*+genv2_generation.json"))
        v21_run = GenerationRun.load(only(env.out, "*+genv2.1_generation.json"))
        assert v2_run.label == "method1/rerank+genv2"
        assert v21_run.label == "method1/rerank+genv2.1"
        assert v2_run.generation["prompt_version"] == PROMPT_VERSION_V2
        assert v21_run.generation["prompt_version"] == PROMPT_VERSION_V2_1
        assert {r.pipeline for r in v21_run.records} == {"v2.1"}
        assert v21_run.cache["hits"] == 0
