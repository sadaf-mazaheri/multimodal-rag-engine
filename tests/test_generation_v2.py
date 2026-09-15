"""Generation V2: evidence pack, answerer, validator, and evaluation isolation.

No network and no models: providers are scripted and token counts heuristic.
"""

from __future__ import annotations

import pytest

from mmrag.config import GenerationConfig
from mmrag.evaluation.generation_eval import (
    PROMPT_VERSION,
    GenerationRecord,
    GenerationRun,
    aggregate,
    run_generation,
    sources_block_of,
)
from mmrag.evaluation.llm_cache import CachingProvider
from mmrag.generation import build_answerer, prompt_version_for
from mmrag.generation.answerer import Answerer, resolve_citations
from mmrag.generation.answerer_v2 import (
    PROMPT_VERSION_V2,
    SYSTEM_PROMPT_V2,
    SYSTEM_PROMPT_V2_NO_REFUSAL,
    AnswererV2,
    PromptBuildV2,
)
from mmrag.generation.evidence import (
    GROUP_SEPARATOR,
    NO_SOURCES,
    build_evidence_pack,
    format_source,
)
from mmrag.generation.providers.base import Message
from mmrag.generation.validation import AnswerValidator, answer_sentences, identifier_tokens
from mmrag.schemas import BBox, Chunk, ChunkType, Modality, ScoredChunk
from mmrag.textify.tokens import HeuristicTokenCounter
from tests.test_generation_eval import CHUNKS as EVAL_CHUNKS
from tests.test_generation_eval import GEN_GOLD, GOLD, RUN, ScriptedProvider, select_work

TOKENS = HeuristicTokenCounter()
TITLE = "Attention Is All You Need"


def scored(
    cid,
    page,
    *,
    rank,
    kind=ChunkType.TEXT,
    body="Body text.",
    doc="arxiv_attention",
    section="3.5 Positional Encoding",
    header=True,
    **metadata,
):
    head = f"{TITLE} > {section}" if header else ""
    chunk = Chunk(
        chunk_id=cid,
        doc_id=doc,
        page_number=page,
        chunk_type=kind,
        text=f"{head}\n\n{body}" if head else body,
        element_ids=[f"{doc}#p{page}#x000"],
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
        section=section,
        variant="method2",
        metadata={"doc_title": TITLE, "context_header": head, **metadata},
    )
    return ScoredChunk(
        chunk=chunk, score=1.0 / rank, rank=rank, retriever="stub", modality=Modality.TEXT
    )


RETRIEVED = [
    scored("a6", 6, rank=1, body="Positional encodings are added to the input embeddings."),
    scored(
        "b9",
        9,
        rank=2,
        kind=ChunkType.TABLE,
        section="6.2 Model Variations",
        body="Table 3: Variations on the Transformer architecture.\n| N | BLEU |\n| 6 | 25.8 |",
    ),
    scored("c6", 6, rank=3, body="The encodings use sine and cosine functions."),
    scored(
        "d3",
        3,
        rank=4,
        kind=ChunkType.FIGURE,
        section="3 Model Architecture",
        body="Figure 1: The Transformer - model architecture.\nMulti-Head Attention",
        has_caption=True,
    ),
]


def v2(provider=None, **config) -> AnswererV2:
    return AnswererV2(
        GenerationConfig(pipeline="v2", **config),
        provider or ScriptedProvider(),
        token_counter=TOKENS,
    )


def v1(provider=None, **config) -> Answerer:
    return Answerer(
        GenerationConfig(**config), provider or ScriptedProvider(), token_counter=TOKENS
    )


# ---------------------------------------------------------------------------
# EvidencePack
# ---------------------------------------------------------------------------


class TestEvidencePack:
    def test_numbers_follow_retrieval_order(self):
        pack = build_evidence_pack(RETRIEVED, budget=6000, tokens=TOKENS)
        assert [s.number for s in pack.sources] == [1, 2, 3, 4]
        assert [s.scored.chunk.chunk_id for s in pack.sources] == ["a6", "b9", "c6", "d3"]

    def test_sources_are_displayed_grouped_by_page_best_rank_first(self):
        body = build_evidence_pack(RETRIEVED, budget=6000, tokens=TOKENS).render()
        groups = body.split(GROUP_SEPARATOR)
        assert [
            [line.split(" ")[0] for line in g.split("\n") if line.startswith("[")] for g in groups
        ] == [["[1]", "[3]"], ["[2]"], ["[4]"]]

    def test_header_format_and_breadcrumb_removed(self):
        block, _ = format_source(1, RETRIEVED[0])
        assert block.split("\n")[0] == f"[1] {TITLE} · page 6 · text · 3.5 Positional Encoding"
        assert f"{TITLE} > " not in block
        assert block.endswith("Positional encodings are added to the input embeddings.")

    def test_header_omits_a_missing_section(self):
        item = scored("x", 2, rank=1, section=None, header=False, body="Plain.")
        assert format_source(5, item)[0].split("\n")[0] == f"[5] {TITLE} · page 2 · text"

    def test_table_and_figure_captions_are_labelled_first(self):
        table, caption = format_source(2, RETRIEVED[1])
        assert (
            table.split("\n")[1] == "Caption: Table 3: Variations on the Transformer architecture."
        )
        assert caption == "Table 3: Variations on the Transformer architecture."
        figure, _ = format_source(4, RETRIEVED[3])
        assert figure.split("\n")[1] == "Caption: Figure 1: The Transformer - model architecture."

    def test_a_later_caption_is_moved_up_without_losing_any_line(self):
        item = scored(
            "f",
            4,
            rank=1,
            kind=ChunkType.FIGURE,
            body="Note: values are medians.\nFigure 7. Projected path\naxis 2020 2030",
        )
        block, caption = format_source(1, item)
        body_lines = block.split("\n")[1:]
        assert body_lines[0] == "Caption: Figure 7. Projected path"
        assert sorted(line.removeprefix("Caption: ") for line in body_lines) == sorted(
            ["Note: values are medians.", "Figure 7. Projected path", "axis 2020 2030"]
        )
        assert caption == "Figure 7. Projected path"

    def test_text_chunks_never_get_a_caption(self):
        item = scored("t", 1, rank=1, body="Table 2 lists the results.")
        assert format_source(1, item) == (
            f"[1] {TITLE} · page 1 · text · 3.5 Positional Encoding\nTable 2 lists the results.",
            None,
        )

    def test_budget_rule_matches_v1_skip_but_keep_going(self):
        items = [
            scored("first", 1, rank=1, body="short " * 5),
            scored("huge", 2, rank=2, body="word " * 3000),
            scored("small", 3, rank=3, body="tiny " * 5),
        ]
        pack = build_evidence_pack(items, budget=200, tokens=TOKENS)
        assert [s.scored.chunk.chunk_id for s in pack.sources] == ["first", "small"]
        assert [s.number for s in pack.sources] == [1, 2]
        assert pack.dropped_for_budget == 1
        v1_prompt = v1(max_context_tokens=256).build_prompt("q", items)
        assert [s.chunk.chunk_id for s in v1_prompt.sources] == ["first", "small"]

    def test_the_first_source_survives_an_impossible_budget(self):
        pack = build_evidence_pack(
            [scored("big", 1, rank=1, body="word " * 3000)], budget=1, tokens=TOKENS
        )
        assert len(pack.sources) == 1

    def test_no_sources(self):
        pack = build_evidence_pack([], budget=6000, tokens=TOKENS)
        assert pack.render() == NO_SOURCES and pack.scored == []


# ---------------------------------------------------------------------------
# AnswererV2
# ---------------------------------------------------------------------------


class TestAnswererV2:
    def test_same_sources_and_citation_resolution_as_v1(self):
        answer = "Positional encodings are added [1]. BLEU is in Table 3 [2][4]."
        old = v1().build_prompt("q", RETRIEVED)
        new = v2().build_prompt("q", RETRIEVED)
        assert [s.chunk.chunk_id for s in new.sources] == [s.chunk.chunk_id for s in old.sources]
        assert resolve_citations(answer, new.sources) == resolve_citations(answer, old.sources)

    def test_exactly_one_call_with_v1_settings(self):
        provider = ScriptedProvider({"Question": "Positional encodings are added [1]."})
        config = {"text_model": "gpt-4o-mini", "temperature": 0.0, "max_output_tokens": 1024}
        v2(provider, **config).answer("What is positional encoding? Question", RETRIEVED)
        old = ScriptedProvider({"Question": "x [1]."})
        v1(old, **config).answer("What is positional encoding? Question", RETRIEVED)
        assert len(provider.calls) == 1
        keys = ("model", "temperature", "max_output_tokens", "seed")
        assert {k: provider.calls[0][k] for k in keys} == {k: old.calls[0][k] for k in keys}

    def test_the_answer_contract(self):
        system = SYSTEM_PROMPT_V2
        assert "Be concise" not in system and "describe the sources" not in system
        for phrase in (
            "direct answer",
            "units",
            "which table, figure or section",
            "answer only part",
            "Only if no source contains information relevant",
            "{refusal}",
            "Do not use prior knowledge",
            "[2] or [1][3]",
            "exactly as they appear",
            "2 to 6 sentences",
        ):
            assert phrase in system, phrase
        assert "{refusal}" not in SYSTEM_PROMPT_V2_NO_REFUSAL

    def test_the_refusal_marker_is_filled_in_and_configurable(self):
        assert (
            "exactly INSUFFICIENT_EVIDENCE" in v2().build_prompt("q", RETRIEVED).messages[0].content
        )
        assert (
            "INSUFFICIENT_EVIDENCE"
            not in v2(refuse_without_evidence=False)
            .build_prompt("q", RETRIEVED)
            .messages[0]
            .content
        )

    def test_sources_block_is_exactly_what_the_user_message_carries(self):
        prompt = v2().build_prompt("Which table?", RETRIEVED)
        assert isinstance(prompt, PromptBuildV2)
        assert prompt.messages[1].content == (
            f"Sources:\n\n{prompt.sources_block}\n\nQuestion: Which table?\n\nAnswer:"
        )
        assert sources_block_of(prompt, "Which table?") == prompt.sources_block

    def test_answer_is_unchanged_and_carries_a_validation_report(self):
        text = "Encodings use sine functions [3]. The BLEU is 99.9 [2]."
        answer = v2(ScriptedProvider({"Question": text})).answer(
            "Question", RETRIEVED, method="method2"
        )
        assert answer.text == text
        meta = answer.metadata
        assert meta["pipeline"] == "v2" and meta["answer_chars"] == len(text)
        assert meta["validation"]["issues"] == ["ungrounded_identifier"]
        assert [c.chunk_id for c in answer.citations] == ["c6", "b9"]

    def test_prompt_versions_are_distinct_and_selectable(self):
        assert PROMPT_VERSION_V2 != PROMPT_VERSION == prompt_version_for("v1")
        assert prompt_version_for("v2") == PROMPT_VERSION_V2
        assert isinstance(
            build_answerer(GenerationConfig(pipeline="v2"), ScriptedProvider()), AnswererV2
        )
        with pytest.raises(ValueError, match="pipeline"):
            prompt_version_for("v3")


# ---------------------------------------------------------------------------
# AnswerValidator
# ---------------------------------------------------------------------------


def validate(text, sources=RETRIEVED):
    return AnswerValidator().validate(text, sources)


class TestValidator:
    def test_a_well_cited_grounded_answer_passes(self):
        report = validate(
            "Table 3 reports BLEU of 25.8 for N of 6 [2]. "
            "Encodings use sine and cosine functions [3]."
        )
        assert report.passed and report.sentence_citation_coverage == 1.0
        assert report.identifiers_checked == 3 and report.ungrounded_identifiers == []

    def test_uncited_factual_sentences_are_reported(self):
        report = validate("Encodings use sine functions [3]. They are added to embeddings.")
        assert report.uncited_sentences == [1] and "uncited_sentence" in report.issues
        assert report.sentence_citation_coverage == 0.5

    def test_citations_after_the_full_stop_belong_to_the_sentence_before(self):
        assert answer_sentences("Encodings use sine functions. [3] BLEU is 25.8. [2]") == [
            "Encodings use sine functions. [3]",
            "BLEU is 25.8. [2]",
        ]
        assert validate("Encodings use sine functions. [3] BLEU is 25.8. [2]").passed

    def test_statements_about_missing_evidence_need_no_citation(self):
        report = validate(
            "Encodings use sine functions [3]. The sources do not state the learning rate."
        )
        assert report.n_factual_sentences == 1 and report.passed

    def test_unresolved_citations(self):
        report = validate("Encodings use sine functions [9].")
        assert report.unresolved_citations == [9] and "unresolved_citation" in report.issues

    def test_identifiers_must_be_in_a_cited_source(self):
        # 25.8 is in source 2, but this sentence cites source 1.
        report = validate("The BLEU score is 25.8 [1].")
        assert report.ungrounded_identifiers == [{"token": "25.8", "sentence": 0, "cited": [1]}]

    def test_an_uncited_identifier_is_checked_against_every_source(self):
        report = validate("The best BLEU score was 25.8 overall.")
        assert report.ungrounded_identifiers == [] and report.uncited_sentences == [0]

    def test_comma_formatted_numbers_match_either_way(self):
        source = [scored("n", 1, rank=1, body="Float reached 169000 dollars.")]
        assert validate("Float reached 169,000 dollars [1].", source).passed

    def test_identifier_tokens(self):
        assert identifier_tokens("Figure SPM.6 shows 2.9% and 22,360 on the A100 [3].") == [
            "SPM.6",
            "2.9%",
            "22,360",
            "A100",
            "3",
        ]

    def test_mixed_refusal_and_plain_refusal(self):
        assert validate("INSUFFICIENT_EVIDENCE. The sources do not say.").passed
        mixed = validate("INSUFFICIENT_EVIDENCE. Encodings use sine functions [3].")
        assert mixed.mixed_refusal and "mixed_refusal" in mixed.issues

    def test_the_validator_only_reports(self):
        text = "Encodings use sine functions [9]."
        report = validate(text)
        assert report.sentences[0].text == text and not report.passed


# ---------------------------------------------------------------------------
# Evaluation: isolation and round trips
# ---------------------------------------------------------------------------


def work():
    return select_work(RUN, GOLD, GEN_GOLD, EVAL_CHUNKS, include_unanswerable=False)


class TestEvaluation:
    def test_v1_and_v2_cache_keys_never_collide(self, tmp_path):
        messages = [Message(role="user", content="identical")]
        keys = {
            pipeline: CachingProvider(
                ScriptedProvider(),
                tmp_path,
                kind="generation",
                prompt_version=prompt_version_for(pipeline),
                seed=42,
            ).key_for(messages, model="gpt-4o-mini")
            for pipeline in ("v1", "v2")
        }
        assert keys["v1"] != keys["v2"]

    def test_a_v1_cache_is_never_replayed_for_v2(self, tmp_path):
        provider = ScriptedProvider({"What is the answer?": "The answer is 42 percent [2]."})
        for pipeline in ("v1", "v2"):
            cache = CachingProvider(
                provider,
                tmp_path,
                kind="generation",
                prompt_version=prompt_version_for(pipeline),
                seed=42,
            )
            answerer = build_answerer(
                GenerationConfig(pipeline=pipeline), cache, token_counter=TOKENS
            )
            run_generation(work()[:1], answerer, cache, concurrency=1, backoff_s=0)
            assert cache.stats.hits == 0 and cache.stats.misses == 1

    def test_v2_records_carry_pipeline_validation_and_shape(self, tmp_path):
        provider = ScriptedProvider(
            {"What is the answer?": "The answer is 42 percent in 2023 [2]."}
        )
        cache = CachingProvider(
            provider, tmp_path, kind="generation", prompt_version=PROMPT_VERSION_V2, seed=42
        )
        records = run_generation(work(), v2(cache), cache, concurrency=1, backoff_s=0)
        first = records[0]
        assert first.pipeline == "v2" and first.answer_chars == len(first.answer)
        assert first.answer_sentences == 1 and first.validation["passed"] is True
        assert "sentences" not in first.validation
        sent = provider.calls[0]["messages"][1][1]
        assert (
            sent == f"Sources:\n\n{first.sources_block}\n\nQuestion: What is the answer?\n\nAnswer:"
        )
        assert " · page 3 · text" in first.sources_block
        metrics = aggregate(records)
        assert metrics["validation"]["n"] == 2 and "answer_shape" in metrics

    def test_v1_records_are_marked_v1_and_keep_the_v1_source_text(self, tmp_path):
        provider = ScriptedProvider({"What is the answer?": "The answer is 42 percent [2]."})
        cache = CachingProvider(
            provider, tmp_path, kind="generation", prompt_version=PROMPT_VERSION, seed=42
        )
        record = run_generation(work()[:1], v1(cache), cache, concurrency=1, backoff_s=0)[0]
        assert record.pipeline == "v1"
        assert record.sources_block.startswith("[1] DOC - page 9 (text)")

    def test_runs_round_trip_and_old_records_still_load(self, tmp_path):
        provider = ScriptedProvider({"What is the answer?": "The answer is 42 percent [2]."})
        cache = CachingProvider(
            provider, tmp_path, kind="generation", prompt_version=PROMPT_VERSION_V2, seed=42
        )
        records = run_generation(work(), v2(cache), cache, concurrency=1, backoff_s=0)
        run = GenerationRun(
            method="method1",
            label="method1/rerank+genv2",
            created_at="t",
            elapsed_s=0.0,
            generation={"pipeline": "v2"},
            records=records,
        )
        loaded = GenerationRun.load(run.save(tmp_path / "run.json"))
        assert loaded.records[0].validation == records[0].validation

        legacy = records[0].model_dump(mode="json")
        for key in ("pipeline", "answer_chars", "answer_sentences", "validation"):
            legacy.pop(key)
        old = GenerationRecord.model_validate(legacy)
        assert old.pipeline is None and old.validation is None
        assert aggregate([old])["validation"]["n"] == 0


# ---------------------------------------------------------------------------
# V2.1: wrong-entity and sentence-citation prompt fixes
# ---------------------------------------------------------------------------

V2_0_PROMPT_VERSION = "55aea920ebb01b97"  # the version the V2.0 runs were generated with


class TestV21:
    def test_v2_0_is_unchanged_and_isolated(self):
        from mmrag.generation.answerer_v2 import PROMPT_VERSION_V2_1

        assert PROMPT_VERSION_V2 == V2_0_PROMPT_VERSION == prompt_version_for("v2")
        assert len({PROMPT_VERSION, PROMPT_VERSION_V2, PROMPT_VERSION_V2_1}) == 3
        assert prompt_version_for("v2.1") == PROMPT_VERSION_V2_1
        assert "Only if no source contains information relevant" in SYSTEM_PROMPT_V2

    def test_the_wrong_entity_rule(self):
        from mmrag.generation.answerer_v2 import SYSTEM_PROMPT_V2_1

        for phrase in (
            "Answer only about the subject the question asks about",
            "use only evidence about that same subject",
            "Never substitute a similar, related, newer, older or otherwise different one",
            "only part of the question about that same subject",
            "including when they only describe a different entity, version or year, reply with "
            "exactly {refusal}",
        ):
            assert phrase in SYSTEM_PROMPT_V2_1, phrase
        # Generic: no product, model or query named.
        for specific in ("RP2", "u003", "GPIO"):
            assert specific not in SYSTEM_PROMPT_V2_1

    def test_the_sentence_citation_rule(self):
        from mmrag.generation.answerer_v2 import SYSTEM_PROMPT_V2_1, SYSTEM_PROMPT_V2_1_NO_REFUSAL

        for phrase in (
            "End every sentence that states a fact with at least one source number",
            "naming the sources that support that sentence",
            "Each sentence needs its own citation",
            "does not cover earlier sentences",
            "leave the detail out rather than stating it without a citation",
            "or the {refusal} reply, needs no citation",
        ):
            assert phrase in SYSTEM_PROMPT_V2_1, phrase
        assert "{refusal}" not in SYSTEM_PROMPT_V2_1_NO_REFUSAL
        assert "Each sentence needs its own citation" in SYSTEM_PROMPT_V2_1_NO_REFUSAL

    def test_v2_1_uses_the_same_evidence_pack_and_call(self):
        provider = ScriptedProvider({"Question": "Encodings use sine functions [3]."})
        old = v2(ScriptedProvider({"Question": "Encodings use sine functions [3]."}))
        new = AnswererV2(
            GenerationConfig(pipeline="v2.1"), provider, token_counter=TOKENS, variant="v2.1"
        )
        p_old, p_new = (
            old.build_prompt("Question", RETRIEVED),
            new.build_prompt("Question", RETRIEVED),
        )
        assert p_new.sources_block == p_old.sources_block
        assert p_new.messages[1].content == p_old.messages[1].content
        assert p_new.messages[0].content != p_old.messages[0].content
        assert "exactly INSUFFICIENT_EVIDENCE" in p_new.messages[0].content
        assert (p_old.pipeline, p_new.pipeline) == ("v2", "v2.1")

        answer = new.answer("Question", RETRIEVED)
        assert len(provider.calls) == 1 and answer.metadata["pipeline"] == "v2.1"

    def test_the_factory_selects_v2_1(self):
        answerer = build_answerer(GenerationConfig(pipeline="v2.1"), ScriptedProvider())
        assert isinstance(answerer, AnswererV2) and answerer.pipeline == "v2.1"
        with pytest.raises(ValueError, match="variant"):
            AnswererV2(GenerationConfig(), ScriptedProvider(), variant="v9")

    def test_v2_v2_1_and_v1_cache_keys_are_distinct(self, tmp_path):
        messages = [Message(role="user", content="identical")]
        keys = {
            pipeline: CachingProvider(
                ScriptedProvider(),
                tmp_path,
                kind="generation",
                prompt_version=prompt_version_for(pipeline),
                seed=42,
            ).key_for(messages, model="gpt-4o-mini")
            for pipeline in ("v1", "v2", "v2.1")
        }
        assert len(set(keys.values())) == 3

    def test_v2_1_records_carry_their_own_pipeline(self, tmp_path):
        provider = ScriptedProvider({"What is the answer?": "The answer is 42 percent [2]."})
        cache = CachingProvider(
            provider,
            tmp_path,
            kind="generation",
            prompt_version=prompt_version_for("v2.1"),
            seed=42,
        )
        answerer = build_answerer(GenerationConfig(pipeline="v2.1"), cache, token_counter=TOKENS)
        record = run_generation(work()[:1], answerer, cache, concurrency=1, backoff_s=0)[0]
        assert record.pipeline == "v2.1" and cache.stats.hits == 0


class TestValidatorMetaStatements:
    def test_identifiers_in_a_refusal_are_not_grounding_failures(self):
        report = validate("INSUFFICIENT_EVIDENCE. The sources do not mention the H100 or GPT-4.")
        assert report.ungrounded_identifiers == [] and report.passed
        assert report.sentences[1].identifiers == []

    def test_identifiers_in_a_statement_about_missing_evidence_are_skipped(self):
        report = validate(
            "Encodings use sine functions [3]. The sources do not state the "
            "2017 learning rate for the RTX 4090."
        )
        assert report.ungrounded_identifiers == [] and report.passed

    def test_a_cited_sentence_is_still_checked_even_if_it_mentions_the_sources(self):
        report = validate("The sources do not state 99.9, but report BLEU of 25.8 [2].")
        assert [u["token"] for u in report.ungrounded_identifiers] == ["99.9"]

    def test_a_short_factual_claim_is_still_checked(self):
        report = validate("It has 99 pins [1].")
        assert [u["token"] for u in report.ungrounded_identifiers] == ["99"]

    def test_uncited_factual_sentences_are_still_caught(self):
        report = validate(
            "Encodings use sine and cosine functions [3]. Table 3 reports a BLEU of 25.8."
        )
        assert report.uncited_sentences == [1] and "uncited_sentence" in report.issues
