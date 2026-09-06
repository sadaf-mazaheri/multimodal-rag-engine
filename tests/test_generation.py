"""Tests for prompt assembly, providers, and citation resolution.

Citation resolution is the part that matters most. A citation is the claim that
an answer is grounded in a specific region of a specific page, and Step 6 will
score exactly that. A resolver that quietly accepts a citation pointing at
nothing would make every grounding metric meaningless while looking perfect.

Nothing here calls a network: the echo provider exists so the whole generation
path can be exercised for free.
"""

from __future__ import annotations

import pytest

from mmrag.config import GenerationConfig, Settings
from mmrag.generation.answerer import (
    REFUSAL_MARKER,
    Answerer,
    is_refusal,
    parse_citation_numbers,
    resolve_citations,
)
from mmrag.generation.providers import EchoProvider, ProviderError, get_provider
from mmrag.generation.providers.base import ImageInput, Message, Usage
from mmrag.schemas import BBox, Chunk, ChunkType, Modality, ScoredChunk


def _scored(
    n: int,
    *,
    text: str = "Revenue rose to 22,360 million.",
    doc: str = "doc1",
    page: int = 3,
    title: str = "Annual Report 2024",
    chunk_type: ChunkType = ChunkType.TEXT,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"method1#{n:04d}",
            doc_id=doc,
            page_number=page,
            chunk_type=chunk_type,
            text=text,
            element_ids=[f"{doc}#p{page}#text{n:03d}"],
            bbox=BBox(x0=0.1, y0=0.2, x1=0.9, y1=0.4),
            section="Financial Review",
            metadata={"doc_title": title, "context_header": ""},
        ),
        score=1.0 / n,
        rank=n,
        retriever="bm25+dense",
        modality=Modality.TEXT,
    )


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------


class TestParseCitationNumbers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Revenue rose [1].", [1]),
            ("Both sources agree [1][3].", [1, 3]),
            ("Combined form [1, 2].", [1, 2]),
            ("Spaced [1 , 2].", [1, 2]),
            ("No citations here.", []),
        ],
    )
    def test_parses_the_common_forms(self, text, expected):
        assert parse_citation_numbers(text) == expected

    def test_deduplicates_but_keeps_first_appearance_order(self):
        assert parse_citation_numbers("[3] then [1] then [3] again.") == [3, 1]

    def test_ignores_bracketed_non_numbers(self):
        assert parse_citation_numbers("See [Figure 3] and [a].") == []


class TestResolveCitations:
    def test_maps_numbers_to_chunk_provenance(self):
        sources = [_scored(1), _scored(2)]
        citations, unresolved = resolve_citations("Revenue rose [1].", sources)

        assert unresolved == []
        assert len(citations) == 1
        citation = citations[0]
        assert citation.doc_id == "doc1"
        assert citation.doc_title == "Annual Report 2024"
        assert citation.page_number == 3
        assert citation.chunk_id == "method1#0001"
        assert citation.bbox is not None
        assert citation.element_ids == ["doc1#p3#text001"]

    def test_out_of_range_citations_are_reported_not_silently_dropped(self):
        """A model citing [7] when five sources were given is a grounding failure."""
        citations, unresolved = resolve_citations("As shown [7].", [_scored(1)])
        assert citations == []
        assert unresolved == [7]

    def test_zero_is_out_of_range(self):
        _, unresolved = resolve_citations("See [0].", [_scored(1)])
        assert unresolved == [0]

    def test_valid_and_invalid_citations_are_separated(self):
        citations, unresolved = resolve_citations("Both [1] and [9].", [_scored(1)])
        assert len(citations) == 1 and unresolved == [9]

    def test_snippet_strips_the_breadcrumb_header(self):
        """The header was for the embedder; showing it as source text misleads."""
        source = _scored(1)
        source.chunk.metadata["context_header"] = "Annual Report 2024 > Financial Review"
        source.chunk.text = "Annual Report 2024 > Financial Review\n\nRevenue rose 8%."
        citations, _ = resolve_citations("[1]", [source])
        assert citations[0].snippet == "Revenue rose 8%."

    def test_snippet_is_truncated(self):
        source = _scored(1, text="word " * 400)
        citations, _ = resolve_citations("[1]", [source])
        assert len(citations[0].snippet) <= 241

    def test_no_sources_resolves_nothing(self):
        citations, unresolved = resolve_citations("[1]", [])
        assert citations == [] and unresolved == [1]


class TestRefusal:
    def test_detects_the_marker(self):
        assert is_refusal(f"{REFUSAL_MARKER} the sources do not give a figure.")

    def test_ordinary_answers_are_not_refusals(self):
        assert not is_refusal("Revenue rose to 22,360 million [1].")


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


@pytest.fixture
def answerer() -> Answerer:
    return Answerer(GenerationConfig(), EchoProvider())


class TestPromptBuilding:
    def test_numbers_sources_from_one_in_retrieval_order(self, answerer):
        sources = [_scored(1, text="First fact."), _scored(2, text="Second fact.")]
        build = answerer.build_prompt("What happened?", sources)
        user = build.messages[1].content
        assert "[1] Annual Report 2024 - page 3 (text)\nFirst fact." in user
        assert "[2] Annual Report 2024 - page 3 (text)\nSecond fact." in user

    def test_includes_the_question(self, answerer):
        build = answerer.build_prompt("What was revenue?", [_scored(1)])
        assert "Question: What was revenue?" in build.messages[1].content

    def test_source_header_names_the_modality(self, answerer):
        """A table read as prose is far likelier to be misread unlabelled."""
        build = answerer.build_prompt("q", [_scored(1, chunk_type=ChunkType.TABLE)])
        assert "(table)" in build.messages[1].content

    def test_system_prompt_demands_citations(self, answerer):
        build = answerer.build_prompt("q", [_scored(1)])
        system = build.messages[0].content
        assert "square brackets" in system
        assert "Do not use prior knowledge" in system

    def test_refusal_instruction_is_configurable(self):
        with_refusal = Answerer(GenerationConfig(refuse_without_evidence=True), EchoProvider())
        without = Answerer(GenerationConfig(refuse_without_evidence=False), EchoProvider())
        assert REFUSAL_MARKER in with_refusal.build_prompt("q", [_scored(1)]).messages[0].content
        assert REFUSAL_MARKER not in without.build_prompt("q", [_scored(1)]).messages[0].content

    def test_over_budget_sources_are_dropped_from_the_tail(self):
        """The best-ranked evidence must never be what gets cut."""
        answerer = Answerer(GenerationConfig(max_context_tokens=256), EchoProvider())
        sources = [_scored(n, text=f"Fact {n}. " + "filler " * 200) for n in range(1, 8)]
        build = answerer.build_prompt("q", sources)

        assert build.dropped_for_budget > 0
        assert build.sources[0].chunk.chunk_id == sources[0].chunk.chunk_id
        assert len(build.sources) < len(sources)

    def test_at_least_one_source_survives_an_impossible_budget(self):
        answerer = Answerer(GenerationConfig(max_context_tokens=256), EchoProvider())
        build = answerer.build_prompt("q", [_scored(1, text="x " * 5000)])
        assert len(build.sources) == 1

    def test_no_sources_is_handled(self, answerer):
        build = answerer.build_prompt("q", [])
        assert "(no sources retrieved)" in build.messages[1].content
        assert build.sources == []

    def test_images_are_attached_to_the_user_message(self, answerer, tmp_path):
        image = tmp_path / "page.png"
        image.write_bytes(b"\x89PNG\r\n")
        build = answerer.build_prompt("q", [_scored(1)], images=[ImageInput(path=image)])
        assert len(build.messages[1].images) == 1
        assert build.metadata["n_images"] == 1


# ---------------------------------------------------------------------------
# End-to-end generation via the echo provider
# ---------------------------------------------------------------------------


class TestAnswerGeneration:
    def test_produces_a_cited_answer(self, answerer):
        sources = [_scored(1), _scored(2)]
        answer = answerer.answer("What was revenue?", sources, method="method1")

        assert answer.query == "What was revenue?"
        assert answer.method == "method1"
        assert answer.citations, "the echo provider cites its sources so parsing is tested"
        assert all(c.doc_id == "doc1" for c in answer.citations)

    def test_records_provider_and_accounting_metadata(self, answerer):
        answer = answerer.answer("q", [_scored(1)])
        assert answer.metadata["provider"] == "echo"
        assert answer.metadata["n_sources"] == 1
        assert answer.metadata["unresolved_citations"] == []
        assert answer.usage["total_tokens"] >= 0
        assert "generation_ms" in answer.latency_ms

    def test_retrieved_chunks_are_kept_on_the_answer(self, answerer):
        """Step 6 scores retrieval and generation separately; it needs both."""
        sources = [_scored(1), _scored(2)]
        answer = answerer.answer("q", sources)
        assert len(answer.retrieved) == 2

    def test_answering_with_no_evidence_still_returns_an_answer(self, answerer):
        answer = answerer.answer("q", [])
        assert answer.text
        assert answer.citations == []


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class TestEchoProvider:
    def test_is_deterministic(self):
        provider = EchoProvider()
        messages = [Message(role="user", content="[1] source\n\nQuestion: q")]
        first = provider.complete(messages, model="echo")
        second = provider.complete(messages, model="echo")
        assert first.text == second.text

    def test_reports_how_many_sources_and_images_it_saw(self, tmp_path):
        image = tmp_path / "p.png"
        image.write_bytes(b"\x89PNG\r\n")
        provider = EchoProvider()
        completion = provider.complete(
            [Message(role="user", content="[1] a\n[2] b", images=[ImageInput(path=image)])],
            model="echo",
        )
        assert completion.metadata["n_sources"] == 2
        assert completion.metadata["n_images"] == 1

    def test_declares_image_support_so_method3_is_exercised(self):
        assert EchoProvider().supports_images()


class TestProviderRegistry:
    def test_echo_needs_no_credentials(self):
        assert get_provider(Settings(_env_file=None), name="echo").name == "echo"

    def test_missing_openai_key_fails_with_actionable_guidance(self):
        settings = Settings(_env_file=None, MMRAG_GENERATION_PROVIDER="openai")
        settings.openai_api_key = None
        with pytest.raises(ProviderError, match="--provider echo"):
            get_provider(settings)

    def test_placeholder_key_is_treated_as_unset(self):
        settings = Settings(_env_file=None, OPENAI_API_KEY="sk-replace-me")
        with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
            get_provider(settings, name="openai")

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ProviderError, match="unknown generation provider"):
            get_provider(Settings(_env_file=None), name="nonsense")


class TestUsage:
    def test_totals_are_summed(self):
        usage = Usage(prompt_tokens=100, completion_tokens=25)
        assert usage.total_tokens == 125
        assert usage.as_dict()["total_tokens"] == 125
