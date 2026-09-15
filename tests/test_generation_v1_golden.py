"""Generation V1 is frozen: its prompt, its version and the exact message it sends.

The published V1 generation and judging results were produced with prompt
version 1ae772d0aa197ff5. Generation V2 was added beside V1, not in place of it,
so none of these may change. If one of these tests fails, V1 has been altered:
revert that change rather than updating the snapshot.
"""

from __future__ import annotations

import hashlib

from mmrag.config import ExperimentConfig, GenerationConfig
from mmrag.evaluation.generation_eval import PROMPT_VERSION
from mmrag.generation import build_answerer, prompt_version_for
from mmrag.generation.answerer import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_REFUSAL, Answerer
from mmrag.schemas import BBox, Chunk, ChunkType, Modality, ScoredChunk
from mmrag.textify.tokens import HeuristicTokenCounter

V1_PROMPT_VERSION = "1ae772d0aa197ff5"
V1_SYSTEM_SHA256 = "24968e7cde8774497cf058539846b83f19e1858406776d8c64e0ba8fe7e5bcb7"
V1_SYSTEM_NO_REFUSAL_SHA256 = "36c5d221af6f63db5c16c3e28827bc212d33d9e8f06e3e001435edef13814f0d"

GOLDEN_SYSTEM = """You answer questions using only the numbered sources provided.

Rules:
- Use only information present in the sources. Do not use prior knowledge.
- Cite every factual claim with the source number in square brackets, like [2].
- A claim drawn from several sources cites each one: [1][3].
- Quote figures, dates and names exactly as they appear in the sources.
- If the sources do not contain the answer, reply with exactly INSUFFICIENT_EVIDENCE and one sentence saying what is missing. Do not guess.
- Be concise. Do not restate the question or describe the sources."""  # noqa: E501

GOLDEN_USER = """Sources:

[1] Attention Is All You Need - page 9 (table)
Attention Is All You Need > 6.2 Model Variations

Table 3: Variations on the Transformer architecture.
| N | BLEU |
| --- | --- |
| 6 | 25.8 |

[2] Attention Is All You Need - page 6 (text)
Attention Is All You Need > 3.5 Positional Encoding

We add positional encodings to the input embeddings.

Question: Which table reports BLEU for the variations?

Answer:"""


def _scored(cid: str, page: int, kind: ChunkType, header: str, body: str, rank: int) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id="arxiv_attention",
        page_number=page,
        chunk_type=kind,
        text=f"{header}\n\n{body}",
        element_ids=[f"arxiv_attention#p{page}#x000"],
        bbox=BBox(x0=0.1, y0=0.1, x1=0.9, y1=0.4),
        section=header.split(" > ")[-1],
        variant="method2",
        metadata={"doc_title": "Attention Is All You Need", "context_header": header},
    )
    return ScoredChunk(
        chunk=chunk, score=1.0 / rank, rank=rank, retriever="stub", modality=Modality.TEXT
    )


RETRIEVED = [
    _scored(
        "t",
        9,
        ChunkType.TABLE,
        "Attention Is All You Need > 6.2 Model Variations",
        "Table 3: Variations on the Transformer architecture.\n"
        "| N | BLEU |\n| --- | --- |\n| 6 | 25.8 |",
        1,
    ),
    _scored(
        "p",
        6,
        ChunkType.TEXT,
        "Attention Is All You Need > 3.5 Positional Encoding",
        "We add positional encodings to the input embeddings.",
        2,
    ),
]


class TestV1IsFrozen:
    def test_prompt_version_is_the_one_the_published_results_used(self):
        assert PROMPT_VERSION == V1_PROMPT_VERSION
        assert prompt_version_for("v1") == V1_PROMPT_VERSION

    def test_system_prompts_are_byte_identical(self):
        assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == V1_SYSTEM_SHA256
        assert (
            hashlib.sha256(SYSTEM_PROMPT_NO_REFUSAL.encode()).hexdigest()
            == V1_SYSTEM_NO_REFUSAL_SHA256
        )

    def test_the_exact_messages_v1_sends(self):
        answerer = Answerer(
            GenerationConfig(),
            provider=None,  # type: ignore[arg-type]
            token_counter=HeuristicTokenCounter(),
        )
        prompt = answerer.build_prompt("Which table reports BLEU for the variations?", RETRIEVED)
        assert prompt.messages[0].content == GOLDEN_SYSTEM
        assert prompt.messages[1].content == GOLDEN_USER
        assert [s.chunk.chunk_id for s in prompt.sources] == ["t", "p"]

    def test_v1_is_the_default_pipeline(self):
        assert GenerationConfig().pipeline == "v1"
        assert isinstance(build_answerer(GenerationConfig(), provider=None), Answerer)  # type: ignore[arg-type]

    def test_configs_saved_before_v2_existed_load_as_v1(self):
        saved = ExperimentConfig().model_dump(mode="json")
        del saved["generation"]["pipeline"]
        assert ExperimentConfig.model_validate(saved).generation.pipeline == "v1"
