"""The generation gold sidecar: schema and cross-validation against v1."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from mmrag.evaluation.generation_gold import (
    GenerationGold,
    describe,
    validate_against_retrieval_gold,
)
from mmrag.evaluation.gold import GoldSet

RETRIEVAL = GoldSet.model_validate({
    "version": 1,
    "queries": [
        {"id": "q001", "query": "a", "stratum": "text", "requires": "text",
         "evidence": [{"doc_id": "d", "page": 1, "modality": "text"}]},
        {"id": "q002", "query": "b", "stratum": "table", "requires": "table",
         "evidence": [{"doc_id": "d", "page": 2, "modality": "table"}]},
    ],
})


def build(**overrides):
    payload = {
        "version": 1,
        "retrieval_gold": "data/eval/gold/v1.yaml",
        "queries": {
            "q001": {"required_facts": ["fact one"]},
            "q002": {"required_facts": ["2.9%", "December 2023"], "note": "NEEDS REVIEW"},
        },
        "unanswerable": [{"id": "u001", "query": "not in corpus?", "note": "checked by BM25"}],
    }
    payload.update(overrides)
    return GenerationGold.model_validate(payload)


class TestSchema:
    def test_a_valid_file_loads(self):
        assert build().facts_for("q002") == ["2.9%", "December 2023"]

    def test_facts_are_required(self):
        with pytest.raises(ValidationError):
            build(queries={"q001": {"required_facts": []}})

    def test_empty_fact_is_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            build(queries={"q001": {"required_facts": ["  "]}})

    def test_duplicate_fact_is_rejected(self):
        with pytest.raises(ValidationError, match="duplicate"):
            build(queries={"q001": {"required_facts": ["X", "x"]}})

    def test_unknown_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            build(queries={"q001": {"required_facts": ["a"], "reference_answer": "no"}})

    def test_unanswerable_id_shape(self):
        with pytest.raises(ValidationError, match="u001"):
            build(unanswerable=[{"id": "q999", "query": "x", "note": "y"}])

    def test_unanswerable_needs_a_note(self):
        with pytest.raises(ValidationError):
            build(unanswerable=[{"id": "u001", "query": "x", "note": ""}])

    def test_duplicate_unanswerable_ids(self):
        with pytest.raises(ValidationError, match="duplicate"):
            build(unanswerable=[{"id": "u001", "query": "x", "note": "y"},
                                {"id": "u001", "query": "z", "note": "y"}])

    def test_needs_review_is_reported(self):
        assert build().needs_review() == ["q002"]

    def test_describe(self):
        info = describe(build())
        assert (info["n_answerable"], info["n_unanswerable"], info["n_required_facts"]) == (2, 1, 3)

    def test_roundtrips_through_yaml(self, tmp_path):
        path = tmp_path / "g.yaml"
        path.write_text(yaml.safe_dump(build().model_dump(mode="json")), encoding="utf-8")
        assert GenerationGold.load(path).facts_for("q001") == ["fact one"]


class TestCrossValidation:
    def test_complete_and_consistent(self):
        assert validate_against_retrieval_gold(build(), RETRIEVAL) == []

    def test_a_retrieval_query_without_facts(self):
        problems = validate_against_retrieval_gold(
            build(queries={"q001": {"required_facts": ["a"]}}), RETRIEVAL)
        assert [(p.kind, p.query_id) for p in problems] == [("missing_facts", "q002")]

    def test_facts_for_a_query_that_does_not_exist(self):
        gen = build(queries={"q001": {"required_facts": ["a"]}, "q002": {"required_facts": ["b"]},
                             "q099": {"required_facts": ["c"]}})
        assert [p.kind for p in validate_against_retrieval_gold(gen, RETRIEVAL)] == ["unknown_query"]


class TestRealFile:
    def test_the_committed_sidecar_is_valid_and_complete(self):
        from pathlib import Path

        path = Path("data/eval/gold/generation_v1.yaml")
        if not path.exists():
            pytest.skip("generation_v1.yaml not authored yet")
        generation = GenerationGold.load(path)
        retrieval = GoldSet.load(generation.retrieval_gold)
        assert validate_against_retrieval_gold(generation, retrieval) == []
        assert generation.unanswerable, "refusal behaviour needs unanswerable queries"
