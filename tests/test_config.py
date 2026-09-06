"""Tests for configuration loading.

The committed configs are exercised directly: if `extends` inheritance silently
broke, every method would run with baseline settings and the benchmark would
compare three copies of the same pipeline while looking perfectly healthy.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mmrag.config import (
    CONFIG_DIR,
    ChunkingConfig,
    ExperimentConfig,
    RetrievalConfig,
    Settings,
    _deep_merge,
    load_experiment_config,
)


class TestDeepMerge:
    def test_override_wins_at_leaf(self):
        assert _deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    def test_nested_dicts_merge_rather_than_replace(self):
        base = {"r": {"top_k": 10, "rrf_k": 60}}
        assert _deep_merge(base, {"r": {"top_k": 5}}) == {"r": {"top_k": 5, "rrf_k": 60}}

    def test_lists_are_replaced_wholesale(self):
        """Merging lists element-wise would make overrides impossible to reason about."""
        assert _deep_merge({"k": [1, 2, 3]}, {"k": [9]}) == {"k": [9]}

    def test_base_is_not_mutated(self):
        base = {"r": {"top_k": 10}}
        _deep_merge(base, {"r": {"top_k": 1}})
        assert base == {"r": {"top_k": 10}}


class TestCommittedConfigs:
    @pytest.mark.parametrize("name", ["default", "method1", "method2", "method3"])
    def test_loads_and_validates(self, name):
        assert isinstance(load_experiment_config(name), ExperimentConfig)

    @pytest.mark.parametrize(
        ("name", "method"),
        [("method1", "method1"), ("method2", "method2"), ("method3", "method3")],
    )
    def test_method_field_matches_filename(self, name, method):
        assert load_experiment_config(name).method == method

    def test_inherits_unstated_baseline_values(self):
        """method1.yaml never mentions chunking or embeddings; it must still get them."""
        cfg = load_experiment_config("method1")
        assert cfg.chunking.target_tokens == 384
        assert cfg.embedding.text_model == "BAAI/bge-small-en-v1.5"

    def test_overrides_what_it_restates(self):
        assert load_experiment_config("method1").enrichment.ocr_enabled is True
        assert load_experiment_config("default").enrichment.ocr_enabled is False

    def test_chunk_variants_are_distinct_per_method(self):
        """Otherwise the three methods' chunks would collide in one namespace."""
        variants = {
            load_experiment_config(n).chunk_variant for n in ("method1", "method2", "method3")
        }
        assert len(variants) == 3

    def test_only_method3_uses_visual_retrieval(self):
        assert "visual_page" not in load_experiment_config("method2").retrieval.fusion_weights
        assert "visual_page" in load_experiment_config("method3").retrieval.fusion_weights

    def test_explicit_overrides_beat_the_file(self):
        cfg = load_experiment_config("method1", overrides={"retrieval": {"top_k": 3}})
        assert cfg.retrieval.top_k == 3
        assert cfg.retrieval.rrf_k == 60  # untouched siblings survive

    def test_missing_config_is_a_clear_error(self):
        with pytest.raises(FileNotFoundError):
            load_experiment_config("no_such_config")


class TestConfigChainCycles:
    def test_cycle_is_detected(self, tmp_path):
        (tmp_path / "a.yaml").write_text("extends: b.yaml\nname: a\n", encoding="utf-8")
        (tmp_path / "b.yaml").write_text("extends: a.yaml\nname: b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="circular"):
            load_experiment_config(tmp_path / "a.yaml")

    def test_multi_level_chain_resolves(self, tmp_path):
        base = CONFIG_DIR / "default.yaml"
        (tmp_path / "mid.yaml").write_text(
            f"extends: {base.as_posix()}\nretrieval:\n  top_k: 7\n", encoding="utf-8"
        )
        (tmp_path / "leaf.yaml").write_text(
            "extends: mid.yaml\nname: leaf\nretrieval:\n  rrf_k: 12\n", encoding="utf-8"
        )
        cfg = load_experiment_config(tmp_path / "leaf.yaml")
        assert (cfg.name, cfg.retrieval.top_k, cfg.retrieval.rrf_k) == ("leaf", 7, 12)


class TestValidators:
    def test_overlap_must_be_smaller_than_target(self):
        with pytest.raises(ValidationError, match="overlap_tokens"):
            ChunkingConfig(target_tokens=128, overlap_tokens=128)

    def test_candidate_pool_must_cover_top_k(self):
        """Fusion needs more candidates than it returns, or it cannot reorder anything."""
        with pytest.raises(ValidationError, match="candidates_per_retriever"):
            RetrievalConfig(top_k=50, candidates_per_retriever=10)

    def test_valid_boundary_values_are_accepted(self):
        assert RetrievalConfig(top_k=10, candidates_per_retriever=10).top_k == 10
        assert ChunkingConfig(target_tokens=128, overlap_tokens=127).overlap_tokens == 127


class TestSettings:
    def test_dsn_and_url_are_assembled_from_parts(self):
        s = Settings(
            _env_file=None,
            POSTGRES_USER="u",
            POSTGRES_PASSWORD="p",
            POSTGRES_DB="d",
            POSTGRES_HOST="h",
            POSTGRES_PORT=1234,
            QDRANT_HOST="qh",
            QDRANT_HTTP_PORT=9999,
        )
        assert s.postgres_dsn == "postgresql://u:p@h:1234/d"
        assert s.qdrant_url == "http://qh:9999"

    def test_redacted_hides_every_secret(self):
        s = Settings(_env_file=None, OPENAI_API_KEY="sk-super-secret", POSTGRES_PASSWORD="hunter2")
        blob = str(s.redacted())
        assert "sk-super-secret" not in blob
        assert "hunter2" not in blob
