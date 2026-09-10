"""Configuration.

Two distinct kinds of configuration, deliberately kept apart:

* :class:`Settings` -- *environment*: secrets, hostnames, ports. Comes from
  ``.env`` / real environment variables. Never committed, never varies between
  experiments.
* :class:`ExperimentConfig` -- *research knobs*: chunk sizes, model names,
  ``top_k``, fusion weights. Comes from a YAML file under ``configs/``.
  Always committed, and varying it is the entire point of the benchmark.

Mixing the two is the usual reason a RAG repo becomes unreproducible: you can no
longer tell which knob produced which number. Keeping them separate means a run
is fully described by "this YAML file, on this corpus".
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# src/mmrag/config.py -> src/mmrag -> src -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
EVAL_DIR = DATA_DIR / "eval"
RUNS_DIR = EVAL_DIR / "runs"


def ensure_data_dirs() -> None:
    """Create the data tree if it is missing (fresh clone, or after a wipe)."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, INDEX_DIR, EVAL_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------

ProviderName = Literal["openai", "local", "echo"]


class Settings(BaseSettings):
    """Environment-driven configuration: secrets and service endpoints."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- generation provider -------------------------------------------------
    generation_provider: ProviderName = Field(default="openai", alias="MMRAG_GENERATION_PROVIDER")
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    local_llm_base_url: str = Field(default="http://localhost:11434/v1", alias="LOCAL_LLM_BASE_URL")
    local_llm_api_key: SecretStr = Field(default=SecretStr("not-needed"), alias="LOCAL_LLM_API_KEY")

    # --- postgres ------------------------------------------------------------
    postgres_user: str = Field(default="mmrag", alias="POSTGRES_USER")
    postgres_password: SecretStr = Field(default=SecretStr("mmrag"), alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="mmrag", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5433, alias="POSTGRES_PORT")

    # --- qdrant --------------------------------------------------------------
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_http_port: int = Field(default=6333, alias="QDRANT_HTTP_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")
    qdrant_api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")

    # --- misc ----------------------------------------------------------------
    log_level: str = Field(default="INFO", alias="MMRAG_LOG_LEVEL")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_http_port}"

    def redacted(self) -> dict[str, Any]:
        """Loggable view with every secret masked."""
        out: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            out[name] = "***" if isinstance(value, SecretStr) or "password" in name else value
        if self.openai_api_key is not None:
            out["openai_api_key"] = "***"
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


class IngestionConfig(BaseModel):
    """How PDFs are turned into elements. Shared by all three methods."""

    parser: Literal["pymupdf", "pdfplumber"] = "pymupdf"
    # DPI for the full-page renders. 150 is a good balance for CPU; Method 3
    # visual retrieval wants >= 144 for ColQwen to read small axis labels.
    page_image_dpi: int = Field(default=150, ge=72, le=400)
    # Crops of figures are rendered higher, since captions and axis text on a
    # small chart become illegible at page DPI.
    figure_image_dpi: int = Field(default=200, ge=72, le=600)
    # Ignore figure crops smaller than this fraction of the page: almost always
    # rules, bullets, logos, or background artefacts rather than real content.
    min_figure_area_ratio: float = Field(default=0.01, ge=0.0, le=1.0)
    extract_tables: bool = True
    extract_figures: bool = True
    # Captions are matched to figures by proximity; this is the max vertical gap
    # as a fraction of page height.
    caption_max_gap_ratio: float = Field(default=0.06, ge=0.0, le=0.5)
    max_pages_per_doc: int | None = None  # for fast smoke runs


class EnrichmentConfig(BaseModel):
    """Optional passes that add derived text to visual elements.

    Both are off by default so a first ingestion run needs no extra setup and
    no API key.
    """

    ocr_enabled: bool = False
    # ``rapidocr`` installs from PyPI with its models inside the wheel, so
    # ``pip install -e ".[ocr]"`` is enough to reproduce an ingestion run.
    # ``tesseract`` additionally needs the binary on PATH.
    ocr_backend: Literal["rapidocr", "tesseract"] = "rapidocr"
    ocr_languages: str = "eng"
    # Recovered text shorter than this is speckle, not content.
    ocr_min_chars: int = 8

    vlm_captions_enabled: bool = False
    vlm_max_elements: int | None = None  # cost guard
    vlm_prompt_style: Literal["describe", "describe_and_transcribe"] = "describe_and_transcribe"


class ChunkingConfig(BaseModel):
    target_tokens: int = Field(default=384, ge=64, le=2048)
    overlap_tokens: int = Field(default=64, ge=0, le=512)
    # Never merge across a page break: it would make a chunk uncitable to a
    # single page, which breaks provenance.
    respect_page_boundaries: bool = True
    # Tables are kept whole up to this size rather than split mid-row.
    max_table_tokens: int = Field(default=1024, ge=128)
    # Prepend "Document title > Section" to each chunk. Cheap and reliably
    # improves both BM25 and dense recall on multi-document corpora.
    prepend_context_header: bool = True

    @model_validator(mode="after")
    def _overlap_smaller_than_target(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < "
                f"target_tokens ({self.target_tokens})"
            )
        return self


class EmbeddingConfig(BaseModel):
    """Local, open-source embedders. Provider-independent by design."""

    text_model: str = "BAAI/bge-small-en-v1.5"
    text_dim: int = 384
    # bge models are trained with an asymmetric query prefix; omitting it costs
    # a few points of recall.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""
    normalize: bool = True
    batch_size: int = 32
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    max_seq_length: int | None = 512

    # Method 2 image retrieval.
    image_model: str = "sentence-transformers/clip-ViT-B-32"
    image_dim: int = 512

    # Method 3 late-interaction visual retrieval.
    visual_model: str = "vidore/colqwen2-v1.0"
    visual_dim: int = 128


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=10, ge=1, le=200)
    # Each retriever returns this many before fusion; must exceed top_k for
    # fusion to have anything to choose between.
    candidates_per_retriever: int = Field(default=50, ge=1, le=500)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    # RRF constant. 60 is the value from Cormack et al. and is a sane default;
    # lower values weight the top ranks more aggressively.
    rrf_k: int = Field(default=60, ge=1)
    # Per-retriever weights applied on top of RRF. Keys are retriever names.
    fusion_weights: dict[str, float] = Field(default_factory=dict)

    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = Field(default=25, ge=1)

    @model_validator(mode="after")
    def _candidates_cover_top_k(self) -> RetrievalConfig:
        if self.candidates_per_retriever < self.top_k:
            raise ValueError(
                f"candidates_per_retriever ({self.candidates_per_retriever}) must be >= "
                f"top_k ({self.top_k})"
            )
        return self


class RouterConfig(BaseModel):
    """Method 2 query router."""

    strategy: Literal["heuristic", "llm", "hybrid", "all"] = "heuristic"
    # When the router is unsure, query everything rather than guess wrong: a
    # false negative here is unrecoverable, a false positive only costs latency.
    fallback_to_all: bool = True
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GenerationConfig(BaseModel):
    """Provider-agnostic generation settings.

    ``provider`` is intentionally absent -- it lives in :class:`Settings`,
    because which provider you can reach is a property of the machine, not of
    the experiment.
    """

    text_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1)
    # Hard cap on context assembled from retrieved chunks.
    max_context_tokens: int = Field(default=6000, ge=256)
    # Method 3: how many full page images to attach to the vision call.
    max_images: int = Field(default=3, ge=0, le=20)
    image_detail: Literal["low", "high", "auto"] = "high"
    cite_sources: bool = True
    refuse_without_evidence: bool = True
    request_timeout_s: float = 120.0
    max_retries: int = 3


class EvaluationConfig(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    # Retrieval metrics computed from the gold page/element labels.
    retrieval_metrics: list[str] = Field(
        default_factory=lambda: ["recall", "precision", "mrr", "ndcg", "hit_rate"]
    )
    # Answer metrics. The LLM judge is separated so it can be turned off to run
    # the whole benchmark for free.
    answer_metrics: list[str] = Field(
        default_factory=lambda: ["exact_match", "token_f1", "citation_accuracy"]
    )
    llm_judge_enabled: bool = False
    llm_judge_model: str = "gpt-4o-mini"
    # Bootstrap resamples for confidence intervals on the headline numbers.
    bootstrap_samples: int = Field(default=1000, ge=0)
    random_seed: int = 42


class ExperimentConfig(BaseModel):
    """A complete, reproducible description of one benchmark run."""

    name: str = "default"
    description: str = ""
    method: Literal["method1", "method2", "method3"] = "method1"

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @property
    def chunk_variant(self) -> str:
        """Namespace for this config's chunks and indexes.

        Method 1 and Method 2 build different chunk sets from the same elements;
        tagging them keeps both resident at once so a comparison run does not
        have to re-ingest between methods.
        """
        return self.method


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping at the top level: {path}")
    return data


def load_experiment_config(
    name_or_path: str | Path = "default",
    *,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load a config by name (``configs/<name>.yaml``) or explicit path.

    A config may declare ``extends: <other>`` to inherit from another file, so
    the per-method configs only have to state what actually differs from the
    baseline. Inheritance chains are followed to any depth, with cycle detection.
    """
    merged = _resolve_config_chain(name_or_path, seen=[])
    if overrides:
        merged = _deep_merge(merged, overrides)
    return ExperimentConfig.model_validate(merged)


def _resolve_config_path(name_or_path: str | Path, base_dir: Path | None) -> Path:
    """Turn a config reference into a concrete path.

    * A bare name (``method1``) always means ``configs/method1.yaml``.
    * An absolute path is used as given.
    * A *relative* path is resolved against the directory of the file that
      referenced it, not against the process working directory -- so a config
      chain keeps working regardless of where the command was run from.
    """
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        return (CONFIG_DIR / f"{name_or_path}.yaml").resolve()
    if path.is_absolute():
        return path.resolve()
    return ((base_dir or Path.cwd()) / path).resolve()


def _resolve_config_chain(
    name_or_path: str | Path,
    seen: list[str],
    base_dir: Path | None = None,
) -> dict[str, Any]:
    path = _resolve_config_path(name_or_path, base_dir)

    key = str(path)
    if key in seen:
        cycle = " -> ".join([*seen, key])
        raise ValueError(f"circular 'extends' in config chain: {cycle}")

    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _resolve_config_chain(parent, seen=[*seen, key], base_dir=path.parent)
    return _deep_merge(base, data)
