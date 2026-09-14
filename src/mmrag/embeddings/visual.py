"""ColQwen2 late-interaction embeddings for rendered pages.

Used by the visual page index (:mod:`mmrag.indexing.visual_pages`); in the
benchmark, only Method 3 enables it. Methods 1 and 2 never see a page image.

A late-interaction model does not compress a page into one vector. It keeps one
128-dimensional vector per image patch, and a query keeps one per token; a page
scores the sum, over query tokens, of each token's best-matching patch (MaxSim).
That is what lets it match a query word to a small axis label or a table cell
without any text having been extracted from the page first.

Nothing here is imported at module load except numpy, so the CLI and the test
suite work on a machine with no GPU and without the ``visual`` extra installed.
The model is loaded only when something is actually encoded, and every way that
can fail -- the package is missing, the device is not there, the weights produce
NaNs in half precision -- raises an error that says what to do about it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from mmrag.config import VisualRetrievalConfig
from mmrag.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

log = get_logger(__name__)


class VisualModelUnavailableError(RuntimeError):
    """The visual model cannot be loaded or used in this environment."""


class DeviceUnavailableError(RuntimeError):
    """An explicitly requested device is not present."""


class VisualEncodingError(RuntimeError):
    """The model ran but produced embeddings that cannot be used."""


# ---------------------------------------------------------------------------
# Device and dtype
# ---------------------------------------------------------------------------


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch ships with sentence-transformers
        raise VisualModelUnavailableError("torch is not installed; pip install torch") from exc
    return torch


def resolve_visual_device(preference: str = "auto", *, torch_module: Any = None) -> str:
    """The device to run on, validated rather than trusted.

    ``auto`` picks cuda, then mps, then cpu. An explicit choice is honoured only
    if it exists: asking for ``cuda`` on a CPU-only laptop is an error, not a
    silent fallback, because a fallback would turn a minutes-long GPU index
    build into an hours-long CPU one without saying so.
    """
    if not re.fullmatch(r"auto|cpu|mps|cuda(:\d+)?", preference):
        raise DeviceUnavailableError(
            f"unrecognised device {preference!r}; use auto, cpu, mps, cuda or cuda:N"
        )
    if preference == "cpu":
        return "cpu"

    torch = torch_module or _torch()
    cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend and mps_backend.is_available())

    if preference == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if preference == "mps":
        if not mps:
            raise DeviceUnavailableError(
                "device 'mps' was requested but Apple MPS is not available"
            )
        return "mps"

    if not cuda:
        raise DeviceUnavailableError(
            f"device {preference!r} was requested but CUDA is not available here. "
            "Run on a GPU machine, or set visual.device to cpu (slow) for small tests."
        )
    if ":" in preference:
        index = int(preference.split(":", 1)[1])
        count = torch.cuda.device_count()
        if index >= count:
            raise DeviceUnavailableError(
                f"device {preference!r} was requested but only {count} CUDA device(s) exist"
            )
    return preference


def resolve_dtype(preference: str, device: str, *, torch_module: Any = None) -> str:
    """The numeric precision to load the model in, as a dtype name.

    bfloat16 where the GPU supports it; float16 on older GPUs such as the T4,
    which cannot run bfloat16; float32 on cpu and mps, where half precision is
    either slow or unreliable for this architecture.
    """
    if preference != "auto":
        return preference
    if device.startswith("cuda"):
        torch = torch_module or _torch()
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


# ---------------------------------------------------------------------------
# Encoder contract
# ---------------------------------------------------------------------------


@runtime_checkable
class VisualEncoder(Protocol):
    """Pages and queries in, one matrix of token vectors per item out."""

    def encode_images(self, images: Sequence[Image]) -> list[np.ndarray]: ...

    def encode_queries(self, texts: Sequence[str]) -> list[np.ndarray]: ...

    def identity(self) -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...


def compatible_identity(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether embeddings from two encoders live in the same space.

    Same model and dimension, and -- where both sides know it -- the same
    revision. Query vectors from one model scored against page vectors from
    another produce numbers that look plausible and mean nothing.
    """
    if a.get("model") != b.get("model") or a.get("dim") != b.get("dim"):
        return False
    for key in ("revision", "commit"):
        if a.get(key) and b.get(key) and a[key] != b[key]:
            return False
    return True


def check_embeddings(vectors: np.ndarray, *, expected_dim: int, what: str) -> np.ndarray:
    """Reject embeddings that would silently corrupt an index."""
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise VisualEncodingError(
            f"{what}: expected token vectors of width {expected_dim}, got shape {vectors.shape}"
        )
    if vectors.shape[0] == 0:
        raise VisualEncodingError(f"{what}: the model returned no token vectors")
    if not np.isfinite(vectors).all():
        raise VisualEncodingError(
            f"{what}: the model produced NaN or infinite values. This usually means half "
            "precision overflowed; set visual.dtype to float32, or use a GPU with bfloat16 "
            "support (A10, L4, A100 or newer)."
        )
    return vectors


# ---------------------------------------------------------------------------
# ColQwen2
# ---------------------------------------------------------------------------


class ColQwen2Encoder:
    """ColQwen2 via ``colpali-engine``, loaded lazily on an explicit device."""

    def __init__(
        self,
        model_name: str,
        config: VisualRetrievalConfig,
        *,
        expected_dim: int,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.config = config
        self.expected_dim = expected_dim
        self.device = resolve_visual_device(device or config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)
        self._model: Any = None
        self._processor: Any = None

    # -- loading -------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ImportError as exc:
            raise VisualModelUnavailableError(
                "ColQwen2 needs the visual extra: pip install -e \".[visual]\". "
                "Indexing the full corpus also needs a GPU; see the README's Method 3 section."
            ) from exc

        torch = _torch()
        kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, self.dtype),
            "device_map": self.device,
        }
        if self.config.model_revision:
            kwargs["revision"] = self.config.model_revision
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        log.info("loading %s on %s as %s", self.model_name, self.device, self.dtype)
        try:
            self._model = ColQwen2.from_pretrained(self.model_name, **kwargs).eval()
            processor_kwargs = (
                {"revision": self.config.model_revision} if self.config.model_revision else {}
            )
            self._processor = ColQwen2Processor.from_pretrained(self.model_name, **processor_kwargs)
        except Exception as exc:
            self._model = None
            raise VisualModelUnavailableError(
                f"could not load {self.model_name} on {self.device} ({type(exc).__name__}: {exc})"
            ) from exc

    # -- encoding ------------------------------------------------------------

    def _forward(self, batch: Any, *, what: str) -> list[np.ndarray]:
        torch = _torch()
        batch = batch.to(self._model.device)
        with torch.no_grad():
            output = self._model(**batch)
        mask = batch["attention_mask"].bool()
        vectors = []
        for row, keep in zip(output, mask, strict=True):
            # Padding positions are zeroed by the model, and a zero vector would
            # still win MaxSim against every negative similarity. Dropping them
            # makes a page's score independent of what it was batched with.
            kept = row[keep].to(torch.float32).cpu().numpy()
            vectors.append(check_embeddings(kept, expected_dim=self.expected_dim, what=what))
        return vectors

    def encode_images(self, images: Sequence[Image]) -> list[np.ndarray]:
        self._load()
        return self._forward(self._processor.process_images(list(images)), what="page image")

    def encode_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        self._load()
        return self._forward(self._processor.process_queries(list(texts)), what="query")

    # -- identity ------------------------------------------------------------

    def identity(self) -> dict[str, Any]:
        self._load()
        return {
            "model": self.model_name,
            "revision": self.config.model_revision,
            "commit": getattr(getattr(self._model, "config", None), "_commit_hash", None),
            "dim": self.expected_dim,
        }

    def describe(self) -> dict[str, Any]:
        self._load()
        from importlib.metadata import PackageNotFoundError, version

        # Distribution metadata, not module attributes: colpali_engine does not
        # define __version__, and a manifest recording None is not reproducible.
        versions: dict[str, str | None] = {}
        for package in ("colpali-engine", "transformers", "torch", "peft"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:  # pragma: no cover
                versions[package] = None
        image_processor = getattr(self._processor, "image_processor", None)
        return {
            **self.identity(),
            "device": self.device,
            "dtype": self.dtype,
            "attn_implementation": self.config.attn_implementation,
            "processor": {
                "class": type(self._processor).__name__,
                "max_num_visual_tokens": getattr(self._processor, "max_num_visual_tokens", None),
                "min_pixels": getattr(image_processor, "min_pixels", None),
                "max_pixels": getattr(image_processor, "max_pixels", None),
            },
            "versions": versions,
        }
