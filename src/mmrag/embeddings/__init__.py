"""Local, provider-independent embedding models."""

from mmrag.embeddings.image import ImageEmbedder
from mmrag.embeddings.text import TextEmbedder, resolve_device

__all__ = ["ImageEmbedder", "TextEmbedder", "resolve_device"]
