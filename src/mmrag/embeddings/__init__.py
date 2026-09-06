"""Local, provider-independent embedding models."""

from mmrag.embeddings.text import TextEmbedder, resolve_device

__all__ = ["TextEmbedder", "resolve_device"]
