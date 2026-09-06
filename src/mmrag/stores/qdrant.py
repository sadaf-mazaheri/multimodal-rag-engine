"""Qdrant dense vector index.

Holds vectors plus the *filterable* subset of each chunk's metadata, and nothing
else. The chunk text and full provenance live in Postgres; duplicating them here
would create two sources of truth that drift.

The payload is the practical expression of the "metadata stays structured" rule:
doc_id, page number, chunk type, section, table/figure type. Those are exactly
the fields Method 2 will filter and route on, and they are queryable here
*without* having been mixed into the embedded text.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from mmrag.config import get_settings
from mmrag.logging_utils import get_logger
from mmrag.schemas import Chunk

log = get_logger(__name__)

UPSERT_BATCH = 256


def point_id_for(chunk_id: str) -> str:
    """Deterministic UUID for a chunk id.

    Qdrant accepts only unsigned integers or UUIDs as point ids, and our chunk
    ids are strings like ``method1#a1b2c3``. A UUID5 keeps the mapping stable
    across rebuilds, so re-indexing updates points in place rather than
    accumulating duplicates.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@dataclass
class DenseHit:
    chunk_id: str
    score: float
    rank: int
    payload: dict[str, Any]


class QdrantStore:
    """Dense vector index for one chunk variant."""

    def __init__(self, collection: str, *, url: str | None = None, api_key: str | None = None):
        from qdrant_client import QdrantClient

        settings = get_settings()
        self.collection = collection
        key = api_key or (
            settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        )
        self.client = QdrantClient(url=url or settings.qdrant_url, api_key=key, timeout=30)

    # -- lifecycle -----------------------------------------------------------

    def recreate(self, dimension: int) -> None:
        """Drop and recreate the collection.

        Index builds are all-or-nothing: a partially rebuilt collection mixing
        vectors from two encoders would produce silently meaningless similarities.
        """
        from qdrant_client import models

        # Delete-then-create rather than the deprecated recreate_collection.
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=dimension,
                # Vectors are normalised at encode time, so cosine is a dot
                # product; naming it cosine keeps the intent explicit.
                distance=models.Distance.COSINE,
            ),
        )
        # Payload indexes for the fields Method 2 will filter on. Creating them
        # up front costs nothing on an empty collection.
        for field, schema in (
            ("doc_id", models.PayloadSchemaType.KEYWORD),
            ("chunk_type", models.PayloadSchemaType.KEYWORD),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("section", models.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(
                collection_name=self.collection, field_name=field, field_schema=schema
            )
        log.info("recreated collection %s (dim=%d)", self.collection, dimension)

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def delete(self) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)

    # -- writes --------------------------------------------------------------

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        from qdrant_client import models

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        points = [
            models.PointStruct(
                id=point_id_for(chunk.chunk_id),
                vector=vector.tolist(),
                payload=self._payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), UPSERT_BATCH):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + UPSERT_BATCH],
                wait=True,
            )
        return len(points)

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, Any]:
        """Filterable metadata only -- never the embedded text's provenance dump."""
        payload: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type.value,
            "variant": chunk.variant,
            "element_ids": chunk.element_ids,
        }
        if chunk.section:
            payload["section"] = chunk.section
        for key in ("doc_type", "domain", "table_type", "figure_type"):
            if chunk.metadata.get(key) is not None:
                payload[key] = chunk.metadata[key]
        return payload

    # -- search --------------------------------------------------------------

    def search(
        self,
        vector: np.ndarray,
        k: int = 10,
        *,
        doc_ids: list[str] | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[DenseHit]:
        from qdrant_client import models

        conditions: list[Any] = []
        if doc_ids:
            conditions.append(
                models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids)))
            )
        if chunk_types:
            conditions.append(
                models.FieldCondition(
                    key="chunk_type", match=models.MatchAny(any=list(chunk_types))
                )
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=k,
            with_payload=True,
            query_filter=models.Filter(must=conditions) if conditions else None,
        )
        return [
            DenseHit(
                chunk_id=str((point.payload or {}).get("chunk_id")),
                score=float(point.score),
                rank=rank,
                payload=dict(point.payload or {}),
            )
            for rank, point in enumerate(response.points, start=1)
        ]

    def describe(self) -> dict[str, Any]:
        return {"backend": "qdrant", "collection": self.collection, "n_points": self.count()}
