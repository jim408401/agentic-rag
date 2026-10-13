"""純 Python 的記憶體向量庫：給單元測試與沒有 Docker 時快速實驗用。"""

import math
from typing import Any

from agentic_rag.models import Chunk, RetrievedChunk


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    return not filters or all(metadata.get(k) == v for k, v in filters.items())


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._rows[chunk.chunk_id] = (chunk, vec)

    def delete_document(self, doc_id: str) -> None:
        self._rows = {k: v for k, v in self._rows.items() if v[0].doc_id != doc_id}

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=chunk, score=cosine(vector, vec))
            for chunk, vec in self._rows.values()
            if matches(chunk.metadata, filters)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._rows)
