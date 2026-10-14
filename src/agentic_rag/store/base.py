from typing import Any, Protocol

from agentic_rag.models import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def delete_document(self, doc_id: str) -> None: ...

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...

    def all_chunks(self) -> list[Chunk]:
        """取出所有 chunk（不含向量），給 BM25 建索引用。"""
        ...

    def count(self) -> int: ...
