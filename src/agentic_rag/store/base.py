from typing import Any, Protocol

from agentic_rag.models import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def delete_document(self, doc_id: str) -> None: ...

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...
