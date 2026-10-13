"""Phase 1：純向量檢索。

Phase 2 會在這一層加入 Hybrid（BM25 + RRF）、Reranker、Query Rewriting，
並維持同一個 `retrieve()` 介面，讓上層與評估程式不用改。
"""

from typing import Any, Protocol

from agentic_rag.llm import Embedder
from agentic_rag.models import RetrievedChunk
from agentic_rag.store.base import VectorStore


class Retriever(Protocol):
    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        [vector] = self.embedder.embed([query])
        return self.store.search(vector, k, filters)
