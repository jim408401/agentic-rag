import math
import zlib
from pathlib import Path

import pytest

from agentic_rag.ingestion.pipeline import ingest_directory
from agentic_rag.rag import RAGPipeline
from agentic_rag.retrieval.retriever import VectorRetriever
from agentic_rag.store.memory import InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]
HANDBOOK = ROOT / "data" / "handbook"


class HashingEmbedder:
    """不需要模型的假 embedding：中文字元 bigram 雜湊成固定維度向量。

    相似的字詞會落在相同維度，足以讓單元測試驗證整條管線的行為。
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for a, b in zip(text, text[1:], strict=False):
                vec[zlib.crc32((a + b).encode()) % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class EchoChat:
    """記錄收到的 messages，回傳固定答案。"""

    def __init__(self, reply: str = "測試答案 [1]"):
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def indexed_store(embedder: HashingEmbedder) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    ingest_directory(HANDBOOK, embedder, store, chunk_size=400, chunk_overlap=60)
    return store


@pytest.fixture
def pipeline(embedder, indexed_store) -> RAGPipeline:
    return RAGPipeline(VectorRetriever(embedder, indexed_store), EchoChat(), top_k=5)
