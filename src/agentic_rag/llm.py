"""LLM 與 Embedding 的介面與 Ollama 實作。

其餘程式碼只依賴 Protocol，之後要換成 vLLM、llama.cpp 或雲端 API，只需新增一個實作。
"""

from typing import Protocol

import httpx

from agentic_rag.config import Settings


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str: ...


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OllamaChat:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self._client = client or httpx.Client(
            base_url=settings.ollama_url, timeout=settings.request_timeout
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        resp = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OllamaEmbedder:
    def __init__(
        self, settings: Settings, client: httpx.Client | None = None, batch_size: int = 32
    ):
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        self.batch_size = batch_size
        self._client = client or httpx.Client(
            base_url=settings.ollama_url, timeout=settings.request_timeout
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self._client.post("/api/embed", json={"model": self.model, "input": batch})
            resp.raise_for_status()
            vectors.extend(resp.json()["embeddings"])
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(
                    f"embedding 維度為 {len(v)}，與設定的 RAG_EMBED_DIM={self.dim} 不符"
                )
        return vectors
