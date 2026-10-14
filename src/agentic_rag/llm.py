"""LLM 與 Embedding 的介面與 Ollama 實作。

其餘程式碼只依賴 Protocol，之後要換成 vLLM、llama.cpp 或雲端 API，只需新增一個實作。
"""

from typing import Protocol

import httpx

from agentic_rag.config import Settings


class ServiceUnavailable(RuntimeError):
    """外部服務（Ollama、資料庫）連不上或設定不對，訊息會直接顯示給使用者。"""


def check_ollama(settings: Settings, models: list[str]) -> None:
    """確認 Ollama 有在跑、需要的模型已下載，避免後面卡住或噴出難懂的錯誤。"""
    try:
        resp = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ServiceUnavailable(
            f"連不上 Ollama（{settings.ollama_url}）。請確認 Ollama 已啟動：{exc}"
        ) from exc
    installed = {m["name"] for m in resp.json().get("models", [])}
    installed |= {name.removesuffix(":latest") for name in installed}
    missing = [m for m in models if m not in installed]
    if missing:
        raise ServiceUnavailable(
            "Ollama 缺少模型："
            + "、".join(missing)
            + "。請先執行 "
            + "；".join(f"ollama pull {m}" for m in missing)
        )


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
