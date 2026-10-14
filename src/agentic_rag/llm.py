"""LLM 與 Embedding 的介面與 Ollama 實作。

其餘程式碼只依賴 Protocol，之後要換成 vLLM、llama.cpp 或雲端 API，只需新增一個實作。
"""

import json
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlparse

import httpx

from agentic_rag.config import Settings

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def use_env_proxy(url: str) -> bool:
    """連本機服務時不走 HTTP_PROXY。

    公司電腦常設定 HTTP_PROXY，httpx 預設會照用，連 localhost 也送去公司 proxy 而被擋（403）。
    """
    return urlparse(url).hostname not in LOCAL_HOSTS


class ServiceUnavailable(RuntimeError):
    """外部服務（Ollama、資料庫）連不上或設定不對，訊息會直接顯示給使用者。"""


def check_ollama(settings: Settings, models: list[str]) -> None:
    """確認 Ollama 有在跑、需要的模型已下載，避免後面卡住或噴出難懂的錯誤。"""
    try:
        resp = httpx.get(
            f"{settings.ollama_url}/api/tags",
            timeout=5,
            trust_env=use_env_proxy(settings.ollama_url),
        )
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


TokenCallback = Callable[[str], None]


class ChatModel(Protocol):
    def chat(
        self, messages: list[dict[str, str]], on_token: TokenCallback | None = None
    ) -> str: ...


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _timeout(settings: Settings) -> httpx.Timeout:
    # 連線 5 秒就該成功；讀取要等模型載入與計算，CPU 上可能要好幾分鐘
    return httpx.Timeout(settings.request_timeout, connect=5)


class OllamaChat:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.think = settings.llm_think
        self.timeout_s = settings.request_timeout
        self._client = client or httpx.Client(
            base_url=settings.ollama_url,
            timeout=_timeout(settings),
            trust_env=use_env_proxy(settings.ollama_url),
        )

    def chat(self, messages: list[dict[str, str]], on_token: TokenCallback | None = None) -> str:
        """以串流方式生成：每收到一段文字就呼叫 on_token，讓使用者不必乾等整段答案。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        if self.think is not None:
            payload["think"] = self.think
        parts: list[str] = []
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise ServiceUnavailable(f"Ollama 回傳錯誤：{event['error']}")
                    # 思考內容在 message.thinking，不算進答案
                    token = event.get("message", {}).get("content", "")
                    if token:
                        parts.append(token)
                        if on_token:
                            on_token(token)
                    if event.get("done"):
                        break
        except httpx.TimeoutException as exc:
            raise ServiceUnavailable(
                f"模型 {self.model} 超過 {self.timeout_s:.0f} 秒沒有回應，多半是在 CPU 上執行太慢。"
                "可以在 .env 調高 RAG_REQUEST_TIMEOUT、改用更小的 RAG_LLM_MODEL，"
                "或讓 Ollama 使用 GPU。"
            ) from exc
        return "".join(parts)


class OllamaEmbedder:
    def __init__(
        self, settings: Settings, client: httpx.Client | None = None, batch_size: int = 32
    ):
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        self.batch_size = batch_size
        self._client = client or httpx.Client(
            base_url=settings.ollama_url,
            timeout=_timeout(settings),
            trust_env=use_env_proxy(settings.ollama_url),
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
