"""Reranker：先用便宜的檢索取回較多候選（例如 20 筆），再用 cross-encoder 逐筆精排。

Embedding 是把問題和文件「分開」編碼後比相似度；cross-encoder 則把「問題 + 文件」一起讀，
能看出「這段到底有沒有回答這個問題」，準確很多，但每筆都要算一次，所以只用在少量候選上。

模型 bge-reranker-v2-m3 由 Hugging Face Text Embeddings Inference（TEI）提供服務，
見 docker-compose.yml 的 reranker。
"""

from typing import Any, Protocol

import httpx

from agentic_rag.config import Settings
from agentic_rag.llm import ServiceUnavailable, use_env_proxy
from agentic_rag.models import RetrievedChunk
from agentic_rag.retrieval.retriever import Retriever


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """回傳每段文字與問題的相關分數，順序與 texts 相同。"""
        ...


class TEIReranker:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.url = settings.reranker_url
        self._client = client or httpx.Client(
            base_url=settings.reranker_url,
            timeout=httpx.Timeout(settings.request_timeout, connect=5),
            trust_env=use_env_proxy(settings.reranker_url),
        )

    def score(self, query: str, texts: list[str]) -> list[float]:
        try:
            resp = self._client.post(
                "/rerank", json={"query": query, "texts": texts, "truncate": True}
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ServiceUnavailable(f"Reranker（{self.url}）呼叫失敗：{exc}") from exc
        scores = [0.0] * len(texts)
        for item in resp.json():
            scores[item["index"]] = float(item["score"])
        return scores


def check_reranker(settings: Settings) -> None:
    try:
        resp = httpx.get(
            f"{settings.reranker_url}/health",
            timeout=5,
            trust_env=use_env_proxy(settings.reranker_url),
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ServiceUnavailable(
            f"連不上 Reranker（{settings.reranker_url}）。請先執行 "
            "docker compose --profile rerank up -d reranker，第一次啟動要下載模型，"
            f"可用 docker compose logs -f reranker 查看進度：{exc}"
        ) from exc


class RerankingRetriever:
    def __init__(self, base: Retriever, reranker: Reranker, candidate_k: int = 20):
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        candidates = self.base.retrieve(query, max(k, self.candidate_k), filters)
        if not candidates:
            return []
        scores = self.reranker.score(query, [c.chunk.text_for_embedding() for c in candidates])
        reranked = [
            RetrievedChunk(chunk=c.chunk, score=s) for c, s in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:k]
