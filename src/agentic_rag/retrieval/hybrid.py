"""Hybrid 檢索：用 RRF（Reciprocal Rank Fusion）合併多個檢索器的結果。

向量分數（0~1 的 cosine）和 BM25 分數（沒有上限）尺度不同，直接相加沒有意義。
RRF 只看名次：每份結果得 1 / (rrf_k + 名次)，在多個檢索器都排得前面的 chunk 總分最高。
rrf_k 通常取 60，越大越不偏袒第一名。
"""

from typing import Any

from agentic_rag.models import RetrievedChunk
from agentic_rag.retrieval.retriever import Retriever


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], rrf_k: int = 60
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(cid, hit)
    fused = [RetrievedChunk(chunk=chunks[cid].chunk, score=s) for cid, s in scores.items()]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


class HybridRetriever:
    def __init__(self, retrievers: list[Retriever], candidate_k: int = 20, rrf_k: int = 60):
        self.retrievers = retrievers
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        n = max(k, self.candidate_k)
        rankings = [r.retrieve(query, n, filters) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.rrf_k)[:k]
