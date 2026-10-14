"""把檢索與生成串起來的 RAG 主流程，以及依設定組裝元件的工廠函式。"""

from typing import Any

from agentic_rag.config import Settings
from agentic_rag.generation.prompts import NO_ANSWER, build_messages
from agentic_rag.llm import ChatModel, OllamaChat, OllamaEmbedder, TokenCallback
from agentic_rag.models import Answer, RetrievedChunk
from agentic_rag.retrieval.query import infer_filters
from agentic_rag.retrieval.retriever import Retriever, VectorRetriever


class RAGPipeline:
    def __init__(
        self, retriever: Retriever, llm: ChatModel, top_k: int, version_filter: bool = False
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.version_filter = version_filter

    def retrieve(
        self, question: str, filters: dict[str, Any] | None = None, all_versions: bool = False
    ) -> list[RetrievedChunk]:
        merged = dict(filters or {})
        if self.version_filter and not all_versions and "status" not in merged:
            merged.update(infer_filters(question))
        return self.retriever.retrieve(question, self.top_k, merged or None)

    def generate(
        self, question: str, hits: list[RetrievedChunk], on_token: TokenCallback | None = None
    ) -> Answer:
        if not hits:
            if on_token:
                on_token(NO_ANSWER)
            return Answer(question=question, text=NO_ANSWER, sources=[])
        text = self.llm.chat(build_messages(question, hits), on_token)
        return Answer(question=question, text=text, sources=hits)

    def ask(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        on_token: TokenCallback | None = None,
        all_versions: bool = False,
    ) -> Answer:
        return self.generate(question, self.retrieve(question, filters, all_versions), on_token)


def build_retriever(settings: Settings, embedder, store) -> Retriever:
    retriever: Retriever = VectorRetriever(embedder, store)
    if settings.retrieval_mode == "hybrid":
        from agentic_rag.retrieval.bm25 import BM25Retriever
        from agentic_rag.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(
            [retriever, BM25Retriever(store)], settings.candidate_k, settings.rrf_k
        )
    if settings.reranker_url:
        from agentic_rag.retrieval.rerank import RerankingRetriever, TEIReranker

        retriever = RerankingRetriever(retriever, TEIReranker(settings), settings.candidate_k)
    return retriever


def build_pipeline(settings: Settings) -> RAGPipeline:
    from agentic_rag.store.pgvector import PgVectorStore

    embedder = OllamaEmbedder(settings)
    store = PgVectorStore(settings.database_url)
    return RAGPipeline(
        retriever=build_retriever(settings, embedder, store),
        llm=OllamaChat(settings),
        top_k=settings.top_k,
        version_filter=settings.version_filter,
    )
