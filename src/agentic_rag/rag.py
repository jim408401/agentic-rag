"""把檢索與生成串起來的 RAG 主流程，以及依設定組裝元件的工廠函式。"""

from typing import Any

from agentic_rag.config import Settings
from agentic_rag.generation.prompts import NO_ANSWER, build_messages
from agentic_rag.llm import ChatModel, OllamaChat, OllamaEmbedder, TokenCallback
from agentic_rag.models import Answer
from agentic_rag.retrieval.retriever import Retriever, VectorRetriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: ChatModel, top_k: int):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def ask(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        on_token: TokenCallback | None = None,
    ) -> Answer:
        hits = self.retriever.retrieve(question, self.top_k, filters)
        if not hits:
            if on_token:
                on_token(NO_ANSWER)
            return Answer(question=question, text=NO_ANSWER, sources=[])
        text = self.llm.chat(build_messages(question, hits), on_token)
        return Answer(question=question, text=text, sources=hits)


def build_pipeline(settings: Settings) -> RAGPipeline:
    from agentic_rag.store.pgvector import PgVectorStore

    embedder = OllamaEmbedder(settings)
    store = PgVectorStore(settings.database_url)
    return RAGPipeline(
        retriever=VectorRetriever(embedder, store),
        llm=OllamaChat(settings),
        top_k=settings.top_k,
    )
