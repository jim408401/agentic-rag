"""FastAPI 服務：uv run uvicorn agentic_rag.api.main:app --reload"""

from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from agentic_rag.config import get_settings
from agentic_rag.rag import RAGPipeline, build_pipeline

app = FastAPI(title="Agentic RAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filters: dict[str, Any] | None = Field(
        default=None, description='metadata 過濾，例如 {"department": "人力資源部"}'
    )
    all_versions: bool = Field(
        default=False,
        description="連已廢止的舊版文件也查；預設只查現行版，問題提到年份或舊版時例外",
    )


class Source(BaseModel):
    index: int
    doc_id: str
    title: str
    section_path: str
    score: float
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@lru_cache
def get_pipeline() -> RAGPipeline:
    return build_pipeline(get_settings())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> AskResponse:
    answer = pipeline.ask(req.question, req.filters, all_versions=req.all_versions)
    return AskResponse(
        answer=answer.text,
        sources=[
            Source(
                index=i,
                doc_id=r.chunk.doc_id,
                title=r.chunk.title,
                section_path=r.chunk.section_path,
                score=round(r.score, 4),
                content=r.chunk.content,
            )
            for i, r in enumerate(answer.sources, start=1)
        ],
    )
