"""資料匯入管線：讀檔 → 切塊 → embedding → 寫入向量庫。"""

from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingestion.chunking import chunk_document
from agentic_rag.ingestion.loaders import load_directory
from agentic_rag.llm import Embedder
from agentic_rag.models import Document
from agentic_rag.store.base import VectorStore


@dataclass
class IngestReport:
    documents: int
    chunks: int


def ingest_documents(
    docs: list[Document],
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestReport:
    total = 0
    for doc in docs:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        vectors = embedder.embed([c.text_for_embedding() for c in chunks])
        # 先刪後寫：文件變短時，舊的多餘 chunk 不會殘留在索引裡
        store.delete_document(doc.doc_id)
        store.upsert(chunks, vectors)
        total += len(chunks)
    return IngestReport(documents=len(docs), chunks=total)


def ingest_directory(
    root: Path, embedder: Embedder, store: VectorStore, chunk_size: int, chunk_overlap: int
) -> IngestReport:
    return ingest_documents(load_directory(root), embedder, store, chunk_size, chunk_overlap)
