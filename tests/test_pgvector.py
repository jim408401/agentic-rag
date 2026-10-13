"""需要 Postgres + pgvector 並已載入 db/init；執行：uv run pytest -m postgres"""

import os

import pytest

from agentic_rag.ingestion.pipeline import ingest_directory
from agentic_rag.retrieval.retriever import VectorRetriever
from tests.conftest import HANDBOOK, HashingEmbedder

pytestmark = pytest.mark.postgres

DATABASE_URL = os.environ.get("RAG_DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")


@pytest.fixture
def store():
    from agentic_rag.store.pgvector import PgVectorStore

    return PgVectorStore(DATABASE_URL)


def test_ingest_is_idempotent_and_searchable(store):
    embedder = HashingEmbedder(dim=1024)  # 與資料表 vector(1024) 一致
    ingest_directory(HANDBOOK, embedder, store, 400, 60)
    first = store.count()
    ingest_directory(HANDBOOK, embedder, store, 400, 60)
    assert store.count() == first

    retriever = VectorRetriever(embedder, store)
    assert retriever.retrieve("連續請病假要附證明嗎", k=3)[0].chunk.doc_id == "HR-001"
    current = retriever.retrieve("特休可以遞延嗎", k=10, filters={"status": "現行"})
    assert current and all(h.chunk.metadata["status"] == "現行" for h in current)


def test_sales_schema_is_seeded(store):
    row = store._conn.execute(
        """
        SELECT sum(oi.quantity * oi.unit_price) AS revenue
        FROM sales.orders o
        JOIN sales.order_items oi USING (order_id)
        JOIN sales.customers c USING (customer_id)
        WHERE c.name LIKE '宏遠%%' AND o.status IN ('已出貨', '已送達')
        """
    ).fetchone()
    # 對應評估集 q024 的標準答案
    assert row["revenue"] == 586600
