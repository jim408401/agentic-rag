"""需要先 docker compose up -d 並下載模型；執行：uv run pytest -m ollama"""

import pytest

from agentic_rag.config import get_settings
from agentic_rag.ingestion.pipeline import ingest_directory
from tests.conftest import HANDBOOK

pytestmark = pytest.mark.ollama


def test_end_to_end_with_pgvector_and_ollama():
    from agentic_rag.llm import OllamaEmbedder
    from agentic_rag.rag import build_pipeline
    from agentic_rag.store.pgvector import PgVectorStore

    s = get_settings()
    ingest_directory(HANDBOOK, OllamaEmbedder(s), PgVectorStore(s.database_url), 400, 60)
    answer = build_pipeline(s).ask("到職滿三年半有幾天特休？")
    assert answer.sources[0].chunk.doc_id == "HR-001"
    assert "14" in answer.text
