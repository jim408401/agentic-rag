import httpx
import pytest

from agentic_rag.config import Settings
from agentic_rag.ingestion.pipeline import ingest_directory
from agentic_rag.llm import ServiceUnavailable, check_ollama
from agentic_rag.store.memory import InMemoryVectorStore
from tests.conftest import HANDBOOK


def fake_tags(monkeypatch, names: list[str]) -> None:
    def get(url, timeout):
        return httpx.Response(
            200, json={"models": [{"name": n} for n in names]}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", get)


def test_check_ollama_accepts_installed_models(monkeypatch):
    fake_tags(monkeypatch, ["bge-m3:latest", "qwen2.5:7b"])
    check_ollama(Settings(), ["bge-m3", "qwen2.5:7b"])


def test_check_ollama_reports_missing_model(monkeypatch):
    fake_tags(monkeypatch, ["bge-m3:latest"])
    with pytest.raises(ServiceUnavailable, match="ollama pull qwen2.5:7b"):
        check_ollama(Settings(), ["bge-m3", "qwen2.5:7b"])


def test_check_ollama_reports_connection_error(monkeypatch):
    def get(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", get)
    with pytest.raises(ServiceUnavailable, match="連不上 Ollama"):
        check_ollama(Settings(), ["bge-m3"])


def test_ingest_reports_progress_per_document(embedder):
    calls = []
    ingest_directory(
        HANDBOOK, embedder, InMemoryVectorStore(), 400, 60, on_progress=lambda *a: calls.append(a)
    )
    assert [c[0] for c in calls] == list(range(1, len(calls) + 1))
    assert all(c[1] == len(calls) and c[3] > 0 for c in calls)
