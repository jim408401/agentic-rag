import httpx
import pytest

from agentic_rag.config import Settings
from agentic_rag.ingestion.pipeline import ingest_directory
from agentic_rag.llm import (
    OllamaEmbedder,
    ServiceUnavailable,
    check_ollama,
    use_env_proxy,
)
from agentic_rag.store.memory import InMemoryVectorStore
from tests.conftest import HANDBOOK


def fake_tags(monkeypatch, names: list[str]) -> None:
    def get(url, **kwargs):
        return httpx.Response(
            200, json={"models": [{"name": n} for n in names]}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", get)


def test_check_ollama_accepts_installed_models(monkeypatch):
    fake_tags(monkeypatch, ["bge-m3:latest", "qwen3.5:0.8b"])
    check_ollama(Settings(), ["bge-m3", "qwen3.5:0.8b"])


def test_check_ollama_reports_missing_model(monkeypatch):
    fake_tags(monkeypatch, ["bge-m3:latest"])
    with pytest.raises(ServiceUnavailable, match="ollama pull qwen3.5:0.8b"):
        check_ollama(Settings(), ["bge-m3", "qwen3.5:0.8b"])


def test_check_ollama_reports_connection_error(monkeypatch):
    def get(url, **kwargs):
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


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434", False),
        ("http://127.0.0.1:11434", False),
        ("http://[::1]:11434", False),
        ("http://gpu-server.internal:11434", True),
    ],
)
def test_local_services_bypass_env_proxy(url, expected):
    assert use_env_proxy(url) is expected


def test_local_ollama_ignores_http_proxy_env(monkeypatch):
    # 指向不存在的 proxy：若 httpx 仍照用環境變數，送出請求時就會失敗
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    seen = {}

    def get(url, **kwargs):
        seen.update(kwargs)
        return httpx.Response(
            200, json={"models": [{"name": "bge-m3"}]}, request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", get)
    check_ollama(Settings(ollama_url="http://localhost:11434"), ["bge-m3"])
    assert seen["trust_env"] is False
    assert OllamaEmbedder(Settings(ollama_url="http://localhost:11434"))._client._trust_env is False
