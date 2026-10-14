import json

import httpx
import pytest

from agentic_rag.config import Settings
from agentic_rag.llm import OllamaChat, ServiceUnavailable


def chat_with(handler) -> OllamaChat:
    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return OllamaChat(Settings(), client=client)


def test_chat_streams_tokens():
    lines = [
        {"message": {"content": "特休"}, "done": False},
        {"message": {"content": " 14 日 [1]"}, "done": False},
        {"message": {"content": ""}, "done": True},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        body = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)
        return httpx.Response(200, content=body.encode())

    tokens: list[str] = []
    text = chat_with(handler).chat([{"role": "user", "content": "q"}], on_token=tokens.append)
    assert text == "特休 14 日 [1]"
    assert tokens == ["特休", " 14 日 [1]"]


def test_chat_timeout_gives_actionable_message():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(ServiceUnavailable, match="RAG_REQUEST_TIMEOUT"):
        chat_with(handler).chat([{"role": "user", "content": "q"}])


def test_chat_surfaces_ollama_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"error": "model requires more system memory"}')

    with pytest.raises(ServiceUnavailable, match="more system memory"):
        chat_with(handler).chat([{"role": "user", "content": "q"}])


def test_chat_sends_think_flag_and_ignores_thinking_text():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["think"] is False
        lines = [
            {"message": {"thinking": "先想一下", "content": ""}, "done": False},
            {"message": {"content": "答案"}, "done": True},
        ]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    assert chat_with(handler).chat([{"role": "user", "content": "q"}]) == "答案"


def test_chat_omits_think_when_unset():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "think" not in json.loads(request.content)
        return httpx.Response(200, content=b'{"message": {"content": "ok"}, "done": true}')

    client = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    assert OllamaChat(Settings(llm_think=None), client=client).chat([]) == "ok"
