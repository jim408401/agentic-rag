from fastapi.testclient import TestClient

from agentic_rag.api.main import app, get_pipeline


def test_ask_endpoint(pipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        resp = client.post("/ask", json={"question": "密碼多久要換一次"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "測試答案 [1]"
        assert body["sources"][0]["index"] == 1
        assert client.post("/ask", json={"question": ""}).status_code == 422
    finally:
        app.dependency_overrides.clear()
