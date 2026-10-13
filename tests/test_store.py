from agentic_rag.models import Chunk
from agentic_rag.store.memory import InMemoryVectorStore


def make_chunk(cid: str, doc_id: str, status: str) -> Chunk:
    return Chunk(cid, doc_id, "t", "", "s", "c", {"status": status})


def test_search_orders_by_similarity_and_filters():
    store = InMemoryVectorStore()
    store.upsert(
        [make_chunk("a", "D1", "現行"), make_chunk("b", "D2", "已廢止")],
        [[1.0, 0.0], [0.9, 0.1]],
    )
    assert [r.chunk.chunk_id for r in store.search([1.0, 0.0], k=2)] == ["a", "b"]
    filtered = store.search([0.0, 1.0], k=2, filters={"status": "現行"})
    assert [r.chunk.chunk_id for r in filtered] == ["a"]


def test_delete_document():
    store = InMemoryVectorStore()
    store.upsert([make_chunk("a", "D1", "現行"), make_chunk("b", "D2", "現行")], [[1, 0], [0, 1]])
    store.delete_document("D1")
    assert store.count() == 1
