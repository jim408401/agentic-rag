import httpx
import pytest

from agentic_rag.config import Settings
from agentic_rag.llm import ServiceUnavailable
from agentic_rag.models import Chunk, RetrievedChunk
from agentic_rag.rag import RAGPipeline, build_retriever
from agentic_rag.retrieval.bm25 import BM25Retriever, tokenize
from agentic_rag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from agentic_rag.retrieval.query import infer_filters
from agentic_rag.retrieval.rerank import RerankingRetriever, TEIReranker
from agentic_rag.retrieval.retriever import VectorRetriever
from agentic_rag.store.memory import InMemoryVectorStore
from tests.conftest import EchoChat


def _hit(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    chunk = Chunk(chunk_id, chunk_id.split("#")[0], "t", "", "s", chunk_id)
    return RetrievedChunk(chunk=chunk, score=score)


def test_tokenize_mixes_cjk_bigrams_and_words():
    tokens = tokenize("訂單 SO-2026-0007 退貨")
    assert "so-2026-0007" in tokens
    assert {"訂單", "退貨"} <= set(tokens)
    assert tokenize("特休假") == ["特休", "休假"]


def test_bm25_finds_exact_identifiers(indexed_store):
    hits = BM25Retriever(indexed_store).retrieve("24681357", k=3)
    # 現行版與 2024 舊版的差旅辦法都寫了這個統編
    assert hits[0].chunk.doc_id.startswith("FIN-002")


def test_bm25_respects_filters(indexed_store):
    hits = BM25Retriever(indexed_store).retrieve("特休遞延", k=10, filters={"status": "現行"})
    assert hits and all(h.chunk.metadata["status"] == "現行" for h in hits)


def test_bm25_rebuilds_index_after_reingest(embedder):
    store = InMemoryVectorStore()
    retriever = BM25Retriever(store)
    assert retriever.retrieve("密碼", k=3) == []
    chunk = Chunk("IT#0", "IT", "資安", "", "s", "密碼每 180 天更換")
    store.upsert([chunk], embedder.embed([chunk.text_for_embedding()]))
    assert retriever.retrieve("密碼", k=3)[0].chunk.chunk_id == "IT#0"


def test_rrf_rewards_chunks_ranked_high_by_both():
    vector = [_hit("A#0"), _hit("B#0"), _hit("C#0")]
    keyword = [_hit("B#0"), _hit("C#0"), _hit("D#0")]
    fused = reciprocal_rank_fusion([vector, keyword], rrf_k=60)
    assert [h.chunk.chunk_id for h in fused] == ["B#0", "C#0", "A#0", "D#0"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_hybrid_retriever(embedder, indexed_store):
    hybrid = HybridRetriever(
        [VectorRetriever(embedder, indexed_store), BM25Retriever(indexed_store)], candidate_k=10
    )
    hits = hybrid.retrieve("發票統一編號 24681357", k=3, filters={"status": "現行"})
    assert len(hits) == 3
    assert hits[0].chunk.doc_id == "FIN-002"


class KeywordReranker:
    """假 reranker：內容含關鍵字者得高分。"""

    def __init__(self, keyword: str):
        self.keyword = keyword

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 if self.keyword in t else 0.0 for t in texts]


class FixedRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.asked_k = None

    def retrieve(self, query, k, filters=None):
        self.asked_k = k
        return self.hits[:k]


def test_reranking_retriever_reorders_candidates():
    base = FixedRetriever([_hit("A#0"), _hit("B#0"), _hit("C#0")])
    hits = RerankingRetriever(base, KeywordReranker("C#0"), candidate_k=20).retrieve("q", k=2)
    assert base.asked_k == 20
    assert [h.chunk.chunk_id for h in hits] == ["C#0", "A#0"]


def test_reranking_retriever_handles_no_candidates():
    assert RerankingRetriever(FixedRetriever([]), KeywordReranker("x")).retrieve("q", 5) == []


def _tei(handler) -> TEIReranker:
    settings = Settings(reranker_url="http://localhost:18080")
    client = httpx.Client(base_url=settings.reranker_url, transport=httpx.MockTransport(handler))
    return TEIReranker(settings, client)


def test_tei_reranker_maps_scores_back_to_input_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        # TEI 依分數高低回傳，需用 index 對回原本順序
        return httpx.Response(200, json=[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])

    assert _tei(handler).score("q", ["a", "b"]) == [0.1, 0.9]


def test_tei_reranker_errors_are_readable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    with pytest.raises(ServiceUnavailable, match="Reranker"):
        _tei(handler).score("q", ["a"])


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("特休可以遞延嗎？", {"status": "現行"}),
        ("2023 年的時候特休可以遞延嗎？", {}),
        ("舊版規定病假怎麼請？", {}),
    ],
)
def test_infer_filters(question, expected):
    assert infer_filters(question) == expected


def test_pipeline_hides_obsolete_versions_unless_asked(embedder, indexed_store):
    pipe = RAGPipeline(BM25Retriever(indexed_store), EchoChat(), top_k=10, version_filter=True)
    current = {h.chunk.doc_id for h in pipe.retrieve("特休可以遞延嗎")}
    assert "HR-001" in current
    assert not {"HR-001-2021", "HR-001-2023"} & current
    assert "HR-001-2023" in {h.chunk.doc_id for h in pipe.retrieve("2023 年特休可以遞延嗎")}
    assert "HR-001-2023" in {
        h.chunk.doc_id for h in pipe.retrieve("特休可以遞延嗎", all_versions=True)
    }


def test_build_retriever_follows_settings(embedder, indexed_store):
    vector = build_retriever(Settings(retrieval_mode="vector"), embedder, indexed_store)
    assert isinstance(vector, VectorRetriever)
    hybrid = build_retriever(Settings(retrieval_mode="hybrid"), embedder, indexed_store)
    assert isinstance(hybrid, HybridRetriever)
    reranked = build_retriever(
        Settings(retrieval_mode="hybrid", reranker_url="http://localhost:18080"),
        embedder,
        indexed_store,
    )
    assert isinstance(reranked, RerankingRetriever)
    assert isinstance(reranked.base, HybridRetriever)
