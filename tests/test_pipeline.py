from agentic_rag.generation.prompts import NO_ANSWER
from agentic_rag.ingestion.pipeline import ingest_directory
from agentic_rag.rag import RAGPipeline
from agentic_rag.retrieval.retriever import VectorRetriever
from agentic_rag.store.memory import InMemoryVectorStore
from tests.conftest import HANDBOOK, EchoChat


def test_reingest_does_not_duplicate_chunks(embedder, indexed_store):
    before = indexed_store.count()
    ingest_directory(HANDBOOK, embedder, indexed_store, chunk_size=400, chunk_overlap=60)
    assert indexed_store.count() == before


def test_retrieval_finds_relevant_document(pipeline):
    hits = pipeline.retriever.retrieve(
        "連續請病假要檢附醫療院所證明嗎", k=3, filters={"status": "現行"}
    )
    assert "HR-001" in {h.chunk.doc_id for h in hits}


def test_filter_excludes_obsolete_version(pipeline):
    hits = pipeline.retriever.retrieve("特休可以遞延嗎", k=10, filters={"status": "現行"})
    assert hits
    assert all(h.chunk.doc_id != "HR-001-2023" for h in hits)


def test_ask_builds_cited_prompt(pipeline):
    answer = pipeline.ask("金卡客戶退貨期限幾天")
    assert answer.text == "測試答案 [1]"
    assert len(answer.sources) == 5
    system, user = pipeline.llm.calls[0]
    assert NO_ANSWER in system["content"]
    assert "<context>" in user["content"] and "[1]" in user["content"]
    assert "問題：金卡客戶退貨期限幾天" in user["content"]


def test_empty_index_returns_no_answer_without_calling_llm(embedder):
    llm = EchoChat()
    pipe = RAGPipeline(VectorRetriever(embedder, InMemoryVectorStore()), llm, top_k=5)
    assert pipe.ask("任何問題").text == NO_ANSWER
    assert llm.calls == []
