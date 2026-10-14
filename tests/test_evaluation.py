import pytest

from agentic_rag.evaluation.metrics import (
    fact_recall,
    has_citation,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    section_matches,
    section_recall,
    section_reciprocal_rank,
)
from agentic_rag.evaluation.runner import evaluate, load_golden
from agentic_rag.ingestion.chunking import chunk_document
from agentic_rag.ingestion.loaders import load_directory
from tests.conftest import HANDBOOK, ROOT


def test_metrics():
    assert hit_at_k(["A", "B"], ["B"]) == 1.0
    assert hit_at_k(["A"], ["B"]) == 0.0
    assert recall_at_k(["A", "B"], ["B", "C"]) == 0.5
    # 同一份文件的多個 chunk 只算一次名次
    assert reciprocal_rank(["A", "A", "B"], ["B"]) == pytest.approx(0.5)


def test_fact_recall_ignores_formatting_and_accepts_alternatives():
    answer = "台北每晚上限 ３,２００ 元，需經部門主管核准 [1]"
    assert fact_recall(answer, ["3200"]) == 1.0
    assert fact_recall(answer, ["3200", "財務長|部門主管"]) == 1.0
    assert fact_recall(answer, ["3200", "財務長"]) == 0.5
    assert fact_recall("每年 14 日", ["14 日"]) == 1.0


def test_section_metrics():
    retrieved = ["HR-001#第二章 > 第 3 條 特別休假", "HR-001#第三章 > 第 8 條 申請方式"]
    assert section_matches(retrieved[1], "HR-001#第 8 條")
    assert not section_matches(retrieved[1], "HR-006#第 8 條")
    assert section_recall(retrieved, ["HR-001#第 8 條", "FIN-002#交通費"]) == 0.5
    assert section_reciprocal_rank(retrieved, ["HR-001#第 8 條"]) == 0.5
    assert section_reciprocal_rank(retrieved, ["HR-001#第 4 條"]) == 0.0


def test_has_citation():
    assert has_citation("每年 14 日 [1]")
    assert not has_citation("每年 14 日")


def test_golden_set_is_valid():
    items = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=3)
    assert len({i.id for i in items}) == len(items)
    assert any(i.type == "unanswerable" for i in items)
    phase1 = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=1)
    assert all(i.phase == 1 for i in phase1)


def test_golden_set_matches_handbook():
    """題目引用的文件都存在；有答案的文件題都要列出關鍵事實，答案正確性才評得到分。"""
    docs = load_directory(HANDBOOK)
    doc_ids = {d.doc_id for d in docs}
    sections = {f"{c.doc_id}#{c.section_path}" for d in docs for c in chunk_document(d, 400, 60)}
    items = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=3)
    for item in items:
        assert set(item.expected_doc_ids) <= doc_ids, item.id
        if item.type != "unanswerable":
            assert item.must_include, item.id
        if item.expected_doc_ids:
            assert item.expected_sections, item.id
        for expected in item.expected_sections:
            matched = [s for s in sections if section_matches(s, expected)]
            # 每個章節標註都要剛好對到一個真實章節，避免片段太短而誤配
            assert len(matched) == 1, (item.id, expected, matched)


def test_evaluate_reports_summary(pipeline):
    items = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=1)
    report = evaluate(pipeline, items, generate=True)
    assert report.summary["n"] == len(items)
    assert 0.0 <= report.summary["hit@5"] <= 1.0
    assert "refusal_accuracy" in report.summary
    assert 0.0 <= report.summary["fact_recall"] <= 1.0
    # 假 LLM 固定回「測試答案 [1]」，每題都有引用
    assert report.summary["citation_rate"] == 1.0
    assert 0.0 <= report.summary["obsolete@5"] <= 1.0
    assert 0.0 <= report.summary["section_mrr"] <= 1.0
