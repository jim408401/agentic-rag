import pytest

from agentic_rag.evaluation.metrics import (
    fact_recall,
    has_citation,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from agentic_rag.evaluation.runner import evaluate, load_golden
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
    doc_ids = {d.doc_id for d in load_directory(HANDBOOK)}
    items = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=3)
    for item in items:
        assert set(item.expected_doc_ids) <= doc_ids, item.id
        if item.type != "unanswerable":
            assert item.must_include, item.id


def test_evaluate_reports_summary(pipeline):
    items = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=1)
    report = evaluate(pipeline, items, k=5, generate=True)
    assert report.summary["n"] == len(items)
    assert 0.0 <= report.summary["hit@5"] <= 1.0
    assert "refusal_accuracy" in report.summary
    assert 0.0 <= report.summary["fact_recall"] <= 1.0
    # 假 LLM 固定回「測試答案 [1]」，每題都有引用
    assert report.summary["citation_rate"] == 1.0
