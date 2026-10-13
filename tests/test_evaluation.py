import pytest

from agentic_rag.evaluation.metrics import hit_at_k, recall_at_k, reciprocal_rank
from agentic_rag.evaluation.runner import evaluate, load_golden
from tests.conftest import ROOT


def test_metrics():
    assert hit_at_k(["A", "B"], ["B"]) == 1.0
    assert hit_at_k(["A"], ["B"]) == 0.0
    assert recall_at_k(["A", "B"], ["B", "C"]) == 0.5
    # 同一份文件的多個 chunk 只算一次名次
    assert reciprocal_rank(["A", "A", "B"], ["B"]) == pytest.approx(0.5)


def test_golden_set_is_valid():
    items = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=3)
    assert len({i.id for i in items}) == len(items)
    assert any(i.type == "unanswerable" for i in items)
    phase1 = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=1)
    assert all(i.phase == 1 for i in phase1)


def test_evaluate_reports_summary(pipeline):
    items = load_golden(ROOT / "eval" / "golden.jsonl", max_phase=1)
    report = evaluate(pipeline, items, k=5, generate=True)
    assert report.summary["n"] == len(items)
    assert 0.0 <= report.summary["hit@5"] <= 1.0
    assert "refusal_accuracy" in report.summary
