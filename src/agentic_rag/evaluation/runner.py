"""用 eval/golden.jsonl 跑評估，輸出逐題結果與彙總分數。"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentic_rag.evaluation.metrics import (
    fact_recall,
    has_citation,
    hit_at_k,
    is_refusal,
    recall_at_k,
    reciprocal_rank,
    section_recall,
    section_reciprocal_rank,
)
from agentic_rag.generation.prompts import NO_ANSWER
from agentic_rag.rag import RAGPipeline


@dataclass
class GoldenItem:
    id: str
    question: str
    reference_answer: str
    expected_doc_ids: list[str]
    type: str
    phase: int
    # 答案所在章節，格式「文件#章節片段」，用來檢查是否取回「對的段落」而不只是對的文件
    expected_sections: list[str] = field(default_factory=list)
    # 答案必須提到的關鍵事實，「|」分隔同義寫法；無答案題留空
    must_include: list[str] = field(default_factory=list)


@dataclass
class ItemResult:
    id: str
    type: str
    question: str
    retrieved_doc_ids: list[str]
    expected_doc_ids: list[str]
    hit: float | None
    recall: float | None
    rr: float | None
    retrieved_sections: list[str] = field(default_factory=list)
    section_hit: float | None = None
    section_recall: float | None = None
    section_rr: float | None = None
    # 前 k 筆中來自已廢止文件的比例（版本題不計）
    obsolete: float | None = None
    answer: str | None = None
    refusal_correct: float | None = None
    fact_recall: float | None = None
    cited: float | None = None
    latency_s: float = 0.0


@dataclass
class Report:
    summary: dict[str, float]
    items: list[ItemResult] = field(default_factory=list)


def load_golden(path: Path, max_phase: int) -> list[GoldenItem]:
    items = [
        GoldenItem(**json.loads(line)) for line in path.read_text("utf-8").splitlines() if line
    ]
    return [i for i in items if i.phase <= max_phase]


def _mean(values: list[float | None]) -> float:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def evaluate(
    pipeline: RAGPipeline,
    items: list[GoldenItem],
    generate: bool,
    on_item: Callable[[ItemResult], None] | None = None,
) -> Report:
    k = pipeline.top_k
    results: list[ItemResult] = []
    for item in items:
        start = time.perf_counter()
        hits = pipeline.retrieve(item.question)
        retrieved = [h.chunk.doc_id for h in hits]
        answerable = bool(item.expected_doc_ids)
        result = ItemResult(
            id=item.id,
            type=item.type,
            question=item.question,
            retrieved_doc_ids=retrieved,
            expected_doc_ids=item.expected_doc_ids,
            hit=hit_at_k(retrieved, item.expected_doc_ids) if answerable else None,
            recall=recall_at_k(retrieved, item.expected_doc_ids) if answerable else None,
            rr=reciprocal_rank(retrieved, item.expected_doc_ids) if answerable else None,
        )
        if item.expected_sections:
            sections = [f"{h.chunk.doc_id}#{h.chunk.section_path}" for h in hits]
            result.retrieved_sections = sections
            result.section_recall = section_recall(sections, item.expected_sections)
            result.section_hit = float(result.section_recall > 0)
            result.section_rr = section_reciprocal_rank(sections, item.expected_sections)
        if hits and item.type != "version":
            result.obsolete = sum(h.chunk.metadata.get("status") == "已廢止" for h in hits) / len(
                hits
            )
        if generate:
            answer = pipeline.generate(item.question, hits)
            result.answer = answer.text
            refused = is_refusal(answer.text, NO_ANSWER)
            # 無答案題應拒答；有答案題不應拒答
            result.refusal_correct = float(refused != answerable)
            if answerable:
                result.fact_recall = fact_recall(answer.text, item.must_include)
                result.cited = float(has_citation(answer.text))
        result.latency_s = round(time.perf_counter() - start, 3)
        results.append(result)
        if on_item:
            on_item(result)

    summary = {
        "n": len(results),
        f"hit@{k}": _mean([r.hit for r in results]),
        f"recall@{k}": _mean([r.recall for r in results]),
        "mrr": _mean([r.rr for r in results]),
        f"section_hit@{k}": _mean([r.section_hit for r in results]),
        f"section_recall@{k}": _mean([r.section_recall for r in results]),
        "section_mrr": _mean([r.section_rr for r in results]),
        f"obsolete@{k}": _mean([r.obsolete for r in results]),
        "avg_latency_s": _mean([r.latency_s for r in results]),
    }
    if generate:
        summary["refusal_accuracy"] = _mean([r.refusal_correct for r in results])
        summary["fact_recall"] = _mean([r.fact_recall for r in results])
        summary["citation_rate"] = _mean([r.cited for r in results])
    return Report(summary=summary, items=results)


def save_report(report: Report, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    payload = {"summary": report.summary, "items": [asdict(i) for i in report.items]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
