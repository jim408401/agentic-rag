"""用 eval/golden.jsonl 跑評估，輸出逐題結果與彙總分數。"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agentic_rag.evaluation.metrics import hit_at_k, is_refusal, recall_at_k, reciprocal_rank
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
    answer: str | None = None
    refusal_correct: float | None = None
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
    k: int,
    generate: bool,
    on_item: Callable[[ItemResult], None] | None = None,
) -> Report:
    results: list[ItemResult] = []
    for item in items:
        start = time.perf_counter()
        hits = pipeline.retriever.retrieve(item.question, k)
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
        if generate:
            answer = pipeline.ask(item.question)
            result.answer = answer.text
            refused = is_refusal(answer.text, NO_ANSWER)
            # 無答案題應拒答；有答案題不應拒答
            result.refusal_correct = float(refused != answerable)
        result.latency_s = round(time.perf_counter() - start, 3)
        results.append(result)
        if on_item:
            on_item(result)

    summary = {
        "n": len(results),
        f"hit@{k}": _mean([r.hit for r in results]),
        f"recall@{k}": _mean([r.recall for r in results]),
        "mrr": _mean([r.rr for r in results]),
        "avg_latency_s": _mean([r.latency_s for r in results]),
    }
    if generate:
        summary["refusal_accuracy"] = _mean([r.refusal_correct for r in results])
    return Report(summary=summary, items=results)


def save_report(report: Report, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    payload = {"summary": report.summary, "items": [asdict(i) for i in report.items]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
