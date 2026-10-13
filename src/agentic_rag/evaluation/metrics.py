"""檢索與生成的基本指標。Phase 2 會再加上 RAGAS 的 Faithfulness 等 LLM 評分。"""


def dedupe(doc_ids: list[str]) -> list[str]:
    seen: list[str] = []
    for d in doc_ids:
        if d not in seen:
            seen.append(d)
    return seen


def hit_at_k(retrieved: list[str], expected: list[str]) -> float:
    """至少一份應引用的文件出現在結果中就算命中。"""
    return float(any(d in expected for d in retrieved))


def recall_at_k(retrieved: list[str], expected: list[str]) -> float:
    """應引用的文件中，有幾成被取回。"""
    if not expected:
        return 0.0
    return sum(d in retrieved for d in expected) / len(expected)


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """第一份正確文件排第 n 名就得 1/n，用來看正確文件排得夠不夠前面。"""
    for rank, d in enumerate(dedupe(retrieved), start=1):
        if d in expected:
            return 1.0 / rank
    return 0.0


def is_refusal(answer: str, refusal_phrase: str) -> bool:
    return refusal_phrase in answer
