"""檢索與生成的基本指標。Phase 2 會再加上 RAGAS 的 Faithfulness 等 LLM 評分。"""

import re
import unicodedata

CITATION = re.compile(r"\[\d+\]")


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


def normalize(text: str) -> str:
    """全形轉半形、去掉空白與千分位逗號，讓「3,200 元」和「３２００元」比對得到。"""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[\s,]", "", text).lower()


def fact_recall(answer: str, facts: list[str]) -> float:
    """答案涵蓋了幾成關鍵事實。每個事實可用「|」列出多種寫法，任一種出現即算涵蓋。"""
    if not facts:
        return 0.0
    text = normalize(answer)
    covered = sum(any(normalize(alt) in text for alt in fact.split("|")) for fact in facts)
    return covered / len(facts)


def has_citation(answer: str) -> bool:
    return bool(CITATION.search(answer))
