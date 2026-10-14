"""BM25 關鍵字檢索。

向量檢索擅長「意思相近」，但對編號、金額、專有名詞（例如 SO-2026-0007、24681357、MFA）
常常不夠準；BM25 只看字面是否出現，兩者互補。

中文沒有空白分詞，這裡用「相鄰兩字」（bigram）當作詞，英數字則整串當一個詞。
不需要額外的斷詞套件，對規章這類用語固定的文件效果已經不錯。
索引放在記憶體，適合數萬個 chunk 以內；更大量時可改用 ParadeDB（pg_search）等資料庫內建的 BM25。
"""

import math
import re
from collections import Counter
from typing import Any

from agentic_rag.models import Chunk, RetrievedChunk
from agentic_rag.store.base import VectorStore
from agentic_rag.store.memory import matches

CJK_RUN = re.compile(r"[㐀-鿿]+")
WORD = re.compile(r"[a-z0-9]+(?:[-.@][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = WORD.findall(text)
    for run in CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(c.text_for_embedding())) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_len = sum(self.lengths) / len(chunks) if chunks else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        results = []
        for chunk, tf, length in zip(self.chunks, self.term_freqs, self.lengths, strict=True):
            if not matches(chunk.metadata, filters):
                continue
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                    score += self.idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                results.append(RetrievedChunk(chunk=chunk, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


class BM25Retriever:
    """從 store 讀出所有 chunk 建索引；chunk 數量變了（重新匯入）就自動重建。"""

    def __init__(self, store: VectorStore):
        self.store = store
        self._index: BM25Index | None = None
        self._indexed_count = -1

    def _get_index(self) -> BM25Index:
        count = self.store.count()
        if self._index is None or count != self._indexed_count:
            self._index = BM25Index(self.store.all_chunks())
            self._indexed_count = count
        return self._index

    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        return self._get_index().search(query, k, filters)
