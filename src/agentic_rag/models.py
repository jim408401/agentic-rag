"""在各層之間傳遞的資料結構。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section_path: str
    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def text_for_embedding(self) -> str:
        # 把標題與章節路徑一起 embed，切塊後才不會失去「這段在講哪份文件的哪一節」
        header = f"{self.title} > {self.section_path}" if self.section_path else self.title
        return f"{header}\n{self.content}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk]
