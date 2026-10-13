"""Phase 1 基線切塊：先依 Markdown 標題分節，節內再用固定長度 + 重疊切開。

Phase 2 會在這裡加入 Semantic Chunking、Parent-Child 等策略，並用評估集比較。
"""

import re
from dataclasses import dataclass

from agentic_rag.models import Chunk, Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# 優先在段落、句號、換行處斷開，避免把一句話切成兩半
BREAK_CHARS = ("\n\n", "。", "\n", "；", "，", " ")


@dataclass
class Section:
    path: list[str]
    text: str


def split_sections(markdown: str) -> list[Section]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(Section(path=[t for _, t in stack], text=text))
        buf.clear()

    for line in markdown.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def window_split(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("chunk_overlap 必須小於 chunk_size")
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # 在視窗後半段找最後一個自然斷點
            for sep in BREAK_CHARS:
                cut = text.rfind(sep, start + size // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in split_sections(doc.text):
        # 第一層標題通常就是文件名稱，不放進章節路徑
        path = section.path[1:] if section.path and section.path[0] == doc.title else section.path
        section_path = " > ".join(path)
        for piece in window_split(section.text, size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):04d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section_path=section_path,
                    source=doc.source,
                    content=piece,
                    metadata=dict(doc.metadata),
                )
            )
    return chunks
