import pytest

from agentic_rag.ingestion.chunking import chunk_document, split_sections, window_split
from agentic_rag.models import Document


def test_sections_keep_heading_path():
    md = "# 辦法\n\n## 第一章\n\n### 第 1 條\n內容一\n\n## 第二章\n內容二"
    sections = split_sections(md)
    assert [s.path for s in sections] == [["辦法", "第一章", "第 1 條"], ["辦法", "第二章"]]
    assert sections[0].text == "內容一"


def test_window_split_respects_size_and_overlap():
    text = "。".join(f"第{i}句話內容" for i in range(200))
    pieces = window_split(text, size=100, overlap=20)
    assert len(pieces) > 1
    assert all(len(p) <= 100 for p in pieces)
    # 相鄰兩塊之間有重疊
    assert pieces[0][-5:] in pieces[1]


def test_window_split_prefers_sentence_boundary():
    text = "甲" * 70 + "。" + "乙" * 70
    first = window_split(text, size=100, overlap=10)[0]
    assert first.endswith("。")


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        window_split("abc" * 100, size=10, overlap=10)


def test_chunk_document_ids_and_section_path():
    doc = Document(
        doc_id="D1",
        title="辦法",
        source="x.md",
        text="# 辦法\n## 第一章\n" + "內容。" * 300,
        metadata={"status": "現行"},
    )
    chunks = chunk_document(doc, size=200, overlap=20)
    assert len(chunks) > 1
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].section_path == "第一章"
    assert chunks[0].metadata == {"status": "現行"}
    assert chunks[0].text_for_embedding().startswith("辦法 > 第一章\n")
