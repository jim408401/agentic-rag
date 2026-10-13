from agentic_rag.ingestion.loaders import load_directory, split_front_matter
from tests.conftest import HANDBOOK


def test_front_matter_is_parsed_into_metadata():
    meta, body = split_front_matter("---\ndoc_id: X-1\nstatus: 現行\n---\n# 標題\n內文")
    assert meta == {"doc_id": "X-1", "status": "現行"}
    assert body.startswith("# 標題")


def test_text_without_front_matter_is_unchanged():
    assert split_front_matter("# 標題") == ({}, "# 標題")


def test_load_handbook():
    docs = {d.doc_id: d for d in load_directory(HANDBOOK)}
    assert {"HR-001", "HR-001-2023", "FIN-002", "IT-003", "CS-004", "PUR-005"} <= docs.keys()
    leave = docs["HR-001"]
    assert leave.title == "請假管理辦法"
    assert leave.metadata["status"] == "現行"
    # YAML 日期轉成字串，才能存進 JSONB
    assert leave.metadata["effective_date"] == "2026-01-01"
    assert docs["HR-001-2023"].metadata["status"] == "已廢止"
