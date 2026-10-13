"""把原始檔案讀成 Document。

Phase 1 只處理 Markdown / 純文字；PDF、Word、PPT 安裝 `docling` 選配套件後走 Docling 轉成 Markdown。
"""

from pathlib import Path

import yaml

from agentic_rag.models import Document

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
DOCLING_SUFFIXES = {".pdf", ".docx", ".pptx", ".html"}


def split_front_matter(raw: str) -> tuple[dict, str]:
    """拆出檔頭 YAML（doc_id、版本、生效日、權限群組等 metadata）。"""
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    meta = yaml.safe_load(parts[1]) or {}
    return meta, parts[2].lstrip("\n")


def _convert_with_docling(path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:  # pragma: no cover - 選配套件
        raise RuntimeError(
            f"要解析 {path.suffix} 請先安裝選配套件：uv sync --extra docling"
        ) from exc
    return DocumentConverter().convert(str(path)).document.export_to_markdown()


def load_file(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    elif suffix in DOCLING_SUFFIXES:
        meta, body = {}, _convert_with_docling(path)
    else:
        raise ValueError(f"不支援的檔案類型：{path}")

    meta = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in meta.items()}
    doc_id = str(meta.pop("doc_id", path.stem))
    title = str(meta.pop("title", path.stem))
    return Document(doc_id=doc_id, title=title, source=str(path), text=body, metadata=meta)


def load_directory(root: Path) -> list[Document]:
    supported = TEXT_SUFFIXES | DOCLING_SUFFIXES
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in supported)
    return [load_file(p) for p in paths]
