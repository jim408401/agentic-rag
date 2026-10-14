"""PostgreSQL + pgvector 實作。資料表定義在 db/init/001_rag_schema.sql。"""

import json
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agentic_rag.llm import ServiceUnavailable
from agentic_rag.models import Chunk, RetrievedChunk


class PgVectorStore:
    def __init__(self, database_url: str):
        try:
            self._conn = psycopg.connect(
                database_url, autocommit=True, row_factory=dict_row, connect_timeout=5
            )
        except psycopg.OperationalError as exc:
            raise ServiceUnavailable(
                f"連不上資料庫。請確認 docker compose up -d postgres 已啟動：{exc}"
            ) from exc
        register_vector(self._conn)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = [
            (
                c.chunk_id,
                c.doc_id,
                c.title,
                c.section_path,
                c.source,
                c.content,
                Jsonb(c.metadata),
                v,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_chunks
                    (chunk_id, doc_id, title, section_path, source, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    title = EXCLUDED.title,
                    section_path = EXCLUDED.section_path,
                    source = EXCLUDED.source,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )

    def delete_document(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM rag_chunks WHERE doc_id = %s", (doc_id,))

    def search(
        self, vector: list[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        # metadata 過濾在資料庫層做（pre-filter），之後的權限控管也走這條路
        where, params = "", []
        if filters:
            where = "WHERE metadata @> %s::jsonb"
            params.append(json.dumps(filters, ensure_ascii=False))
        rows = self._conn.execute(
            f"""
            SELECT *, 1 - (embedding <=> %s::vector) AS score
            FROM rag_chunks {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            [vector, *params, vector, k],
        ).fetchall()
        return [RetrievedChunk(chunk=_to_chunk(r), score=float(r["score"])) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, title, section_path, source, content, metadata"
            " FROM rag_chunks ORDER BY chunk_id"
        ).fetchall()
        return [_to_chunk(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT count(*) AS n FROM rag_chunks").fetchone()["n"]


def _to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        title=row["title"],
        section_path=row["section_path"],
        source=row["source"],
        content=row["content"],
        metadata=row["metadata"],
    )
