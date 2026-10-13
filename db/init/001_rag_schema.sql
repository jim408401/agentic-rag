-- 文件檢索用的 chunk 表（Phase 1 起使用）
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    title         TEXT NOT NULL,
    section_path  TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL,
    content       TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding     vector(1024) NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_chunks_doc_id_idx ON rag_chunks (doc_id);
CREATE INDEX IF NOT EXISTS rag_chunks_metadata_idx ON rag_chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
