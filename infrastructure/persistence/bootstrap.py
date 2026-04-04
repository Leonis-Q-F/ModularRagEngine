from __future__ import annotations

from psycopg import sql

from .connection import PostgresConnectionFactory


def bootstrap_schema(connection_factory: PostgresConnectionFactory) -> None:
    """初始化 schema、表和索引。"""
    with connection_factory.connect() as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(connection_factory.schema)))

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    namespace_id UUID PRIMARY KEY,
                    namespace_key TEXT NOT NULL UNIQUE,
                    namespace_name TEXT NOT NULL,
                    namespace_type TEXT,
                    external_ref TEXT,
                    status TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    deleted_at TIMESTAMPTZ
                )
                """
            ).format(connection_factory.table("rag_namespaces"))
        )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    doc_id UUID PRIMARY KEY,
                    namespace_id UUID NOT NULL REFERENCES {}(namespace_id),
                    dedupe_key TEXT NOT NULL,
                    external_doc_id TEXT,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    source_uri TEXT,
                    language TEXT,
                    status TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    parser_name TEXT NOT NULL,
                    parser_version TEXT,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    parsed_md_content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    deleted_at TIMESTAMPTZ,
                    UNIQUE(namespace_id, dedupe_key)
                )
                """
            ).format(
                connection_factory.table("rag_source_documents"),
                connection_factory.table("rag_namespaces"),
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    parent_id UUID PRIMARY KEY,
                    namespace_id UUID NOT NULL REFERENCES {}(namespace_id),
                    doc_id UUID NOT NULL REFERENCES {}(doc_id),
                    chunk_version TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    language TEXT NOT NULL,
                    token_count INT NOT NULL,
                    heading_level INT,
                    header_path TEXT[] NOT NULL DEFAULT '{{}}',
                    split_route TEXT NOT NULL,
                    start_line INT,
                    end_line INT,
                    start_char INT,
                    end_char INT,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL,
                    deleted_at TIMESTAMPTZ
                )
                """
            ).format(
                connection_factory.table("rag_parent_chunks"),
                connection_factory.table("rag_namespaces"),
                connection_factory.table("rag_source_documents"),
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    block_id UUID PRIMARY KEY,
                    namespace_id UUID NOT NULL REFERENCES {}(namespace_id),
                    doc_id UUID NOT NULL REFERENCES {}(doc_id),
                    parent_id UUID NOT NULL REFERENCES {}(parent_id),
                    chunk_version TEXT NOT NULL,
                    child_index INT NOT NULL,
                    content TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    language TEXT NOT NULL,
                    token_count INT NOT NULL,
                    start_char INT,
                    end_char INT,
                    start_token INT,
                    end_token INT,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL,
                    deleted_at TIMESTAMPTZ
                )
                """
            ).format(
                connection_factory.table("rag_child_blocks"),
                connection_factory.table("rag_namespaces"),
                connection_factory.table("rag_source_documents"),
                connection_factory.table("rag_parent_chunks"),
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    index_id UUID PRIMARY KEY,
                    namespace_id UUID NOT NULL REFERENCES {}(namespace_id),
                    index_version TEXT NOT NULL,
                    chunk_version TEXT NOT NULL,
                    index_name TEXT NOT NULL,
                    retrieval_strategy TEXT NOT NULL,
                    retrieval_text_policy TEXT NOT NULL,
                    embedding_provider TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INT NOT NULL,
                    sparse_provider TEXT NOT NULL,
                    zh_collection_name TEXT,
                    en_collection_name TEXT,
                    status TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT FALSE,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    activated_at TIMESTAMPTZ,
                    deleted_at TIMESTAMPTZ,
                    UNIQUE(namespace_id, index_version)
                )
                """
            ).format(
                connection_factory.table("rag_retrieval_indexes"),
                connection_factory.table("rag_namespaces"),
            )
        )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    entry_id UUID PRIMARY KEY,
                    index_id UUID NOT NULL REFERENCES {}(index_id),
                    namespace_id UUID NOT NULL REFERENCES {}(namespace_id),
                    doc_id UUID NOT NULL REFERENCES {}(doc_id),
                    parent_id UUID NOT NULL REFERENCES {}(parent_id),
                    block_id UUID NOT NULL REFERENCES {}(block_id),
                    chunk_version TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    child_index INT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    language TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    vector_status TEXT NOT NULL,
                    vector_collection TEXT NOT NULL,
                    vector_primary_key TEXT NOT NULL,
                    indexed_at TIMESTAMPTZ,
                    last_error TEXT,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    deleted_at TIMESTAMPTZ
                )
                """
            ).format(
                connection_factory.table("rag_index_entries"),
                connection_factory.table("rag_retrieval_indexes"),
                connection_factory.table("rag_namespaces"),
                connection_factory.table("rag_source_documents"),
                connection_factory.table("rag_parent_chunks"),
                connection_factory.table("rag_child_blocks"),
            )
        )

        for statement in _index_statements(connection_factory):
            cur.execute(statement)


def _index_statements(connection_factory: PostgresConnectionFactory) -> list[sql.Composed]:
    """返回需要补齐的数据库索引语句。"""
    return [
        sql.SQL("CREATE INDEX IF NOT EXISTS rag_source_documents_namespace_idx ON {} (namespace_id)").format(
            connection_factory.table("rag_source_documents")
        ),
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS rag_parent_chunks_doc_active_idx ON {} (doc_id, chunk_index) WHERE deleted_at IS NULL"
        ).format(connection_factory.table("rag_parent_chunks")),
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS rag_child_blocks_doc_active_idx ON {} (doc_id, child_index) WHERE deleted_at IS NULL"
        ).format(connection_factory.table("rag_child_blocks")),
        sql.SQL("CREATE INDEX IF NOT EXISTS rag_indexes_namespace_idx ON {} (namespace_id, is_active)").format(
            connection_factory.table("rag_retrieval_indexes")
        ),
        sql.SQL("CREATE INDEX IF NOT EXISTS rag_index_entries_index_idx ON {} (index_id, doc_id)").format(
            connection_factory.table("rag_index_entries")
        ),
    ]
