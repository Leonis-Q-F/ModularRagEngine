from __future__ import annotations

from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from ...application.contracts import ChunkBundle
from ...domain.entities import IndexEntry, Namespace, ParentChunk, RetrievalIndex, SourceDocument, utc_now
from ...domain.exceptions import IndexNotFoundError, NamespaceNotFoundError
from .connection import PostgresConnectionFactory
from .mappers import (
    child_block_from_row,
    index_entry_from_row,
    namespace_from_row,
    parent_chunk_from_row,
    retrieval_index_from_row,
    source_document_from_row,
)


class BaseRepository:
    """为具体仓储提供共享连接能力。"""

    def __init__(self, connection_factory: PostgresConnectionFactory) -> None:
        self._connection_factory = connection_factory

    def _table(self, name: str):
        return self._connection_factory.table(name)

    def _connect(self):
        return self._connection_factory.connect()


class NamespaceRepository(BaseRepository):
    """管理 namespace 元数据。"""

    def ensure_namespace(self, namespace_key: str, namespace_name: str | None = None) -> Namespace:
        now = utc_now()
        namespace = Namespace(namespace_key=namespace_key, namespace_name=namespace_name or namespace_key, updated_at=now)
        query = sql.SQL(
            """
            INSERT INTO {} (
                namespace_id,
                namespace_key,
                namespace_name,
                namespace_type,
                external_ref,
                status,
                metadata,
                created_at,
                updated_at,
                deleted_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (namespace_key)
            DO UPDATE SET
                updated_at = EXCLUDED.updated_at,
                deleted_at = NULL
            RETURNING *
            """
        ).format(self._table("rag_namespaces"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                query,
                (
                    namespace.namespace_id,
                    namespace.namespace_key,
                    namespace.namespace_name,
                    namespace.namespace_type,
                    namespace.external_ref,
                    namespace.status,
                    Jsonb(namespace.metadata),
                    namespace.created_at,
                    namespace.updated_at,
                    namespace.deleted_at,
                ),
            )
            return namespace_from_row(cur.fetchone())

    def get_namespace(self, namespace_id: UUID | None = None, namespace_key: str | None = None) -> Namespace:
        if namespace_id is None and not namespace_key:
            raise NamespaceNotFoundError("namespace 不存在。")

        if namespace_id is not None:
            query = sql.SQL("SELECT * FROM {} WHERE namespace_id = %s AND deleted_at IS NULL").format(
                self._table("rag_namespaces")
            )
            params = (namespace_id,)
        else:
            query = sql.SQL("SELECT * FROM {} WHERE namespace_key = %s AND deleted_at IS NULL").format(
                self._table("rag_namespaces")
            )
            params = (namespace_key,)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            return namespace_from_row(cur.fetchone())


class ContentRepository(BaseRepository):
    """管理文档、父块与子块内容。"""

    def upsert_source_documents(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        query = sql.SQL(
            """
            INSERT INTO {} (
                doc_id,
                namespace_id,
                dedupe_key,
                external_doc_id,
                file_name,
                file_type,
                source_uri,
                language,
                status,
                content_sha256,
                parser_name,
                parser_version,
                metadata,
                parsed_md_content,
                created_at,
                updated_at,
                deleted_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (namespace_id, dedupe_key)
            DO UPDATE SET
                external_doc_id = EXCLUDED.external_doc_id,
                file_name = EXCLUDED.file_name,
                file_type = EXCLUDED.file_type,
                source_uri = EXCLUDED.source_uri,
                language = EXCLUDED.language,
                status = EXCLUDED.status,
                content_sha256 = EXCLUDED.content_sha256,
                parser_name = EXCLUDED.parser_name,
                parser_version = EXCLUDED.parser_version,
                metadata = EXCLUDED.metadata,
                parsed_md_content = EXCLUDED.parsed_md_content,
                updated_at = EXCLUDED.updated_at,
                deleted_at = NULL
            RETURNING *
            """
        ).format(self._table("rag_source_documents"))

        stored: list[SourceDocument] = []
        with self._connect() as conn, conn.cursor() as cur:
            for document in documents:
                dedupe_key = document.external_doc_id or document.content_sha256
                cur.execute(
                    query,
                    (
                        document.doc_id,
                        document.namespace_id,
                        dedupe_key,
                        document.external_doc_id,
                        document.file_name,
                        document.file_type,
                        document.source_uri,
                        document.language,
                        document.status,
                        document.content_sha256,
                        document.parser_name,
                        document.parser_version,
                        Jsonb(document.metadata),
                        document.parsed_md_content,
                        document.created_at,
                        utc_now(),
                        document.deleted_at,
                    ),
                )
                stored.append(source_document_from_row(cur.fetchone()))
        return stored

    def replace_document_chunks(self, bundle: ChunkBundle) -> ChunkBundle:
        now = utc_now()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "UPDATE {} SET is_active = FALSE, deleted_at = %s WHERE doc_id = %s AND is_active = TRUE AND deleted_at IS NULL"
                ).format(self._table("rag_parent_chunks")),
                (now, bundle.source_document.doc_id),
            )
            cur.execute(
                sql.SQL(
                    "UPDATE {} SET is_active = FALSE, deleted_at = %s WHERE doc_id = %s AND is_active = TRUE AND deleted_at IS NULL"
                ).format(self._table("rag_child_blocks")),
                (now, bundle.source_document.doc_id),
            )

            parent_query = sql.SQL(
                """
                INSERT INTO {} (
                    parent_id,
                    namespace_id,
                    doc_id,
                    chunk_version,
                    chunk_index,
                    content,
                    content_sha256,
                    language,
                    token_count,
                    heading_level,
                    header_path,
                    split_route,
                    start_line,
                    end_line,
                    start_char,
                    end_char,
                    metadata,
                    is_active,
                    created_at,
                    deleted_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            ).format(self._table("rag_parent_chunks"))
            for parent_chunk in bundle.parent_chunks:
                cur.execute(
                    parent_query,
                    (
                        parent_chunk.parent_id,
                        parent_chunk.namespace_id,
                        parent_chunk.doc_id,
                        parent_chunk.chunk_version,
                        parent_chunk.chunk_index,
                        parent_chunk.content,
                        parent_chunk.content_sha256,
                        parent_chunk.language,
                        parent_chunk.token_count,
                        parent_chunk.heading_level,
                        list(parent_chunk.header_path),
                        parent_chunk.split_route,
                        parent_chunk.start_line,
                        parent_chunk.end_line,
                        parent_chunk.start_char,
                        parent_chunk.end_char,
                        Jsonb(parent_chunk.metadata),
                        parent_chunk.is_active,
                        parent_chunk.created_at,
                        parent_chunk.deleted_at,
                    ),
                )

            child_query = sql.SQL(
                """
                INSERT INTO {} (
                    block_id,
                    namespace_id,
                    doc_id,
                    parent_id,
                    chunk_version,
                    child_index,
                    content,
                    content_sha256,
                    language,
                    token_count,
                    start_char,
                    end_char,
                    start_token,
                    end_token,
                    metadata,
                    is_active,
                    created_at,
                    deleted_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            ).format(self._table("rag_child_blocks"))
            for child_block in bundle.child_blocks:
                cur.execute(
                    child_query,
                    (
                        child_block.block_id,
                        child_block.namespace_id,
                        child_block.doc_id,
                        child_block.parent_id,
                        child_block.chunk_version,
                        child_block.child_index,
                        child_block.content,
                        child_block.content_sha256,
                        child_block.language,
                        child_block.token_count,
                        child_block.start_char,
                        child_block.end_char,
                        child_block.start_token,
                        child_block.end_token,
                        Jsonb(child_block.metadata),
                        child_block.is_active,
                        child_block.created_at,
                        child_block.deleted_at,
                    ),
                )
        return bundle

    def list_child_blocks(self, namespace_id: UUID, doc_ids: list[UUID] | None = None):
        base = sql.SQL(
            """
            SELECT *
            FROM {}
            WHERE namespace_id = %s
              AND is_active = TRUE
              AND deleted_at IS NULL
            """
        ).format(self._table("rag_child_blocks"))
        params: list[object] = [namespace_id]
        if doc_ids:
            base += sql.SQL(" AND doc_id = ANY(%s)")
            params.append(doc_ids)
        base += sql.SQL(" ORDER BY doc_id::text, child_index, created_at")

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(base, params)
            return [child_block_from_row(row) for row in cur.fetchall()]

    def get_parent_chunks(self, parent_ids: list[UUID]) -> list[ParentChunk]:
        if not parent_ids:
            return []
        query = sql.SQL(
            """
            SELECT *
            FROM {}
            WHERE parent_id = ANY(%s)
              AND is_active = TRUE
              AND deleted_at IS NULL
            ORDER BY doc_id::text, chunk_index
            """
        ).format(self._table("rag_parent_chunks"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (parent_ids,))
            return [parent_chunk_from_row(row) for row in cur.fetchall()]

    def get_parent_chunk_window(self, parent_id: UUID, window: int) -> list[ParentChunk]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT doc_id, chunk_index
                    FROM {}
                    WHERE parent_id = %s
                      AND is_active = TRUE
                      AND deleted_at IS NULL
                    """
                ).format(self._table("rag_parent_chunks")),
                (parent_id,),
            )
            seed = cur.fetchone()
            if seed is None:
                return []

            start = max(seed["chunk_index"] - window, 0)
            end = seed["chunk_index"] + window + 1
            cur.execute(
                sql.SQL(
                    """
                    SELECT *
                    FROM {}
                    WHERE doc_id = %s
                      AND chunk_index >= %s
                      AND chunk_index < %s
                      AND is_active = TRUE
                      AND deleted_at IS NULL
                    ORDER BY chunk_index
                    """
                ).format(self._table("rag_parent_chunks")),
                (seed["doc_id"], start, end),
            )
            return [parent_chunk_from_row(row) for row in cur.fetchall()]


class IndexRepository(BaseRepository):
    """管理索引元数据与索引条目。"""

    def create_index(self, index: RetrievalIndex) -> RetrievalIndex:
        query = sql.SQL(
            """
            INSERT INTO {} (
                index_id,
                namespace_id,
                index_version,
                chunk_version,
                index_name,
                retrieval_strategy,
                retrieval_text_policy,
                embedding_provider,
                embedding_model,
                embedding_dim,
                sparse_provider,
                zh_collection_name,
                en_collection_name,
                status,
                is_active,
                metadata,
                created_at,
                updated_at,
                activated_at,
                deleted_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """
        ).format(self._table("rag_retrieval_indexes"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                query,
                (
                    index.index_id,
                    index.namespace_id,
                    index.index_version,
                    index.chunk_version,
                    index.index_name,
                    index.retrieval_strategy,
                    index.retrieval_text_policy,
                    index.embedding_provider,
                    index.embedding_model,
                    index.embedding_dim,
                    index.sparse_provider,
                    index.zh_collection_name,
                    index.en_collection_name,
                    index.status,
                    index.is_active,
                    Jsonb(index.metadata),
                    index.created_at,
                    index.updated_at,
                    index.activated_at,
                    index.deleted_at,
                ),
            )
            return retrieval_index_from_row(cur.fetchone())

    def get_active_index(self, namespace_id: UUID) -> RetrievalIndex | None:
        query = sql.SQL(
            """
            SELECT *
            FROM {}
            WHERE namespace_id = %s
              AND is_active = TRUE
              AND deleted_at IS NULL
            ORDER BY activated_at DESC NULLS LAST, created_at DESC
            LIMIT 1
            """
        ).format(self._table("rag_retrieval_indexes"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (namespace_id,))
            row = cur.fetchone()
            return retrieval_index_from_row(row) if row else None

    def list_indexes(self, namespace_id: UUID) -> list[RetrievalIndex]:
        query = sql.SQL(
            """
            SELECT *
            FROM {}
            WHERE namespace_id = %s
              AND deleted_at IS NULL
            ORDER BY created_at
            """
        ).format(self._table("rag_retrieval_indexes"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (namespace_id,))
            return [retrieval_index_from_row(row) for row in cur.fetchall()]

    def get_index(self, index_id: UUID) -> RetrievalIndex:
        query = sql.SQL(
            """
            SELECT *
            FROM {}
            WHERE index_id = %s
              AND deleted_at IS NULL
            """
        ).format(self._table("rag_retrieval_indexes"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (index_id,))
            row = cur.fetchone()
            if row is None:
                raise IndexNotFoundError("索引不存在。")
            return retrieval_index_from_row(row)

    def save_index_entries(self, entries: list[IndexEntry]) -> list[IndexEntry]:
        query = sql.SQL(
            """
            INSERT INTO {} (
                entry_id,
                index_id,
                namespace_id,
                doc_id,
                parent_id,
                block_id,
                chunk_version,
                index_version,
                child_index,
                file_type,
                file_name,
                language,
                retrieval_text,
                vector_status,
                vector_collection,
                vector_primary_key,
                indexed_at,
                last_error,
                metadata,
                is_active,
                created_at,
                updated_at,
                deleted_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entry_id)
            DO UPDATE SET
                vector_status = EXCLUDED.vector_status,
                vector_collection = EXCLUDED.vector_collection,
                vector_primary_key = EXCLUDED.vector_primary_key,
                indexed_at = EXCLUDED.indexed_at,
                last_error = EXCLUDED.last_error,
                metadata = EXCLUDED.metadata,
                is_active = EXCLUDED.is_active,
                updated_at = EXCLUDED.updated_at,
                deleted_at = EXCLUDED.deleted_at
            RETURNING *
            """
        ).format(self._table("rag_index_entries"))

        saved: list[IndexEntry] = []
        with self._connect() as conn, conn.cursor() as cur:
            for entry in entries:
                entry.vector_primary_key = str(entry.entry_id)
                entry.updated_at = utc_now()
                cur.execute(
                    query,
                    (
                        entry.entry_id,
                        entry.index_id,
                        entry.namespace_id,
                        entry.doc_id,
                        entry.parent_id,
                        entry.block_id,
                        entry.chunk_version,
                        entry.index_version,
                        entry.child_index,
                        entry.file_type,
                        entry.file_name,
                        entry.language,
                        entry.retrieval_text,
                        entry.vector_status,
                        entry.vector_collection,
                        entry.vector_primary_key,
                        entry.indexed_at,
                        entry.last_error,
                        Jsonb(entry.metadata),
                        entry.is_active,
                        entry.created_at,
                        entry.updated_at,
                        entry.deleted_at,
                    ),
                )
                saved.append(index_entry_from_row(cur.fetchone()))
        return saved

    def deactivate_index_entries(self, index_id: UUID, doc_ids: list[UUID] | None = None) -> None:
        now = utc_now()
        query = sql.SQL(
            """
            UPDATE {}
            SET is_active = FALSE,
                deleted_at = %s,
                updated_at = %s
            WHERE index_id = %s
              AND is_active = TRUE
              AND deleted_at IS NULL
            """
        ).format(self._table("rag_index_entries"))
        params: list[object] = [now, now, index_id]
        if doc_ids:
            query += sql.SQL(" AND doc_id = ANY(%s)")
            params.append(doc_ids)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)

    def get_index_entries(self, entry_ids: list[UUID]) -> list[IndexEntry]:
        if not entry_ids:
            return []
        query = sql.SQL(
            """
            SELECT *
            FROM {}
            WHERE entry_id = ANY(%s)
              AND is_active = TRUE
              AND deleted_at IS NULL
            """
        ).format(self._table("rag_index_entries"))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, (entry_ids,))
            return [index_entry_from_row(row) for row in cur.fetchall()]

    def activate_index(self, index_id: UUID) -> RetrievalIndex:
        now = utc_now()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT namespace_id FROM {} WHERE index_id = %s").format(self._table("rag_retrieval_indexes")),
                (index_id,),
            )
            target = cur.fetchone()
            if target is None:
                raise NamespaceNotFoundError("namespace 不存在。")

            namespace_id = target["namespace_id"]
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}
                    SET is_active = FALSE,
                        status = CASE WHEN status != 'failed' THEN 'retired' ELSE status END,
                        updated_at = %s
                    WHERE namespace_id = %s
                      AND index_id != %s
                      AND deleted_at IS NULL
                    """
                ).format(self._table("rag_retrieval_indexes")),
                (now, namespace_id, index_id),
            )
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}
                    SET is_active = TRUE,
                        status = 'ready',
                        activated_at = %s,
                        updated_at = %s,
                        deleted_at = NULL
                    WHERE index_id = %s
                    RETURNING *
                    """
                ).format(self._table("rag_retrieval_indexes")),
                (now, now, index_id),
            )
            active_row = cur.fetchone()

            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}
                    SET is_active = CASE WHEN index_id = %s THEN TRUE ELSE FALSE END,
                        updated_at = %s
                    WHERE namespace_id = %s
                      AND deleted_at IS NULL
                    """
                ).format(self._table("rag_index_entries")),
                (index_id, now, namespace_id),
            )
            return retrieval_index_from_row(active_row)

    def update_index_status(self, index_id: UUID, status: str, is_active: bool | None = None) -> RetrievalIndex:
        now = utc_now()
        if is_active is None:
            query = sql.SQL(
                "UPDATE {} SET status = %s, updated_at = %s WHERE index_id = %s RETURNING *"
            ).format(self._table("rag_retrieval_indexes"))
            params = (status, now, index_id)
        else:
            query = sql.SQL(
                "UPDATE {} SET status = %s, is_active = %s, updated_at = %s WHERE index_id = %s RETURNING *"
            ).format(self._table("rag_retrieval_indexes"))
            params = (status, is_active, now, index_id)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            return retrieval_index_from_row(cur.fetchone())

    def delete_index(self, index_id: UUID) -> RetrievalIndex:
        now = utc_now()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}
                    SET is_active = FALSE,
                        deleted_at = %s,
                        updated_at = %s
                    WHERE index_id = %s
                      AND deleted_at IS NULL
                    """
                ).format(self._table("rag_index_entries")),
                (now, now, index_id),
            )
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {}
                    SET is_active = FALSE,
                        deleted_at = %s,
                        updated_at = %s
                    WHERE index_id = %s
                      AND deleted_at IS NULL
                    RETURNING *
                    """
                ).format(self._table("rag_retrieval_indexes")),
                (now, now, index_id),
            )
            row = cur.fetchone()
            if row is None:
                raise IndexNotFoundError("索引不存在。")
            return retrieval_index_from_row(row)
