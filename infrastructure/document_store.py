from __future__ import annotations

from uuid import UUID

from ..config import settings
from ..domain.entities import IndexEntry, Namespace, ParentChunk, RetrievalIndex, SourceDocument
from ..domain.value_objects import ChunkBundle
from .persistence.bootstrap import bootstrap_schema
from .persistence.connection import PostgresConnectionConfig, PostgresConnectionFactory
from .persistence.repositories import ContentRepository, IndexRepository, NamespaceRepository


class DocumentStore:
    """基于 PostgreSQL 的内容层与索引元数据门面。"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        schema: str | None = None,
        sslmode: str | None = None,
        connect_timeout: int | None = None,
    ) -> None:
        """初始化连接工厂、建表引导和各子仓储。"""
        config = PostgresConnectionConfig(
            host=host or settings.postgres_host,
            port=port or settings.postgres_port,
            dbname=dbname or settings.postgres_db,
            user=user or settings.postgres_user,
            password=password if password is not None else settings.postgres_password,
            schema=schema or settings.postgres_schema,
            sslmode=sslmode or settings.postgres_sslmode,
            connect_timeout=connect_timeout or settings.postgres_timeout_seconds,
        )
        self._connection_factory = PostgresConnectionFactory(config)
        bootstrap_schema(self._connection_factory)
        self._namespace_repository = NamespaceRepository(self._connection_factory)
        self._content_repository = ContentRepository(self._connection_factory)
        self._index_repository = IndexRepository(self._connection_factory)

    def ensure_namespace(self, namespace_key: str, namespace_name: str | None = None) -> Namespace:
        return self._namespace_repository.ensure_namespace(namespace_key=namespace_key, namespace_name=namespace_name)

    def get_namespace(self, namespace_id: UUID | None = None, namespace_key: str | None = None) -> Namespace:
        return self._namespace_repository.get_namespace(namespace_id=namespace_id, namespace_key=namespace_key)

    def upsert_source_documents(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        return self._content_repository.upsert_source_documents(documents)

    def replace_document_chunks(self, bundle: ChunkBundle) -> ChunkBundle:
        return self._content_repository.replace_document_chunks(bundle)

    def list_child_blocks(self, namespace_id: UUID, doc_ids: list[UUID] | None = None):
        return self._content_repository.list_child_blocks(namespace_id=namespace_id, doc_ids=doc_ids)

    def get_parent_chunks(self, parent_ids: list[UUID]) -> list[ParentChunk]:
        return self._content_repository.get_parent_chunks(parent_ids)

    def get_parent_chunk_window(self, parent_id: UUID, window: int) -> list[ParentChunk]:
        return self._content_repository.get_parent_chunk_window(parent_id=parent_id, window=window)

    def create_index(self, index: RetrievalIndex) -> RetrievalIndex:
        return self._index_repository.create_index(index)

    def get_active_index(self, namespace_id: UUID) -> RetrievalIndex | None:
        return self._index_repository.get_active_index(namespace_id)

    def list_indexes(self, namespace_id: UUID) -> list[RetrievalIndex]:
        return self._index_repository.list_indexes(namespace_id)

    def get_index(self, index_id: UUID) -> RetrievalIndex:
        return self._index_repository.get_index(index_id)

    def save_index_entries(self, entries: list[IndexEntry]) -> list[IndexEntry]:
        return self._index_repository.save_index_entries(entries)

    def deactivate_index_entries(self, index_id: UUID, doc_ids: list[UUID] | None = None) -> None:
        self._index_repository.deactivate_index_entries(index_id=index_id, doc_ids=doc_ids)

    def get_index_entries(self, entry_ids: list[UUID]) -> list[IndexEntry]:
        return self._index_repository.get_index_entries(entry_ids)

    def activate_index(self, index_id: UUID) -> RetrievalIndex:
        return self._index_repository.activate_index(index_id)

    def update_index_status(self, index_id: UUID, status: str, is_active: bool | None = None) -> RetrievalIndex:
        return self._index_repository.update_index_status(index_id=index_id, status=status, is_active=is_active)

    def delete_index(self, index_id: UUID) -> RetrievalIndex:
        return self._index_repository.delete_index(index_id)
