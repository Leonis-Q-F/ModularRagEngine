from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from ..domain.entities import IndexEntry, RetrievalIndex
from ..domain.exceptions import NamespaceNotFoundError
from ..domain.ports import DocumentStorePort, EmbeddingPort, VectorStorePort
from ..domain.value_objects import VectorRecord


class IndexService:
    """编排索引快照投影、向量写入与激活切换。"""

    def __init__(
        self,
        document_store: DocumentStorePort,
        vector_store: VectorStorePort,
        embedding_service: EmbeddingPort,
        retrieval_text_policy: str = "header_path_plus_content",
        chunk_version: str = "chunk-v1",
    ) -> None:
        self._document_store = document_store
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._retrieval_text_policy = retrieval_text_policy
        self._chunk_version = chunk_version

    def sync_documents_to_active_index(self, namespace_id: UUID, doc_ids: list[UUID]) -> RetrievalIndex:
        index = self._ensure_compatible_active_index(namespace_id)
        self._document_store.deactivate_index_entries(index.index_id, doc_ids=doc_ids)
        blocks = self._document_store.list_child_blocks(namespace_id=namespace_id, doc_ids=doc_ids)
        entries = self._project_entries(index=index, blocks=blocks)
        saved_entries = self._document_store.save_index_entries(entries)
        self._write_vectors(index=index, entries=saved_entries)
        self._document_store.update_index_status(index.index_id, status="ready", is_active=True)
        return self._document_store.activate_index(index.index_id)

    def rebuild_index(self, namespace_id: UUID, namespace_key: str | None = None) -> RetrievalIndex:
        namespace = self._document_store.get_namespace(namespace_id=namespace_id, namespace_key=namespace_key)
        if namespace is None:
            raise NamespaceNotFoundError("namespace 不存在。")

        index = self._create_new_index(namespace_id=namespace.namespace_id)
        self._vector_store.ensure_collections(index)
        blocks = self._document_store.list_child_blocks(namespace_id=namespace.namespace_id)
        entries = self._project_entries(index=index, blocks=blocks)
        saved_entries = self._document_store.save_index_entries(entries)
        self._write_vectors(index=index, entries=saved_entries)
        self._document_store.update_index_status(index.index_id, status="ready", is_active=False)
        return self._document_store.activate_index(index.index_id)

    def _ensure_compatible_active_index(self, namespace_id: UUID) -> RetrievalIndex:
        active_index = self._document_store.get_active_index(namespace_id)
        if active_index is not None and self._is_compatible(active_index):
            return active_index

        new_index = self._create_new_index(namespace_id=namespace_id)
        self._vector_store.ensure_collections(new_index)
        blocks = self._document_store.list_child_blocks(namespace_id=namespace_id)
        entries = self._project_entries(index=new_index, blocks=blocks)
        saved_entries = self._document_store.save_index_entries(entries)
        self._write_vectors(index=new_index, entries=saved_entries)
        self._document_store.update_index_status(new_index.index_id, status="ready", is_active=False)
        return self._document_store.activate_index(new_index.index_id)

    def _create_new_index(self, namespace_id: UUID) -> RetrievalIndex:
        existing_indexes = self._document_store.list_indexes(namespace_id)
        version_number = len(existing_indexes) + 1
        index = RetrievalIndex(
            namespace_id=namespace_id,
            index_version=f"index-v{version_number}",
            chunk_version=self._chunk_version,
            index_name=f"default-index-v{version_number}",
            retrieval_text_policy=self._retrieval_text_policy,
            embedding_provider=self._embedding_service.provider_name,
            embedding_model=self._embedding_service.model_name,
            embedding_dim=self._embedding_service.dimension,
            reranker_provider=None,
            reranker_model=None,
            status="building",
            is_active=False,
        )
        return self._document_store.create_index(index)

    def _is_compatible(self, index: RetrievalIndex) -> bool:
        return (
            index.chunk_version == self._chunk_version
            and index.embedding_provider == self._embedding_service.provider_name
            and index.embedding_model == self._embedding_service.model_name
            and index.retrieval_text_policy == self._retrieval_text_policy
            and index.sparse_provider == "simple_lexical"
        )

    def _project_entries(self, index: RetrievalIndex, blocks: list) -> list[IndexEntry]:
        entries: list[IndexEntry] = []
        for block in blocks:
            file_name = str(block.metadata.get("file_name", "unknown"))
            file_type = str(block.metadata.get("file_type", "txt"))
            retrieval_text = self._build_retrieval_text(block=block)
            entries.append(
                IndexEntry(
                    index_id=index.index_id,
                    namespace_id=block.namespace_id,
                    doc_id=block.doc_id,
                    parent_id=block.parent_id,
                    block_id=block.block_id,
                    chunk_version=block.chunk_version,
                    index_version=index.index_version,
                    child_index=block.child_index,
                    file_type=file_type,
                    file_name=file_name,
                    language=block.language,
                    retrieval_text=retrieval_text,
                    vector_collection=self._collection_name(index=index, language=block.language),
                    vector_primary_key="",
                    metadata=dict(block.metadata),
                    is_active=True,
                )
            )
        return entries

    def _build_retrieval_text(self, block) -> str:
        if self._retrieval_text_policy == "content_only":
            return block.content

        header_path = block.metadata.get("header_path") or []
        if not header_path:
            return block.content
        return "\n".join([" > ".join(header_path), block.content])

    def _write_vectors(self, index: RetrievalIndex, entries: list[IndexEntry]) -> None:
        if not entries:
            return

        vectors = self._embedding_service.embed_texts([entry.retrieval_text for entry in entries])
        records = [
            VectorRecord(
                entry_id=entry.entry_id,
                index_id=entry.index_id,
                namespace_id=entry.namespace_id,
                doc_id=entry.doc_id,
                parent_id=entry.parent_id,
                block_id=entry.block_id,
                child_index=entry.child_index,
                language=entry.language,
                file_type=entry.file_type,
                file_name=entry.file_name,
                retrieval_text=entry.retrieval_text,
                dense_vector=vector,
                metadata=dict(entry.metadata),
                index_version=entry.index_version,
                chunk_version=entry.chunk_version,
                is_active=entry.is_active,
            )
            for entry, vector in zip(entries, vectors, strict=True)
        ]
        self._vector_store.upsert_entries(index=index, records=records)

    def _collection_name(self, index: RetrievalIndex, language: str) -> str:
        return f"rag_idx_{index.index_id}_{language}"
