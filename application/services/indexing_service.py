from __future__ import annotations

from uuid import UUID

from ...domain.constants import DEFAULT_CHUNK_VERSION, DEFAULT_SPARSE_PROVIDER, IndexStatus, RetrievalTextPolicy
from ...domain.entities import ChildBlock, IndexEntry, RetrievalIndex
from ...domain.exceptions import ActiveIndexDeletionError
from ..contracts import VectorRecord
from ..ports import DocumentStorePort, EmbeddingPort, VectorStorePort


class IndexingService:
    """编排索引快照投影、向量写入与激活切换。"""

    def __init__(
        self,
        document_store: DocumentStorePort,
        vector_store: VectorStorePort,
        embedding_service: EmbeddingPort,
        retrieval_text_policy: str = RetrievalTextPolicy.HEADER_PATH_PLUS_CONTENT.value,
        chunk_version: str = DEFAULT_CHUNK_VERSION,
    ) -> None:
        """注入索引构建所需的存储与向量依赖。"""
        self._document_store = document_store
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._retrieval_text_policy = RetrievalTextPolicy(retrieval_text_policy).value
        self._chunk_version = chunk_version

    def sync_documents_to_active_index(self, namespace_id: UUID, doc_ids: list[UUID]) -> RetrievalIndex:
        """把指定文档增量投影到当前激活索引。"""
        index = self._ensure_compatible_active_index(namespace_id)
        self._document_store.deactivate_index_entries(index.index_id, doc_ids=doc_ids)
        blocks = self._document_store.list_child_blocks(namespace_id=namespace_id, doc_ids=doc_ids)
        entries = self._project_entries(
            index=index,
            blocks=blocks,
            retrieval_text_policy=index.retrieval_text_policy,
        )
        saved_entries = self._document_store.save_index_entries(entries)
        self._write_vectors(index=index, entries=saved_entries)
        self._document_store.update_index_status(index.index_id, status=IndexStatus.READY.value, is_active=True)
        return self._document_store.activate_index(index.index_id)

    def rebuild_index(
        self,
        namespace_id: UUID,
        namespace_key: str | None = None,
        retrieval_text_policy: str | None = None,
    ) -> RetrievalIndex:
        """基于 namespace 的全部子块重建整份索引。"""
        namespace = self._document_store.get_namespace(namespace_id=namespace_id, namespace_key=namespace_key)
        policy = self._resolve_retrieval_text_policy(retrieval_text_policy)
        index = self._create_new_index(namespace_id=namespace.namespace_id, retrieval_text_policy=policy)
        self._vector_store.ensure_collections(index)
        blocks = self._document_store.list_child_blocks(namespace_id=namespace.namespace_id)
        entries = self._project_entries(index=index, blocks=blocks, retrieval_text_policy=policy)
        saved_entries = self._document_store.save_index_entries(entries)
        self._write_vectors(index=index, entries=saved_entries)
        self._document_store.update_index_status(index.index_id, status=IndexStatus.READY.value, is_active=False)
        return self._document_store.activate_index(index.index_id)

    def delete_index(self, index_id: UUID, allow_active: bool = False) -> RetrievalIndex:
        """删除指定索引及其向量数据，默认禁止删除激活索引。"""
        index = self._document_store.get_index(index_id)
        if index.is_active and not allow_active:
            raise ActiveIndexDeletionError("不允许删除当前激活索引，请先切换到其他索引。")

        self._vector_store.delete_index(index)
        return self._document_store.delete_index(index_id)

    def _ensure_compatible_active_index(self, namespace_id: UUID) -> RetrievalIndex:
        """获取可复用的激活索引，不兼容时自动新建。"""
        active_index = self._document_store.get_active_index(namespace_id)
        if active_index is not None and self._is_compatible(active_index):
            return active_index

        new_index = self._create_new_index(namespace_id=namespace_id, retrieval_text_policy=self._retrieval_text_policy)
        self._vector_store.ensure_collections(new_index)
        blocks = self._document_store.list_child_blocks(namespace_id=namespace_id)
        entries = self._project_entries(index=new_index, blocks=blocks, retrieval_text_policy=self._retrieval_text_policy)
        saved_entries = self._document_store.save_index_entries(entries)
        self._write_vectors(index=new_index, entries=saved_entries)
        self._document_store.update_index_status(new_index.index_id, status=IndexStatus.READY.value, is_active=False)
        return self._document_store.activate_index(new_index.index_id)

    def _create_new_index(self, namespace_id: UUID, retrieval_text_policy: str) -> RetrievalIndex:
        """创建新的索引元数据记录。"""
        existing_indexes = self._document_store.list_indexes(namespace_id)
        version_number = len(existing_indexes) + 1
        index = RetrievalIndex(
            namespace_id=namespace_id,
            index_version=f"index-v{version_number}",
            chunk_version=self._chunk_version,
            index_name=f"default-index-v{version_number}",
            retrieval_text_policy=retrieval_text_policy,
            embedding_provider=self._embedding_service.provider_name,
            embedding_model=self._embedding_service.model_name,
            embedding_dim=self._embedding_service.dimension,
            status=IndexStatus.BUILDING.value,
            is_active=False,
        )
        index.zh_collection_name = f"rag_idx_{index.index_id.hex}_zh"
        index.en_collection_name = f"rag_idx_{index.index_id.hex}_en"
        return self._document_store.create_index(index)

    def _is_compatible(self, index: RetrievalIndex) -> bool:
        """判断现有索引是否兼容当前构建参数。"""
        return (
            index.chunk_version == self._chunk_version
            and index.embedding_provider == self._embedding_service.provider_name
            and index.embedding_model == self._embedding_service.model_name
            and index.retrieval_text_policy == self._retrieval_text_policy
            and index.sparse_provider == DEFAULT_SPARSE_PROVIDER
        )

    def _project_entries(
        self,
        index: RetrievalIndex,
        blocks: list[ChildBlock],
        retrieval_text_policy: str,
    ) -> list[IndexEntry]:
        """把子块投影为可写入索引的 entry 列表。"""
        entries: list[IndexEntry] = []
        for block in blocks:
            file_name = str(block.metadata.get("file_name", "unknown"))
            file_type = str(block.metadata.get("file_type", "txt"))
            retrieval_text = self._build_retrieval_text(block=block, retrieval_text_policy=retrieval_text_policy)
            vector_collection = (
                (index.zh_collection_name or f"rag_idx_{index.index_id.hex}_zh")
                if block.language == "zh"
                else (index.en_collection_name or f"rag_idx_{index.index_id.hex}_en")
            )
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
                    vector_collection=vector_collection,
                    vector_primary_key="",
                    metadata=dict(block.metadata),
                    is_active=True,
                )
            )
        return entries

    def _build_retrieval_text(self, block: ChildBlock, retrieval_text_policy: str | None = None) -> str:
        """根据检索文本策略组装 entry 文本。"""
        policy = self._resolve_retrieval_text_policy(retrieval_text_policy)
        if policy == RetrievalTextPolicy.CONTENT_ONLY.value:
            return block.content

        file_name = str(block.metadata.get("file_name", "unknown"))
        file_type = str(block.metadata.get("file_type", "txt"))
        header_path = block.metadata.get("header_path") or []
        parts = [
            f"文件名: {file_name}",
            f"文件类型: {file_type}",
        ]
        if header_path:
            parts.append(f"标题路径: {' > '.join(header_path)}")
        parts.extend(["正文:", block.content])
        return "\n".join(parts)

    def _resolve_retrieval_text_policy(self, retrieval_text_policy: str | None) -> str:
        """统一解析请求级或服务级检索文本策略。"""
        if retrieval_text_policy is None:
            return self._retrieval_text_policy
        return RetrievalTextPolicy(retrieval_text_policy).value

    def _write_vectors(self, index: RetrievalIndex, entries: list[IndexEntry]) -> None:
        """为 entry 生成向量并写入向量库。"""
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
