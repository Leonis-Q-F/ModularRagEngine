from __future__ import annotations

from typing import Protocol
from uuid import UUID

from ...domain.entities import ChildBlock, IndexEntry, Namespace, ParentChunk, RetrievalIndex, SourceDocument
from ..contracts import ChunkBundle, ParsedDocument, SearchFilters, VectorHit, VectorRecord
from ..dto import ContextBlock


class LoaderPort(Protocol):
    def load(self, file_paths: list[str], use_ocr: bool = False) -> list[ParsedDocument]:
        """加载文件并返回统一的解析结果。"""
        ...


class ChunkerPort(Protocol):
    def split_document(self, doc: SourceDocument, chunk_version: str) -> ChunkBundle:
        """把源文档切分为父块和子块。"""
        ...


class EmbeddingPort(Protocol):
    provider_name: str
    model_name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本向量。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """生成查询向量。"""
        ...


class RerankerPort(Protocol):
    provider_name: str | None
    model_name: str | None

    def rerank(self, query: str, entries: list[IndexEntry], top_k: int) -> list[tuple[IndexEntry, float]]:
        """对召回条目进行重排并返回分数。"""
        ...


class DocumentStorePort(Protocol):
    def ensure_namespace(self, namespace_key: str, namespace_name: str | None = None) -> Namespace:
        """按 key 获取或创建 namespace。"""
        ...

    def get_namespace(self, namespace_id: UUID | None = None, namespace_key: str | None = None) -> Namespace:
        """按 id 或 key 查询单个 namespace。"""
        ...

    def upsert_source_documents(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        """批量写入或更新源文档。"""
        ...

    def replace_document_chunks(self, bundle: ChunkBundle) -> ChunkBundle:
        """替换指定文档当前激活的父块与子块。"""
        ...

    def list_child_blocks(self, namespace_id: UUID, doc_ids: list[UUID] | None = None) -> list[ChildBlock]:
        """列出 namespace 下激活的子块，可按文档过滤。"""
        ...

    def create_index(self, index: RetrievalIndex) -> RetrievalIndex:
        """创建索引元数据记录。"""
        ...

    def get_active_index(self, namespace_id: UUID) -> RetrievalIndex | None:
        """获取 namespace 当前激活的索引。"""
        ...

    def list_indexes(self, namespace_id: UUID) -> list[RetrievalIndex]:
        """列出 namespace 的全部索引记录。"""
        ...

    def get_index(self, index_id: UUID) -> RetrievalIndex:
        """按 index_id 获取单个索引记录。"""
        ...

    def save_index_entries(self, entries: list[IndexEntry]) -> list[IndexEntry]:
        """批量保存索引 entry 元数据。"""
        ...

    def deactivate_index_entries(self, index_id: UUID, doc_ids: list[UUID] | None = None) -> None:
        """批量停用索引 entry，可限制到指定文档。"""
        ...

    def get_index_entries(self, entry_ids: list[UUID]) -> list[IndexEntry]:
        """按 entry_id 批量获取激活 entry。"""
        ...

    def get_parent_chunks(self, parent_ids: list[UUID]) -> list[ParentChunk]:
        """按 parent_id 批量获取父块。"""
        ...

    def get_parent_chunk_window(self, parent_id: UUID, window: int) -> list[ParentChunk]:
        """按窗口范围回填指定父块附近的兄弟块。"""
        ...

    def activate_index(self, index_id: UUID) -> RetrievalIndex:
        """激活目标索引并退役同 namespace 的其他索引。"""
        ...

    def update_index_status(self, index_id: UUID, status: str, is_active: bool | None = None) -> RetrievalIndex:
        """更新索引状态，并可同步修改激活标记。"""
        ...

    def delete_index(self, index_id: UUID) -> RetrievalIndex:
        """软删除索引记录及其 entry 元数据。"""
        ...


class VectorStorePort(Protocol):
    def ensure_collections(self, index: RetrievalIndex) -> RetrievalIndex:
        """确保索引对应的向量 collection 已存在。"""
        ...

    def upsert_entries(self, index: RetrievalIndex, records: list[VectorRecord]) -> None:
        """把索引记录写入向量库。"""
        ...

    def hybrid_search(
        self,
        index: RetrievalIndex,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[VectorHit]:
        """执行混合检索并返回向量命中列表。"""
        ...

    def delete_index(self, index: RetrievalIndex) -> None:
        """删除索引对应的向量集合。"""
        ...


class ContextPresenterPort(Protocol):
    def build(self, contexts: list[ContextBlock]) -> str:
        """把上下文块格式化为稳定的对外文本。"""
        ...
