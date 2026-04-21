from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from ..domain.constants import RetrievalTextPolicy
from .contracts import NamespaceReference, SearchFilters


def _validate_positive_text(value: str, field_name: str) -> str:
    """统一校验必须非空的文本字段。"""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空。")
    return normalized

class NamespaceScopedRequest(BaseModel):
    namespace_id: UUID | None = None
    namespace_key: str | None = None

    @model_validator(mode="after")
    def validate_namespace_scope(self) -> "NamespaceScopedRequest":
        """确保请求至少带有一种 namespace 标识。"""
        if self.namespace_id is None and not self.namespace_key:
            raise ValueError("必须提供 namespace_id 或 namespace_key。")
        return self

    def namespace_reference(self) -> NamespaceReference:
        """导出统一的 namespace 引用对象。"""
        return NamespaceReference(namespace_id=self.namespace_id, namespace_key=self.namespace_key)


class InputDocument(BaseModel):
    external_doc_id: str | None = None
    file_name: str
    file_type: str
    parsed_md_content: str
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_name: str = "host"
    parser_version: str | None = None

    @field_validator("file_name", "file_type", "parsed_md_content")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        """收紧文档输入的基础文本字段。"""
        return _validate_positive_text(value, info.field_name)


class IngestFilesRequest(NamespaceScopedRequest):
    file_paths: list[str] = Field(min_length=1)
    use_ocr: bool = False

    @field_validator("file_paths")
    @classmethod
    def validate_file_paths(cls, value: list[str]) -> list[str]:
        """确保文件路径列表中不存在空路径。"""
        return [_validate_positive_text(item, "file_paths") for item in value]


class IngestDocumentsRequest(NamespaceScopedRequest):
    documents: list[InputDocument] = Field(min_length=1)


class RebuildIndexRequest(NamespaceScopedRequest):
    retrieval_text_policy: str = RetrievalTextPolicy.HEADER_PATH_PLUS_CONTENT.value

    @field_validator("retrieval_text_policy")
    @classmethod
    def validate_retrieval_text_policy(cls, value: str) -> str:
        """确保索引重建策略在受支持列表内。"""
        normalized = _validate_positive_text(value, "retrieval_text_policy")
        return RetrievalTextPolicy(normalized).value


class DeleteIndexRequest(BaseModel):
    index_id: UUID


class SearchRequest(NamespaceScopedRequest):
    query: str
    top_k_recall: int = Field(default=8, ge=1)
    top_k_rerank: int = Field(default=5, ge=1)
    top_k_candidates: int | None = Field(default=None, ge=1)
    top_k_context: int = Field(default=3, ge=1)
    parent_window: int = Field(default=0, ge=0)
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """确保检索查询非空。"""
        return _validate_positive_text(value, "query")

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, value: dict[str, Any]) -> dict[str, Any]:
        """在 DTO 层提前校验过滤条件结构。"""
        return SearchFilters.from_raw(value).to_legacy_payload()

    def normalized_filters(self) -> SearchFilters:
        """返回经过标准化的检索过滤对象。"""
        return SearchFilters.from_raw(self.filters)


class IndexedDocument(BaseModel):
    doc_id: UUID
    file_name: str
    file_type: str


class IngestResult(BaseModel):
    namespace_id: UUID
    namespace_key: str
    doc_ids: list[UUID]
    documents: list[IndexedDocument]
    chunk_version: str
    index_id: UUID
    index_version: str


class RebuildIndexResult(BaseModel):
    namespace_id: UUID
    namespace_key: str
    index_id: UUID
    index_version: str
    status: str


class DeleteIndexResult(BaseModel):
    index_id: UUID
    namespace_id: UUID
    index_version: str
    deleted: bool


class SearchHit(BaseModel):
    entry_id: UUID
    block_id: UUID
    parent_id: UUID
    recall_score: float
    rerank_score: float | None = None
    retrieval_text: str


class ContextBlock(BaseModel):
    parent_id: UUID
    doc_id: UUID
    file_name: str
    chunk_index: int
    score: float
    content: str


class SearchResult(BaseModel):
    namespace_id: UUID
    namespace_key: str
    index_version: str
    hits: list[SearchHit]
    contexts: list[ContextBlock]
    llm_context: str
