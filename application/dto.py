from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class NamespaceScopedRequest(BaseModel):
    namespace_id: UUID | None = None
    namespace_key: str | None = None

    @model_validator(mode="after")
    def validate_namespace_scope(self) -> "NamespaceScopedRequest":
        if self.namespace_id is None and not self.namespace_key:
            raise ValueError("必须提供 namespace_id 或 namespace_key。")
        return self


class InputDocument(BaseModel):
    external_doc_id: str | None = None
    file_name: str
    file_type: str
    parsed_md_content: str
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_name: str = "host"
    parser_version: str | None = None


class IngestFilesRequest(NamespaceScopedRequest):
    file_paths: list[str]
    use_ocr: bool = False
    document_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    index_after_ingest: bool = True


class IngestDocumentsRequest(NamespaceScopedRequest):
    documents: list[InputDocument]
    index_after_ingest: bool = True


class RebuildIndexRequest(NamespaceScopedRequest):
    retrieval_text_policy: str = "header_path_plus_content"


class SearchRequest(NamespaceScopedRequest):
    query: str
    top_k_recall: int = 8
    top_k_rerank: int = 5
    top_k_context: int = 3
    parent_window: int = 0
    filters: dict[str, Any] = Field(default_factory=dict)


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
    index_id: UUID | None = None
    index_version: str | None = None


class RebuildIndexResult(BaseModel):
    namespace_id: UUID
    namespace_key: str
    index_id: UUID
    index_version: str
    status: str


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
