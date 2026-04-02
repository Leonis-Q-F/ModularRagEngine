from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Namespace(BaseModel):
    namespace_id: UUID = Field(default_factory=uuid4)
    namespace_key: str
    namespace_name: str
    namespace_type: str | None = None
    external_ref: str | None = None
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None


class SourceDocument(BaseModel):
    doc_id: UUID = Field(default_factory=uuid4)
    namespace_id: UUID
    external_doc_id: str | None = None
    file_name: str
    file_type: str
    source_uri: str | None = None
    language: str | None = None
    status: str = "completed"
    content_sha256: str
    parser_name: str = "native"
    parser_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    parsed_md_content: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None


class ParentChunk(BaseModel):
    parent_id: UUID = Field(default_factory=uuid4)
    namespace_id: UUID
    doc_id: UUID
    chunk_version: str
    chunk_index: int
    content: str
    content_sha256: str
    language: str
    token_count: int
    heading_level: int | None = None
    header_path: list[str] = Field(default_factory=list)
    split_route: str
    start_line: int | None = None
    end_line: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None


class ChildBlock(BaseModel):
    block_id: UUID = Field(default_factory=uuid4)
    namespace_id: UUID
    doc_id: UUID
    parent_id: UUID
    chunk_version: str
    child_index: int
    content: str
    content_sha256: str
    language: str
    token_count: int
    start_char: int | None = None
    end_char: int | None = None
    start_token: int | None = None
    end_token: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None


class RetrievalIndex(BaseModel):
    index_id: UUID = Field(default_factory=uuid4)
    namespace_id: UUID
    index_version: str
    chunk_version: str
    index_name: str
    retrieval_strategy: str = "hybrid"
    retrieval_text_policy: str = "header_path_plus_content"
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    sparse_provider: str = "simple_lexical"
    reranker_provider: str | None = None
    reranker_model: str | None = None
    zh_collection_name: str | None = None
    en_collection_name: str | None = None
    status: str = "building"
    is_active: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    activated_at: datetime | None = None
    deleted_at: datetime | None = None


class IndexEntry(BaseModel):
    entry_id: UUID = Field(default_factory=uuid4)
    index_id: UUID
    namespace_id: UUID
    doc_id: UUID
    parent_id: UUID
    block_id: UUID
    chunk_version: str
    index_version: str
    child_index: int
    file_type: str
    file_name: str
    language: str
    retrieval_text: str
    vector_status: str = "pending"
    vector_collection: str
    vector_primary_key: str
    indexed_at: datetime | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deleted_at: datetime | None = None
