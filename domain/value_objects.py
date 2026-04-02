from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from .entities import ChildBlock, ParentChunk, SourceDocument


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ParsedDocument(BaseModel):
    external_doc_id: str | None = None
    file_name: str
    file_type: str
    source_uri: str | None = None
    parsed_md_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    parser_name: str = "native"
    parser_version: str | None = None


class ChunkBundle(BaseModel):
    source_document: SourceDocument
    parent_chunks: list[ParentChunk] = Field(default_factory=list)
    child_blocks: list[ChildBlock] = Field(default_factory=list)


class VectorRecord(BaseModel):
    entry_id: UUID
    index_id: UUID
    namespace_id: UUID
    doc_id: UUID
    parent_id: UUID
    block_id: UUID
    child_index: int
    language: str
    file_type: str
    file_name: str
    retrieval_text: str
    dense_vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    index_version: str
    chunk_version: str
    is_active: bool = True


class VectorHit(BaseModel):
    entry_id: UUID
    score: float
    dense_score: float
    sparse_score: float

