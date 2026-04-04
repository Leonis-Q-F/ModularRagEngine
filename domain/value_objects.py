from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from .constants import SupportedLanguage
from .entities import ChildBlock, ParentChunk, SourceDocument


def sha256_text(text: str) -> str:
    """计算文本内容的 SHA-256 摘要。"""
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


class NamespaceReference(BaseModel):
    namespace_id: UUID | None = None
    namespace_key: str | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> "NamespaceReference":
        """确保至少提供一种 namespace 标识。"""
        if self.namespace_id is None and not self.namespace_key:
            raise ValueError("必须提供 namespace_id 或 namespace_key。")
        return self


class ResolvedNamespace(BaseModel):
    namespace_id: UUID
    namespace_key: str


class SearchFilters(BaseModel):
    language: SupportedLanguage | None = None
    file_type: str | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, payload: dict[str, Any] | None) -> "SearchFilters":
        """兼容旧式平铺 filters，并收敛为强类型过滤模型。"""
        payload = dict(payload or {})
        metadata = payload.pop("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("filters.metadata 必须是对象。")

        normalized_metadata: dict[str, str | int | float | bool] = {}
        for key, value in dict(metadata).items():
            normalized_metadata[key] = _validate_filter_value(key, value)

        for key, value in payload.items():
            if key == "language":
                continue
            if key == "file_type":
                continue
            normalized_metadata[key] = _validate_filter_value(key, value)

        language_value = payload.get("language")
        file_type_value = payload.get("file_type")
        return cls(
            language=SupportedLanguage(language_value) if language_value is not None else None,
            file_type=str(file_type_value).strip() or None if file_type_value is not None else None,
            metadata=normalized_metadata,
        )

    def to_legacy_payload(self) -> dict[str, Any]:
        """为兼容现有向量层接口，导出标准化字典。"""
        payload: dict[str, Any] = {"metadata": dict(self.metadata)}
        if self.language is not None:
            payload["language"] = self.language.value
        if self.file_type is not None:
            payload["file_type"] = self.file_type
        return payload


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


def _validate_filter_value(key: str, value: Any) -> str | int | float | bool:
    """确保搜索过滤值是标量，避免弱类型对象泄漏到底层。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    raise ValueError(f"filters 中的字段 {key!r} 只能使用字符串、数字或布尔值。")
