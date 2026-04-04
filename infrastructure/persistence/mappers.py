from __future__ import annotations

from ...domain.entities import ChildBlock, IndexEntry, Namespace, ParentChunk, RetrievalIndex, SourceDocument
from ...domain.exceptions import NamespaceNotFoundError


def namespace_from_row(row: dict | None) -> Namespace:
    """把数据库行转换为 Namespace 实体。"""
    if row is None:
        raise NamespaceNotFoundError("namespace 不存在。")
    return Namespace(**normalize_json_fields(row, ["metadata"]))


def source_document_from_row(row: dict) -> SourceDocument:
    """把数据库行转换为 SourceDocument 实体。"""
    return SourceDocument(**normalize_json_fields(row, ["metadata"]))


def parent_chunk_from_row(row: dict) -> ParentChunk:
    """把数据库行转换为 ParentChunk 实体。"""
    payload = normalize_json_fields(row, ["metadata"])
    payload["header_path"] = payload.get("header_path") or []
    return ParentChunk(**payload)


def child_block_from_row(row: dict) -> ChildBlock:
    """把数据库行转换为 ChildBlock 实体。"""
    return ChildBlock(**normalize_json_fields(row, ["metadata"]))


def retrieval_index_from_row(row: dict | None) -> RetrievalIndex:
    """把数据库行转换为 RetrievalIndex 实体。"""
    if row is None:
        raise NamespaceNotFoundError("namespace 不存在。")
    return RetrievalIndex(**normalize_json_fields(row, ["metadata"]))


def index_entry_from_row(row: dict) -> IndexEntry:
    """把数据库行转换为 IndexEntry 实体。"""
    return IndexEntry(**normalize_json_fields(row, ["metadata"]))


def normalize_json_fields(row: dict, fields: list[str]) -> dict:
    """确保指定 JSON 字段至少是空字典。"""
    payload = dict(row)
    for field_name in fields:
        if payload.get(field_name) is None:
            payload[field_name] = {}
    return payload
