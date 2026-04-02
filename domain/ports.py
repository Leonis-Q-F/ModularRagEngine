from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from .entities import IndexEntry, Namespace, ParentChunk, RetrievalIndex, SourceDocument
from .value_objects import ChunkBundle, ParsedDocument, VectorHit, VectorRecord


class LoaderPort(Protocol):
    def load(self, file_paths: list[str], use_ocr: bool = False) -> list[ParsedDocument]:
        ...


class ChunkerPort(Protocol):
    def split_document(self, doc: SourceDocument, chunk_version: str) -> ChunkBundle:
        ...


class EmbeddingPort(Protocol):
    provider_name: str
    model_name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class RerankerPort(Protocol):
    provider_name: str | None
    model_name: str | None

    def rerank(self, query: str, entries: list[IndexEntry], top_k: int) -> list[tuple[IndexEntry, float]]:
        ...


class DocumentStorePort(Protocol):
    def ensure_namespace(self, namespace_key: str, namespace_name: str | None = None) -> Namespace:
        ...

    def get_namespace(self, namespace_id: UUID | None = None, namespace_key: str | None = None) -> Namespace:
        ...

    def upsert_source_documents(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        ...

    def replace_document_chunks(self, bundle: ChunkBundle) -> ChunkBundle:
        ...

    def list_child_blocks(self, namespace_id: UUID, doc_ids: list[UUID] | None = None) -> list[Any]:
        ...

    def create_index(self, index: RetrievalIndex) -> RetrievalIndex:
        ...

    def get_active_index(self, namespace_id: UUID) -> RetrievalIndex | None:
        ...

    def list_indexes(self, namespace_id: UUID) -> list[RetrievalIndex]:
        ...

    def save_index_entries(self, entries: list[IndexEntry]) -> list[IndexEntry]:
        ...

    def deactivate_index_entries(self, index_id: UUID, doc_ids: list[UUID] | None = None) -> None:
        ...

    def get_index_entries(self, entry_ids: list[UUID]) -> list[IndexEntry]:
        ...

    def get_parent_chunks(self, parent_ids: list[UUID]) -> list[ParentChunk]:
        ...

    def get_parent_chunk_window(self, parent_id: UUID, window: int) -> list[ParentChunk]:
        ...

    def activate_index(self, index_id: UUID) -> RetrievalIndex:
        ...

    def update_index_status(self, index_id: UUID, status: str, is_active: bool | None = None) -> RetrievalIndex:
        ...


class VectorStorePort(Protocol):
    def ensure_collections(self, index: RetrievalIndex) -> RetrievalIndex:
        ...

    def upsert_entries(self, index: RetrievalIndex, records: list[VectorRecord]) -> None:
        ...

    def hybrid_search(
        self,
        index: RetrievalIndex,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        ...

    def delete_index(self, index: RetrievalIndex) -> None:
        ...
