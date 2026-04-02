from __future__ import annotations

from typing import Any

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

try:
    from ..config import settings
except ImportError:  # pragma: no cover - 兼容直接从仓库根目录运行
    from config import settings

from ..domain.entities import RetrievalIndex
from ..domain.value_objects import VectorHit, VectorRecord


class MilvusStore:
    """基于真实 Milvus 服务的向量检索适配器。"""

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        db_name: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._uri = (uri or settings.milvus_uri).strip()
        self._token = token if token is not None else settings.milvus_token
        self._db_name = db_name if db_name is not None else (settings.milvus_db_name or "")
        self._timeout = timeout if timeout is not None else float(settings.milvus_timeout_seconds)
        self._client = MilvusClient(
            uri=self._uri,
            token=self._token or "",
            db_name=self._db_name,
            timeout=self._timeout,
        )

    def ensure_collections(self, index: RetrievalIndex) -> RetrievalIndex:
        if index.zh_collection_name is None:
            index.zh_collection_name = f"rag_idx_{index.index_id.hex}_zh"
        if index.en_collection_name is None:
            index.en_collection_name = f"rag_idx_{index.index_id.hex}_en"

        self._ensure_collection(
            collection_name=index.zh_collection_name,
            dim=index.embedding_dim,
            language="zh",
        )
        self._ensure_collection(
            collection_name=index.en_collection_name,
            dim=index.embedding_dim,
            language="en",
        )
        return index

    def upsert_entries(self, index: RetrievalIndex, records: list[VectorRecord]) -> None:
        if not records:
            return

        self.ensure_collections(index)
        grouped_records: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            collection_name = self._collection_name(index=index, language=record.language)
            grouped_records.setdefault(collection_name, []).append(
                {
                    "entry_id": str(record.entry_id),
                    "index_id": str(record.index_id),
                    "namespace_id": str(record.namespace_id),
                    "doc_id": str(record.doc_id),
                    "parent_id": str(record.parent_id),
                    "block_id": str(record.block_id),
                    "child_index": record.child_index,
                    "language": record.language,
                    "file_type": record.file_type,
                    "file_name": record.file_name,
                    "retrieval_text": record.retrieval_text,
                    "dense_vector": record.dense_vector,
                    "metadata": dict(record.metadata),
                    "index_version": record.index_version,
                    "chunk_version": record.chunk_version,
                    "is_active": record.is_active,
                }
            )

        for collection_name, payload in grouped_records.items():
            self._client.upsert(collection_name=collection_name, data=payload, timeout=self._timeout)
            self._client.flush(collection_name=collection_name, timeout=self._timeout)
            self._client.load_collection(collection_name=collection_name, timeout=self._timeout)

    def hybrid_search(
        self,
        index: RetrievalIndex,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        filters = filters or {}
        expr = self._build_filter_expr(filters)
        search_limit = max(top_k * 5, 20)
        hits_by_entry_id: dict[str, VectorHit] = {}

        for collection_name in self._target_collections(index=index, filters=filters):
            hybrid_hits = self._hybrid_search_collection(
                collection_name=collection_name,
                query_text=query_text,
                query_vector=query_vector,
                expr=expr,
                limit=search_limit,
            )
            dense_hits = self._search_dense_collection(
                collection_name=collection_name,
                query_vector=query_vector,
                expr=expr,
                limit=search_limit,
            )
            sparse_hits = self._search_sparse_collection(
                collection_name=collection_name,
                query_text=query_text,
                expr=expr,
                limit=search_limit,
            )

            dense_scores = {item["entry_id"]: float(item["distance"]) for item in dense_hits}
            sparse_scores = {item["entry_id"]: float(item["distance"]) for item in sparse_hits}

            for item in hybrid_hits:
                entity = item.get("entity", {})
                if not self._match_post_filters(entity=entity, filters=filters):
                    continue

                entry_id = str(item["entry_id"])
                hit = VectorHit(
                    entry_id=item["entry_id"],
                    score=float(item["distance"]),
                    dense_score=dense_scores.get(entry_id, 0.0),
                    sparse_score=sparse_scores.get(entry_id, 0.0),
                )
                current = hits_by_entry_id.get(entry_id)
                if current is None or hit.score > current.score:
                    hits_by_entry_id[entry_id] = hit

        hits = list(hits_by_entry_id.values())
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def delete_index(self, index: RetrievalIndex) -> None:
        for collection_name in filter(None, [index.zh_collection_name, index.en_collection_name]):
            if self._client.has_collection(collection_name=collection_name, timeout=self._timeout):
                self._client.drop_collection(collection_name=collection_name, timeout=self._timeout)

    def _ensure_collection(self, collection_name: str, dim: int, language: str) -> None:
        if self._client.has_collection(collection_name=collection_name, timeout=self._timeout):
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="entry_id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="index_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="namespace_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="block_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="child_index", datatype=DataType.INT64)
        schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=8)
        schema.add_field(field_name="file_type", datatype=DataType.VARCHAR, max_length=16)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(
            field_name="retrieval_text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params=self._analyzer_params(language),
        )
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="index_version", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="chunk_version", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="is_active", datatype=DataType.BOOL)
        schema.add_function(
            Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["retrieval_text"],
                output_field_names=["sparse_vector"],
            )
        )

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_autoinex",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_bm25",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self._client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            timeout=self._timeout,
        )
        self._client.load_collection(collection_name=collection_name, timeout=self._timeout)

    def _hybrid_search_collection(
        self,
        collection_name: str,
        query_text: str,
        query_vector: list[float],
        expr: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        dense_request = AnnSearchRequest(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=limit,
            expr=expr,
        )
        sparse_request = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=limit,
            expr=expr,
        )
        results = self._client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_request, sparse_request],
            ranker=RRFRanker(),
            limit=limit,
            output_fields=["file_type", "language", "metadata", "is_active"],
            timeout=self._timeout,
        )
        return results[0] if results else []

    def _search_dense_collection(
        self,
        collection_name: str,
        query_vector: list[float],
        expr: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        results = self._client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            limit=limit,
            filter=expr,
            output_fields=["file_type", "language", "metadata", "is_active"],
            search_params={"metric_type": "COSINE", "params": {}},
            timeout=self._timeout,
        )
        return results[0] if results else []

    def _search_sparse_collection(
        self,
        collection_name: str,
        query_text: str,
        expr: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        results = self._client.search(
            collection_name=collection_name,
            data=[query_text],
            anns_field="sparse_vector",
            limit=limit,
            filter=expr,
            output_fields=["file_type", "language", "metadata", "is_active"],
            search_params={"metric_type": "BM25", "params": {}},
            timeout=self._timeout,
        )
        return results[0] if results else []

    def _target_collections(self, index: RetrievalIndex, filters: dict[str, Any]) -> list[str]:
        language = filters.get("language")
        if language == "zh":
            return [index.zh_collection_name] if index.zh_collection_name else []
        if language == "en":
            return [index.en_collection_name] if index.en_collection_name else []
        return [collection for collection in [index.zh_collection_name, index.en_collection_name] if collection]

    def _build_filter_expr(self, filters: dict[str, Any]) -> str:
        clauses = ["is_active == true"]

        if "language" in filters and filters["language"] is not None:
            clauses.append(f'language == "{self._escape_string(str(filters["language"]))}"')
        if "file_type" in filters and filters["file_type"] is not None:
            clauses.append(f'file_type == "{self._escape_string(str(filters["file_type"]))}"')

        return " and ".join(clauses)

    def _match_post_filters(self, entity: dict[str, Any], filters: dict[str, Any]) -> bool:
        if not entity.get("is_active", True):
            return False

        metadata = entity.get("metadata") or {}
        for key, value in filters.items():
            if key in {"language", "file_type"}:
                if entity.get(key) != value:
                    return False
                continue
            if metadata.get(key) != value:
                return False
        return True

    def _collection_name(self, index: RetrievalIndex, language: str) -> str:
        return index.zh_collection_name if language == "zh" else index.en_collection_name

    def _analyzer_params(self, language: str) -> dict[str, Any]:
        if language == "zh":
            return {
                "tokenizer": "jieba",
                "filter": ["cnalphanumonly"],
            }
        return {
            "tokenizer": "standard",
            "filter": ["lowercase"],
        }

    def _escape_string(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
