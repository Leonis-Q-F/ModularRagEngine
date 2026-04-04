from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker, WeightedRanker

from ..config import settings
from ..domain.entities import RetrievalIndex
from ..domain.value_objects import SearchFilters, VectorHit, VectorRecord


def build_milvus_ranker(
    strategy: str,
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
):
    """按配置构造 Milvus 混合检索融合器。"""
    if strategy == "rrf":
        return RRFRanker(k=rrf_k)
    if strategy == "weighted":
        total = dense_weight + sparse_weight
        if total <= 0:
            raise ValueError("weighted ranker 的 dense/sparse 权重和必须大于 0。")
        return WeightedRanker(dense_weight / total, sparse_weight / total)
    raise ValueError(f"不支持的 Milvus 融合策略：{strategy}")


@dataclass(slots=True)
class MilvusRawHit:
    """承载 Milvus 原始结果，避免弱类型字典直接泄漏到领域层。"""

    entry_id: str
    distance: float
    entity: dict[str, Any]


class MilvusStore:
    """基于 Milvus 服务的向量检索适配器。"""

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        db_name: str | None = None,
        timeout: float | None = None,
        collect_score_breakdown: bool | None = None,
        sparse_inverted_index_algo: str | None = None,
        client: MilvusClient | None = None,
    ) -> None:
        """初始化 Milvus 客户端连接。"""
        self._uri = (uri or settings.milvus_uri).strip()
        self._token = token if token is not None else settings.milvus_token
        self._db_name = db_name if db_name is not None else (settings.milvus_db_name or "")
        self._timeout = timeout if timeout is not None else float(settings.milvus_timeout_seconds)
        self._ranker_strategy = settings.milvus_ranker_strategy
        self._dense_weight = float(settings.milvus_dense_weight)
        self._sparse_weight = float(settings.milvus_sparse_weight)
        self._rrf_k = int(settings.milvus_rrf_k)
        self._collect_score_breakdown = (
            bool(collect_score_breakdown)
            if collect_score_breakdown is not None
            else bool(settings.milvus_collect_score_breakdown)
        )
        self._sparse_inverted_index_algo = sparse_inverted_index_algo or settings.milvus_sparse_inverted_index_algo
        self._client = client or MilvusClient(
            uri=self._uri,
            token=self._token or "",
            db_name=self._db_name,
            timeout=self._timeout,
        )

    def ensure_collections(self, index: RetrievalIndex) -> RetrievalIndex:
        """确保当前索引的中英文 collection 已存在。"""
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
        """把索引记录写入对应语言的 collection。"""
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
        filters: SearchFilters | None = None,
    ) -> list[VectorHit]:
        """执行 dense+sparse 混合检索并合并结果。"""
        normalized_filters = filters or SearchFilters()
        expr = self._build_filter_expr(normalized_filters)
        search_limit = max(top_k * 5, 20)
        hits_by_entry_id: dict[str, VectorHit] = {}

        for collection_name in self._target_collections(index=index, filters=normalized_filters):
            hybrid_hits = self._hybrid_search_collection(
                collection_name=collection_name,
                query_text=query_text,
                query_vector=query_vector,
                expr=expr,
                limit=search_limit,
            )
            dense_scores: dict[str, float] = {}
            sparse_scores: dict[str, float] = {}
            if self._collect_score_breakdown:
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
                dense_scores = {item.entry_id: item.distance for item in dense_hits}
                sparse_scores = {item.entry_id: item.distance for item in sparse_hits}

            for item in hybrid_hits:
                if not self._match_post_filters(entity=item.entity, filters=normalized_filters):
                    continue

                entry_id = item.entry_id
                hit = VectorHit(
                    entry_id=entry_id,
                    score=item.distance,
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
        """删除索引对应的 Milvus collection。"""
        for collection_name in filter(None, [index.zh_collection_name, index.en_collection_name]):
            if self._client.has_collection(collection_name=collection_name, timeout=self._timeout):
                self._client.drop_collection(collection_name=collection_name, timeout=self._timeout)

    def _ensure_collection(self, collection_name: str, dim: int, language: str) -> None:
        """按给定 schema 和索引配置创建 collection。"""
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
            index_name="dense_autoindex",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_bm25",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": self._sparse_inverted_index_algo},
        )
        index_params.add_index(
            field_name="language",
            index_name="language_inverted",
            index_type="INVERTED",
        )
        index_params.add_index(
            field_name="file_type",
            index_name="file_type_inverted",
            index_type="INVERTED",
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
    ) -> list[MilvusRawHit]:
        """在单个 collection 上执行混合检索。"""
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
            ranker=build_milvus_ranker(
                strategy=self._ranker_strategy,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
                rrf_k=self._rrf_k,
            ),
            limit=limit,
            output_fields=["file_type", "language", "metadata", "is_active"],
            timeout=self._timeout,
        )
        return self._raw_hits_from_result(results[0] if results else [])

    def _search_dense_collection(
        self,
        collection_name: str,
        query_vector: list[float],
        expr: str,
        limit: int,
    ) -> list[MilvusRawHit]:
        """在单个 collection 上执行 dense 向量检索。"""
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
        return self._raw_hits_from_result(results[0] if results else [])

    def _search_sparse_collection(
        self,
        collection_name: str,
        query_text: str,
        expr: str,
        limit: int,
    ) -> list[MilvusRawHit]:
        """在单个 collection 上执行 BM25 稀疏检索。"""
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
        return self._raw_hits_from_result(results[0] if results else [])

    def _target_collections(self, index: RetrievalIndex, filters: SearchFilters) -> list[str]:
        """根据过滤条件选择要搜索的 collection。"""
        language = filters.language.value if filters.language is not None else None
        if language == "zh":
            return [index.zh_collection_name] if index.zh_collection_name else []
        if language == "en":
            return [index.en_collection_name] if index.en_collection_name else []
        return [collection for collection in [index.zh_collection_name, index.en_collection_name] if collection]

    def _build_filter_expr(self, filters: SearchFilters) -> str:
        """把通用过滤条件转换为 Milvus 表达式。"""
        clauses = ["is_active == true"]

        if filters.language is not None:
            clauses.append(f'language == "{self._escape_string(filters.language.value)}"')
        if filters.file_type is not None:
            clauses.append(f'file_type == "{self._escape_string(filters.file_type)}"')

        return " and ".join(clauses)

    def _match_post_filters(self, entity: dict[str, Any], filters: SearchFilters) -> bool:
        """对 Milvus 返回实体执行补充过滤。"""
        if not entity.get("is_active", True):
            return False

        metadata = entity.get("metadata") or {}
        if filters.language is not None and entity.get("language") != filters.language.value:
            return False
        if filters.file_type is not None and entity.get("file_type") != filters.file_type:
            return False
        for key, value in filters.metadata.items():
            if metadata.get(key) != value:
                return False
        return True

    def _collection_name(self, index: RetrievalIndex, language: str) -> str:
        """根据语言获取 collection 名。"""
        return index.zh_collection_name if language == "zh" else index.en_collection_name

    def _analyzer_params(self, language: str) -> dict[str, Any]:
        """为不同语言返回 analyzer 配置。"""
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
        """转义 Milvus 过滤表达式中的字符串。"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _raw_hits_from_result(self, items: list[dict[str, Any]]) -> list[MilvusRawHit]:
        """把 Milvus SDK 返回的原始字典转换为局部强类型对象。"""
        return [
            MilvusRawHit(
                entry_id=str(item["entry_id"]),
                distance=float(item["distance"]),
                entity=dict(item.get("entity", {})),
            )
            for item in items
        ]
