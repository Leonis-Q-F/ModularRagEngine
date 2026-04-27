from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker, WeightedRanker

from ..application.contracts import SearchFilters, VectorHit, VectorRecord
from ..config import settings
from ..domain.entities import RetrievalIndex


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

    _DENSE_INDEX_NAME = "dense_autoindex"
    _SPARSE_INDEX_NAME = "sparse_bm25"
    _LANGUAGE_INDEX_NAME = "language_inverted"
    _FILE_TYPE_INDEX_NAME = "file_type_inverted"
    _SEARCH_OUTPUT_FIELDS = ["entry_id", "file_type", "language", "metadata", "is_active"]

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        db_name: str | None = None,
        timeout: float | None = None,
        collect_score_breakdown: bool | None = None,
        sparse_inverted_index_algo: str | None = None,
        upsert_batch_size: int | None = None,
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
        self._upsert_batch_size = max(int(upsert_batch_size or settings.milvus_upsert_batch_size), 1)

        client_kwargs: dict[str, Any] = {
            "uri": self._uri,
            "timeout": self._timeout,
        }
        if self._db_name:
            client_kwargs["db_name"] = self._db_name
        if self._token:
            client_kwargs["token"] = self._token
        self._client = client or MilvusClient(**client_kwargs)

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

    def delete_entries(self, index: RetrievalIndex, doc_ids: list[UUID] | None = None) -> None:
        """删除索引下指定文档的旧向量，避免 stale hit 挤占召回池。"""
        delete_filters = self._build_delete_filters(index=index, doc_ids=doc_ids)
        if not delete_filters:
            return

        for collection_name in filter(None, [index.zh_collection_name, index.en_collection_name]):
            if not self._client.has_collection(collection_name=collection_name, timeout=self._timeout):
                continue
            self._ensure_collection_indexed(collection_name=collection_name)
            self._ensure_collection_loaded(collection_name=collection_name)
            for delete_filter in delete_filters:
                self._client.delete(
                    collection_name=collection_name,
                    filter=delete_filter,
                    timeout=self._timeout,
                )

    def insert_entries(self, index: RetrievalIndex, records: list[VectorRecord]) -> None:
        """向新建索引的 collection 分批写入记录，不在批次中途 flush/index/load。"""
        self._write_entries(index=index, records=records, operation="insert")

    def upsert_entries(self, index: RetrievalIndex, records: list[VectorRecord]) -> None:
        """向已存在索引的 collection 分批 upsert 记录。"""
        self._write_entries(index=index, records=records, operation="upsert")

    def prepare_index_for_search(self, index: RetrievalIndex, languages: set[str] | None = None) -> None:
        """在批量写入完成后统一 flush、建索引并加载 collection。"""
        self.ensure_collections(index)
        target_collections = self._collections_for_languages(index=index, languages=languages)
        if not target_collections:
            return

        for collection_name in target_collections:
            if not self._client.has_collection(collection_name=collection_name, timeout=self._timeout):
                continue
            self._client.flush(collection_name=collection_name, timeout=self._timeout)
            self._ensure_collection_indexed(collection_name=collection_name)
            self._ensure_collection_loaded(collection_name=collection_name)

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

        self.ensure_collections(index)
        for collection_name in self._target_collections(index=index, filters=normalized_filters):
            self._ensure_collection_indexed(collection_name=collection_name)
            self._ensure_collection_loaded(collection_name=collection_name)
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
        """按给定 schema 创建 collection，索引与加载在写入后单独保证。"""
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

        self._client.create_collection(
            collection_name=collection_name,
            schema=schema,
            timeout=self._timeout,
        )

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
            output_fields=self._SEARCH_OUTPUT_FIELDS,
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
            output_fields=self._SEARCH_OUTPUT_FIELDS,
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
            output_fields=self._SEARCH_OUTPUT_FIELDS,
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
            clauses.append(f'language == {self._format_filter_value(filters.language.value)}')
        if filters.file_type is not None:
            clauses.append(f'file_type == {self._format_filter_value(filters.file_type)}')
        for key, value in filters.metadata.items():
            clauses.append(f'metadata["{self._escape_string(key)}"] == {self._format_filter_value(value)}')

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

    def _format_filter_value(self, value: str | int | float | bool) -> str:
        """把 Python 标量值转换为 Milvus 表达式字面量。"""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f'"{self._escape_string(value)}"'
        return str(value)

    def _build_delete_filters(self, index: RetrievalIndex, doc_ids: list[UUID] | None) -> list[str]:
        """按文档分批构造删除过滤条件，避免旧向量继续参与召回。"""
        base_clause = f'index_id == {self._format_filter_value(str(index.index_id))}'
        if not doc_ids:
            return [base_clause]

        filters: list[str] = []
        normalized_doc_ids = [str(doc_id) for doc_id in doc_ids]
        for batch in self._batched(normalized_doc_ids, self._upsert_batch_size):
            doc_id_literals = ", ".join(self._format_filter_value(doc_id) for doc_id in batch)
            filters.append(f"{base_clause} and doc_id in [{doc_id_literals}]")
        return filters

    def _ensure_collection_indexed(self, collection_name: str) -> None:
        """保证 collection 的向量与标量索引都已创建。"""
        existing_indexes = set(self._client.list_indexes(collection_name=collection_name))
        index_params = MilvusClient.prepare_index_params()
        has_missing_indexes = False

        if self._DENSE_INDEX_NAME not in existing_indexes:
            has_missing_indexes = True
            index_params.add_index(
                field_name="dense_vector",
                index_name=self._DENSE_INDEX_NAME,
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
        if self._SPARSE_INDEX_NAME not in existing_indexes:
            has_missing_indexes = True
            index_params.add_index(
                field_name="sparse_vector",
                index_name=self._SPARSE_INDEX_NAME,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"inverted_index_algo": self._sparse_inverted_index_algo},
            )
        if self._LANGUAGE_INDEX_NAME not in existing_indexes:
            has_missing_indexes = True
            index_params.add_index(
                field_name="language",
                index_name=self._LANGUAGE_INDEX_NAME,
                index_type="INVERTED",
            )
        if self._FILE_TYPE_INDEX_NAME not in existing_indexes:
            has_missing_indexes = True
            index_params.add_index(
                field_name="file_type",
                index_name=self._FILE_TYPE_INDEX_NAME,
                index_type="INVERTED",
            )

        if not has_missing_indexes:
            return

        if self._is_collection_loaded(collection_name):
            self._client.release_collection(collection_name=collection_name, timeout=self._timeout)
        self._client.create_index(collection_name=collection_name, index_params=index_params, timeout=self._timeout)

    def _ensure_collection_loaded(self, collection_name: str) -> None:
        """保证 collection 已加载，Milvus 重启或 release 后可自愈。"""
        if self._is_collection_loaded(collection_name):
            return
        self._client.load_collection(collection_name=collection_name, timeout=self._timeout)

    def _is_collection_loaded(self, collection_name: str) -> bool:
        """判断 collection 当前是否已加载到内存。"""
        state = self._client.get_load_state(collection_name=collection_name, timeout=self._timeout).get("state")
        state_name = getattr(state, "name", None) or str(state)
        return "Loaded" in state_name

    def _collections_for_languages(self, index: RetrievalIndex, languages: set[str] | None = None) -> list[str]:
        """根据语言集合返回需要处理的 collection 名称。"""
        if languages is None:
            return [collection for collection in [index.zh_collection_name, index.en_collection_name] if collection]

        target_collections: list[str] = []
        if "zh" in languages and index.zh_collection_name:
            target_collections.append(index.zh_collection_name)
        if "en" in languages and index.en_collection_name:
            target_collections.append(index.en_collection_name)
        return target_collections

    def _serialize_record(self, record: VectorRecord) -> dict[str, Any]:
        """把领域层向量记录转换为 Milvus 可写入字典。"""
        return {
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

    def _write_entries(self, index: RetrievalIndex, records: list[VectorRecord], operation: str) -> None:
        """把记录按语言和批次写入 Milvus，不触发构建尾部动作。"""
        if not records:
            return

        if operation not in {"insert", "upsert"}:
            raise ValueError(f"不支持的 Milvus 写入操作：{operation}")

        self.ensure_collections(index)
        grouped_records: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            collection_name = index.zh_collection_name if record.language == "zh" else index.en_collection_name
            grouped_records.setdefault(collection_name, []).append(self._serialize_record(record))

        write_fn = getattr(self._client, operation, None)
        if write_fn is None:
            raise AttributeError(f"MilvusClient 缺少 {operation} 方法。")

        for collection_name, payload in grouped_records.items():
            for batch in self._batched(payload, self._upsert_batch_size):
                write_fn(collection_name=collection_name, data=batch, timeout=self._timeout)

    def _batched(self, items: list[Any], size: int) -> list[list[Any]]:
        """按固定大小切分列表。"""
        return [items[start : start + size] for start in range(0, len(items), size)]

    def _raw_hits_from_result(self, items: list[dict[str, Any]]) -> list[MilvusRawHit]:
        """把 Milvus SDK 返回的原始字典转换为局部强类型对象。"""
        raw_hits: list[MilvusRawHit] = []
        for item in items:
            entity = dict(item.get("entity", {}))
            raw_entry_id = item.get("entry_id") or item.get("id") or entity.get("entry_id") or entity.get("id")
            if raw_entry_id is None:
                raise KeyError("Milvus 返回结果缺少主键字段 entry_id/id。")
            raw_hits.append(
                MilvusRawHit(
                    entry_id=str(raw_entry_id),
                    distance=float(item["distance"]),
                    entity=entity,
                )
            )
        return raw_hits
