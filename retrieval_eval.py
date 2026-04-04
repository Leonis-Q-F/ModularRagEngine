from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from ModularRagEngine import RAGEngine
from ModularRagEngine.application.dto import IngestFilesRequest, SearchRequest
from ModularRagEngine.application.index_service import IndexService
from ModularRagEngine.config import settings
from ModularRagEngine.infrastructure.document_store import DocumentStore
from ModularRagEngine.infrastructure.milvus_store import MilvusStore


DEFAULT_RUNTIME_GUIDE_CASES: list[dict[str, Any]] = [
    {"query": "DocumentStore 的作用是什么", "relevant_markers": ["`infrastructure/document_store.py`"]},
    {"query": "MilvusStore 负责什么", "relevant_markers": ["`infrastructure/milvus_store.py`"]},
    {"query": "SearchService 做什么", "relevant_markers": ["`application/search_service.py`"]},
    {"query": "ingest_files 流程是什么", "relevant_markers": ["#### `ingest_files`"]},
    {"query": "rebuild_index 流程是什么", "relevant_markers": ["### 2.3 索引重建流程"]},
    {"query": "RAGEngine 初始化会创建哪些组件", "relevant_markers": ["### 2.1 初始化流程"]},
    {"query": "NamespaceScopedRequest 校验什么", "relevant_markers": ["`application/dto.py`"]},
    {"query": "MarkdownChunker 做什么", "relevant_markers": ["`infrastructure/markdown_chunker.py`"]},
    {"query": "当前默认运行链路重点是什么", "relevant_markers": ["## 5. 当前默认运行链路的重点"]},
    {"query": "ContextAssembler 做什么", "relevant_markers": ["`application/context_assembler.py`"]},
]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对当前 RAGEngine 执行真实检索评测。")
    parser.add_argument("--doc", required=True, help="待评测文档路径。")
    parser.add_argument(
        "--cases-file",
        help="评测集 JSON 文件。格式为 query + relevant_markers 列表；不传则尝试使用内置评测集。",
    )
    parser.add_argument(
        "--ks",
        default="1,3,5",
        help="要统计的 K 列表，逗号分隔，例如 1,3,5。",
    )
    parser.add_argument("--top-context", type=int, default=2, help="上下文回填数量。")
    parser.add_argument(
        "--namespace-key",
        help="评测时使用的 namespace_key；不传则自动生成临时值。",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="保留 PostgreSQL / Milvus 中的评测数据，默认评测结束自动清理。",
    )
    return parser.parse_args()


def load_cases(doc_path: Path, cases_file: str | None) -> list[dict[str, Any]]:
    """加载评测集，必要时回退到内置样例。"""
    if cases_file:
        with Path(cases_file).expanduser().open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, list) or not payload:
            raise ValueError("评测集 JSON 必须是非空数组。")
        return payload

    if doc_path.name == "project_runtime_guide.md":
        return DEFAULT_RUNTIME_GUIDE_CASES

    raise ValueError("未提供 --cases-file，且当前文档没有内置评测集。")


def parse_ks(raw: str) -> list[int]:
    """解析并校验 K 值列表。"""
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or values[0] <= 0:
        raise ValueError("--ks 必须是正整数列表。")
    return values


def pg_conninfo() -> str:
    """构造 PostgreSQL 连接字符串。"""
    parts = [
        f"host={settings.postgres_host}",
        f"port={settings.postgres_port}",
        f"dbname={settings.postgres_db}",
        f"user={settings.postgres_user}",
        f"sslmode={settings.postgres_sslmode}",
        f"connect_timeout={settings.postgres_timeout_seconds}",
    ]
    if settings.postgres_password:
        parts.append(f"password={settings.postgres_password}")
    return " ".join(parts)


class CleanupEmbeddingStub:
    """为 cleanup 路径提供最小 embedding 元信息占位。"""

    provider_name = "cleanup"
    model_name = "cleanup"
    dimension = 0


def is_relevant(text: str, markers: list[str]) -> bool:
    """判断文本是否命中任一相关标记。"""
    return any(marker in text for marker in markers)


def recall_at_k(hit_block_ids: list[str], relevant_block_ids: set[str], k: int) -> float:
    """计算给定 K 的 Recall。"""
    if not relevant_block_ids:
        return 0.0
    return len(set(hit_block_ids[:k]) & relevant_block_ids) / len(relevant_block_ids)


def mrr_at_k(hit_block_ids: list[str], relevant_block_ids: set[str], k: int) -> float:
    """计算给定 K 的 MRR。"""
    for rank, block_id in enumerate(hit_block_ids[:k], start=1):
        if block_id in relevant_block_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hit_block_ids: list[str], relevant_block_ids: set[str], k: int) -> float:
    """计算给定 K 的 nDCG。"""
    dcg = 0.0
    for rank, block_id in enumerate(hit_block_ids[:k], start=1):
        rel = 1.0 if block_id in relevant_block_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    ideal_rels = min(len(relevant_block_ids), k)
    if ideal_rels == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_rels + 1))
    return dcg / idcg if idcg > 0 else 0.0


def resolve_case_ground_truth(index_id, cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """从真实索引 entry 中解析每个 case 的相关集合。"""
    schema = settings.postgres_schema
    ground_truth: dict[str, dict[str, Any]] = {}
    with psycopg.connect(pg_conninfo()) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "select block_id, parent_id, retrieval_text "
                "from {}.rag_index_entries "
                "where index_id = %s and is_active = true and deleted_at is null"
            ).format(sql.Identifier(schema)),
            (index_id,),
        )
        rows = cur.fetchall()

    for case in cases:
        query = case["query"]
        markers = list(case["relevant_markers"])
        relevant_block_ids: set[str] = set()
        relevant_parent_ids: set[str] = set()
        for block_id, parent_id, retrieval_text in rows:
            if is_relevant(retrieval_text, markers):
                relevant_block_ids.add(str(block_id))
                relevant_parent_ids.add(str(parent_id))

        if not relevant_block_ids:
            raise RuntimeError(f"评测失败：query={query!r} 没有解析出任何相关 block。")

        ground_truth[query] = {
            "relevant_markers": markers,
            "relevant_block_ids": relevant_block_ids,
            "relevant_parent_ids": relevant_parent_ids,
        }

    return ground_truth


def build_case_result(
    case: dict[str, Any],
    search_result,
    case_ground_truth: dict[str, Any],
    ks: list[int],
    top_context: int,
) -> dict[str, Any]:
    """把单条查询结果整理为评测明细。"""
    markers = list(case["relevant_markers"])
    relevant_block_ids = set(case_ground_truth["relevant_block_ids"])
    relevant_parent_ids = set(case_ground_truth["relevant_parent_ids"])
    hit_texts = [hit.retrieval_text for hit in search_result.hits]
    hit_block_ids = [str(hit.block_id) for hit in search_result.hits]
    context_texts = [context.content for context in search_result.contexts]
    context_parent_ids = [str(context.parent_id) for context in search_result.contexts]

    rank = None
    for idx, block_id in enumerate(hit_block_ids, start=1):
        if block_id in relevant_block_ids:
            rank = idx
            break

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(hit_block_ids, relevant_block_ids, k)
        metrics[f"mrr@{k}"] = mrr_at_k(hit_block_ids, relevant_block_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(hit_block_ids, relevant_block_ids, k)

    metrics[f"context_hit@{top_context}"] = (
        1.0 if any(parent_id in relevant_parent_ids for parent_id in context_parent_ids[:top_context]) else 0.0
    )

    return {
        "query": case["query"],
        "relevant_markers": markers,
        "relevant_block_count": len(relevant_block_ids),
        "relevant_parent_count": len(relevant_parent_ids),
        "rank_of_first_relevant": rank,
        "metrics": metrics,
        "top_hit_preview": hit_texts[0][:180] if hit_texts else None,
        "top_context_preview": context_texts[0][:180] if context_texts else None,
    }


def aggregate_metrics(case_results: list[dict[str, Any]], ks: list[int], top_context: int) -> dict[str, float]:
    """汇总全部 case 的平均指标。"""
    summary: dict[str, float] = {}
    num_cases = len(case_results)
    for k in ks:
        summary[f"Recall@{k}"] = sum(item["metrics"][f"recall@{k}"] for item in case_results) / num_cases
        summary[f"MRR@{k}"] = sum(item["metrics"][f"mrr@{k}"] for item in case_results) / num_cases
        summary[f"nDCG@{k}"] = sum(item["metrics"][f"ndcg@{k}"] for item in case_results) / num_cases
    summary[f"ContextHit@{top_context}"] = sum(item["metrics"][f"context_hit@{top_context}"] for item in case_results) / num_cases
    return summary


def build_cleanup_index_service() -> IndexService:
    """构造只用于清理路径的索引服务。"""
    return IndexService(
        document_store=DocumentStore(),
        vector_store=MilvusStore(),
        embedding_service=CleanupEmbeddingStub(),
    )


def cleanup(
    namespace_id,
    index_id,
    keep_artifacts: bool,
    index_service: IndexService | None = None,
) -> dict[str, Any]:
    """清理评测产生的 PGSQL 和 Milvus 数据。"""
    if keep_artifacts:
        return {"namespace_deleted": False, "dropped_collections": []}

    service = index_service or build_cleanup_index_service()
    deleted_index = service.delete_index(index_id=index_id, allow_active=True)
    dropped_collections = [
        name
        for name in [deleted_index.zh_collection_name, deleted_index.en_collection_name]
        if name
    ]

    schema = settings.postgres_schema
    with psycopg.connect(pg_conninfo()) as conn, conn.cursor() as cur:
        for table in [
            "rag_index_entries",
            "rag_retrieval_indexes",
            "rag_child_blocks",
            "rag_parent_chunks",
            "rag_source_documents",
            "rag_namespaces",
        ]:
            cur.execute(
                sql.SQL("delete from {}.{} where namespace_id = %s").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                ),
                (namespace_id,),
            )
        conn.commit()

    return {"namespace_deleted": True, "dropped_collections": dropped_collections}


def main() -> None:
    """执行整条真实检索评测流程。"""
    args = parse_args()
    doc_path = Path(args.doc).expanduser().resolve()
    if not doc_path.exists():
        raise FileNotFoundError(doc_path)

    ks = parse_ks(args.ks)
    cases = load_cases(doc_path=doc_path, cases_file=args.cases_file)
    namespace_key = args.namespace_key or f"eval-{uuid.uuid4().hex[:8]}"

    engine = RAGEngine()
    result = None
    active_index = None
    payload: dict[str, Any] = {
        "doc": str(doc_path),
        "namespace_key": namespace_key,
    }
    try:
        result = engine.ingest_files(
            IngestFilesRequest(
                namespace_key=namespace_key,
                file_paths=[str(doc_path)],
            )
        )
        active_index = engine.document_store.get_active_index(result.namespace_id)
        if active_index is None:
            raise RuntimeError("评测失败：未生成激活索引。")

        case_results = []
        case_ground_truth = resolve_case_ground_truth(active_index.index_id, cases)
        top_k = max(ks)
        for case in cases:
            search_result = engine.search(
                SearchRequest(
                    namespace_key=namespace_key,
                    query=case["query"],
                    top_k_recall=top_k,
                    top_k_rerank=top_k,
                    top_k_context=args.top_context,
                )
            )
            case_results.append(
                build_case_result(
                    case=case,
                    search_result=search_result,
                    case_ground_truth=case_ground_truth[case["query"]],
                    ks=ks,
                    top_context=args.top_context,
                )
            )

        payload.update(
            {
            "doc": str(doc_path),
            "namespace_key": namespace_key,
            "namespace_id": str(result.namespace_id),
            "index_id": str(active_index.index_id),
            "index_version": active_index.index_version,
            "embedding_provider": active_index.embedding_provider,
            "embedding_model": active_index.embedding_model,
            "embedding_dim": active_index.embedding_dim,
            "metrics": aggregate_metrics(case_results, ks, args.top_context),
            "cases": case_results,
            }
        )
    finally:
        if result is not None and active_index is not None:
            payload["cleanup"] = cleanup(
                namespace_id=result.namespace_id,
                index_id=active_index.index_id,
                keep_artifacts=args.keep_artifacts,
            )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
