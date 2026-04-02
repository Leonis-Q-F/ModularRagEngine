from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ..domain.exceptions import ActiveIndexNotFoundError
from ..domain.ports import DocumentStorePort, EmbeddingPort, RerankerPort, VectorStorePort
from .context_assembler import ContextAssembler
from .dto import ContextBlock, SearchHit, SearchRequest, SearchResult


class SearchService:
    """编排召回、重排、父块回填和上下文组装。"""

    def __init__(
        self,
        document_store: DocumentStorePort,
        vector_store: VectorStorePort,
        embedding_service: EmbeddingPort,
        reranker: RerankerPort | None,
        context_assembler: ContextAssembler,
    ) -> None:
        self._document_store = document_store
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._reranker = reranker
        self._context_assembler = context_assembler

    def search(self, request: SearchRequest) -> SearchResult:
        namespace = self._document_store.get_namespace(
            namespace_id=request.namespace_id,
            namespace_key=request.namespace_key,
        )
        active_index = self._document_store.get_active_index(namespace.namespace_id)
        if active_index is None:
            raise ActiveIndexNotFoundError("当前 namespace 没有可用的激活索引。")

        query_vector = self._embedding_service.embed_query(request.query)
        vector_hits = self._vector_store.hybrid_search(
            index=active_index,
            query_text=request.query,
            query_vector=query_vector,
            top_k=request.top_k_recall,
            filters=request.filters,
        )
        entries = self._document_store.get_index_entries([hit.entry_id for hit in vector_hits])
        entry_by_id = {entry.entry_id: entry for entry in entries}
        recall_scores = {hit.entry_id: hit.score for hit in vector_hits}

        reranked = self._rerank(
            query=request.query,
            entries=[entry_by_id[hit.entry_id] for hit in vector_hits if hit.entry_id in entry_by_id],
            top_k=request.top_k_rerank,
            recall_scores=recall_scores,
        )
        contexts = self._fill_parent_contexts(
            reranked=reranked,
            top_k_context=request.top_k_context,
            parent_window=request.parent_window,
        )

        return SearchResult(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
            index_version=active_index.index_version,
            hits=[
                SearchHit(
                    entry_id=item["entry"].entry_id,
                    block_id=item["entry"].block_id,
                    parent_id=item["entry"].parent_id,
                    recall_score=item["recall_score"],
                    rerank_score=item["score"],
                    retrieval_text=item["entry"].retrieval_text,
                )
                for item in reranked
            ],
            contexts=contexts,
            llm_context=self._context_assembler.build(contexts),
        )

    def _rerank(self, query: str, entries: list, top_k: int, recall_scores: dict) -> list[dict[str, Any]]:
        if not entries:
            return []

        if self._reranker is None:
            ranked_pairs = [(entry, recall_scores.get(entry.entry_id, 0.0)) for entry in entries]
        else:
            ranked_pairs = self._reranker.rerank(query=query, entries=entries, top_k=top_k)

        ranked = [
            {
                "entry": entry,
                "score": score,
                "recall_score": recall_scores.get(entry.entry_id, 0.0),
            }
            for entry, score in ranked_pairs
        ]
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def _fill_parent_contexts(self, reranked: list[dict[str, Any]], top_k_context: int, parent_window: int) -> list[ContextBlock]:
        parent_scores: OrderedDict = OrderedDict()
        for item in reranked:
            parent_id = item["entry"].parent_id
            if parent_id not in parent_scores:
                parent_scores[parent_id] = item["score"]
            else:
                parent_scores[parent_id] = max(parent_scores[parent_id], item["score"])

        seed_parent_ids = list(parent_scores.keys())[:top_k_context]
        ordered_parents: OrderedDict = OrderedDict()

        for parent_id in seed_parent_ids:
            if parent_window > 0:
                parent_chunks = self._document_store.get_parent_chunk_window(parent_id=parent_id, window=parent_window)
            else:
                parent_chunks = self._document_store.get_parent_chunks([parent_id])

            for parent_chunk in sorted(parent_chunks, key=lambda item: (str(item.doc_id), item.chunk_index)):
                if parent_chunk.parent_id not in ordered_parents:
                    ordered_parents[parent_chunk.parent_id] = parent_chunk

        contexts: list[ContextBlock] = []
        for parent_id, parent_chunk in ordered_parents.items():
            contexts.append(
                ContextBlock(
                    parent_id=parent_chunk.parent_id,
                    doc_id=parent_chunk.doc_id,
                    file_name=str(parent_chunk.metadata.get("file_name", "unknown")),
                    chunk_index=parent_chunk.chunk_index,
                    score=parent_scores.get(parent_id, 0.0),
                    content=parent_chunk.content,
                )
            )
        return contexts
