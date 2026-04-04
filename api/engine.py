from __future__ import annotations

from ..application.context_assembler import ContextAssembler
from ..application.dto import (
    DeleteIndexRequest,
    DeleteIndexResult,
    IngestDocumentsRequest,
    IngestFilesRequest,
    IngestResult,
    RebuildIndexRequest,
    RebuildIndexResult,
    SearchRequest,
    SearchResult,
)
from ..composition import build_engine_components
from ..domain.ports import ChunkerPort, DocumentStorePort, EmbeddingPort, LoaderPort, RerankerPort, VectorStorePort


class RAGEngine:
    """包的统一对外入口。"""

    def __init__(
        self,
        document_store: DocumentStorePort | None = None,
        vector_store: VectorStorePort | None = None,
        loader: LoaderPort | None = None,
        chunker: ChunkerPort | None = None,
        embedding_service: EmbeddingPort | None = None,
        reranker: RerankerPort | None = None,
        context_assembler: ContextAssembler | None = None,
    ) -> None:
        """通过独立装配模块构造完整检索链路。"""
        components = build_engine_components(
            document_store=document_store,
            vector_store=vector_store,
            loader=loader,
            chunker=chunker,
            embedding_service=embedding_service,
            reranker=reranker,
            context_assembler=context_assembler,
        )
        self.document_store = components.document_store
        self.vector_store = components.vector_store
        self.loader = components.loader
        self.chunker = components.chunker
        self.embedding_service = components.embedding_service
        self.reranker = components.reranker
        self.context_assembler = components.context_assembler
        self.namespace_resolver = components.namespace_resolver
        self.index_service = components.index_service
        self.ingest_service = components.ingest_service
        self.search_service = components.search_service

    def ingest_files(self, request: IngestFilesRequest) -> IngestResult:
        """加载文件、切分文档并按需同步到索引。"""
        return self.ingest_service.ingest_files(request)

    def ingest_documents(self, request: IngestDocumentsRequest) -> IngestResult:
        """接收宿主传入的已解析文档并执行入库。"""
        return self.ingest_service.ingest_documents(request)

    def rebuild_index(self, request: RebuildIndexRequest) -> RebuildIndexResult:
        """为指定 namespace 重建并激活新的索引快照。"""
        namespace = self.namespace_resolver.resolve_existing(request.namespace_reference())
        index = self.index_service.rebuild_index(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
            retrieval_text_policy=request.retrieval_text_policy,
        )
        return RebuildIndexResult(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
            index_id=index.index_id,
            index_version=index.index_version,
            status=index.status,
        )

    def delete_index(self, request: DeleteIndexRequest) -> DeleteIndexResult:
        """删除指定的非激活索引。"""
        index = self.index_service.delete_index(request.index_id)
        return DeleteIndexResult(
            index_id=index.index_id,
            namespace_id=index.namespace_id,
            index_version=index.index_version,
            deleted=True,
        )

    def search(self, request: SearchRequest) -> SearchResult:
        """执行检索请求并返回命中结果与上下文。"""
        return self.search_service.search(request)
