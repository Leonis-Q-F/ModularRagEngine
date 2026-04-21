from __future__ import annotations

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
from ..application.ports import (
    ChunkerPort,
    ContextPresenterPort,
    DocumentStorePort,
    EmbeddingPort,
    LoaderPort,
    RerankerPort,
    VectorStorePort,
)
from ..composition import build_engine_components


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
        context_assembler: ContextPresenterPort | None = None,
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
        self.context_presenter = components.context_assembler
        self.context_assembler = self.context_presenter
        self.namespace_service = components.namespace_resolver
        self.namespace_resolver = self.namespace_service
        self.indexing_service = components.index_service
        self.index_service = self.indexing_service
        self.ingest_use_case = components.ingest_service
        self.ingest_service = self.ingest_use_case
        self.search_use_case = components.search_service
        self.search_service = self.search_use_case

    def ingest_files(self, request: IngestFilesRequest) -> IngestResult:
        """加载文件、切分文档并按需同步到索引。"""
        return self.ingest_use_case.ingest_files(request)

    def ingest_documents(self, request: IngestDocumentsRequest) -> IngestResult:
        """接收宿主传入的已解析文档并执行入库。"""
        return self.ingest_use_case.ingest_documents(request)

    def rebuild_index(self, request: RebuildIndexRequest) -> RebuildIndexResult:
        """为指定 namespace 重建并激活新的索引快照。"""
        namespace = self.namespace_service.resolve_existing(request.namespace_reference())
        index = self.indexing_service.rebuild_index(
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
        index = self.indexing_service.delete_index(request.index_id)
        return DeleteIndexResult(
            index_id=index.index_id,
            namespace_id=index.namespace_id,
            index_version=index.index_version,
            deleted=True,
        )

    def search(self, request: SearchRequest) -> SearchResult:
        """执行检索请求并返回命中结果与上下文。"""
        return self.search_use_case.search(request)
