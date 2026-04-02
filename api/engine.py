from __future__ import annotations

from ..application.context_assembler import ContextAssembler
from ..application.dto import (
    IngestDocumentsRequest,
    IngestFilesRequest,
    IngestResult,
    RebuildIndexRequest,
    RebuildIndexResult,
    SearchRequest,
    SearchResult,
)
from ..application.index_service import IndexService
from ..application.ingest_service import IngestService
from ..application.search_service import SearchService
from ..infrastructure.document_loader import DocumentLoader
from ..infrastructure.document_store import DocumentStore
from ..infrastructure.embedding_service import EmbeddingService
from ..infrastructure.markdown_chunker import MarkdownChunker
from ..infrastructure.milvus_store import MilvusStore
from ..infrastructure.reranker import SimpleReranker


class RAGEngine:
    """包的统一对外入口。"""

    def __init__(
        self,
        document_store: DocumentStore | None = None,
        vector_store: MilvusStore | None = None,
        loader: DocumentLoader | None = None,
        chunker: MarkdownChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        reranker: SimpleReranker | None = None,
    ) -> None:
        self.document_store = document_store or DocumentStore()
        self.vector_store = vector_store or MilvusStore()
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or MarkdownChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.reranker = reranker or SimpleReranker()
        self.context_assembler = ContextAssembler()
        self.index_service = IndexService(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
        )
        self.ingest_service = IngestService(
            document_store=self.document_store,
            loader=self.loader,
            chunker=self.chunker,
            index_service=self.index_service,
        )
        self.search_service = SearchService(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            reranker=self.reranker,
            context_assembler=self.context_assembler,
        )

    def ingest_files(self, request: IngestFilesRequest) -> IngestResult:
        return self.ingest_service.ingest_files(request)

    def ingest_documents(self, request: IngestDocumentsRequest) -> IngestResult:
        return self.ingest_service.ingest_documents(request)

    def rebuild_index(self, request: RebuildIndexRequest) -> RebuildIndexResult:
        namespace = self.document_store.get_namespace(
            namespace_id=request.namespace_id,
            namespace_key=request.namespace_key,
        )
        index = self.index_service.rebuild_index(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
        )
        return RebuildIndexResult(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
            index_id=index.index_id,
            index_version=index.index_version,
            status=index.status,
        )

    def search(self, request: SearchRequest) -> SearchResult:
        return self.search_service.search(request)
