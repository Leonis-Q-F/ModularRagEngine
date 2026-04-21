from __future__ import annotations

from dataclasses import dataclass

from .api.presenters.context_presenter import ContextPresenter
from .application.ports import (
    ChunkerPort,
    ContextPresenterPort,
    DocumentStorePort,
    EmbeddingPort,
    LoaderPort,
    RerankerPort,
    VectorStorePort,
)
from .application.services.indexing_service import IndexingService
from .application.services.namespace_resolution_service import NamespaceResolutionService
from .application.use_cases.ingest import IngestUseCase
from .application.use_cases.search import SearchUseCase
from .infrastructure.document_loader import DocumentLoader
from .infrastructure.document_store import DocumentStore
from .infrastructure.embedding_service import EmbeddingService
from .infrastructure.markdown_chunker import MarkdownChunker
from .infrastructure.milvus_store import MilvusStore
from .infrastructure.reranker import SemanticReranker


@dataclass(slots=True)
class EngineComponents:
    """聚合默认引擎所需的全部依赖。"""

    document_store: DocumentStorePort
    vector_store: VectorStorePort
    loader: LoaderPort
    chunker: ChunkerPort
    embedding_service: EmbeddingPort
    reranker: RerankerPort | None
    context_presenter: ContextPresenterPort
    namespace_service: NamespaceResolutionService
    indexing_service: IndexingService
    ingest_use_case: IngestUseCase
    search_use_case: SearchUseCase


def build_engine_components(
    document_store: DocumentStorePort | None = None,
    vector_store: VectorStorePort | None = None,
    loader: LoaderPort | None = None,
    chunker: ChunkerPort | None = None,
    embedding_service: EmbeddingPort | None = None,
    reranker: RerankerPort | None = None,
    context_presenter: ContextPresenterPort | None = None,
) -> EngineComponents:
    """构造默认运行链路，并允许调用方覆盖底层端口实现。"""
    resolved_document_store = document_store or DocumentStore()
    resolved_vector_store = vector_store or MilvusStore()
    resolved_loader = loader or DocumentLoader()
    resolved_chunker = chunker or MarkdownChunker()
    resolved_embedding_service = embedding_service or EmbeddingService()
    resolved_reranker = reranker or SemanticReranker()
    resolved_context_presenter = context_presenter or ContextPresenter()
    namespace_service = NamespaceResolutionService(resolved_document_store)
    indexing_service = IndexingService(
        document_store=resolved_document_store,
        vector_store=resolved_vector_store,
        embedding_service=resolved_embedding_service,
    )
    ingest_use_case = IngestUseCase(
        document_store=resolved_document_store,
        loader=resolved_loader,
        chunker=resolved_chunker,
        indexing_service=indexing_service,
        namespace_service=namespace_service,
    )
    search_use_case = SearchUseCase(
        document_store=resolved_document_store,
        vector_store=resolved_vector_store,
        embedding_service=resolved_embedding_service,
        reranker=resolved_reranker,
        context_presenter=resolved_context_presenter,
    )
    return EngineComponents(
        document_store=resolved_document_store,
        vector_store=resolved_vector_store,
        loader=resolved_loader,
        chunker=resolved_chunker,
        embedding_service=resolved_embedding_service,
        reranker=resolved_reranker,
        context_presenter=resolved_context_presenter,
        namespace_service=namespace_service,
        indexing_service=indexing_service,
        ingest_use_case=ingest_use_case,
        search_use_case=search_use_case,
    )
