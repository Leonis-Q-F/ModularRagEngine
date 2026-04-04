from __future__ import annotations

from dataclasses import dataclass

from .application.context_assembler import ContextAssembler
from .application.index_service import IndexService
from .application.ingest_service import IngestService
from .application.namespace_resolver import NamespaceResolver
from .application.search_service import SearchService
from .domain.ports import ChunkerPort, DocumentStorePort, EmbeddingPort, LoaderPort, RerankerPort, VectorStorePort
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
    context_assembler: ContextAssembler
    namespace_resolver: NamespaceResolver
    index_service: IndexService
    ingest_service: IngestService
    search_service: SearchService


def build_engine_components(
    document_store: DocumentStorePort | None = None,
    vector_store: VectorStorePort | None = None,
    loader: LoaderPort | None = None,
    chunker: ChunkerPort | None = None,
    embedding_service: EmbeddingPort | None = None,
    reranker: RerankerPort | None = None,
    context_assembler: ContextAssembler | None = None,
) -> EngineComponents:
    """构造默认运行链路，并允许调用方覆盖底层端口实现。"""
    resolved_document_store = document_store or DocumentStore()
    resolved_vector_store = vector_store or MilvusStore()
    resolved_loader = loader or DocumentLoader()
    resolved_chunker = chunker or MarkdownChunker()
    resolved_embedding_service = embedding_service or EmbeddingService()
    resolved_reranker = reranker or SemanticReranker()
    resolved_context_assembler = context_assembler or ContextAssembler()
    namespace_resolver = NamespaceResolver(resolved_document_store)
    index_service = IndexService(
        document_store=resolved_document_store,
        vector_store=resolved_vector_store,
        embedding_service=resolved_embedding_service,
    )
    ingest_service = IngestService(
        document_store=resolved_document_store,
        loader=resolved_loader,
        chunker=resolved_chunker,
        index_service=index_service,
        namespace_resolver=namespace_resolver,
    )
    search_service = SearchService(
        document_store=resolved_document_store,
        vector_store=resolved_vector_store,
        embedding_service=resolved_embedding_service,
        reranker=resolved_reranker,
        context_assembler=resolved_context_assembler,
    )
    return EngineComponents(
        document_store=resolved_document_store,
        vector_store=resolved_vector_store,
        loader=resolved_loader,
        chunker=resolved_chunker,
        embedding_service=resolved_embedding_service,
        reranker=resolved_reranker,
        context_assembler=resolved_context_assembler,
        namespace_resolver=namespace_resolver,
        index_service=index_service,
        ingest_service=ingest_service,
        search_service=search_service,
    )
