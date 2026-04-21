from __future__ import annotations

from uuid import UUID

from ...domain.constants import DEFAULT_CHUNK_VERSION
from ...domain.entities import SourceDocument
from ..contracts import ParsedDocument, sha256_text
from ..dto import IndexedDocument, IngestDocumentsRequest, IngestFilesRequest, IngestResult
from ..ports import ChunkerPort, DocumentStorePort, LoaderPort
from ..services.indexing_service import IndexingService
from ..services.namespace_resolution_service import NamespaceResolutionService


class IngestUseCase:
    """编排文档入库、切分和索引同步。"""

    def __init__(
        self,
        document_store: DocumentStorePort,
        loader: LoaderPort,
        chunker: ChunkerPort,
        index_service: IndexingService,
        namespace_resolver: NamespaceResolutionService | None = None,
        chunk_version: str = DEFAULT_CHUNK_VERSION,
    ) -> None:
        """注入入库流程所需的存储、加载和索引依赖。"""
        self._document_store = document_store
        self._loader = loader
        self._chunker = chunker
        self._index_service = index_service
        self._namespace_resolver = namespace_resolver or NamespaceResolutionService(document_store)
        self._chunk_version = chunk_version

    def ingest_files(self, request: IngestFilesRequest) -> IngestResult:
        """把文件路径输入转换为标准入库与可检索索引流程。"""
        namespace = self._namespace_resolver.resolve_for_ingest(request.namespace_reference())
        parsed_documents = self._loader.load(request.file_paths, use_ocr=request.use_ocr)
        return self._ingest_parsed_documents(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
            parsed_documents=parsed_documents,
        )

    def ingest_documents(self, request: IngestDocumentsRequest) -> IngestResult:
        """接收宿主传入的解析结果并执行统一入库与索引。"""
        namespace = self._namespace_resolver.resolve_for_ingest(request.namespace_reference())
        parsed_documents = [
            ParsedDocument(
                external_doc_id=document.external_doc_id,
                file_name=document.file_name,
                file_type=document.file_type,
                source_uri=document.source_uri,
                parsed_md_content=document.parsed_md_content,
                metadata=dict(document.metadata),
                parser_name=document.parser_name,
                parser_version=document.parser_version,
            )
            for document in request.documents
        ]
        return self._ingest_parsed_documents(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
            parsed_documents=parsed_documents,
        )

    def _ingest_parsed_documents(
        self,
        namespace_id: UUID,
        namespace_key: str,
        parsed_documents: list[ParsedDocument],
    ) -> IngestResult:
        """把解析后的文档写入存储、切分并同步到可检索索引。"""
        source_documents = [
            SourceDocument(
                namespace_id=namespace_id,
                external_doc_id=document.external_doc_id,
                file_name=document.file_name,
                file_type=document.file_type,
                source_uri=document.source_uri,
                language=document.metadata.get("language"),
                content_sha256=sha256_text(document.parsed_md_content),
                parser_name=document.parser_name,
                parser_version=document.parser_version,
                metadata=dict(document.metadata),
                parsed_md_content=document.parsed_md_content,
            )
            for document in parsed_documents
        ]

        stored_documents = self._document_store.upsert_source_documents(source_documents)
        for document in stored_documents:
            bundle = self._chunker.split_document(doc=document, chunk_version=self._chunk_version)
            self._document_store.replace_document_chunks(bundle)

        active_index = self._index_service.sync_documents_to_active_index(
            namespace_id=namespace_id,
            doc_ids=[document.doc_id for document in stored_documents],
        )

        return IngestResult(
            namespace_id=namespace_id,
            namespace_key=namespace_key,
            doc_ids=[document.doc_id for document in stored_documents],
            documents=[
                IndexedDocument(
                    doc_id=document.doc_id,
                    file_name=document.file_name,
                    file_type=document.file_type,
                )
                for document in stored_documents
            ],
            chunk_version=self._chunk_version,
            index_id=active_index.index_id,
            index_version=active_index.index_version,
        )
