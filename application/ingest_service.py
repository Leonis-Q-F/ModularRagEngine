from __future__ import annotations

from ..domain.entities import SourceDocument
from ..domain.ports import ChunkerPort, DocumentStorePort, LoaderPort
from ..domain.value_objects import ParsedDocument, sha256_text
from .dto import IndexedDocument, IngestDocumentsRequest, IngestFilesRequest, IngestResult, InputDocument
from .index_service import IndexService


class IngestService:
    """编排文档入库、切分和增量索引同步。"""

    def __init__(
        self,
        document_store: DocumentStorePort,
        loader: LoaderPort,
        chunker: ChunkerPort,
        index_service: IndexService,
        chunk_version: str = "chunk-v1",
    ) -> None:
        self._document_store = document_store
        self._loader = loader
        self._chunker = chunker
        self._index_service = index_service
        self._chunk_version = chunk_version

    def ingest_files(self, request: IngestFilesRequest) -> IngestResult:
        namespace = self._document_store.ensure_namespace(
            namespace_key=request.namespace_key or str(request.namespace_id),
        )
        parsed_documents = self._loader.load(request.file_paths, use_ocr=request.use_ocr)
        return self._ingest_parsed_documents(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
            parsed_documents=parsed_documents,
            index_after_ingest=request.index_after_ingest,
        )

    def ingest_documents(self, request: IngestDocumentsRequest) -> IngestResult:
        namespace = self._document_store.ensure_namespace(
            namespace_key=request.namespace_key or str(request.namespace_id),
        )
        parsed_documents = [self._to_parsed_document(document) for document in request.documents]
        return self._ingest_parsed_documents(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
            parsed_documents=parsed_documents,
            index_after_ingest=request.index_after_ingest,
        )

    def _ingest_parsed_documents(
        self,
        namespace_id,
        namespace_key: str,
        parsed_documents: list[ParsedDocument],
        index_after_ingest: bool,
    ) -> IngestResult:
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

        active_index = None
        if index_after_ingest:
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
            index_id=active_index.index_id if active_index else None,
            index_version=active_index.index_version if active_index else None,
        )

    def _to_parsed_document(self, document: InputDocument) -> ParsedDocument:
        return ParsedDocument(
            external_doc_id=document.external_doc_id,
            file_name=document.file_name,
            file_type=document.file_type,
            source_uri=document.source_uri,
            parsed_md_content=document.parsed_md_content,
            metadata=dict(document.metadata),
            parser_name=document.parser_name,
            parser_version=document.parser_version,
        )
