from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from pydantic import BaseModel, Field

from ..domain.entities import ChildBlock, ParentChunk, SourceDocument
from ..domain.value_objects import ChunkBundle, sha256_text


@dataclass(slots=True)
class SplitSection:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkerConfig(BaseModel):
    parent_max_tokens: int = Field(default=1200, ge=100)
    child_max_tokens: int = Field(default=320, ge=50)
    tokenizer_encoding: str = Field(default="cl100k_base")


class TokenCounter:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        text = text.strip()
        if not text:
            return 0
        return len(self._encoding.encode(text))


class MarkdownChunker(BaseModel):
    """执行父子分层切分。"""

    config: ChunkerConfig = Field(default_factory=ChunkerConfig)

    def model_post_init(self, __context: Any) -> None:
        self._token_counter = TokenCounter(self.config.tokenizer_encoding)

    def split_document(self, doc: SourceDocument, chunk_version: str) -> ChunkBundle:
        sections = self._split_markdown(doc.parsed_md_content)
        parent_chunks = self._build_parent_chunks(doc=doc, sections=sections, chunk_version=chunk_version)
        child_blocks = self._build_child_blocks(doc=doc, parent_chunks=parent_chunks, chunk_version=chunk_version)
        return ChunkBundle(source_document=doc, parent_chunks=parent_chunks, child_blocks=child_blocks)

    def _split_markdown(self, markdown: str) -> list[SplitSection]:
        markdown = markdown.strip()
        if not markdown:
            return []

        sections = self._split_by_headers(markdown)
        if sections:
            return sections
        language = self._detect_language(markdown)
        return [SplitSection(content=markdown, metadata={"header_path": [], "heading_level": None, "split_route": "语义切分", "language": language})]

    def _split_by_headers(self, markdown: str) -> list[SplitSection]:
        pattern = re.compile(r"(?m)^(#{1,6})\s+(.+)$")
        matches = list(pattern.finditer(markdown))
        if not matches:
            return []

        sections: list[SplitSection] = []
        header_stack: list[str] = []
        for index, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
            content = markdown[start:end].strip()
            if not content:
                continue

            header_stack = header_stack[: level - 1] + [title]
            sections.append(
                SplitSection(
                    content=content,
                    metadata={
                        "header_path": list(header_stack),
                        "heading_level": len(header_stack),
                        "split_route": "标题结构切分",
                        "language": self._detect_language(content),
                    },
                )
            )
        return sections

    def _build_parent_chunks(self, doc: SourceDocument, sections: list[SplitSection], chunk_version: str) -> list[ParentChunk]:
        parent_chunks: list[ParentChunk] = []
        for index, section in enumerate(sections):
            content = section.content.strip()
            if not content:
                continue
            parent_chunks.append(
                ParentChunk(
                    namespace_id=doc.namespace_id,
                    doc_id=doc.doc_id,
                    chunk_version=chunk_version,
                    chunk_index=index,
                    content=content,
                    content_sha256=sha256_text(content),
                    language=section.metadata["language"],
                    token_count=self._token_counter.count(content),
                    heading_level=section.metadata.get("heading_level"),
                    header_path=list(section.metadata.get("header_path", [])),
                    split_route=section.metadata.get("split_route", "语义切分"),
                    start_char=doc.parsed_md_content.find(content),
                    end_char=doc.parsed_md_content.find(content) + len(content),
                    metadata={
                        "file_name": doc.file_name,
                        "file_type": doc.file_type,
                        "header_path": list(section.metadata.get("header_path", [])),
                        "language": section.metadata["language"],
                    },
                )
            )
        return parent_chunks

    def _build_child_blocks(self, doc: SourceDocument, parent_chunks: list[ParentChunk], chunk_version: str) -> list[ChildBlock]:
        child_blocks: list[ChildBlock] = []
        for parent in parent_chunks:
            child_blocks.append(
                ChildBlock(
                    namespace_id=doc.namespace_id,
                    doc_id=doc.doc_id,
                    parent_id=parent.parent_id,
                    chunk_version=chunk_version,
                    child_index=0,
                    content=parent.content,
                    content_sha256=sha256_text(parent.content),
                    language=parent.language,
                    token_count=parent.token_count,
                    start_char=0,
                    end_char=len(parent.content),
                    start_token=0,
                    end_token=parent.token_count,
                    metadata={
                        **parent.metadata,
                        "header_path": list(parent.header_path),
                    },
                )
            )
        return child_blocks

    def _detect_language(self, text: str) -> str:
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_word_count = len(re.findall(r"[A-Za-z]+", text))
        return "zh" if cjk_count >= latin_word_count else "en"
