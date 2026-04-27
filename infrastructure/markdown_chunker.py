from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import tiktoken
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, PrivateAttr

from ..application.contracts import ChunkBundle, sha256_text
from ..domain.entities import ChildBlock, ParentChunk, SourceDocument
from ..utils.embeddings import build_embedding_model

Language = Literal["zh", "en"]


@dataclass(slots=True)
class SplitSection:
    """表示一级切分后的中间结果。"""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkerConfig(BaseModel):
    """控制结构切分、语义切分和长度兜底的配置。"""

    parent_max_tokens: int = Field(default=1200, ge=100)
    parent_min_tokens: int = Field(default=240, ge=1)
    parent_chunk_overlap: int = Field(default=120, ge=0)
    child_max_tokens: int = Field(default=320, ge=50)
    child_chunk_overlap: int = Field(default=50, ge=0)
    semantic_buffer_size: int = Field(default=1, ge=1)
    semantic_breakpoint_threshold_type: Literal[
        "percentile",
        "standard_deviation",
        "interquartile",
        "gradient",
    ] = Field(default="percentile")
    semantic_breakpoint_threshold_amount: float | None = Field(default=None)
    zh_semantic_sentence_split_regex: str = Field(default=r"(?<=[。！？；])(?:\s+)?|\n+")
    en_semantic_sentence_split_regex: str = Field(default=r"(?<=[.!?;:])\s+|\n+")
    language_detect_sample_size: int = Field(default=4000, ge=200)
    tokenizer_encoding: str = Field(default="cl100k_base")

    def model_post_init(self, __context: Any) -> None:
        """校验长度参数，避免出现无效配置。"""
        if self.parent_min_tokens >= self.parent_max_tokens:
            raise ValueError("父块最小长度必须小于父块最大长度。")
        if self.parent_chunk_overlap >= self.parent_max_tokens:
            raise ValueError("父块重叠长度必须小于父块最大长度。")
        if self.child_chunk_overlap >= self.child_max_tokens:
            raise ValueError("子块重叠长度必须小于子块最大长度。")


class TokenCounter:
    """负责估算文本长度，优先返回真实 token 数。"""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """初始化计数器并加载 tiktoken 编码器。"""
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """统计文本长度。"""
        text = text.strip()
        if not text:
            return 0
        return len(self._encoding.encode(text))


class MarkdownChunker(BaseModel):
    """先做路由，再做一级父块和二级子块切分。"""

    config: ChunkerConfig = Field(default_factory=ChunkerConfig)

    _token_counter: TokenCounter = PrivateAttr()
    _header_splitter: MarkdownHeaderTextSplitter = PrivateAttr()
    _parent_splitters: dict[Language, RecursiveCharacterTextSplitter] = PrivateAttr(default_factory=dict)
    _child_splitters: dict[Language, RecursiveCharacterTextSplitter] = PrivateAttr(default_factory=dict)
    _semantic_splitters: dict[Language, SemanticChunker] = PrivateAttr(default_factory=dict)
    _embedding_model: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """初始化结构切分器和长度计数器。"""
        self._token_counter = TokenCounter(self.config.tokenizer_encoding)
        self._header_splitter = self._build_header_splitter()

    def split_document(self, doc: SourceDocument, chunk_version: str) -> ChunkBundle:
        """对单个文档执行父子分层切分。"""
        sections = self._split_markdown(doc.parsed_md_content)
        parent_chunks = self._to_parent_chunks(doc, sections, chunk_version)
        parent_chunks = self._merge_small_parent_chunks(doc, parent_chunks)
        child_blocks = self._to_child_blocks(doc, parent_chunks, chunk_version)
        return ChunkBundle(source_document=doc, parent_chunks=parent_chunks, child_blocks=child_blocks)

    def _split_markdown(self, markdown: str) -> list[SplitSection]:
        """对 Markdown 文本执行一级切分，并自动选择切分路由。"""
        markdown = self._sanitize_text(markdown, keep_newlines=True)
        if not markdown:
            return []

        language = self._detect_language(markdown)
        if self._has_markdown_headings(markdown):
            sections = self._split_by_headers(markdown, language)
            if sections:
                return sections

        return self._split_by_semantic(markdown, language)

    def _has_markdown_headings(self, markdown: str) -> bool:
        """判断输入文本是否包含明确的 Markdown 标题。"""
        return bool(re.search(r"(?m)^\s{0,3}#{1,6}\s+\S", markdown))

    def _detect_language(self, text: str) -> Language:
        """自动识别文本主语言，目前只区分中文和英文。"""
        sample = self._normalize_plain_text(text)[: self.config.language_detect_sample_size]
        if not sample:
            return "en"

        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", sample))
        latin_word_count = len(re.findall(r"[A-Za-z]+", sample))
        return "zh" if cjk_count >= latin_word_count else "en"

    def _split_by_headers(self, markdown: str, language: Language) -> list[SplitSection]:
        """当存在明确标题时，优先按标题结构切分。"""
        documents = self._header_splitter.split_text(markdown)
        sections: list[SplitSection] = []

        for document in documents:
            content = self._sanitize_text(document.page_content, keep_newlines=True)
            if not content:
                continue

            header_path = self._extract_header_path(document.metadata)
            metadata = {
                "header_path": header_path,
                "heading_level": len(header_path) or None,
                "split_route": "标题结构切分",
                "language": language,
            }
            sections.extend(self._split_section_content(content, metadata, language))

        return sections

    def _split_by_semantic(self, markdown: str, language: Language) -> list[SplitSection]:
        """当不存在明确标题时，退回到 embedding 语义切分。"""
        text = self._normalize_plain_text(markdown)
        if not text:
            return []

        pieces = [
            self._sanitize_text(piece, keep_newlines=True)
            for piece in self._get_semantic_splitter(language).split_text(text)
            if self._sanitize_text(piece, keep_newlines=True)
        ]
        if not pieces:
            pieces = [text]

        sections: list[SplitSection] = []
        metadata = {
            "header_path": [],
            "heading_level": None,
            "split_route": "语义切分",
            "language": language,
        }
        for content in pieces:
            sections.extend(self._split_section_content(content, metadata, language))
        return sections

    def _split_section_content(
        self,
        content: str,
        metadata: dict[str, Any],
        language: Language,
    ) -> list[SplitSection]:
        """对单个一级块应用长度兜底切分。"""
        content = self._sanitize_text(content, keep_newlines=True)
        if not content:
            return []

        if self._token_counter.count(content) <= self.config.parent_max_tokens:
            return [SplitSection(content=content, metadata=dict(metadata))]

        pieces = self._split_by_length(content, self._get_parent_splitter(language))
        if len(pieces) <= 1:
            return [SplitSection(content=content, metadata=dict(metadata))]

        sections: list[SplitSection] = []
        total = len(pieces)
        for index, piece in enumerate(pieces):
            sections.append(
                SplitSection(
                    content=piece,
                    metadata={**metadata, "split_part": index, "split_total": total},
                )
            )
        return sections

    def _to_parent_chunks(
        self,
        doc: SourceDocument,
        sections: list[SplitSection],
        chunk_version: str,
    ) -> list[ParentChunk]:
        """把一级切分结果转换成 ParentChunk 模型。"""
        parent_chunks: list[ParentChunk] = []
        raw_contents = [self._sanitize_text(section.content, keep_newlines=True) for section in sections]
        spans = self._locate_spans(doc.parsed_md_content, raw_contents)

        for index, (section, (start_char, end_char)) in enumerate(zip(sections, spans, strict=True)):
            content = self._sanitize_text(section.content, keep_newlines=True)
            if not content:
                continue

            language = self._ensure_language(section.metadata.get("language"))
            header_path = list(section.metadata.get("header_path") or [])
            token_count = self._token_counter.count(content)
            parent_chunks.append(
                ParentChunk(
                    parent_id=uuid4(),
                    namespace_id=doc.namespace_id,
                    doc_id=doc.doc_id,
                    chunk_version=chunk_version,
                    chunk_index=index,
                    content=content,
                    content_sha256=sha256_text(content),
                    language=language,
                    token_count=token_count,
                    heading_level=section.metadata.get("heading_level"),
                    header_path=header_path,
                    split_route=str(section.metadata.get("split_route", "语义切分")),
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        **doc.metadata,
                        "file_name": doc.file_name,
                        "file_type": doc.file_type,
                        "token_count": token_count,
                        **section.metadata,
                        "header_path": header_path,
                        "language": language,
                    },
                )
            )

        return parent_chunks

    def _to_child_blocks(
        self,
        doc: SourceDocument,
        parent_chunks: list[ParentChunk],
        chunk_version: str,
    ) -> list[ChildBlock]:
        """把父块继续切成用于向量检索的子块。"""
        child_blocks: list[ChildBlock] = []

        for parent in parent_chunks:
            language = self._ensure_language(parent.metadata.get("language"))
            pieces = self._split_by_length(parent.content, self._get_child_splitter(language))
            if not pieces:
                pieces = [parent.content]

            spans = self._locate_spans(parent.content, pieces)
            total = len(pieces)
            for index, (piece, (local_start, local_end)) in enumerate(zip(pieces, spans, strict=True)):
                child_token_count = self._token_counter.count(piece)
                start_char = None
                end_char = None
                start_token = None
                end_token = None
                if local_start is not None and local_end is not None:
                    if parent.start_char is not None:
                        start_char = parent.start_char + local_start
                        end_char = parent.start_char + local_end
                    start_token = self._token_counter.count(parent.content[:local_start])
                    end_token = start_token + child_token_count

                child_blocks.append(
                    ChildBlock(
                        block_id=uuid4(),
                        namespace_id=doc.namespace_id,
                        doc_id=doc.doc_id,
                        parent_id=parent.parent_id,
                        chunk_version=chunk_version,
                        child_index=index,
                        content=piece,
                        content_sha256=sha256_text(piece),
                        language=language,
                        token_count=child_token_count,
                        start_char=start_char,
                        end_char=end_char,
                        start_token=start_token,
                        end_token=end_token,
                        metadata={
                            **parent.metadata,
                            "child_index": index,
                            "child_total": total,
                            "child_token_count": child_token_count,
                            "embedding_ready": False,
                            "header_path": list(parent.header_path),
                            "language": language,
                        },
                    )
                )

        return child_blocks

    def _merge_small_parent_chunks(self, doc: SourceDocument, parent_chunks: list[ParentChunk]) -> list[ParentChunk]:
        """把过小的父块与相邻父块合并，尽量满足最小长度约束。"""
        if not parent_chunks:
            return []

        groups: list[list[ParentChunk]] = []
        current_group: list[ParentChunk] = [parent_chunks[0]]
        current_tokens = parent_chunks[0].token_count

        for next_chunk in parent_chunks[1:]:
            next_tokens = next_chunk.token_count
            candidate_group = [*current_group, next_chunk]
            candidate_tokens = self._count_parent_group_tokens(doc, candidate_group)

            if current_tokens < self.config.parent_min_tokens and candidate_tokens <= self.config.parent_max_tokens:
                current_group = candidate_group
                current_tokens = candidate_tokens
                continue

            if (
                current_tokens >= self.config.parent_min_tokens
                and next_tokens < self.config.parent_min_tokens
                and candidate_tokens <= self.config.parent_max_tokens
            ):
                current_group = candidate_group
                current_tokens = candidate_tokens
                continue

            groups.append(current_group)
            current_group = [next_chunk]
            current_tokens = next_tokens

        if groups and current_tokens < self.config.parent_min_tokens:
            merged_tail = [*groups[-1], *current_group]
            merged_tail_tokens = self._count_parent_group_tokens(doc, merged_tail)
            if merged_tail_tokens <= self.config.parent_max_tokens:
                groups[-1] = merged_tail
            else:
                groups.append(current_group)
        else:
            groups.append(current_group)

        merged_chunks: list[ParentChunk] = []
        for index, group in enumerate(groups):
            merged_chunks.append(self._merge_parent_group(doc=doc, group=group, chunk_index=index))
        return merged_chunks

    def _build_header_splitter(self) -> MarkdownHeaderTextSplitter:
        """构造按 Markdown 标题切分的结构切分器。"""
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "一级标题"),
                ("##", "二级标题"),
                ("###", "三级标题"),
            ],
            strip_headers=False,
        )

    def _get_semantic_splitter(self, language: Language) -> SemanticChunker:
        """按语言懒加载语义切分器。"""
        if language not in self._semantic_splitters:
            self._semantic_splitters[language] = self._build_semantic_splitter(language)
        return self._semantic_splitters[language]

    def _build_semantic_splitter(self, language: Language) -> SemanticChunker:
        """构造指定语言的语义切分器。"""
        return SemanticChunker(
            embeddings=self._get_embedding_model(),
            buffer_size=self.config.semantic_buffer_size,
            breakpoint_threshold_type=self.config.semantic_breakpoint_threshold_type,
            breakpoint_threshold_amount=self.config.semantic_breakpoint_threshold_amount,
            sentence_split_regex=self._get_sentence_split_regex(language),
        )

    def _get_embedding_model(self) -> Any:
        """按需初始化并返回 embedding 模型。"""
        if self._embedding_model is None:
            self._embedding_model = build_embedding_model()
        return self._embedding_model

    def _get_parent_splitter(self, language: Language) -> RecursiveCharacterTextSplitter:
        """按语言懒加载父块长度兜底切分器。"""
        if language not in self._parent_splitters:
            self._parent_splitters[language] = self._build_length_splitter(
                chunk_size=self.config.parent_max_tokens,
                chunk_overlap=self.config.parent_chunk_overlap,
                language=language,
            )
        return self._parent_splitters[language]

    def _get_child_splitter(self, language: Language) -> RecursiveCharacterTextSplitter:
        """按语言懒加载子块长度兜底切分器。"""
        if language not in self._child_splitters:
            self._child_splitters[language] = self._build_length_splitter(
                chunk_size=self.config.child_max_tokens,
                chunk_overlap=self.config.child_chunk_overlap,
                language=language,
            )
        return self._child_splitters[language]

    def _build_length_splitter(
        self,
        chunk_size: int,
        chunk_overlap: int,
        language: Language,
    ) -> RecursiveCharacterTextSplitter:
        """构造指定语言的长度兜底切分器。"""
        separators = self._get_length_separators(language)
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=self.config.tokenizer_encoding,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

    def _get_sentence_split_regex(self, language: Language) -> str:
        """返回指定语言的语义切分句子边界规则。"""
        if language == "zh":
            return self.config.zh_semantic_sentence_split_regex
        return self.config.en_semantic_sentence_split_regex

    def _get_length_separators(self, language: Language) -> list[str]:
        """返回指定语言优先使用的长度切分边界。"""
        if language == "zh":
            return [
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                "：",
                "，",
                "、",
                ". ",
                "! ",
                "? ",
                "; ",
                ": ",
                ", ",
                " ",
                "",
            ]
        return [
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ": ",
            ", ",
            "。",
            "！",
            "？",
            "；",
            "：",
            "，",
            "、",
            " ",
            "",
        ]

    def _extract_header_path(self, metadata: dict[str, Any]) -> list[str]:
        """从标题切分器的 metadata 中提取标题路径。"""
        header_path = [
            metadata.get("一级标题"),
            metadata.get("二级标题"),
            metadata.get("三级标题"),
        ]
        return [item for item in header_path if item]

    def _split_by_length(self, text: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
        """使用统一长度切分器切分文本。"""
        return [
            self._sanitize_text(piece, keep_newlines=True)
            for piece in splitter.split_text(text)
            if self._sanitize_text(piece, keep_newlines=True)
        ]

    def _count_parent_group_tokens(self, doc: SourceDocument, group: list[ParentChunk]) -> int:
        """估算候选父块分组在合并后的 token 数。"""
        content = self._build_parent_group_content(doc=doc, group=group)
        return self._token_counter.count(content)

    def _merge_parent_group(self, doc: SourceDocument, group: list[ParentChunk], chunk_index: int) -> ParentChunk:
        """把一个或多个相邻父块合并为最终父块。"""
        if len(group) == 1:
            return group[0].model_copy(update={"chunk_index": chunk_index})

        first_chunk = group[0]
        merged_content = self._build_parent_group_content(doc=doc, group=group)
        merged_token_count = self._token_counter.count(merged_content)
        header_path = self._common_header_path([chunk.header_path for chunk in group])
        split_route = group[0].split_route if all(chunk.split_route == group[0].split_route for chunk in group) else "混合切分"
        start_char = group[0].start_char if all(chunk.start_char is not None for chunk in group) else None
        end_char = group[-1].end_char if all(chunk.end_char is not None for chunk in group) else None

        merged_metadata = {
            **doc.metadata,
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "token_count": merged_token_count,
            "header_path": header_path,
            "heading_level": len(header_path) or None,
            "split_route": split_route,
            "language": first_chunk.language,
            "merge_from_small_sections": True,
            "merged_parent_count": len(group),
        }

        return first_chunk.model_copy(
            update={
                "chunk_index": chunk_index,
                "content": merged_content,
                "content_sha256": sha256_text(merged_content),
                "token_count": merged_token_count,
                "heading_level": len(header_path) or None,
                "header_path": header_path,
                "split_route": split_route,
                "start_char": start_char,
                "end_char": end_char,
                "metadata": merged_metadata,
            }
        )

    def _build_parent_group_content(self, doc: SourceDocument, group: list[ParentChunk]) -> str:
        """基于原始文档范围构造父块合并后的内容，优先保留原始间隔。"""
        if group and all(chunk.start_char is not None and chunk.end_char is not None for chunk in group):
            raw_content = doc.parsed_md_content[group[0].start_char : group[-1].end_char]
            content = self._sanitize_text(raw_content, keep_newlines=True)
            if content:
                return content

        fallback = "\n\n".join(chunk.content for chunk in group if chunk.content)
        return self._sanitize_text(fallback, keep_newlines=True)

    def _common_header_path(self, header_paths: list[list[str]]) -> list[str]:
        """计算多个标题路径的最长公共前缀。"""
        if not header_paths:
            return []

        prefix = list(header_paths[0])
        for header_path in header_paths[1:]:
            match_length = 0
            for left, right in zip(prefix, header_path):
                if left != right:
                    break
                match_length += 1
            prefix = prefix[:match_length]
            if not prefix:
                break
        return prefix

    def _sanitize_text(self, text: str, keep_newlines: bool) -> str:
        """清理非法字符，并按需要保留换行。"""
        if not text:
            return ""

        text = re.sub(r"[\ud800-\udfff]", "", text)
        text = text.replace("\u00ad", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
        if keep_newlines:
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_plain_text(self, text: str) -> str:
        """把 Markdown 或纯文本规范化为适合语义切分的连续文本。"""
        text = self._sanitize_text(text, keep_newlines=False)
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
        return text.strip()

    def _ensure_language(self, value: Any) -> Language:
        """把 metadata 中的语言值规范化为受支持的语言。"""
        return "zh" if value == "zh" else "en"

    def _locate_spans(self, text: str, contents: list[str]) -> list[tuple[int | None, int | None]]:
        """按顺序为多个切分结果做最佳努力的字符定位。"""
        spans: list[tuple[int | None, int | None]] = []
        last_start: int | None = None
        for content in contents:
            start, end = self._locate_next_span(text=text, content=content, last_start=last_start)
            spans.append((start, end))
            if start is not None:
                last_start = start
        return spans

    def _locate_next_span(
        self,
        text: str,
        content: str,
        last_start: int | None,
    ) -> tuple[int | None, int | None]:
        """在文本中查找下一个切分片段的位置。"""
        if not content:
            return None, None

        search_from = 0 if last_start is None else min(last_start + 1, len(text))
        start = text.find(content, search_from)
        if start == -1 and last_start is not None:
            fallback_from = max(last_start - len(content), 0)
            start = text.find(content, fallback_from)
        if start == -1:
            start = text.find(content)
        if start == -1:
            return None, None
        return start, start + len(content)
