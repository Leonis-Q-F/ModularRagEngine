from __future__ import annotations

import re
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, PrivateAttr

from models.schemes import ChildBlock
from utils.embeddings import build_embedding_model

try:
    import tiktoken
except ImportError:  # pragma: no cover - 可选依赖
    tiktoken = None


class VectorizeConfig(BaseModel):
    """控制子块向量化行为的配置。"""

    batch_size: int = Field(default=16, ge=1, description="批量请求 embedding 时的批大小。")
    min_chars: int = Field(default=10, ge=0, description="小于该字符数的文本默认跳过向量化。")
    min_tokens: int = Field(default=3, ge=0, description="小于该 token 数的文本默认跳过向量化。")
    skip_short_blocks: bool = Field(default=True, description="是否跳过过短文本和明显噪声块。")
    overwrite_existing: bool = Field(default=False, description="是否覆盖已有 embedding 的子块。")
    fail_on_error: bool = Field(default=False, description="批量向量化失败时是否直接抛错。")
    tokenizer_encoding: str = Field(default="cl100k_base", description="tiktoken 使用的编码名称。")


class VectorizeResult(BaseModel):
    """表示子块向量化后的结果。"""

    child_blocks: list[ChildBlock] = Field(default_factory=list, description="已更新 embedding 的子块列表。")
    embedded_count: int = Field(default=0, ge=0, description="成功向量化的子块数量。")
    skipped_count: int = Field(default=0, ge=0, description="被跳过的子块数量。")
    failed_count: int = Field(default=0, ge=0, description="向量化失败的子块数量。")

class ChildBlockVectorizer(BaseModel):
    """负责把 ChildBlock 列表转换成带 embedding 的结果。"""

    config: VectorizeConfig = Field(default_factory=VectorizeConfig)

    _embedding_model: Embeddings | None = PrivateAttr(default=None)
    _encoding: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """初始化 token 计数依赖。"""
        if tiktoken is not None:
            try:
                self._encoding = tiktoken.get_encoding(self.config.tokenizer_encoding)
            except Exception:
                self._encoding = None

    def vectorize_blocks(self, blocks: list[ChildBlock]) -> VectorizeResult:
        """对一组子块执行批量向量化。"""
        if not blocks:
            return VectorizeResult()

        output_blocks = list(blocks) # 保留原始顺序进行替换
        pending_items: list[tuple[int, ChildBlock]] = []
        skipped_count = 0
        embedded_count = 0
        failed_count = 0

        # 过滤无效子快
        for index, block in enumerate(blocks):
            skip_reason = self._get_skip_reason(block)
            if skip_reason is not None:
                output_blocks[index] = self._mark_block_skipped(block, skip_reason)
                skipped_count += 1
                continue

            pending_items.append((index, block))

        # 批量向量化
        for batch in self._iter_batches(pending_items):
            batch_indices = [item[0] for item in batch]
            batch_blocks = [item[1] for item in batch]
            batch_texts = [block.content for block in batch_blocks]

            try:
                vectors = self._embed_batch(batch_texts)
                for index, block, vector in zip(batch_indices, batch_blocks, vectors, strict=True):
                    output_blocks[index] = self._attach_embedding(block, vector)
                    embedded_count += 1
            except Exception as exc:
                if self.config.fail_on_error:
                    raise RuntimeError(f"批量向量化失败: {exc}") from exc

                for index, block in batch:
                    try:
                        vector = self._embed_single(block.content)
                        output_blocks[index] = self._attach_embedding(block, vector)
                        embedded_count += 1
                    except Exception as single_exc:
                        output_blocks[index] = self._mark_block_failed(block, single_exc)
                        failed_count += 1

        return VectorizeResult(
            child_blocks=output_blocks,
            embedded_count=embedded_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
        )



    def _get_skip_reason(self, block: ChildBlock) -> str | None:
        """判断一个子块是否应跳过向量化。"""
        if not self.config.skip_short_blocks:
            return None

        text = block.content.strip()
        if not text:
            return "空文本"
        if len(text) < self.config.min_chars:
            return "字符数过短"
        if self._count_tokens(text) < self.config.min_tokens:
            return "token 数过少"
        if re.fullmatch(r"[\d\W_]+", text):
            return "纯数字或符号"

        return None

    def _iter_batches(self, items: list[tuple[int, ChildBlock]]) -> list[list[tuple[int, ChildBlock]]]:
        """把待处理子块按批大小切成多个批次。"""
        batches: list[list[tuple[int, ChildBlock]]] = []
        for start in range(0, len(items), self.config.batch_size):
            batches.append(items[start:start + self.config.batch_size])
        return batches

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """调用 embedding 模型批量计算向量。"""
        vectors = self._get_embedding_model().embed_documents(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("embedding 返回数量与输入文本数量不一致。")
        return vectors

    def _embed_single(self, text: str) -> list[float]:
        """对单条文本执行向量化。"""
        vectors = self._embed_batch([text])
        return vectors[0]

    def _attach_embedding(self, block: ChildBlock, vector: list[float]) -> ChildBlock:
        """把 embedding 结果回填到子块中。"""
        metadata = {
            **block.metadata,
            "embedding_ready": True,
            "embedding_dimension": len(vector),
        }
        metadata.pop("embedding_error", None)
        metadata.pop("embedding_skipped", None)
        metadata.pop("embedding_skip_reason", None)

        return block.model_copy(
            update={
                "embedding": vector,
                "metadata": metadata,
            },
            deep=True,
        )

    def _mark_block_skipped(self, block: ChildBlock, reason: str) -> ChildBlock:
        """标记一个子块被跳过向量化。"""
        metadata = {
            **block.metadata,
            "embedding_ready": False,
            "embedding_skipped": True,
            "embedding_skip_reason": reason,
        }
        return block.model_copy(update={"metadata": metadata}, deep=True)

    def _mark_block_failed(self, block: ChildBlock, exc: Exception) -> ChildBlock:
        """标记一个子块向量化失败。"""
        metadata = {
            **block.metadata,
            "embedding_ready": False,
            "embedding_error": str(exc),
        }
        return block.model_copy(update={"metadata": metadata}, deep=True)

    def _get_embedding_model(self) -> Embeddings:
        """按需初始化并返回 embedding 模型。"""
        if self._embedding_model is None:
            self._embedding_model = build_embedding_model()
        return self._embedding_model

    def _count_tokens(self, text: str) -> int:
        """统计文本长度，优先按 token 计数。"""
        text = text.strip()
        if not text:
            return 0

        if self._encoding is not None:
            return len(self._encoding.encode(text))

        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text))


__all__ = [
    "ChildBlockVectorizer",
    "VectorizeConfig",
    "VectorizeResult",
]
