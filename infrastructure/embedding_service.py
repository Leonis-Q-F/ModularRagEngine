from __future__ import annotations

from typing import Any, Callable

from ..config import settings
from ..utils.embeddings import build_embedding_model


class EmbeddingService:
    """embedding provider 适配层。"""

    def __init__(
        self,
        model: Any | None = None,
        model_factory: Callable[[], Any] | None = None,
        provider_name: str | None = None,
    ) -> None:
        """初始化当前启用的 embedding 模型及其元信息。"""
        self._model = model or (model_factory or build_embedding_model)()
        self.provider_name = provider_name or settings.embedding_provider
        self.model_name = self._resolve_model_name()
        self._dimension: int | None = self._resolve_dimension_from_model()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成文本向量。"""
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """生成单条查询向量。"""
        return self._model.embed_query(text)

    @property
    def dimension(self) -> int:
        """按需解析 embedding 向量维度，并缓存结果。"""
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def _resolve_model_name(self) -> str:
        """从底层模型对象中解析可读模型名。"""
        model_name = getattr(self._model, "model", None)
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()

        internal_model = getattr(self._model, "_model", None)
        if isinstance(internal_model, str) and internal_model.strip():
            return internal_model.strip()

        return type(self._model).__name__

    def _resolve_dimension_from_model(self) -> int | None:
        """优先尝试从模型元信息中读取维度，避免构造阶段远程探测。"""
        for attr_name in ("dimension", "dimensions", "embedding_dim"):
            value = getattr(self._model, attr_name, None)
            if isinstance(value, int) and value > 0:
                return value
        return None
