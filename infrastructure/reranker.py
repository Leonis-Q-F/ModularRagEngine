from __future__ import annotations
from typing import Any, Callable, Protocol

from ..config import settings
from ..domain.entities import IndexEntry


class CrossEncoderLike(Protocol):
    """描述 cross-encoder 模型所需的最小接口。"""

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
    ) -> Any:
        """为查询-文档对批量打分。"""
        ...


class SemanticReranker:
    """基于 cross-encoder 的语义 reranker。"""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        device: str | None = None,
        model: CrossEncoderLike | None = None,
        model_factory: Callable[..., CrossEncoderLike] | None = None,
    ) -> None:
        """初始化 cross-encoder 模型并缓存推理参数。"""
        self.provider_name = "sentence-transformers"
        self.model_name = model_name or settings.reranker_model_name
        self._batch_size = batch_size or settings.reranker_batch_size
        self._max_length = max_length if max_length is not None else settings.reranker_max_length
        self._device = device if device is not None else settings.reranker_device
        if model is not None:
            self._model = model
            return

        factory = model_factory or self._default_model_factory
        self._model = factory(
            self.model_name,
            max_length=self._max_length,
            device=self._device,
        )

    def rerank(self, query: str, entries: list[IndexEntry], top_k: int) -> list[tuple[IndexEntry, float]]:
        """按 cross-encoder 分数重新排序候选块。"""
        if not entries:
            return []

        pairs = [(query, entry.retrieval_text) for entry in entries]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scored: list[tuple[IndexEntry, float]] = []
        for entry, score in zip(entries, raw_scores, strict=True):
            scored.append((entry, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _default_model_factory(
        self,
        model_name: str,
        *,
        max_length: int | None,
        device: str | None,
    ) -> CrossEncoderLike:
        """按需加载 sentence-transformers CrossEncoder。"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - 依赖缺失时走真实环境错误
            raise ImportError(
                "缺少 sentence-transformers 依赖，无法启用 cross-encoder reranker。"
            ) from exc

        kwargs: dict[str, Any] = {}
        if max_length is not None:
            kwargs["max_length"] = max_length
        if device:
            kwargs["device"] = device
        return CrossEncoder(model_name, **kwargs)
