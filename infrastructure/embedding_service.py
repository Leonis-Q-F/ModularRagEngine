from __future__ import annotations

try:
    from ..config import settings
except ImportError:  # pragma: no cover - 兼容直接从仓库根目录运行
    from config import settings

from ..utils.embeddings import build_embedding_model


class EmbeddingService:
    """基于当前配置构建真实 embedding provider。"""

    def __init__(self) -> None:
        self._model = build_embedding_model()
        self.provider_name = settings.embedding_provider
        self.model_name = self._resolve_model_name()
        self.dimension = len(self.embed_query("dimension probe"))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    def _resolve_model_name(self) -> str:
        model_name = getattr(self._model, "model", None)
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()

        internal_model = getattr(self._model, "_model", None)
        if isinstance(internal_model, str) and internal_model.strip():
            return internal_model.strip()

        return type(self._model).__name__
