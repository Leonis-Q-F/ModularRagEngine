from __future__ import annotations

from typing import Any, Protocol

try:
    from ..config import Settings, settings
except ImportError:  # pragma: no cover - 兼容直接从仓库根目录运行
    from config import Settings, settings

try:
    from langchain_core.embeddings import Embeddings
except ImportError:  # pragma: no cover - 运行时按 provider 抛错
    class Embeddings(Protocol):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            ...

        def embed_query(self, text: str) -> list[float]:
            ...

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - 按需抛错
    OpenAIEmbeddings = None

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover - 运行时按 provider 抛错
    OllamaClient = None


class EmbeddingAdapter(Embeddings):
    """把官方 ollama Python 客户端适配为 LangChain Embeddings 接口。"""

    def __init__(self, base_url: str, model: str) -> None:
        """初始化 Ollama 客户端和模型配置。"""
        if OllamaClient is None:
            raise ImportError("缺少 `ollama` 依赖，请先安装 `ollama`。")

        self._client = OllamaClient(host=base_url)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量计算多个文本的向量。"""
        normalized_texts = [text.strip() or " " for text in texts]
        if not normalized_texts:
            return []

        response = self._client.embed(model=self._model, input=normalized_texts)
        embeddings = self._extract_embeddings(response)
        if len(embeddings) != len(normalized_texts):
            raise RuntimeError("Ollama 返回的向量数量与输入文本数量不一致。")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """计算单条查询文本的向量。"""
        embeddings = self.embed_documents([text])
        if not embeddings:
            raise RuntimeError("Ollama 未返回任何查询向量。")
        return embeddings[0]

    def _extract_embeddings(self, response: Any) -> list[list[float]]:
        """从 Ollama 响应中提取 embeddings 字段。"""
        if isinstance(response, dict):
            embeddings = response.get("embeddings")
        else:
            embeddings = getattr(response, "embeddings", None)

        if embeddings is None:
            raise RuntimeError("Ollama 返回中缺少 embeddings 字段。")
        return embeddings


def build_embedding_model() -> Embeddings:
    """基于全局配置构建当前启用的 embedding 模型。"""
    return build_embedding_model_from_settings(settings)


def build_embedding_model_from_settings(app_settings: Settings) -> Embeddings:
    """根据指定配置构建 embedding 模型。"""
    if app_settings.embedding_provider == "openai":
        return _build_openai_embedding_model(app_settings)
    if app_settings.embedding_provider == "ollama":
        return _build_ollama_embedding_model(app_settings)
    raise ValueError(f"不支持的 embedding provider: {app_settings.embedding_provider}")


def _build_openai_embedding_model(app_settings: Settings) -> Embeddings:
    """构建 OpenAI embedding 模型。"""
    if OpenAIEmbeddings is None:
        raise ImportError("缺少 `langchain_openai` 依赖，请先安装。")
    api_key = _require_setting(app_settings.embedding_openai_api_key, "EMBEDDING_OPENAI_API_KEY")
    model = _require_setting(app_settings.embedding_openai_model, "EMBEDDING_OPENAI_MODEL")

    return OpenAIEmbeddings(
        openai_api_key=api_key,
        openai_api_base=app_settings.embedding_openai_api_base,
        model=model,
        request_timeout=app_settings.embedding_timeout_seconds,
    )


def _build_ollama_embedding_model(app_settings: Settings) -> Embeddings:
    """构建基于官方 ollama 客户端的 embedding 模型。"""
    base_url = _require_setting(app_settings.embedding_ollama_base_url, "EMBEDDING_OLLAMA_BASE_URL")
    model = _require_setting(app_settings.embedding_ollama_model, "EMBEDDING_OLLAMA_MODEL")
    return EmbeddingAdapter(base_url=base_url, model=model)


def _require_setting(value: str | None, env_name: str) -> str:
    """确保必需配置存在，并返回清理后的值。"""
    if value and value.strip():
        return value.strip()
    raise ValueError(f"缺少 embedding 配置：{env_name}")
