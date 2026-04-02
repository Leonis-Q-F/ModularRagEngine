from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    ocr_provider: Literal["paddle", "openai"] = "paddle"
    ocr_timeout_seconds: int = Field(default=60, ge=1)
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_timeout_seconds: int = Field(default=60, ge=1)
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1)
    postgres_db: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: Optional[str] = None
    postgres_schema: str = "public"
    postgres_sslmode: str = "prefer"
    postgres_timeout_seconds: int = Field(default=30, ge=1)
    milvus_uri: str = "http://localhost:19530"
    milvus_token: Optional[str] = None
    milvus_db_name: Optional[str] = None
    milvus_timeout_seconds: int = Field(default=30, ge=1)

    # Paddle
    paddle_ocr_api_key: Optional[str] = None
    paddle_ocr_base_url: Optional[str] = None

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_model: Optional[str] = None

    # OpenAI Embedding
    embedding_openai_api_key: Optional[str] = None
    embedding_openai_api_base: Optional[str] = "https://api.openai.com/v1"
    embedding_openai_model: Optional[str] = None

    # Ollama Embedding
    embedding_ollama_base_url: Optional[str] = "http://localhost:11434"
    embedding_ollama_model: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
