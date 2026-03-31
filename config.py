from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    ocr_provider: Literal["paddle", "openai"] = "paddle"
    ocr_timeout_seconds: int = Field(default=60, ge=1)

    # Paddle
    paddle_ocr_api_key: Optional[str] = None
    paddle_ocr_base_url: Optional[str] = None

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_model: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
