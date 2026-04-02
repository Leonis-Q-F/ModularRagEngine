import base64
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field, PrivateAttr

try:
    from ..config import settings
except ImportError:  # pragma: no cover - 兼容直接从仓库根目录运行
    from config import settings


class PaddleOCRClient(BaseModel):
    """基于百度 AI Studio 的 PaddleOCR 布局解析客户端"""

    api_token: str | None = Field(
        default_factory=lambda: settings.paddle_ocr_api_key,
        description="Paddle AI Studio 的 Token",
    )
    api_url: str | None = Field(
        default_factory=lambda: settings.paddle_ocr_base_url,
        description="PaddleOCR API 地址",
    )
    timeout_seconds: int = Field(
        default_factory=lambda: settings.ocr_timeout_seconds,
        ge=1,
        description="OCR 请求超时时间（秒）",
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.api_token:
            raise ValueError("缺少 Paddle OCR API Key 配置")
        if not self.api_url:
            raise ValueError("缺少 Paddle OCR API 地址配置")

    def run(self, file_path: str) -> str:
        """调用 PaddleOCR API 将图片转换为纯 Markdown 文本"""
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            file_data = base64.b64encode(file_bytes).decode("ascii")

        headers = {
            "Authorization": f"token {self.api_token}",
            "Content-Type": "application/json",
        }

        ext = Path(file_path).suffix.lower().strip(".")
        file_type = 0 if ext == "pdf" else 1

        payload = {
            "file": file_data,
            "fileType": file_type,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Paddle OCR 请求失败: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Paddle OCR 请求失败，状态码: {response.status_code}, 返回: {response.text}"
            )

        try:
            result = response.json().get("result", {})
        except ValueError as exc:
            raise RuntimeError(f"Paddle OCR 返回了无法解析的 JSON: {response.text}") from exc

        layout_results = result.get("layoutParsingResults", [])

        md_texts = []
        for res in layout_results:
            markdown_data = res.get("markdown", {})
            text_content = markdown_data.get("text", "")
            if text_content:
                md_texts.append(text_content)

        return "\n\n".join(md_texts)


class OpenAIVisionClient(BaseModel):
    """基于 OpenAI 原生视觉模型的兜底 OCR 客户端"""

    api_key: str | None = Field(default_factory=lambda: settings.openai_api_key, description="OpenAI API Key")
    api_base: str | None = Field(default_factory=lambda: settings.openai_api_base, description="OpenAI API Base")
    api_model: str | None = Field(default_factory=lambda: settings.openai_api_model, description="OpenAI API 模型")
    _client: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if not self.api_key:
            raise ValueError("缺少 OpenAI API Key 配置")
        if not self.api_model:
            raise ValueError("缺少 OpenAI OCR 模型配置")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("请安装 OpenAI SDK：pip install openai") from exc

        client_kwargs = {"api_key": self.api_key}
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        self._client = OpenAI(**client_kwargs)

    def run(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            ext = Path(file_path).suffix.lower().strip(".")
            mime_type = f"image/{ext}" if ext in ["jpeg", "png", "gif", "webp"] else "image/jpeg"

            response = self._client.chat.completions.create(
                model=self.api_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请将这张图片中的所有文本、表格和版面结构完整地提取出来，并使用标准的 Markdown 格式输出。不需要任何额外的解释或对话。",
                            },
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=2000,
                timeout=60,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            raise RuntimeError(f"OpenAI API 调用失败: {exc}") from exc
