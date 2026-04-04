from __future__ import annotations

import json
from pathlib import Path

from ..config import settings
from ..domain.exceptions import UnsupportedFileError
from ..domain.value_objects import ParsedDocument
from ..utils.ocr import OpenAIVisionClient, PaddleOCRClient


class DocumentLoader:
    """把文件加载为统一的 ParsedDocument。"""

    def load(self, file_paths: list[str], use_ocr: bool = False) -> list[ParsedDocument]:
        """批量加载多个文件。"""
        return [self._load_single(file_path, use_ocr=use_ocr) for file_path in file_paths]

    def _load_single(self, file_path: str, use_ocr: bool = False) -> ParsedDocument:
        """加载单个文件并转换为 ParsedDocument。"""
        path = Path(file_path)
        file_type = path.suffix.lower().strip(".")
        if not file_type:
            raise UnsupportedFileError(f"无法识别文件类型: {file_path}")

        content = self._convert_to_markdown(path=path, file_type=file_type, use_ocr=use_ocr)
        return ParsedDocument(
            external_doc_id=file_path,
            file_name=path.name,
            file_type=file_type,
            source_uri=str(path),
            parsed_md_content=content,
        )

    def _convert_to_markdown(self, path: Path, file_type: str, use_ocr: bool) -> str:
        """按文件类型把原始文件转换为 Markdown。"""
        if file_type in {"md", "txt"}:
            return self._read_text(path)
        if file_type == "json":
            return self._json_to_markdown(path)
        if file_type == "csv":
            return self._csv_to_markdown(path)
        if file_type in {"doc", "docx", "html", "htm"}:
            return self._markitdown_convert(path)
        if file_type == "pdf":
            return self._pdf_to_markdown(path, use_ocr=use_ocr)
        if file_type in {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}:
            return self._ocr_to_markdown(path)
        raise UnsupportedFileError(f"不支持的文件类型: {path}")

    def _read_text(self, path: Path) -> str:
        """按候选编码读取文本文件。"""
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"文件编码无法识别: {path}")

    def _json_to_markdown(self, path: Path) -> str:
        """把 JSON 文件格式化为 Markdown 代码块。"""
        data = json.loads(self._read_text(path))
        return f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"

    def _csv_to_markdown(self, path: Path) -> str:
        """把 CSV 表格转换为 Markdown 表格。"""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("处理 CSV 需要安装 pandas。") from exc

        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                df = pd.read_csv(path, encoding=encoding)
                return df.to_markdown(index=False)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"CSV 文件编码无法识别: {path}")

    def _markitdown_convert(self, path: Path) -> str:
        """使用 MarkItDown 转换 Office 或 HTML 文档。"""
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise ImportError("处理该文件类型需要安装 markitdown。") from exc

        return MarkItDown().convert(str(path)).text_content

    def _pdf_to_markdown(self, path: Path, use_ocr: bool) -> str:
        """优先用 docling 解析 PDF，失败时按需回退 OCR。"""
        try:
            from docling.document_converter import DocumentConverter

            result = DocumentConverter().convert(str(path))
            return result.document.export_to_markdown()
        except Exception:
            if use_ocr:
                return self._ocr_to_markdown(path)
            raise

    def _ocr_to_markdown(self, path: Path) -> str:
        """调用当前配置的 OCR 客户端提取 Markdown。"""
        if settings.ocr_provider == "openai":
            return OpenAIVisionClient().run(str(path))
        return PaddleOCRClient().run(str(path))
