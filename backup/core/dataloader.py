"""
使用 MarkItDown 作为统一的文档转换引擎，将各种格式的文档转换为 Markdown 格式，
以便后续的文本处理和分析。
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from markitdown import MarkItDown
from pydantic import BaseModel, PrivateAttr
from pydantic_settings import BaseSettings

from config import settings
from models.schemes import FileType, SourceDocument
from utils.ocr import OpenAIVisionClient, PaddleOCRClient

logger = logging.getLogger(__name__)

class DocumentLoader(BaseModel):
    """
    文档加载器，用于将各种格式的文档转换为 SourceDocument。
    """

    _md_client: MarkItDown = PrivateAttr(default_factory=MarkItDown) # MarkItDown 实例
    _settings: BaseSettings = PrivateAttr(default=settings) # 全局配置
    _pdf_converter_client: Any = PrivateAttr(default=None) # PDF 转换模型

    @property
    def _pdf_converter(self) -> Any:
        """懒加载机制：只有在处理 PDF 时才初始化 Docling 模型"""
        if self._pdf_converter_client is None:
            from docling.document_converter import DocumentConverter

            self._pdf_converter_client = DocumentConverter()
        return self._pdf_converter_client

    def load(self, file_paths: str | list[str], use_ocr: bool = False) -> SourceDocument | list[SourceDocument]:
        """
        统一的文档加载入口。
        支持传入单个文件路径或文件路径列表，并返回 SourceDocument 模型。
        """
        if isinstance(file_paths, list):
            return [self._load_single(fp, use_ocr=use_ocr) for fp in file_paths]
        return self._load_single(file_paths, use_ocr=use_ocr)

    def _load_single(self, file_path: str, use_ocr: bool = False) -> SourceDocument:
        """处理单个文件，并封装为 SourceDocument。"""
        if not self.is_supported_file(file_path):
            raise ValueError(f"不支持的文件类型: {file_path}")

        markdown = self._convert_single_to_markdown(file_path, use_ocr=use_ocr)
        return self._build_source_document(file_path, markdown)

    def _convert_single_to_markdown(self, file_path: str, use_ocr: bool = False) -> str:
        """把单个文件转换为标准 Markdown 文本。"""
        ext = Path(file_path).suffix.lower().strip(".")

        if ext in ["txt", "md", "doc", "docx", "html", "htm"]:
            return self._standard_markdown_loader(file_path)
        if ext == "pdf":
            return self._pdf_markdown_loader(file_path, use_ocr=use_ocr)
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff", "tif"]:
            return self._ocr_markdown_loader(file_path)
        if ext in ["json", "csv", "xls", "xlsx"]:
            return self._structured_data_markdown_loader(file_path)
        raise RuntimeError(f"未预期的文件扩展名: {ext}")

    def _build_source_document(self, file_path: str, parsed_md_content: str) -> SourceDocument:
        """根据文件路径和解析后的 Markdown 构建 SourceDocument。"""
        path = Path(file_path)
        ext = path.suffix.lower().strip(".")
        return SourceDocument(
            doc_id=uuid4(),
            file_name=path.name,
            file_type=FileType(ext),
            parsed_md_content=parsed_md_content,
        )

    def is_supported_file(self, file_path: str) -> bool:
        """判断文件扩展名是否在支持列表中。"""
        ext = Path(file_path).suffix.lower().strip(".")
        return ext in FileType._value2member_map_

    # 分支 1 ： 标准 Markdown转换
    def _standard_markdown_loader(self, file_path: str) -> str:
        """处理可直接转换的文本类文档。"""
        try:
            return self._md_client.convert(file_path).text_content
        except Exception as exc:
            raise RuntimeError(f"文档转换失败: {exc}") from exc

    # 分支 2 ： pdf 转 Markdown
    def _pdf_markdown_loader(self, file_path: str, use_ocr: bool = False) -> str:
        """
        PDF 提取引擎：优先使用 Docling，兜底转图片 OCR 或 PyPDFLoader。
        """
        errors: list[str] = []

        try:
            result = self._pdf_converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as doc_converter_error:
            errors.append(f"Docling: {doc_converter_error}")
            logger.warning("Docling 解析失败，准备回退: %s", doc_converter_error)

        if use_ocr:
            try:
                return self._pdf_to_ocr_flow(file_path)
            except Exception as ocr_error:
                errors.append(f"PDF OCR 回退: {ocr_error}")
                logger.warning("PDF 转图片 OCR 回退失败，继续尝试文本回退: %s", ocr_error)

        try:
            from langchain_community.document_loaders import PyPDFLoader

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            md_content = []
            for doc in docs:
                text = doc.page_content.strip()

                if not text:
                    continue
                else:
                    md_content.append(text)

            if md_content:
                return "\n\n".join(md_content)
            errors.append("PyPDFLoader: 未提取到任何可用文本")
        except Exception as pypdf_error:
            errors.append(f"PyPDFLoader: {pypdf_error}")

        raise RuntimeError("PDF 提取彻底失败，所有回退方案均失败: " + " | ".join(errors))

    def _pdf_to_ocr_flow(self, file_path: str) -> str:
        """将 PDF 转换为图片并调用 OCR"""
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("需要安装 pymupdf 才能进行 PDF 转图片：pip install pymupdf") from exc

        md_results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            doc = fitz.open(file_path)
            image_paths = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_path = os.path.join(temp_dir, f"page_{i}.png")
                pix.save(img_path)
                image_paths.append(img_path)
            doc.close()

            for img_path in image_paths:
                content = self._ocr_markdown_loader(img_path)
                if content:
                    md_results.append(content)

        if not md_results:
            raise RuntimeError("PDF 已转为图片，但 OCR 未提取到任何内容")

        return "\n\n---\n\n".join(md_results)

    # 分支 3 ： OCR 转 Markdown
    def _ocr_markdown_loader(self, file_path: str) -> str:
        """对图片文件执行 OCR，并返回 Markdown 文本。"""
        try:
            if self._settings.ocr_provider == "paddle":
                client = PaddleOCRClient()
            elif self._settings.ocr_provider == "openai":
                client = OpenAIVisionClient()
            else:
                raise ValueError("不支持的 OCR provider")

            content = client.run(file_path).strip()
            if not content:
                raise RuntimeError(f"OCR 未提取到任何内容: {file_path}")
            return content
        except Exception as exc:
            raise RuntimeError(f"OCR 提取失败: {exc}") from exc

    # 分支 4 ： 结构化数据转换
    def _structured_data_markdown_loader(self, file_path: str) -> str:
        """把 JSON、CSV、Excel 等结构化数据转换为 Markdown。"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".json":
            data = self._read_json(path)
            return self._json_to_markdown(data)

        if ext == ".csv":
            df = self._read_csv(path)
            return self._dataframe_to_markdown(df)

        if ext in {".xls", ".xlsx"}:
            sheets = self._read_excel(path)
            parts = []

            for sheet_name, df in sheets.items():
                parts.append(f"## {sheet_name}")
                parts.append(self._dataframe_to_markdown(df))

            return "\n\n".join(parts)

        raise RuntimeError(f"不支持的结构化数据格式: {path.suffix}")

    def _read_json(self, file_path: Path) -> Any:
        """读取 JSON 文件，并自动尝试常见中文编码。"""
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return json.loads(file_path.read_text(encoding=encoding))
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON 解析失败: {exc}") from exc

        raise RuntimeError(f"JSON 文件编码无法识别: {file_path}")

    def _read_csv(self, file_path: Path):
        """读取 CSV 文件，并自动尝试常见中文编码。"""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("处理 CSV 需要安装 pandas") from exc

        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise RuntimeError(f"CSV 文件编码无法识别: {file_path}")

    def _read_excel(self, file_path: Path):
        """读取 Excel 文件，返回全部工作表。"""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("处理 Excel 需要安装 pandas") from exc

        try:
            return pd.read_excel(file_path, sheet_name=None)
        except Exception as exc:
            raise RuntimeError(f"Excel 读取失败: {exc}") from exc

    def _json_to_markdown(self, data: Any) -> str:
        """把 JSON 数据转换为 Markdown 表格或代码块。"""
        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            try:
                import pandas as pd
            except ImportError:
                return f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"

            df = pd.json_normalize(data)
            return self._dataframe_to_markdown(df)

        return f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"

    def _dataframe_to_markdown(self, df) -> str:
        """把 DataFrame 规范化后转换为 Markdown 表格。"""
        normalized = df.fillna("").copy()

        for column in normalized.columns:
            normalized[column] = normalized[column].map(
                lambda value: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
            )

        return normalized.to_markdown(index=False)

__all__ = [
    "DocumentLoader"
]
