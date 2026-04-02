from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field ,UUID4

# ==========================================
# 1. 枚举类型定义 (规范数据字典)
# ==========================================

class FileType(str, Enum):
    TXT = "txt"
    MD = "md"
    DOC = "doc"
    DOCX = "docx"
    HTML = "html"
    HTM = "htm"
    JSON = "json"
    CSV = "csv"
    XLS = "xls"
    XLSX = "xlsx"
    PDF = "pdf"
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    TIFF = "tiff"
    TIF = "tif"

class DocStatus(str, Enum):
    PENDING = "pending"         # 等待处理
    PARSING = "parsing"         # 正在解析为 Markdown
    SPLITTING = "splitting"     # 正在进行混合切分
    EMBEDDING = "embedding"     # 正在向量化
    COMPLETED = "completed"     # 处理完成
    FAILED = "failed"           # 处理失败

# ==========================================
# 2. 核心数据模型定义
# ==========================================

class SourceDocument(BaseModel):
    """
    源文档模型
    """
    doc_id: UUID4 = Field(..., description="文档唯一标识符")
    file_name: str = Field(..., description="文件名")
    file_type: FileType = Field(..., description="文件类型")
    parsed_md_content: str = Field(..., description="解析后的 Markdown 内容")
    status: DocStatus = Field(DocStatus.PENDING, description="文档处理状态")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="入库时间")

class ParentChunk(BaseModel):
    """
    父文档表：存储包含完整语义和上下文的“大块”（存入关系型/文档数据库）
    """
    parent_id: UUID4 = Field(..., description="主键：唯一标识一个语义大块")
    doc_id: UUID4 = Field(..., description="外键：关联的原始文件 ID")
    content: str = Field(..., min_length=1, description="大块的完整纯文本内容，用于丢给 LLM 的上下文")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Markdown 标题层级等提取信息，如 {'Header 1': '背景', 'Header 2': '目标'}")
    chunk_index: int = Field(..., ge=0, description="该大块在原文档中的顺序索引（从 0 开始）")


class ChildBlock(BaseModel):
    """
    子数据块表：存储细粒度“小块”及其向量（存入向量数据库）
    """
    block_id: UUID4 = Field(..., description="主键：唯一标识一个用于检索的小块")
    parent_id: UUID4 = Field(..., description="核心映射键：强关联指向 ParentChunk 表")
    content: str = Field(..., description="小块的具体文本")
    embedding: List[float] = Field(..., description="文本转化后的高维向量数组")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="继承自父级的 metadata，并可追加特有属性（用于向量库的 Metadata Filtering）")