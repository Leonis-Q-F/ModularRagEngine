"""旧路径兼容层：入库用例已经迁移到 application.use_cases。"""

from .use_cases.ingest import IngestUseCase

IngestService = IngestUseCase

__all__ = [
    "IngestService",
    "IngestUseCase",
]
