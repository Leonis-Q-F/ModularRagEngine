"""旧路径兼容层：索引服务已经迁移到 application.services。"""

from .services.indexing_service import IndexingService

IndexService = IndexingService

__all__ = [
    "IndexService",
    "IndexingService",
]
