"""旧路径兼容层：检索用例已经迁移到 application.use_cases。"""

from .use_cases.search import SearchUseCase

SearchService = SearchUseCase

__all__ = [
    "SearchService",
    "SearchUseCase",
]
