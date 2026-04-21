"""旧路径兼容层：namespace 解析已经迁移到 application.services。"""

from .services.namespace_resolution_service import NamespaceResolutionService

NamespaceResolver = NamespaceResolutionService

__all__ = [
    "NamespaceResolver",
    "NamespaceResolutionService",
]
