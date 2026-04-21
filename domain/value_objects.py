"""旧路径兼容层：值对象已经迁移到 application.contracts。"""

from ..application.contracts import (
    ChunkBundle,
    NamespaceReference,
    ParsedDocument,
    ResolvedNamespace,
    SearchFilters,
    VectorHit,
    VectorRecord,
    sha256_text,
)

__all__ = [
    "ChunkBundle",
    "NamespaceReference",
    "ParsedDocument",
    "ResolvedNamespace",
    "SearchFilters",
    "VectorHit",
    "VectorRecord",
    "sha256_text",
]
