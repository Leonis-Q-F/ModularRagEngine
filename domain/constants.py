from __future__ import annotations

from enum import Enum


DEFAULT_CHUNK_VERSION = "chunk-v1"
DEFAULT_SPARSE_PROVIDER = "simple_lexical"


class SupportedLanguage(str, Enum):
    ZH = "zh"
    EN = "en"


class RetrievalTextPolicy(str, Enum):
    CONTENT_ONLY = "content_only"
    HEADER_PATH_PLUS_CONTENT = "header_path_plus_content"


class IndexStatus(str, Enum):
    BUILDING = "building"
    READY = "ready"
