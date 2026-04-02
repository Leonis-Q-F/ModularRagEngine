class RAGError(Exception):
    """RAG 组件基础异常。"""


class NamespaceNotFoundError(RAGError):
    """指定 namespace 不存在。"""


class ActiveIndexNotFoundError(RAGError):
    """当前 namespace 不存在可用激活索引。"""


class UnsupportedFileError(RAGError):
    """文件类型不受支持。"""
