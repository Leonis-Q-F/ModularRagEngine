from __future__ import annotations

from ...domain.entities import Namespace
from ...domain.exceptions import NamespaceConflictError
from ..contracts import NamespaceReference, ResolvedNamespace
from ..ports import DocumentStorePort


class NamespaceResolutionService:
    """统一处理 namespace 的创建、查询与一致性校验。"""

    def __init__(self, document_store: DocumentStorePort) -> None:
        """注入 namespace 解析所需的存储端口。"""
        self._document_store = document_store

    def resolve_for_ingest(self, reference: NamespaceReference) -> ResolvedNamespace:
        """入库路径允许用 key 创建 namespace，但 id 只能指向已存在 namespace。"""
        if reference.namespace_id is not None:
            namespace = self._document_store.get_namespace(namespace_id=reference.namespace_id)
            self._ensure_namespace_match(namespace, reference)
            return ResolvedNamespace(
                namespace_id=namespace.namespace_id,
                namespace_key=namespace.namespace_key,
            )

        namespace = self._document_store.ensure_namespace(namespace_key=reference.namespace_key or "")
        return ResolvedNamespace(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
        )

    def resolve_existing(self, reference: NamespaceReference) -> ResolvedNamespace:
        """查询与索引路径只允许解析既有 namespace。"""
        namespace = self._document_store.get_namespace(
            namespace_id=reference.namespace_id,
            namespace_key=reference.namespace_key,
        )
        self._ensure_namespace_match(namespace, reference)
        return ResolvedNamespace(
            namespace_id=namespace.namespace_id,
            namespace_key=namespace.namespace_key,
        )

    def _ensure_namespace_match(self, namespace: Namespace, reference: NamespaceReference) -> None:
        """当请求同时携带 id 与 key 时，校验它们指向同一个 namespace。"""
        if reference.namespace_key is None:
            return
        if namespace.namespace_key != reference.namespace_key:
            raise NamespaceConflictError("namespace_id 与 namespace_key 不匹配。")
