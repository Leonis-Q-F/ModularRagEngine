from __future__ import annotations

from ...application.dto import ContextBlock


class ContextPresenter:
    """把父块结果组装为稳定的 LLM 上下文文本。"""

    def build(self, contexts: list[ContextBlock]) -> str:
        """把上下文块拼接为稳定的 LLM 输入文本。"""
        parts: list[str] = []
        for context in contexts:
            parts.append(
                "\n".join(
                    [
                        f"[文档] {context.file_name}",
                        f"[文档ID] {context.doc_id}",
                        f"[父块ID] {context.parent_id}",
                        f"[相关性] {context.score:.4f}",
                        "[内容]",
                        context.content,
                    ]
                )
            )
        return "\n\n".join(parts)
