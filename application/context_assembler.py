"""旧路径兼容层：输出组装已经迁移到 api.presenters。"""

from ..api.presenters.context_presenter import ContextPresenter

ContextAssembler = ContextPresenter

__all__ = [
    "ContextAssembler",
    "ContextPresenter",
]
