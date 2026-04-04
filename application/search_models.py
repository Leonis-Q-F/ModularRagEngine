from __future__ import annotations

from dataclasses import dataclass

from ..domain.entities import IndexEntry


@dataclass(slots=True)
class RankedEntry:
    """表示完成召回与重排后的稳定中间结果。"""

    entry: IndexEntry
    score: float
    recall_score: float
