from __future__ import annotations

import re

from ..domain.entities import IndexEntry


class SimpleReranker:
    """基于词面重叠的轻量 reranker。"""

    provider_name = "builtin"
    model_name = "lexical-reranker-v1"

    def rerank(self, query: str, entries: list[IndexEntry], top_k: int) -> list[tuple[IndexEntry, float]]:
        query_terms = set(self._tokenize(query))
        scored: list[tuple[IndexEntry, float]] = []
        for entry in entries:
            terms = set(self._tokenize(entry.retrieval_text))
            overlap = len(query_terms & terms)
            score = overlap / max(len(query_terms), 1)
            scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text)
