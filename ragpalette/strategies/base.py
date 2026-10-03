from typing import Protocol, runtime_checkable

from ragpalette.core import RAGResult


@runtime_checkable
class RAGStrategy(Protocol):
    def run(self, query: str) -> RAGResult:
        ...
