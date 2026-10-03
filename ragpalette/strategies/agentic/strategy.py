from ragpalette.core import RAGResult


class AgenticRAG:
    """Plan and execute retrieval steps to answer a query."""

    def run(self, query: str) -> RAGResult:
        raise NotImplementedError
