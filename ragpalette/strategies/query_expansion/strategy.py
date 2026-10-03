from ragpalette.core import RAGResult


class QueryExpansionRAG:
    """Expand a query before retrieving context."""

    def run(self, query: str) -> RAGResult:
        raise NotImplementedError
