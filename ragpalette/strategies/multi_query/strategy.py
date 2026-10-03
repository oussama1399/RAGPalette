from ragpalette.core import RAGResult


class MultiQueryRAG:
    """Retrieve context using multiple versions of a query."""

    def run(self, query: str) -> RAGResult:
        raise NotImplementedError
