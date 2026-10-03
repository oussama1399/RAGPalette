from ragpalette.core import RAGResult


class GraphRAG:
    """Use graph relationships to retrieve context for an answer."""

    def run(self, query: str) -> RAGResult:
        raise NotImplementedError
