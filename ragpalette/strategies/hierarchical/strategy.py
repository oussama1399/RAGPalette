from ragpalette.core import RAGResult


class HierarchicalRAG:
    """Retrieve context across multiple levels of detail."""

    def run(self, query: str) -> RAGResult:
        raise NotImplementedError
