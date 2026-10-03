from ragpalette.core import RAGResult


class ClassicRAG:
    """Retrieve relevant context and generate an answer."""

    def run(self, query: str) -> RAGResult:
        raise NotImplementedError
