from ragpalette.core import RAGResult


class RerankedRAG:
    """Rerank retrieved context before generating an answer."""

    def run(self, query: str) -> RAGResult:
        raise NotImplementedError
