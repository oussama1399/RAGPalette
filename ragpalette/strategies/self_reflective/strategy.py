from ragpalette.core import RAGResult


class SelfReflectiveRAG:
    """Evaluate retrieved context and refine the answer."""

    def run(self, query: str) -> RAGResult:
        raise NotImplementedError
