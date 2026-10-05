from ragpalette.core import RAGResult
from ragpalette.core.protocols import Generator, Retriever


class ClassicRAG:
    """Retrieve relevant context and generate an answer."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> RAGResult:
        # TODO 1: Retrieve results using query and self.top_k.
        results = self.retriever.retrieve(query, self.top_k)

        # TODO 2: Generate an answer using query and those results.
        answer = self.generator.generate(query, results)

        return RAGResult(answer=answer, context=results)
