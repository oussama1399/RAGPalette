from ragpalette.core import RAGResult
from ragpalette.core.protocols import Generator, Reranker, Retriever


class RerankedRAG:
    """Rerank retrieved context before generating an answer."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k.")

        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.candidate_k = candidate_k

    def run(self, query: str) -> RAGResult:
        results = self.retriever.retrieve(query, self.candidate_k)
        context = self.reranker.rerank(query, results)[:self.top_k] if results else []
        answer = self.generator.generate(query, context)
        return RAGResult(answer=answer, context=context)
