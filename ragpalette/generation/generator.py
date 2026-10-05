from collections.abc import Callable

from ragpalette.core.models import SearchResult
from ragpalette.generation.context import ContextBuilder


class LLMGenerator:
    """Generate an answer using retrieved context."""

    def __init__(self, llm: Callable[[str], str]) -> None:
        self.llm = llm
        self.context_builder = ContextBuilder()

    def generate(
        self,
        query: str,
        context: list[SearchResult],
    ) -> str:
        formatted_context = self.context_builder.build(context)
        prompt = (
            "Answer the following question using only the provided context. "
            "If the context is insufficient, please indicate that you cannot answer the question.\n\n"
            f"Context:\n{formatted_context}\n\n"
            f"Question: {query}"
        )
        answer = self.llm(prompt)
        return answer
