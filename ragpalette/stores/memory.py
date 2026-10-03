import math

from ragpalette.core.models import Chunk, SearchResult


class InMemoryVectorStore:
    """Store chunks and embeddings in memory."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )
            
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        results: list[SearchResult] = []

        for chunk, embedding in zip(self._chunks, self._embeddings):
            score = self._cosine_similarity(query_embedding, embedding)

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    source=str(
                        chunk.metadata.get("source", chunk.document_id)
                    ),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(first: list[float], second: list[float]) -> float:
        """Calculate the cosine similarity between two vectors."""
        if len(first) != len(second):
            raise ValueError("Vectors must be of the same length.")

        dot_product = sum(a * b for a, b in zip(first, second))
        first_norm = math.sqrt(sum(a * a for a in first))
        second_norm = math.sqrt(sum(b * b for b in second))

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return dot_product / (first_norm * second_norm)
