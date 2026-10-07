from collections.abc import Callable

from ragpalette.core.models import SearchResult


class ScoringReranker:
    """Rerank using a query/text scoring function; higher scores rank first."""

    def __init__(self, scorer: Callable[[str, str], float]) -> None:
        self.scorer = scorer

    def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        rescored = [
            SearchResult(
                chunk=result.chunk,
                score=self.scorer(query, result.chunk.text),
                source=result.source,
            )
            for result in results
        ]
        return sorted(rescored, key=lambda result: result.score, reverse=True)
