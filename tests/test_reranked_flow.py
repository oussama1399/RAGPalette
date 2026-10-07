import pytest

from ragpalette.core.models import Chunk, SearchResult
from ragpalette.generation.generator import LLMGenerator
from ragpalette.reranking import ScoringReranker
from ragpalette.strategies.reranked import RerankedRAG


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_reranked_flow_selects_context_before_generation() -> None:
    results = [
        SearchResult(Chunk("1", "general", "Manage your account."), 0.9, "general"),
        SearchResult(Chunk("2", "reset", "Reset your password."), 0.7, "reset"),
        SearchResult(Chunk("3", "refund", "Request a refund."), 0.5, "refund"),
    ]
    retriever = FakeRetriever(results)
    query = "How do I reset my password?"
    scored_pairs: list[tuple[str, str]] = []
    prompts: list[str] = []

    def score(question: str, text: str) -> float:
        scored_pairs.append((question, text))
        return 1.0 if "password" in text else 0.0

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Reset your password."

    strategy = RerankedRAG(
        retriever=retriever,
        reranker=ScoringReranker(score),
        generator=LLMGenerator(llm),
        top_k=1,
        candidate_k=3,
    )
    result = strategy.run(query)

    assert retriever.calls == [(query, 3)]
    assert scored_pairs == [(query, item.chunk.text) for item in results]
    assert result.answer == "Reset your password."
    assert result.context == [SearchResult(results[1].chunk, 1.0, "reset")]
    assert len(prompts) == 1
    assert f"Question: {query}" in prompts[0]
    assert "Source 1: reset\nReset your password." in prompts[0]
    assert results[0].chunk.text not in prompts[0]
    assert results[2].chunk.text not in prompts[0]
    assert [item.score for item in results] == [0.9, 0.7, 0.5]


def test_empty_retrieval_still_generates_with_empty_context() -> None:
    def unexpected_score(query: str, text: str) -> float:
        pytest.fail("Empty retrieval should skip reranking.")

    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Insufficient context."

    strategy = RerankedRAG(
        retriever=FakeRetriever([]),
        reranker=ScoringReranker(unexpected_score),
        generator=LLMGenerator(llm),
    )
    result = strategy.run("Where is my order?")

    assert result.answer == "Insufficient context."
    assert result.context == []
    assert len(prompts) == 1
    assert "Context:\n\n\nQuestion: Where is my order?" in prompts[0]


def test_fewer_candidates_and_equal_scores_preserve_retrieval_order() -> None:
    results = [
        SearchResult(Chunk("1", "first", "First source."), 0.9, "first"),
        SearchResult(Chunk("2", "second", "Second source."), 0.7, "second"),
    ]
    strategy = RerankedRAG(
        retriever=FakeRetriever(results),
        reranker=ScoringReranker(lambda query, text: 1.0),
        generator=LLMGenerator(lambda prompt: "Answer."),
    )

    result = strategy.run("Question?")

    assert [item.source for item in result.context] == ["first", "second"]
    assert [item.score for item in result.context] == [1.0, 1.0]


@pytest.mark.parametrize("top_k,candidate_k", [(0, 20), (-1, 20), (5, 0), (5, 4)])
def test_invalid_candidate_limits(top_k: int, candidate_k: int) -> None:
    with pytest.raises(ValueError):
        RerankedRAG(
            retriever=FakeRetriever([]),
            reranker=ScoringReranker(lambda query, text: 1.0),
            generator=LLMGenerator(lambda prompt: "Answer."),
            top_k=top_k,
            candidate_k=candidate_k,
        )
