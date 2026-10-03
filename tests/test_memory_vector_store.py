import pytest

from ragpalette.core.models import Chunk
from ragpalette.stores.memory import InMemoryVectorStore


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        text=f"Text for {chunk_id}",
        metadata={"source": "test.txt"},
    )


def test_search_orders_chunks_by_similarity() -> None:
    store = InMemoryVectorStore()

    chunks = [
        make_chunk("similar"),
        make_chunk("unrelated"),
        make_chunk("opposite"),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]

    store.add_chunks(chunks, embeddings)

    results = store.search(
        query_embedding=[1.0, 0.0],
        top_k=3,
    )

    assert [result.chunk.id for result in results] == [
        "similar",
        "unrelated",
        "opposite",
    ]

    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)
    assert results[2].score == pytest.approx(-1.0)


def test_search_respects_top_k() -> None:
    store = InMemoryVectorStore()

    store.add_chunks(
        chunks=[make_chunk("one"), make_chunk("two")],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.search(
        query_embedding=[1.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "one"


def test_add_chunks_rejects_different_list_lengths() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.add_chunks(
            chunks=[make_chunk("one"), make_chunk("two")],
            embeddings=[[1.0, 0.0]],
        )


def test_search_rejects_non_positive_top_k() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search(
            query_embedding=[1.0, 0.0],
            top_k=0,
        )


def test_search_rejects_different_vector_dimensions() -> None:
    store = InMemoryVectorStore()

    store.add_chunks(
        chunks=[make_chunk("one")],
        embeddings=[[1.0, 0.0, 0.5]],
    )

    with pytest.raises(ValueError):
        store.search(
            query_embedding=[1.0, 0.0],
            top_k=1,
        )


def test_zero_vector_has_zero_similarity() -> None:
    store = InMemoryVectorStore()

    store.add_chunks(
        chunks=[make_chunk("zero")],
        embeddings=[[0.0, 0.0]],
    )

    results = store.search(
        query_embedding=[1.0, 0.0],
        top_k=1,
    )

    assert results[0].score == pytest.approx(0.0)