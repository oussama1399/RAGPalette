from ragpalette.chunking.fixed import FixedSizeChunker
from ragpalette.core.models import Document
from ragpalette.ingestion.indexer import Indexer
from ragpalette.retrieval.dense import DenseRetriever
from ragpalette.stores.memory import InMemoryVectorStore


class FakeEmbedder:
    """Provide predictable embeddings without downloading a model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    @staticmethod
    def _embed(text: str) -> list[float]:
        text = text.lower()

        if "password" in text:
            return [1.0, 0.0, 0.0]

        if "refund" in text:
            return [0.0, 1.0, 0.0]

        if "delivery" in text:
            return [0.0, 0.0, 1.0]

        return [0.0, 0.0, 0.0]


def test_complete_retrieval_flow() -> None:
    documents = [
        Document(
            id="password-doc",
            text="Reset a forgotten password using the password recovery page.",
        ),
        Document(
            id="refund-doc",
            text="Request a refund from the billing page.",
        ),
        Document(
            id="delivery-doc",
            text="Track your delivery using the tracking number.",
        ),
    ]

    chunker = FixedSizeChunker(chunk_size=500)
    embedder = FakeEmbedder()
    vector_store = InMemoryVectorStore()

    indexer = Indexer(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    for document in documents:
        indexer.index_document(document)

    results = retriever.retrieve(
        query="How can I recover my forgotten password?",
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].chunk.document_id == "password-doc"
    assert results[0].score == 1.0