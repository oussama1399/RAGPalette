from ragpalette.core.models import SearchResult
from ragpalette.core.protocols import Embedder, VectorStore


class DenseRetriever:
    """Retrieve chunks using dense-vector similarity."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        query_embedding = self.embedder.embed_query(query)
        search_results = self.vector_store.search(query_embedding, top_k)

        return search_results