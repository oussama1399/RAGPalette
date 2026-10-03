from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Create embeddings using a Sentence Transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)
        

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode_document(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode_query(query, normalize_embeddings=True)
        return embedding.tolist()