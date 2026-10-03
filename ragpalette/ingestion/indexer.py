from ragpalette.core.models import Chunk, Document
from ragpalette.core.protocols import Chunker, Embedder, VectorStore

class Indexer:
    """Index documents by splitting them into chunks, embedding them, and storing them in a vector store."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index_document(self, document: Document) -> list[Chunk]:
        chunks = self.chunker.split(document)

        texts = [chunk.text for chunk in chunks]
        if not texts:
            return []
        embeddings = self.embedder.embed_documents(texts)
        self.vector_store.add_chunks(chunks, embeddings)
        
        return chunks