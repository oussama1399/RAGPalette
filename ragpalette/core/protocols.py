"""Core protocols."""
from typing import Protocol
from ragpalette.core.models import Chunk, Document, SearchResult


class Chunker(Protocol):
    
    
    def split(self, document: Document) -> list[Chunk]:
        
        """Split the document into smaller chunks."""
        
        ...
        
    
class Embedder(Protocol):
    
    def embed_documents(self, texts:list[str]) -> list[list[float]]:
        
        """Embed the chunk into a vector."""
        
        ...
        
    
    def embed_query(self, query:str) -> list[float]:
        
        """Embed the query into a vector."""
        
        ...
        


class VectorStore(Protocol):
    
    def add_chunks(self, chunks:list[Chunk],embeddings:list[list[float]]) -> None:
        
        """Add chunks to the vector store."""
        
        ...
        


    def search(self, query_embedding:list[float], top_k:int) -> list[SearchResult]:
        
        """Search for similar chunks in the vector store."""
        
        ...
        
    

class Retriever(Protocol):
    
    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        
        """Retrieve relevant chunks from the vector store."""
        
        ...
        
    
class Reranker(Protocol):
    def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        """Return results ordered from most to least relevant to the query."""
        ...


class Generator(Protocol):
    
    def generate(self, query: str, context: list[SearchResult]) -> str:
        
        """Generate a response based on the query and context."""
        
        ...
