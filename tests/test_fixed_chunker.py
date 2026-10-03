from ragpalette.core.models import Chunk, Document
from ragpalette.chunking.fixed import FixedSizeChunker  

document = Document(
    id="doc1",
    text="This is a sample document that will be split into fixed-size chunks.",
    metadata={"author": "John Doe", "date": "2023-10-01"},
)

chunker = FixedSizeChunker(chunk_size=20, overlap=5)
chunks = chunker.split(document)

print(f"Total chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(f"  ID: {chunk.id}")
    print(f"  Document ID: {chunk.document_id}")
    print(f"  Text: {chunk.text}")
    print(f"  Metadata: {chunk.metadata}")
    print() 
    