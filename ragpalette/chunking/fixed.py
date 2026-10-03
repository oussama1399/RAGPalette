
from ragpalette.core.models import Chunk, Document


class FixedSizeChunker:
    """Split documents into fixed-size overlapping chunks."""

    def __init__(self, chunk_size: int, overlap: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_number = 0

        while start < len(document.text):
            end = min(start + self.chunk_size, len(document.text))
            chunk_text = document.text[start:end]

            metadata = dict(document.metadata)
            metadata.update(
                {
                    "start": start,
                    "end": end,
                    "chunk_number": chunk_number,
                }
            )

            chunks.append(
                Chunk(
                    id=f"{document.id}:chunk:{chunk_number}",
                    document_id=document.id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

            # The final chunk has reached the end of the document.
            if end == len(document.text):
                break

            start = end - self.overlap
            chunk_number += 1

        return chunks