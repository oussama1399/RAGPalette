# ragpalette/core/models.py

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    source: str


@dataclass
class RAGResult:
    answer: str
    context: list[SearchResult] = field(default_factory=list)