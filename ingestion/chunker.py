"""Flat chunks for the RAG baseline. Three strategies for the chunking experiment."""
from __future__ import annotations

from dataclasses import dataclass

from graph.schema import Document, Section


@dataclass
class Chunk:
    id: str
    text: str
    document_id: str
    document_name: str
    section_hint: str           # nearest section number, for citation attribution
    source: str                 # "html" | "ocr"


def _windows(words: list[str], size: int, overlap: int) -> list[tuple[int, int]]:
    step = size - overlap
    return [(i, min(i + size, len(words))) for i in range(0, max(len(words) - overlap, 1), step)]


def chunk_fixed(
    sections: list[Section], doc: Document, size: int = 512, overlap: int = 256
) -> list[Chunk]:
    """Fixed-size word windows over the document stream, section hints carried along."""
    stream: list[tuple[str, str, str]] = []  # (word, section_number, source)
    for sec in sections:
        stream.extend((w, sec.number, sec.source) for w in sec.text.split())
    words = [w for w, _, _ in stream]
    return [
        Chunk(
            id=f"{doc.id}::chunk{n}", text=" ".join(words[a:b]), document_id=doc.id,
            document_name=doc.name, section_hint=stream[a][1], source=stream[a][2],
        )
        for n, (a, b) in enumerate(_windows(words, size, overlap))
    ]


def chunk_sections(sections: list[Section], doc: Document, max_words: int = 512) -> list[Chunk]:
    """Section-aware: one chunk per section/subsection, long ones split."""
    chunks = []
    for sec in sections:
        words = sec.text.split()
        if not words:
            continue
        parts = _windows(words, max_words, 0) if len(words) > max_words else [(0, len(words))]
        for n, (a, b) in enumerate(parts):
            suffix = f"::p{n}" if len(parts) > 1 else ""
            chunks.append(Chunk(
                id=f"{sec.id}{suffix}", text=" ".join(words[a:b]), document_id=doc.id,
                document_name=doc.name, section_hint=sec.number, source=sec.source,
            ))
    return chunks
