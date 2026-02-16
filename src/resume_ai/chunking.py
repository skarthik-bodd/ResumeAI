from __future__ import annotations

import re

from .types import Chunk, Document


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            paragraph_break = normalized.rfind("\n\n", start, end)
            if paragraph_break > start + chunk_size // 3:
                end = paragraph_break
            else:
                whitespace_break = normalized.rfind(" ", start, end)
                if whitespace_break > start + chunk_size // 3:
                    end = whitespace_break

        if end <= start:
            end = min(start + chunk_size, text_length)

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(0, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def build_chunks(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            chunk_id = f"{doc.source}::chunk::{i}"
            chunks.append(Chunk(chunk_id=chunk_id, source=doc.source, text=piece))

    return chunks
