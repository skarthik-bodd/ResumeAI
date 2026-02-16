from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .types import Chunk, RetrievalHit


class VectorStoreError(ValueError):
    """Raised for vector store related errors."""


class LocalVectorStore:
    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        self._encoder = SentenceTransformer(embedding_model)
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise VectorStoreError("Cannot build index from empty chunks.")

        texts = [c.text for c in chunks]
        embeddings = self._encoder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self._chunks = chunks
        self._matrix = embeddings.astype(np.float32)

    def save(self, index_path: str | Path) -> None:
        if self._matrix is None:
            raise VectorStoreError("No index to save. Build or load first.")

        target = Path(index_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(target, embeddings=self._matrix)
        metadata_path = _metadata_path(target)
        metadata = {
            "embedding_model": self.embedding_model,
            "chunks": [
                {"chunk_id": c.chunk_id, "source": c.source, "text": c.text}
                for c in self._chunks
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def load(self, index_path: str | Path) -> None:
        target = Path(index_path)
        embeddings_file = _npz_path(target)
        metadata_file = _metadata_path(target)

        if not embeddings_file.exists() or not metadata_file.exists():
            raise VectorStoreError(
                f"Index files not found for base path {target}. Expected {embeddings_file} and {metadata_file}."
            )

        data = np.load(embeddings_file)
        self._matrix = data["embeddings"].astype(np.float32)

        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        model_name = metadata.get("embedding_model")
        if model_name and model_name != self.embedding_model:
            raise VectorStoreError(
                "Embedding model mismatch. "
                f"Index model is `{model_name}` but runtime model is `{self.embedding_model}`."
            )

        self._chunks = [Chunk(**item) for item in metadata.get("chunks", [])]
        if len(self._chunks) != len(self._matrix):
            raise VectorStoreError("Chunk count does not match embedding count in loaded index.")

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        if self._matrix is None or not self._chunks:
            raise VectorStoreError("Index is empty. Build or load before searching.")
        if top_k <= 0:
            raise VectorStoreError("`top_k` must be greater than 0.")

        query_vector = self._encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        scores = self._matrix @ query_vector
        top_k = min(top_k, len(self._chunks))
        indices = np.argsort(scores)[-top_k:][::-1]

        return [
            RetrievalHit(chunk=self._chunks[i], score=float(scores[i]))
            for i in indices
        ]


def _metadata_path(base: Path) -> Path:
    return base.with_suffix(".json")


def _npz_path(base: Path) -> Path:
    return base.with_suffix(".npz")
