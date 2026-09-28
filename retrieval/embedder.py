"""SentenceTransformer embedder + FAISS index, lazy-loaded."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model: str) -> None:
        self._name = model
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from config import settings
            self._model = SentenceTransformer(
                self._name, trust_remote_code=True, device=settings.embed_device
            )
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._get().encode(texts, normalize_embeddings=True, show_progress_bar=False)


class VectorIndex:
    """Flat inner-product FAISS index over chunk texts."""

    def __init__(self, ids: list[str], texts: list[str], index) -> None:
        self.ids = ids
        self.texts = texts
        self._index = index

    @classmethod
    def build(cls, ids: list[str], texts: list[str], embedder: Embedder) -> VectorIndex:
        import faiss
        vecs = embedder.encode(texts).astype(np.float32)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(ids, texts, index)

    def save(self, dir_: Path) -> None:
        import faiss
        dir_.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_ / "chunks.faiss"))
        (dir_ / "chunks_meta.json").write_text(
            json.dumps({"ids": self.ids, "texts": self.texts}, ensure_ascii=False)
        )

    @classmethod
    def load(cls, dir_: Path) -> VectorIndex:
        import faiss
        meta = json.loads((dir_ / "chunks_meta.json").read_text())
        return cls(meta["ids"], meta["texts"], faiss.read_index(str(dir_ / "chunks.faiss")))

    def search(self, query: str, embedder: Embedder, k: int = 30) -> list[tuple[str, str, float]]:
        qv = embedder.encode([query]).astype(np.float32)
        scores, idxs = self._index.search(qv, k)
        return [
            (self.ids[i], self.texts[i], float(s))
            for i, s in zip(idxs[0], scores[0], strict=False) if i >= 0
        ]
