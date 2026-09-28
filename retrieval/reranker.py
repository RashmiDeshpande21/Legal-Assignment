"""CrossEncoder reranker, lazy-loaded."""
from __future__ import annotations


class Reranker:
    def __init__(self, model: str, batch_size: int = 32) -> None:
        self._name = model
        self._batch = batch_size
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from config import settings
            self._model = CrossEncoder(
                self._name, trust_remote_code=True, device=settings.embed_device
            )
        return self._model

    def top_k(
        self, query: str, docs: list[tuple[str, str]], k: int = 10
    ) -> list[tuple[str, str, float]]:
        """docs: (id, text). Returns top-k (id, text, score) by cross-encoder score."""
        if not docs:
            return []
        scores = self._get().predict(
            [(query, t) for _, t in docs], batch_size=self._batch, show_progress_bar=False
        )
        ranked = sorted(zip(docs, scores, strict=True), key=lambda x: -float(x[1]))
        return [(d[0], d[1], float(s)) for d, s in ranked[:k]]
