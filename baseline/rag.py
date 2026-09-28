"""Flat RAG baseline: FAISS recall -> rerank -> same LLM, same prompt. No graph."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from config import settings
from llm.prompts import ANSWER_SYSTEM, build_answer_prompt, build_context
from retrieval.embedder import Embedder, VectorIndex
from retrieval.pipeline import Answer, _context_char_budget, _pack_blocks
from retrieval.reranker import Reranker


def load_chunks(path: Path) -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(path.read_text())}


def baseline_answer(
    question: str, as_of: date | None, index: VectorIndex, chunks: dict[str, dict],
    embedder: Embedder, reranker: Reranker, llm, top_k: int | None = None,
) -> Answer:
    hits = index.search(question, embedder, k=30)
    ranked = reranker.top_k(question, [(i, t) for i, t, _ in hits],
                            k=top_k or settings.reranker_top_k)
    blocks = []
    for cid, text, _ in ranked:
        c = chunks[cid]
        citation = f"{c['document_name']}, near \u00a7{c['section_hint']}"
        blocks.append((citation, text, c["source"]))
    blocks = _pack_blocks(blocks, _context_char_budget())
    prompt = build_answer_prompt(
        question, build_context(blocks), as_of.isoformat() if as_of else None
    )
    text = llm.generate(ANSWER_SYSTEM, prompt)
    return Answer(text=text, citations=[c for c, _, _ in blocks],
                  context_ids=[cid for cid, _, _ in ranked], blocks=blocks)
