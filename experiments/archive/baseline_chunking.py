"""Chunking sweep for the flat RAG baseline: fixed 512/256, fixed 256/64, section-aware.

Metric: context recall@10 (after reranking, i.e. what the LLM actually sees) against the
gold node IDs in mini_eval_set.json, using the winner embedder + reranker. Writes
experiments/results/chunking_results.json.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from config import settings
from eval.harness import _gold_hit_baseline
from ingestion.chunker import chunk_fixed, chunk_sections
from ingestion.parser import parse_all
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker

STRATEGIES = {
    "fixed_512_overlap_256": lambda secs, doc: chunk_fixed(secs, doc, 512, 256),
    "fixed_256_overlap_64": lambda secs, doc: chunk_fixed(secs, doc, 256, 64),
    "section_aware_512": lambda secs, doc: chunk_sections(secs, doc, 512),
}


def main() -> None:
    import faiss
    import numpy as np

    parsed, docs = parse_all(settings.data_dir)
    eval_set = json.loads(Path("experiments/mini_eval_set.json").read_text())
    eval_set = [e for e in eval_set if e["gold"]]
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)

    results = {}
    for name, fn in STRATEGIES.items():
        chunks = [c for doc_id, secs in parsed.items() for c in fn(secs, docs[doc_id])]
        t0 = time.time()
        vecs = embedder.encode([c.text for c in chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(np.asarray(vecs, dtype="float32"))
        t_index = time.time() - t0

        per_q, recalls = {}, []
        for e in eval_set:
            qv = np.asarray(embedder.encode([e["text"]]), dtype="float32")
            _, idx = index.search(qv, 30)
            cands = [(chunks[i].id, chunks[i].text) for i in idx[0]]
            top = reranker.top_k(e["text"], cands, k=10)
            units = [asdict(chunks[[c.id for c in chunks].index(cid)]) for cid, _, _ in top]
            hits = [g for g in e["gold"] if _gold_hit_baseline(g, units)]
            recall = len(hits) / len(e["gold"])
            per_q[e["id"]] = round(recall, 3)
            recalls.append(recall)

        results[name] = {
            "n_chunks": len(chunks),
            "index_build_s": round(t_index, 1),
            "mean_context_recall_at_10": round(statistics.mean(recalls), 3),
            "per_question_recall": per_q,
        }
        print(f"{name}: n={len(chunks)} recall@10={results[name]['mean_context_recall_at_10']}",
              flush=True)

    winner = max(results, key=lambda k: results[k]["mean_context_recall_at_10"])
    out = {"metric": "context recall@10 post-rerank vs mini_eval_set gold",
           "embedder": settings.embedder_model, "reranker": settings.reranker_model,
           "winner": winner, "results": results}
    outp = Path("experiments/results/chunking_results.json")
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(out, indent=1))
    print(f"winner: {winner} -> {outp}")


if __name__ == "__main__":
    main()
