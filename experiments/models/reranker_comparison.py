"""Reranker comparison: rerank the winning embedder's top-30 per question.

Metrics: NDCG@10, MRR@10, latency per query.
Run: CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.models.reranker_comparison
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

CANDIDATES = [
    "lxyuan/LegalBenchRAG-Ettin-150M-Reranker",   # legal-specific
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "BAAI/bge-reranker-large",
]


def ndcg_at_10(ranked: list[str], gold: set[str]) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, n in enumerate(ranked[:10]) if n in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / ideal if ideal else 0.0


def main() -> None:
    import torch
    from sentence_transformers import CrossEncoder

    from graph.loader import load_graph
    G = load_graph(Path("artifacts/graph.json"))
    questions = json.loads(Path("experiments/mini_eval_set.json").read_text())
    emb = json.loads(Path("experiments/models/results/embedder_results.json").read_text())
    ok = {m: r for m, r in emb.items() if "recall@10" in r}
    winner = max(ok, key=lambda m: (ok[m]["recall@10"], ok[m]["mrr@10"]))
    rankings = json.loads(
        Path(f"experiments/models/results/rankings_{winner.split('/')[-1]}.json").read_text())
    print("candidate pool from embedder winner:", winner)

    results = {}
    for model in CANDIDATES:
        try:
            m = CrossEncoder(model, trust_remote_code=True)
            ndcgs, mrrs, times = [], [], []
            for q in questions:
                gold = set(q["gold"])
                if not gold:
                    continue
                pool = rankings[q["id"]][:30]
                pairs = [(q["text"], G.nodes[n].get("text", "")[:2000]) for n in pool]
                t0 = time.time()
                scores = m.predict(pairs, show_progress_bar=False)
                times.append(time.time() - t0)
                ranked = [n for n, _ in sorted(zip(pool, scores, strict=True),
                                               key=lambda x: -float(x[1]))]
                ndcgs.append(ndcg_at_10(ranked, gold))
                mrrs.append(next((1 / (i + 1) for i, n in enumerate(ranked[:10]) if n in gold), 0))
            results[model] = {
                "ndcg@10": round(sum(ndcgs) / len(ndcgs), 3),
                "mrr@10": round(sum(mrrs) / len(mrrs), 3),
                "latency_s": round(sum(times) / len(times), 3),
            }
            print(f"{model}: {results[model]}")
            del m
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 — record and continue
            results[model] = {"error": str(exc)[:300]}
            print(f"{model}: FAILED {exc}")
    Path("experiments/models/results/reranker_results.json").write_text(
        json.dumps(results, indent=1))
    ok = {m: r for m, r in results.items() if "ndcg@10" in r}
    if ok:
        print("WINNER:", max(ok, key=lambda m: ok[m]["ndcg@10"]))


if __name__ == "__main__":
    main()
