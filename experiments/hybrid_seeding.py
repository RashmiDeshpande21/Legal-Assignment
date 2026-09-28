"""Experiment: keyword-only vs hybrid (keyword + dense) graph seeding.

Measures post-rerank context recall/precision of the graph retrieval path on both
eval sets (12 assignment questions + 36 Opus-generated synthetic questions), with
and without dense FAISS entry points feeding the DFS. No LLM involved.
Writes experiments/results/hybrid_seeding_results.json.
Run: CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.hybrid_seeding
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from config import settings
from eval.harness import _gold_hit_graph, _hier_match
from graph.intent import find_seeds
from graph.loader import load_graph
from graph.traversal import dfs_context
from retrieval.embedder import Embedder, VectorIndex
from retrieval.reranker import Reranker


def _retrieve(G, question, as_of, reranker, index=None, embedder=None) -> list[str]:
    """Mirror retrieval in retrieval.pipeline.graph_answer (same params, no LLM)."""
    seeds = find_seeds(G, question, index=index, embedder=embedder)
    effs = dfs_context(G, seeds, as_of, max_depth=2, max_nodes=40)
    ranked = reranker.top_k(
        question, [(e.node_id, e.text) for e in effs if e.text],
        k=settings.reranker_top_k,
    )
    cids = [nid for nid, _, _ in ranked]
    # seed force-inclusion, as the pipeline does
    for s in seeds[:3]:
        if s not in cids and G.nodes[s]["node_type"] in ("Section", "Definition"):
            cids.insert(0, s)
    return cids


def _score(questions: list[dict], G, reranker, index, embedder, mode: str) -> dict:
    recalls, precisions, per_q = [], [], {}
    dense = (index, embedder) if mode == "hybrid" else (None, None)
    for q in questions:
        golds = q["gold"]
        if not golds:
            continue
        text = q.get("retrieval_text") or q.get("question") or q["text"]
        cids = _retrieve(G, text, None, reranker, index=dense[0], embedder=dense[1])
        hits = [g for g in golds if _gold_hit_graph(g, cids)]
        relevant = [c for c in cids if any(_hier_match(g, c) for g in golds)]
        recall = len(hits) / len(golds)
        prec = len(relevant) / len(cids) if cids else 0.0
        recalls.append(recall)
        precisions.append(prec)
        per_q[q["id"]] = {"recall": round(recall, 3), "precision": round(prec, 3)}
    return {
        "mean_recall": round(statistics.mean(recalls), 3),
        "mean_precision": round(statistics.mean(precisions), 3),
        "per_question": per_q,
    }


def main() -> None:
    G = load_graph(settings.artifacts_dir / "graph.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)

    assignment = json.loads(Path("experiments/mini_eval_set.json").read_text())
    synthetic = json.loads(
        Path("experiments/archive/synthetic_eval_set.json").read_text()
    )

    results = {}
    for set_name, questions in (("assignment_12", assignment), ("synthetic_36", synthetic)):
        for mode in ("keyword_only", "hybrid"):
            r = _score(questions, G, reranker, index, embedder, mode)
            results[f"{set_name}::{mode}"] = r
            print(f"{set_name:>14} | {mode:<12} | recall {r['mean_recall']:.3f} "
                  f"| precision {r['mean_precision']:.3f}")

    out = Path("experiments/results/hybrid_seeding_results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
