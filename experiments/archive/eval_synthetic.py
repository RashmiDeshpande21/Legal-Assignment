"""Evaluate Graph vs Baseline retrieval across the 36 Opus-generated synthetic questions.

Computes Tier 1 Context Recall and Context Precision deterministically against gold node IDs.
Writes outputs/synthetic_eval_results.json and prints summary table.
Run: uv run python -m eval.eval_synthetic
"""
from __future__ import annotations

import json
from pathlib import Path

from baseline.rag import load_chunks
from config import settings
from eval.harness import _baseline_units, _gold_hit_baseline, _gold_hit_graph
from graph.intent import find_seeds
from graph.loader import load_graph
from graph.traversal import dfs_context
from retrieval.embedder import Embedder, VectorIndex
from retrieval.reranker import Reranker


def main() -> None:
    print("Loading Graph, Chunks, Embedder, and Reranker...")
    G = load_graph(settings.artifacts_dir / "graph.json")
    chunks = load_chunks(settings.artifacts_dir / "chunks_baseline.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)

    questions = json.loads(Path("eval/synthetic_eval_set.json").read_text())
    print(f"Evaluating {len(questions)} synthetic questions...")

    graph_recalls, graph_precs = [], []
    base_recalls, base_precs = [], []

    per_q = []
    for q in questions:
        golds = q["gold"]
        q_text = q["question"]
        as_of = None

        # --- Graph Path Retrieval (mirrors retrieval.pipeline.graph_answer:
        # hybrid seeding, same DFS params, seed force-inclusion) ---
        seeds = find_seeds(G, q_text, index=index, embedder=embedder)
        effs = dfs_context(G, seeds, as_of, max_depth=2, max_nodes=40)
        # Rerank Graph
        docs_g = [(e.node_id, e.text) for e in effs]
        ranked_g = reranker.top_k(q_text, docs_g, k=settings.reranker_top_k)
        g_cids = [cid for cid, _, _ in ranked_g]
        for s in seeds[:3]:
            if s not in g_cids and G.nodes[s]["node_type"] in ("Section", "Definition"):
                g_cids.insert(0, s)

        # Graph Metrics
        g_hits = sum(1 for g in golds if _gold_hit_graph(g, g_cids))
        g_recall = g_hits / len(golds) if golds else 1.0
        g_prec = g_hits / len(g_cids) if g_cids else 0.0
        graph_recalls.append(g_recall)
        graph_precs.append(g_prec)

        # --- Baseline Path Retrieval ---
        raw_hits = index.search(q_text, embedder, k=30)
        c_docs = [(cid, text) for cid, text, _ in raw_hits if cid in chunks]
        ranked_b = reranker.top_k(q_text, c_docs, k=settings.reranker_top_k)
        b_cids = [cid for cid, _, _ in ranked_b]
        b_units = _baseline_units(b_cids, chunks)

        # Baseline Metrics
        b_hits = sum(1 for g in golds if _gold_hit_baseline(g, b_units))
        b_recall = b_hits / len(golds) if golds else 1.0
        b_prec = b_hits / len(b_cids) if b_cids else 0.0
        base_recalls.append(b_recall)
        base_precs.append(b_prec)

        per_q.append({
            "id": q["id"],
            "category": q["category"],
            "question": q_text,
            "gold": golds,
            "graph": {
                "recall": round(g_recall, 3),
                "precision": round(g_prec, 3),
                "cids": g_cids[:3],
            },
            "baseline": {
                "recall": round(b_recall, 3),
                "precision": round(b_prec, 3),
                "cids": b_cids[:3],
            },
        })

    summary = {
        "n_questions": len(questions),
        "graph": {
            "mean_context_recall": round(sum(graph_recalls) / len(graph_recalls), 3),
            "mean_context_precision": round(sum(graph_precs) / len(graph_precs), 3),
        },
        "baseline": {
            "mean_context_recall": round(sum(base_recalls) / len(base_recalls), 3),
            "mean_context_precision": round(sum(base_precs) / len(base_precs), 3),
        },
    }

    out_file = Path("outputs/synthetic_eval_results.json")
    out_file.write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=2))

    g_rec = summary['graph']['mean_context_recall']
    b_rec = summary['baseline']['mean_context_recall']
    g_pr = summary['graph']['mean_context_precision']
    b_pr = summary['baseline']['mean_context_precision']

    print("\n" + "=" * 60)
    print(f"SYNTHETIC EVALUATION RESULTS ({len(questions)} UNSEEN QUESTIONS)")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Graph':<12} | {'Baseline':<12}")
    print("-" * 60)
    print(f"{'Context Recall (mean)':<30} | {g_rec:<12} | {b_rec:<12}")
    print(f"{'Context Precision (mean)':<30} | {g_pr:<12} | {b_pr:<12}")
    print("-" * 60)
    print(f"Wrote full results to {out_file}")


if __name__ == "__main__":
    main()
