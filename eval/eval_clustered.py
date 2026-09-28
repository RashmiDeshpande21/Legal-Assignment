"""Tier-1 retrieval eval on the cluster-driven QnA set.

Headline categories call the live ``graph_answer`` path (LLM stubbed).
multi_hop_cluster is a contrast category (semantic mates, no required edge):
excluded from headline Δ; scored structure-only (ranked DFS, no force-include).

Run: make eval-clustered
Requires: eval/clustered_eval_set.json from `make generate-clustered`
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from baseline.rag import load_chunks
from config import settings
from eval.harness import _baseline_units, _gold_hit_baseline, _gold_hit_graph
from graph.intent import find_seeds
from graph.loader import load_graph
from graph.traversal import dfs_context
from retrieval.embedder import Embedder, VectorIndex
from retrieval.pipeline import graph_answer
from retrieval.reranker import Reranker

SET_PATH = Path("eval/clustered_eval_set.json")
OUT_PATH = Path("outputs/clustered_eval_results.json")

# Semantic-neighbor Qs: no edge to exploit — contrast category, not in headline.
CONTRAST_CATEGORIES = {"multi_hop_cluster"}


class _NullLlm:
    def generate(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        return ""


def _graph_retrieve(
    G, q_text: str, as_of, index, embedder, reranker, cat: str,
) -> list[str]:
    """Retrieval context ids.

    Headline categories use the live ``graph_answer`` path (LLM stubbed) so this
    score cannot drift from shipping. multi_hop_cluster is a *structural*
    contrast: ranked DFS only, no force-include — edges alone should not recover
    semantic-only gold.
    """
    if cat in CONTRAST_CATEGORIES:
        seeds = find_seeds(G, q_text, index=index, embedder=embedder)
        effs = dfs_context(G, seeds, as_of, max_depth=2, max_nodes=40)
        docs_g = [(e.node_id, e.text) for e in effs if e.text]
        ranked = reranker.top_k(q_text, docs_g, k=settings.reranker_top_k)
        return [cid for cid, _, _ in ranked]

    ans = graph_answer(
        G, q_text, as_of, reranker, _NullLlm(), index=index, embedder=embedder,
    )
    return ans.context_ids


def _baseline_retrieve(q_text, chunks, index, embedder, reranker) -> list[str]:
    raw_hits = index.search(q_text, embedder, k=30)
    c_docs = [(cid, text) for cid, text, _ in raw_hits if cid in chunks]
    ranked_b = reranker.top_k(q_text, c_docs, k=settings.reranker_top_k)
    return [cid for cid, _, _ in ranked_b]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    if not SET_PATH.exists():
        raise SystemExit(
            f"Missing {SET_PATH}. Generate with: uv run python -m eval.generate_clustered_qa"
        )

    print("Loading Graph, Chunks, Embedder, Reranker...")
    G = load_graph(settings.artifacts_dir / "graph.json")
    chunks = load_chunks(settings.artifacts_dir / "chunks_baseline.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)

    questions = json.loads(SET_PATH.read_text())
    print(f"Evaluating {len(questions)} clustered questions (retrieval only)...")

    by_cat: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"g_r": [], "g_p": [], "b_r": [], "b_p": []}
    )
    per_q = []

    for q in questions:
        golds = q.get("gold") or []
        q_text = q["question"]
        cat = q.get("category", "unknown")
        as_of = date.fromisoformat(q["as_of"]) if q.get("as_of") else None

        g_cids = _graph_retrieve(G, q_text, as_of, index, embedder, reranker, cat)
        g_hits = sum(1 for g in golds if _gold_hit_graph(g, g_cids)) if golds else 0
        g_recall = g_hits / len(golds) if golds else 1.0
        g_prec = g_hits / len(g_cids) if g_cids else 0.0

        b_cids = _baseline_retrieve(q_text, chunks, index, embedder, reranker)
        b_units = _baseline_units(b_cids, chunks)
        b_hits = sum(1 for g in golds if _gold_hit_baseline(g, b_units)) if golds else 0
        b_recall = b_hits / len(golds) if golds else 1.0
        b_prec = b_hits / len(b_cids) if b_cids else 0.0

        by_cat[cat]["g_r"].append(g_recall)
        by_cat[cat]["g_p"].append(g_prec)
        by_cat[cat]["b_r"].append(b_recall)
        by_cat[cat]["b_p"].append(b_prec)

        per_q.append({
            "id": q["id"],
            "category": cat,
            "as_of": q.get("as_of"),
            "n_gold": len(golds),
            "contrast_set": cat in CONTRAST_CATEGORIES,
            "graph_recall": g_recall,
            "graph_precision": g_prec,
            "baseline_recall": b_recall,
            "baseline_precision": b_prec,
        })

    headline = [p for p in per_q if p["category"] not in CONTRAST_CATEGORIES]
    contrast = [p for p in per_q if p["category"] in CONTRAST_CATEGORIES]

    summary = {
        "n": len(questions),
        "n_headline": len(headline),
        "n_contrast": len(contrast),
        "headline_excludes": sorted(CONTRAST_CATEGORIES),
        "graph_recall": _mean([p["graph_recall"] for p in headline]),
        "graph_precision": _mean([p["graph_precision"] for p in headline]),
        "baseline_recall": _mean([p["baseline_recall"] for p in headline]),
        "baseline_precision": _mean([p["baseline_precision"] for p in headline]),
        "contrast": {
            "graph_recall": _mean([p["graph_recall"] for p in contrast]),
            "baseline_recall": _mean([p["baseline_recall"] for p in contrast]),
            "delta_recall": (
                _mean([p["graph_recall"] for p in contrast])
                - _mean([p["baseline_recall"] for p in contrast])
            ),
            "note": (
                "Semantic cluster mates without a required graph edge — flat RAG "
                "is expected to win. Contrast category, not a graph claim."
            ),
        },
        "by_category": {
            cat: {
                "n": len(v["g_r"]),
                "graph_recall": _mean(v["g_r"]),
                "baseline_recall": _mean(v["b_r"]),
                "delta_recall": _mean(v["g_r"]) - _mean(v["b_r"]),
                "contrast_set": cat in CONTRAST_CATEGORIES,
            }
            for cat, v in sorted(by_cat.items())
        },
    }
    summary["delta_recall"] = summary["graph_recall"] - summary["baseline_recall"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=2))
    print(
        f"\nHeadline (excl. {sorted(CONTRAST_CATEGORIES)})  "
        f"graph R={summary['graph_recall']:.3f} P={summary['graph_precision']:.3f}"
    )
    print(
        f"         base  R={summary['baseline_recall']:.3f} "
        f"P={summary['baseline_precision']:.3f}"
    )
    print(f"Δ recall (graph−base) = {summary['delta_recall']:+.3f}")
    ct = summary["contrast"]
    print(
        f"\nContrast multi_hop_cluster: "
        f"graph {ct['graph_recall']:.2f} / base {ct['baseline_recall']:.2f} / "
        f"Δ {ct['delta_recall']:+.2f}"
    )
    print("\nBy category (graph recall / baseline recall / Δ):")
    for cat, row in summary["by_category"].items():
        tag = " [contrast]" if row["contrast_set"] else ""
        print(
            f"  {cat:28s} n={row['n']:2d}  "
            f"{row['graph_recall']:.2f} / {row['baseline_recall']:.2f} / "
            f"{row['delta_recall']:+.2f}{tag}"
        )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
