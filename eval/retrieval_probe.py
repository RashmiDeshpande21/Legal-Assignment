"""Tier-1 retrieval check with no generator in the loop.

The graph path is exercised for real (seeding, DFS, rerank, both expansion stages,
projection blocks); only ``llm.generate`` is stubbed out. Gold recall therefore moves
exactly as it would in a full run, in ~20 seconds instead of ~20 minutes, so a
retrieval change can be accepted or rejected before any answer tokens are spent.

Two sets, same code path:
  make probe          the assignment's 12 questions (the tuning set)
  make probe-holdout  the 20 unseen hard questions (the anti-overfitting guard)

A retrieval change is only accepted when it holds on both. Recall on the tuning set
alone says nothing about the questions the reviewer will actually ask.

Run: make probe [SET=holdout]
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from config import settings
from eval.harness import _gold_hit_graph
from eval.questions import QUESTIONS
from graph.loader import load_graph
from retrieval.embedder import Embedder, VectorIndex
from retrieval.pipeline import graph_answer
from retrieval.reranker import Reranker

EVAL_SET = Path("experiments/mini_eval_set.json")
HOLDOUT_SET = Path("eval/holdout_hard_set.json")


class _NullLlm:
    """Stands in for the generator: retrieval is what this script measures."""

    def generate(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        return ""


def _cases(which: str) -> list[tuple[str, str, date | None, list[str]]]:
    """(id, question, as_of, gold) for the requested set."""
    if which == "holdout":
        return [
            (c["id"], c["question"],
             date.fromisoformat(c["as_of"]) if c.get("as_of") else None,
             c["gold"])
            for c in json.loads(HOLDOUT_SET.read_text())
        ]
    gold_by_q = {e["id"]: e["gold"] for e in json.loads(EVAL_SET.read_text())}
    return [
        (q.id, q.retrieval_text or q.text, q.as_of, gold_by_q.get(q.id, []))
        for q in QUESTIONS
    ]


def main() -> None:
    which = os.environ.get("SET", "assignment")
    G = load_graph(settings.artifacts_dir / "graph.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)

    recalls, rows = [], []
    for qid, question, as_of, golds in _cases(which):
        ans = graph_answer(G, question, as_of, reranker,
                           _NullLlm(), index=index, embedder=embedder)
        if not golds:  # q8 is judged on the answer, not on a gold retrieval set
            print(f"{qid}: no gold retrieval set (skipped)")
            continue
        missed = [g for g in golds if not _gold_hit_graph(g, ans.context_ids)]
        recall = 1 - len(missed) / len(golds)
        recalls.append(recall)
        rows.append({"id": qid, "recall": round(recall, 3),
                     "n_context": len(ans.context_ids), "missed": missed})
        flag = "" if not missed else f"  MISSED {missed}"
        print(f"{qid}: recall={recall:.3f} context={len(ans.context_ids)}{flag}")

    mean = sum(recalls) / len(recalls)
    perfect = sum(1 for r in recalls if r == 1.0)
    print(f"\n[{which}] mean gold recall (graph): {mean:.3f} over {len(recalls)} "
          f"questions; {perfect}/{len(recalls)} at 1.000")
    suffix = "" if which == "assignment" else f"_{which}"
    out = settings.outputs_dir / f"retrieval_probe{suffix}.json"
    out.write_text(json.dumps(
        {"set": which, "mean_recall": round(mean, 3), "n_perfect": perfect,
         "per_question": rows}, indent=1, ensure_ascii=False,
    ))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
