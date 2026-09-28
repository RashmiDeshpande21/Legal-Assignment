"""Run the held-out hard set (eval/holdout_hard_set.json) through the LIVE pipeline.

Purpose: anti-overfitting guard. The assignment's 12 questions are the tuning set;
every pipeline change must also be scored on these 20 unseen questions of the same
archetypes (evolution/temporal/aggregation/dependents/false-premise /
conflict). Run BEFORE and AFTER each pipeline change; a change that helps the 12
but not the hold-out is overfitting and gets rejected.

Judged by Bedrock Sonnet-4.6 (eval-only frontier use). Also reports Tier-1 retrieval
recall of gold node ids against the ranked context (context_ids excludes the few
force-included seed blocks, so recall is a slight underestimate — consistent across
runs, fine for A/B).

Run: CUDA_VISIBLE_DEVICES=N LLM_MAIN_GPU=0 EMBED_DEVICE=cpu \
     HOLDOUT_OUT=experiments/results/holdout_<model>.json \
     [CHANGE_HISTORY=1] \
     uv run python -m experiments.holdout_eval <model_name> [tag]
model_name: key from experiments.large_gguf_comparison.MODELS
tag: label for this pipeline variant (default "baseline")
CHANGE_HISTORY=1 enables the change-history context block (A/B vs baseline)
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from config import settings
from eval.harness import JUDGE_PROMPT, JUDGE_SYSTEM, _parse_judge
from experiments.large_gguf_comparison import MODELS
from graph.loader import load_graph
from llm.client import BedrockLlm, LocalLlm
from retrieval.embedder import Embedder, VectorIndex
from retrieval.pipeline import graph_answer
from retrieval.reranker import Reranker

HOLDOUT = Path("eval/holdout_hard_set.json")


def main() -> None:
    name = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    path, n_ctx = MODELS[name]

    G = load_graph(settings.artifacts_dir / "graph.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)
    judge = BedrockLlm(settings.bedrock_model_judge)
    llm = LocalLlm(path, n_ctx=n_ctx)
    _ = llm.generate("You are a legal analyst.", "Say ready.", max_tokens=4)

    questions = json.loads(HOLDOUT.read_text())
    per_q, correct = {}, 0
    for q in questions:
        as_of = date.fromisoformat(q["as_of"]) if q.get("as_of") else None
        t0 = time.time()
        ans = graph_answer(G, q["question"], as_of, reranker, llm,
                           index=index, embedder=embedder,
                           change_history=os.environ.get("CHANGE_HISTORY") == "1",
                           conflict_audit=os.environ.get("CONFLICT_AUDIT") == "1")
        lat = round(time.time() - t0, 2)
        gold_nodes = set(q.get("gold", []))
        recall = (len(gold_nodes & set(ans.context_ids)) / len(gold_nodes)
                  if gold_nodes else None)
        ctx = "\n\n".join(f"[{c}]\n{t}" for c, t, _ in ans.blocks)[:12000]
        jp = JUDGE_PROMPT.format(
            as_of=f" (as of {q['as_of']})" if q.get("as_of") else "",
            question=q["question"], gold_answer=q["gold_answer"],
            answer=ans.text[:6000], context=ctx,
        )
        try:
            j = _parse_judge(judge.generate(JUDGE_SYSTEM, jp, max_tokens=400))
            per_q[q["id"]] = {
                "category": q["category"],
                "correct": bool(j.get("answer_correct")),
                "faithfulness": j.get("faithfulness"),
                "citation_accuracy": j.get("citation_accuracy"),
                "recall": recall,
                "note": j.get("note", ""),
                "latency_s": lat,
                "answer_preview": ans.text[:300],
            }
        except Exception as exc:  # noqa: BLE001
            per_q[q["id"]] = {"category": q["category"], "correct": False,
                              "note": f"judge error: {exc!r}", "latency_s": lat}
        correct += per_q[q["id"]]["correct"]
        print(f"  {q['id']} [{q['category']}]: correct={per_q[q['id']]['correct']} "
              f"recall={recall} lat={lat}s note={str(per_q[q['id']].get('note',''))[:90]}",
              flush=True)

    by_cat: dict[str, list[bool]] = {}
    for v in per_q.values():
        by_cat.setdefault(v["category"], []).append(v["correct"])
    summary = {
        "correct": correct, "of": len(per_q),
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
        "model": name, "tag": tag,
        "per_question": per_q,
    }
    out = Path(os.environ.get("HOLDOUT_OUT",
                              f"experiments/results/holdout_{name}.json"))
    results = json.loads(out.read_text()) if out.exists() else {}
    results[f"{name}::{tag}"] = summary
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nSCORE {name}::{tag}: {correct}/{len(per_q)}  by_category="
          f"{summary['by_category']}\nwrote {out}")


if __name__ == "__main__":
    main()
