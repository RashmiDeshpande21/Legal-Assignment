"""A/B: change-history context block (retrieval/pipeline._change_history_context).

Hypothesis: q12-type failures (missing the First Amendment's Consolidated EBITDA
change when asked about the leverage ratio) are a context-assembly gap, not a model
gap — the amended TEXT is in context but per-instrument attribution and component-
definition changes are not. The block enumerates AMENDS ops on seeds + the
definitions they depend on, with an exhaustiveness statement.

Protocol: tuned ONLY on assignment questions whose wording hits the trigger
(q9, q10, q11, q12). Validation on eval/holdout_hard_set.json happens separately
(experiments/holdout_eval.py) and was NOT used to tune this.

Run: CUDA_VISIBLE_DEVICES=N LLM_MAIN_GPU=0 EMBED_DEVICE=cpu \
     uv run python -m experiments.change_history [model_name]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from config import settings
from eval.harness import JUDGE_PROMPT, JUDGE_SYSTEM, _parse_judge
from eval.questions import QUESTIONS
from experiments.large_gguf_comparison import MODELS
from graph.loader import load_graph
from llm.client import BedrockLlm, LocalLlm
from retrieval.embedder import Embedder, VectorIndex
from retrieval.pipeline import graph_answer
from retrieval.reranker import Reranker

TUNE_IDS = ("q9", "q10", "q11", "q12")
OUT = Path("experiments/results/change_history.json")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen25-14b"
    path, n_ctx = MODELS[name]
    G = load_graph(settings.artifacts_dir / "graph.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)
    judge = BedrockLlm(settings.bedrock_model_judge)
    gold = {e["id"]: e["gold_answer"]
            for e in json.loads(Path("experiments/mini_eval_set.json").read_text())}
    llm = LocalLlm(path, n_ctx=n_ctx)

    qs = [q for q in QUESTIONS if q.id in TUNE_IDS]
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for cond, flag in (("off", False), ("on", True)):
        per_q = {}
        for q in qs:
            t0 = time.time()
            ans = graph_answer(G, q.retrieval_text or q.text, q.as_of, reranker, llm,
                               index=index, embedder=embedder, change_history=flag)
            lat = round(time.time() - t0, 2)
            ctx = "\n\n".join(f"[{c}]\n{t}" for c, t, _ in ans.blocks)[:12000]
            jp = JUDGE_PROMPT.format(
                as_of=f" (as of {q.as_of})" if q.as_of else "", question=q.text,
                gold_answer=gold[q.id], answer=ans.text[:6000], context=ctx,
            )
            j = _parse_judge(judge.generate(JUDGE_SYSTEM, jp, max_tokens=400))
            per_q[q.id] = {
                "correct": bool(j.get("answer_correct")),
                "faithfulness": j.get("faithfulness"),
                "note": j.get("note", ""), "latency_s": lat,
                "answer_preview": ans.text[:300],
            }
            print(f"  [{cond}] {q.id}: correct={per_q[q.id]['correct']} lat={lat}s "
                  f"note={per_q[q.id]['note'][:100]}", flush=True)
        results[f"{name}::{cond}"] = {
            "correct": sum(v["correct"] for v in per_q.values()),
            "of": len(per_q), "per_question": per_q,
        }
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))
    for k, v in results.items():
        print(f"{k}: {v['correct']}/{v['of']}")


if __name__ == "__main__":
    main()
