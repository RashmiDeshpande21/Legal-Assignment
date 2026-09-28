"""A/B: conflict-audit context block (retrieval/pipeline._conflict_audit_context).

Q8 asks the system to FIND a conflict or broken cross-reference. Every generator
(Opus included) failed it: discovery requires comparing all referring clauses to
their targets, which no retrieval context contains. The offline audit
(experiments/crossref_audit.py) does that comparison at index time; this block
injects its findings + the explicit precedence clauses.

Tuned ONLY on q8 (assignment tuning set); holdout conflict_precedence questions
validate generalization separately.

Run: CUDA_VISIBLE_DEVICES=N LLM_MAIN_GPU=0 EMBED_DEVICE=cpu \
     uv run python -m experiments.conflict_audit_ab [model_name]
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

OUT = Path("experiments/results/conflict_audit_ab.json")


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

    q8 = next(q for q in QUESTIONS if q.id == "q8")
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for cond, flag in (("off", False), ("on", True)):
        t0 = time.time()
        ans = graph_answer(G, q8.text, q8.as_of, reranker, llm,
                           index=index, embedder=embedder, conflict_audit=flag)
        lat = round(time.time() - t0, 2)
        ctx = "\n\n".join(f"[{c}]\n{t}" for c, t, _ in ans.blocks)[:12000]
        jp = JUDGE_PROMPT.format(
            as_of="", question=q8.text, gold_answer=gold["q8"],
            answer=ans.text[:6000], context=ctx,
        )
        j = _parse_judge(judge.generate(JUDGE_SYSTEM, jp, max_tokens=400))
        results[f"{name}::{cond}"] = {
            "correct": bool(j.get("answer_correct")),
            "faithfulness": j.get("faithfulness"),
            "note": j.get("note", ""), "latency_s": lat,
            "answer_preview": ans.text[:500],
        }
        print(f"[{cond}] correct={results[f'{name}::{cond}']['correct']} lat={lat}s\n"
              f"  note: {results[f'{name}::{cond}']['note'][:200]}", flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
