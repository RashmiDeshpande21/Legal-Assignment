"""Compare Mistral-Small-24B/Gemma-2-27B/Qwen3.6-27B vs current Qwen2.5-14B.

Same live pipeline context (hybrid seeding + proviso-split + legal CoT), judged by
Bedrock Sonnet-4.6. Models loaded one at a time on a single A10G.
Writes experiments/results/large_gguf_results.json.
Run: uv run python -m experiments.large_gguf_comparison [model_filter]
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

from config import settings
from eval.harness import JUDGE_PROMPT, JUDGE_SYSTEM, _parse_judge
from eval.questions import QUESTIONS
from graph.loader import load_graph
from llm.client import BedrockLlm, LocalLlm
from retrieval.embedder import Embedder, VectorIndex
from retrieval.pipeline import graph_answer
from retrieval.reranker import Reranker

MODELS = {
    "qwen25-14b": ("models/Qwen2.5-14B-Instruct-Q4_K_M.gguf", None),
    "mistral-24b": ("models/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf", None),
    # Gemma-2 train ctx = 8192; llama-cpp refuses n_ctx > n_ctx_train
    "gemma2-27b": ("models/gemma-2-27b-it-Q4_K_M.gguf", 8192),
    "qwen36-27b": ("models/Qwen_Qwen3.6-27B-Q4_K_M.gguf", None),
}

# Hard set from proviso_prompting + full 12 for winner confirmation
HARD = {"q2", "q5", "q7", "q10"}


def _vram_mb() -> int:
    try:
        import torch
        return int(torch.cuda.memory_allocated(0) / 1e6) if torch.cuda.is_available() else -1
    except Exception:  # noqa: BLE001
        return -1


def _nvidia_used_mb() -> int:
    """Used MiB on this process's GPU (CUDA_VISIBLE_DEVICES first id, else max of 1..N)."""
    import os
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    lines = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and "," not in visible and visible.isdigit():
        idx = int(visible)
        return lines[idx] if idx < len(lines) else -1
    return max(lines[1:]) if len(lines) > 1 else (lines[0] if lines else -1)


def smoke(path: str) -> dict:
    """Load model, generate 1 token, report VRAM, unload."""
    t0 = time.time()
    llm = LocalLlm(path)
    text = llm.generate("You are a test harness.", "Reply with the single word: ok", max_tokens=8)
    used = _nvidia_used_mb()
    load_s = round(time.time() - t0, 1)
    del llm
    gc.collect()
    return {"ok": True, "reply": text[:80], "vram_used_mb": used, "load_s": load_s}


def run_questions(llm, G, questions, index, embedder, reranker, judge, gold) -> dict:
    import os
    per_q = {}
    for q in questions:
        q_text = q.retrieval_text or q.text
        t0 = time.time()
        ans = graph_answer(G, q_text, q.as_of, reranker, llm, index=index, embedder=embedder,
                           change_history=os.environ.get("CHANGE_HISTORY") == "1",
                           conflict_audit=os.environ.get("CONFLICT_AUDIT") == "1")
        lat = round(time.time() - t0, 2)
        ctx = "\n\n".join(f"[{c}]\n{t}" for c, t, _ in ans.blocks)[:12000]
        jp = JUDGE_PROMPT.format(
            as_of=f" (as of {q.as_of})" if q.as_of else "", question=q.text,
            gold_answer=gold[q.id], answer=ans.text[:6000], context=ctx,
        )
        try:
            j = _parse_judge(judge.generate(JUDGE_SYSTEM, jp, max_tokens=400))
            per_q[q.id] = {
                "correct": bool(j.get("answer_correct")),
                "faithfulness": j.get("faithfulness"),
                "citation_accuracy": j.get("citation_accuracy"),
                "note": j.get("note", ""),
                "latency_s": lat,
                "answer_preview": ans.text[:300],
            }
        except Exception as exc:  # noqa: BLE001
            per_q[q.id] = {"correct": False, "note": f"judge error: {exc!r}", "latency_s": lat}
        print(f"  {q.id}: correct={per_q[q.id].get('correct')} "
              f"lat={lat}s note={str(per_q[q.id].get('note', ''))[:100]}", flush=True)
    n_ok = sum(1 for v in per_q.values() if v.get("correct"))
    return {"correct": n_ok, "of": len(per_q), "per_question": per_q}


def main() -> None:
    # argv: [model_filter?] [smoke|hard|full]
    args = [a for a in sys.argv[1:] if a]
    mode = "hard"
    only = None
    for a in args:
        if a in ("smoke", "hard", "full"):
            mode = a
        else:
            only = a
    G = load_graph(settings.artifacts_dir / "graph.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)
    judge = BedrockLlm(settings.bedrock_model_judge)
    gold = {e["id"]: e["gold_answer"]
            for e in json.loads(Path("experiments/mini_eval_set.json").read_text())}

    qs = QUESTIONS if mode == "full" else [q for q in QUESTIONS if q.id in HARD]

    # Per-model shard when filtered (safe for parallel GPU runs); else shared file.
    import os
    shard = os.environ.get("LARGE_GGUF_OUT")
    if shard:
        out_path = Path(shard)
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
    elif only:
        out_path = Path(f"experiments/results/large_gguf_{only}.json")
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
    else:
        out_path = Path("experiments/results/large_gguf_results.json")
        results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name, (path, n_ctx) in MODELS.items():
        if only and only not in name:
            continue
        key = f"{name}::{mode}"
        print(f"\n=== {name} ({path}) mode={mode} n_ctx={n_ctx or settings.llm_context_length} "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} ===",
              flush=True)
        try:
            if mode == "smoke":
                results[key] = smoke(path)
                print(results[key], flush=True)
            else:
                llm = LocalLlm(path, n_ctx=n_ctx)
                # warm + report VRAM after first real call would be late; probe once
                _ = llm.generate("You are a legal analyst.", "Say ready.", max_tokens=4)
                results[key] = run_questions(
                    llm, G, qs, index, embedder, reranker, judge, gold)
                results[key]["vram_used_mb"] = _nvidia_used_mb()
                results[key]["model_path"] = path
                results[key]["n_ctx"] = n_ctx or settings.llm_context_length
                results[key]["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
                results[key]["system"] = "ANSWER_SYSTEM (incl. CoT)"
                print(f"SCORE {name}: {results[key]['correct']}/{results[key]['of']} "
                      f"vram={results[key]['vram_used_mb']}MB", flush=True)
                del llm
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[key] = {"ok": False, "error": repr(exc)}
            print(f"FAIL {name}: {exc!r}", flush=True)
            gc.collect()
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))

    print(f"\nwrote {out_path}")
    # summary table
    for k, v in results.items():
        if "correct" in v:
            print(f"{k}: {v['correct']}/{v['of']} vram={v.get('vram_used_mb')}MB")
        elif v.get("ok"):
            print(f"{k}: smoke ok vram={v.get('vram_used_mb')}MB reply={v.get('reply')!r}")
        else:
            print(f"{k}: FAIL {v.get('error')}")


if __name__ == "__main__":
    main()
