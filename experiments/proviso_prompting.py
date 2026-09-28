"""Experiment: proviso-as-first-block context + structured legal CoT prompt.

2x2 grid (context: standard | proviso-split-first) x (prompt: standard | legal CoT),
on the four questions open-weight models fail (Q2 false premise, Q5 negative covenant,
Q7 grace period, Q10 amendment proviso), for Saul-7B and Qwen-14B GGUF.
Judged by Bedrock sonnet-4-6 answer_correct. Earlier inline seam markers failed on
Saul (outputs/qa_notes.md) — this tests block separation + ordering instead.
Writes experiments/results/proviso_prompting_results.json.
Run: uv run python -m experiments.proviso_prompting [model_filter]
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

from config import settings
from eval.harness import JUDGE_PROMPT, JUDGE_SYSTEM, _parse_judge
from eval.questions import QUESTIONS
from graph.intent import find_seeds
from graph.loader import load_graph
from graph.traversal import _amend_ops_before, dfs_context, find_effective
from llm.client import BedrockLlm, LocalLlm
from llm.prompts import ANSWER_SYSTEM_BASE, COT_ADDENDUM, build_answer_prompt, build_context
from retrieval.embedder import Embedder, VectorIndex
from retrieval.reranker import Reranker

QIDS = ["q2", "q5", "q7", "q10"]

# winner: split + CoT — both now wired into retrieval.pipeline/llm.prompts
COT_SYSTEM = ANSWER_SYSTEM_BASE + COT_ADDENDUM

MODELS = {
    "saul-7b": "models/Saul-7B-Instruct-v1.Q5_K_M.gguf",
    "qwen-14b": "models/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
}


def build_blocks(G, question, as_of, reranker, index, embedder, split_proviso: bool):
    """Mirror retrieval.pipeline.graph_answer block assembly, minus catalog/dependents."""
    seeds = find_seeds(G, question, index=index, embedder=embedder)
    effs = dfs_context(G, seeds, as_of, max_depth=2, max_nodes=40)
    by_id = {e.node_id: e for e in effs}
    ranked = reranker.top_k(
        question, [(e.node_id, e.text) for e in effs if e.text], k=settings.reranker_top_k)
    order = [nid for nid, _, _ in ranked]
    for s in seeds[:3]:
        if s not in order and G.nodes[s]["node_type"] in ("Section", "Definition"):
            order.insert(0, s)
            by_id.setdefault(s, find_effective(G, s, as_of))
    blocks = []
    for nid in order:
        eff = by_id[nid]
        if not eff.text:
            continue
        inserts = ([op for op in _amend_ops_before(G, nid, as_of)
                    if op["amendment_type"] == "insert" and G.nodes[nid].get("text")]
                   if split_proviso else [])
        if inserts:
            # amendment text as its own labelled block, BEFORE the parent section
            remaining = eff.text
            for op in inserts:
                blocks.append((
                    f"{eff.citation} — PROVISO added by {op['instrument']} "
                    f"(effective {op['effective_date']}), overrides the clauses of this "
                    "section where they conflict",
                    op["amended_text"], eff.source,
                ))
                remaining = remaining.replace("\n" + op["amended_text"], "")
            blocks.append((eff.citation, remaining, eff.source))
        else:
            blocks.append((eff.citation, eff.text, eff.source))
    return blocks


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    G = load_graph(settings.artifacts_dir / "graph.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)
    judge = BedrockLlm(settings.bedrock_model_judge)
    gold = {e["id"]: e["gold_answer"]
            for e in json.loads(Path("experiments/mini_eval_set.json").read_text())}
    questions = [q for q in QUESTIONS if q.id in QIDS]

    out_path = Path("experiments/results/proviso_prompting_results.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for model_name, model_path in MODELS.items():
        if only and only not in model_name:
            continue
        llm = LocalLlm(model_path)
        for split in (False, True):
            for cot in (False, True):
                cond = f"{model_name}::split={split}::cot={cot}"
                per_q = {}
                for q in questions:
                    q_text = q.retrieval_text or q.text
                    blocks = build_blocks(G, q_text, q.as_of, reranker, index, embedder,
                                          split_proviso=split)
                    prompt = build_answer_prompt(
                        q_text, build_context(blocks),
                        q.as_of.isoformat() if q.as_of else None)
                    ans = llm.generate(COT_SYSTEM if cot else ANSWER_SYSTEM_BASE, prompt)
                    ctx = "\n\n".join(f"[{c}]\n{t}" for c, t, _ in blocks)[:12000]
                    jp = JUDGE_PROMPT.format(
                        as_of=f" (as of {q.as_of})" if q.as_of else "", question=q.text,
                        gold_answer=gold[q.id], answer=ans[:6000], context=ctx)
                    try:
                        j = _parse_judge(judge.generate(JUDGE_SYSTEM, jp, max_tokens=400))
                        per_q[q.id] = {"correct": bool(j.get("answer_correct")),
                                       "note": j.get("note", "")}
                    except Exception as exc:  # noqa: BLE001
                        per_q[q.id] = {"correct": False, "note": f"judge error: {exc!r}"}
                n_ok = sum(1 for v in per_q.values() if v["correct"])
                results[cond] = {"correct": n_ok, "of": len(per_q), "per_question": per_q}
                print(f"{cond}: {n_ok}/{len(per_q)} "
                      f"({[qid for qid, v in per_q.items() if v['correct']]})")
        del llm
        gc.collect()
    out_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
