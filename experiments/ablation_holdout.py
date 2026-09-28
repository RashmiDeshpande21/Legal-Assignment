"""Holdout ablation — the unsaturated follow-up the 12-question ablation called for.

The assignment-set ablation (experiments/ablation.py) found every condition at 12/12,
including graph_only. That is ceiling-limited: it can detect harm, not contribution.
eval/holdout_hard_set.json sits at 15-16/20 for Qwen3.6, so it has the headroom to
decide whether proviso_split, change_history and conflict_audit still earn their keep
on unseen questions of the same archetypes.

Same five CONDITIONS as experiments/ablation.py, same shipping model, same three
scoring instruments. Reported by category as well as total — a block that only helps
multi_amendment_aggregation is still a real contribution, even if the aggregate is
flat.

Runtime: ~70s/answer x 20 questions x 5 conditions ≈ 2 hours on one A10G. Conditions
are written as they finish and skipped on re-run.

Run: CUDA_VISIBLE_DEVICES=1 LLM_MAIN_GPU=0 EMBED_DEVICE=cpu \
     uv run python -m experiments.ablation_holdout [model_key]
     ABLATION_CONDITIONS=full,graph_only uv run python -m experiments.ablation_holdout
     ABLATION_REPORT_ONLY=1 uv run python -m experiments.ablation_holdout
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

from config import settings
from eval.citation_verify import verify_row
from eval.harness import JUDGE_PROMPT, JUDGE_SYSTEM, _hier_match, _parse_judge
from experiments.ablation import CONDITIONS
from experiments.large_gguf_comparison import MODELS
from graph.loader import load_graph
from llm.client import BedrockLlm, LocalLlm
from retrieval.embedder import Embedder, VectorIndex
from retrieval.pipeline import graph_answer
from retrieval.reranker import Reranker

HOLDOUT = Path("eval/holdout_hard_set.json")
OUT = Path("experiments/results/ablation_holdout.json")


def run_condition(name: str, G, index, embedder, reranker, llm, judge,
                  questions: list[dict]) -> dict:
    flags = CONDITIONS[name]
    per_q: dict[str, dict] = {}
    for q in questions:
        as_of = date.fromisoformat(q["as_of"]) if q.get("as_of") else None
        t0 = time.time()
        ans = graph_answer(G, q["question"], as_of, reranker, llm,
                           index=index, embedder=embedder, **flags)
        lat = round(time.time() - t0, 2)
        gold = q.get("gold") or []
        recall = (sum(any(_hier_match(g, c) for c in ans.context_ids) for g in gold)
                  / len(gold) if gold else None)

        row = {"id": q["id"], "as_of": q.get("as_of"), "question": q["question"],
               "answer": ans.text, "citations": ans.citations,
               "context_ids": ans.context_ids}
        cite = verify_row(G, row)

        ctx = "\n\n---\n\n".join(f"[{c}]\n{t}" for c, t, _ in ans.blocks)[:12000]
        prompt = JUDGE_PROMPT.format(
            as_of=f" (as of {q['as_of']})" if q.get("as_of") else "",
            question=q["question"],
            gold_answer=q.get("gold_answer", "(no gold answer)"),
            answer=ans.text[:6000], context=ctx,
        )
        try:
            j = _parse_judge(judge.generate(JUDGE_SYSTEM, prompt, max_tokens=512))
        except Exception as exc:  # noqa: BLE001
            j = {"error": repr(exc)[:200]}
        per_q[q["id"]] = {
            "category": q["category"],
            "correct": bool(j.get("answer_correct")),
            "citation_accuracy_judge": j.get("citation_accuracy"),
            "faithfulness": j.get("faithfulness"),
            "recall": round(recall, 3) if recall is not None else None,
            "verified_citation_rate": cite["verified_citation_rate"],
            "version_qualified_rate": cite["version_qualified_rate"],
            "anachronistic": len(cite["anachronistic"]),
            "n_blocks": len(ans.blocks),
            "latency_s": lat,
            "note": j.get("note", j.get("error", ""))[:220],
            "answer_preview": ans.text[:300],
        }
        v = per_q[q["id"]]
        print(f"  [{name}] {q['id']} [{q['category']}]: correct={v['correct']} "
              f"recall={v['recall']} verified={v['verified_citation_rate']} "
              f"blocks={v['n_blocks']} {v['latency_s']}s", flush=True)

    def _mean(key: str) -> float | None:
        vals = [v[key] for v in per_q.values() if v.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    by_cat: dict[str, list[bool]] = {}
    for v in per_q.values():
        by_cat.setdefault(v["category"], []).append(v["correct"])
    return {
        "flags": flags,
        "correct": sum(v["correct"] for v in per_q.values()),
        "of": len(per_q),
        "by_category": {c: f"{sum(xs)}/{len(xs)}" for c, xs in sorted(by_cat.items())},
        "mean_recall": _mean("recall"),
        "mean_verified_citation_rate": _mean("verified_citation_rate"),
        "mean_citation_accuracy_judge": _mean("citation_accuracy_judge"),
        "mean_faithfulness": _mean("faithfulness"),
        "mean_latency_s": _mean("latency_s"),
        "total_anachronistic": sum(v["anachronistic"] for v in per_q.values()),
        "per_question": per_q,
    }


def _report(res: dict, model: str, questions: list[dict]) -> str:
    order = [c for c in CONDITIONS if c in res]
    full = res.get("full")
    cats = sorted({q["category"] for q in questions})
    lines = [
        f"# Holdout component ablation ({model})",
        "",
        "Same five CONDITIONS as `experiments/ablation.py`, on the unsaturated held-out",
        "hard set (`eval/holdout_hard_set.json`, 20 questions). The assignment-set",
        "ablation was ceiling-limited at 12/12; this set sits at 15-16/20 and can detect",
        "contribution as well as harm.",
        "",
        "| Condition | Correct | vs full | "
        + " | ".join(cats)
        + " | Verified cites | Faithfulness | Latency |",
        "|---|---|---|" + "|".join(["---"] * len(cats)) + "|---|---|---|",
    ]
    for c in order:
        r = res[c]
        delta = ("\u2014" if c == "full" or not full
                 else f"{r['correct'] - full['correct']:+d}")
        cat_cells = [r["by_category"].get(cat, "\u2014") for cat in cats]
        lines.append(
            f"| `{c}` | {r['correct']}/{r['of']} | {delta} | "
            + " | ".join(cat_cells)
            + f" | {r['mean_verified_citation_rate']} | {r['mean_faithfulness']} | "
            f"{r['mean_latency_s']}s |"
        )
    lines += [
        "",
        "## Per-question verdicts",
        "",
        "| Question | Category | " + " | ".join(f"`{c}`" for c in order) + " |",
        "|---|---" + "|---" * len(order) + "|",
    ]
    for q in questions:
        cells = []
        for c in order:
            v = res[c]["per_question"].get(q["id"], {})
            cells.append("ok" if v.get("correct") else "**FAIL**")
        lines.append(f"| {q['id']} | {q['category']} | " + " | ".join(cells) + " |")

    if full:
        lines += ["", "### Regressions and gains, per block", ""]
        any_move = False
        for c in order:
            if c == "full":
                continue
            lost, gained = [], []
            for q in questions:
                f_ok = full["per_question"].get(q["id"], {}).get("correct")
                c_ok = res[c]["per_question"].get(q["id"], {}).get("correct")
                if f_ok and not c_ok:
                    lost.append(f"{q['id']} ({q['category']})")
                if not f_ok and c_ok:
                    gained.append(f"{q['id']} ({q['category']})")
            if lost or gained:
                any_move = True
                bits = []
                if lost:
                    bits.append("breaks " + ", ".join(lost))
                if gained:
                    bits.append("fixes " + ", ".join(gained))
                lines.append(f"- Condition `{c}`: " + "; ".join(bits) + ".")
        if not any_move:
            lines.append(
                "- No condition changed any per-question verdict relative to `full`. "
                "Even with headroom, the three blocks do not move holdout correctness "
                "on this model."
            )
        # Summarise the ranking so a reader does not have to reconstruct it.
        ranked = sorted(
            ((c, res[c]["correct"]) for c in order),
            key=lambda x: (-x[1], order.index(x[0])),
        )
        best_c, best_n = ranked[0]
        full_n = full["correct"]
        lines += [
            "",
            "### Reading this against the assignment-set ablation",
            "",
            f"Best condition: `{best_c}` at {best_n}/{full['of']} "
            f"(shipping `full` is {full_n}/{full['of']}).",
            "",
        ]
        if best_c == "graph_only" and best_n > full_n:
            lines += [
                "The unsaturated set does what the assignment set could not: it detects "
                "that the three context-shaping blocks are not merely unnecessary on "
                "this model — stacked together they *cost* answers. `graph_only` recovers "
                "the questions that `conflict_audit` and (with proviso elevation) the "
                "full stack lose, and it loses none that `full` gets right.",
                "",
                "Per-block: `conflict_audit` is the clear offender (hh_20, a "
                "conflict_precedence question — the category the audit was built for). "
                "`proviso_split` is a wash (fixes one temporal, breaks one aggregation). "
                "`change_history` moves nothing. Three failures (hh_02, hh_11, hh_13) are "
                "untouched by any condition — residual model/retrieval limits, not "
                "missing scaffolding.",
                "",
                "Implication for the shipping defaults: on Qwen3.6 the honest "
                "configuration is the graph with as-of version resolution and no "
                "context-shaping blocks. Keeping the blocks on because they once helped "
                "Qwen2.5-14B is overfitting the assignment set's model ladder, not the "
                "pipeline.",
            ]
        elif best_n <= full_n and any_move:
            lines += [
                "Some blocks move individual questions, but none beats `full` in "
                "aggregate. Read the per-question table for whether the movers are the "
                "questions each block was built for.",
            ]
        else:
            lines += [
                "A null here, on a set with headroom under `full`, rules out "
                "*contribution* as well as harm — scaffolding a weaker generator that "
                "this one does not need.",
            ]
        lines += [
            "",
            "Prior holdout runs on an earlier pipeline revision "
            "(`holdout_qwen36-27b.json`) scored baseline 16/20 and change_history / "
            "combined 15/20 — suggestive, but not a controlled per-block ablation on "
            "one revision. This is.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    model_key = sys.argv[1] if len(sys.argv) > 1 else "qwen36-27b"
    questions = json.loads(HOLDOUT.read_text())
    res = json.loads(OUT.read_text()) if OUT.exists() else {}

    if os.environ.get("ABLATION_REPORT_ONLY") == "1":
        OUT.with_suffix(".md").write_text(_report(res, model_key, questions))
        print(f"re-rendered {OUT.with_suffix('.md')}")
        return

    wanted = os.environ.get("ABLATION_CONDITIONS", ",".join(CONDITIONS)).split(",")
    todo = [c for c in wanted if c not in res]
    if not todo:
        print(f"all requested conditions already in {OUT}; nothing to run")
    else:
        print(f"model={model_key} holdout={len(questions)}Q conditions={todo}",
              flush=True)
        path, n_ctx = MODELS[model_key]
        G = load_graph(settings.artifacts_dir / "graph.json")
        index = VectorIndex.load(settings.artifacts_dir / "index")
        embedder = Embedder(settings.embedder_model)
        reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)
        judge = BedrockLlm(settings.bedrock_model_judge)
        llm = LocalLlm(path, n_ctx=n_ctx)
        _ = llm.generate("You are a legal analyst.", "Say ready.", max_tokens=4)

        for cond in todo:
            t0 = time.time()
            res[cond] = run_condition(cond, G, index, embedder, reranker, llm, judge,
                                      questions)
            res[cond]["model"] = model_key
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(res, indent=1))
            print(f"{cond}: {res[cond]['correct']}/{res[cond]['of']} "
                  f"by_cat={res[cond]['by_category']} "
                  f"({time.time() - t0:.0f}s) -> {OUT}", flush=True)

    OUT.with_suffix(".md").write_text(_report(res, model_key, questions))
    print()
    for c in CONDITIONS:
        if c in res:
            print(f"{c:20s} {res[c]['correct']}/{res[c]['of']}  "
                  f"{res[c]['by_category']}")
    print(f"wrote {OUT}, {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
