"""Run the Frontier Benchmark: Graph + Bedrock Opus 4.6 across all 12 questions.

Uses the exact same graph retrieval and context blocks that SaulLM and Qwen saw.
Judged by Bedrock Sonnet 4.6 (judge) to produce the definitive Frontier comparison.
Writes outputs/opus_answers.json and outputs/frontier_3way_comparison.md.
Run: uv run python -m eval.eval_opus
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import settings
from eval.harness import JUDGE_PROMPT, JUDGE_SYSTEM, _parse_judge
from eval.questions import QUESTIONS
from graph.loader import load_graph
from llm.client import BedrockLlm
from retrieval.pipeline import graph_answer
from retrieval.reranker import Reranker


def main() -> None:
    print(f"Loading Graph, Reranker, and Bedrock Opus 4.6 ({settings.bedrock_model})...")
    G = load_graph(settings.artifacts_dir / "graph.json")
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)
    opus_llm = BedrockLlm(settings.bedrock_model)  # Opus 4.6
    judge_llm = BedrockLlm(settings.bedrock_model_judge)  # Sonnet 4.6 judge

    eval_set = json.loads(Path("experiments/mini_eval_set.json").read_text())
    gold_by_q = {e["id"]: e["gold_answer"] for e in eval_set}

    results = []
    print("\n--- Generating Answers with Claude Opus 4.6 ---")
    for q in QUESTIONS:
        t0 = time.time()
        q_text = q.retrieval_text or q.text
        ans = graph_answer(G, q_text, q.as_of, reranker, opus_llm)
        lat = round(time.time() - t0, 2)

        # Grade with Bedrock Sonnet judge
        as_of_str = f" (as of {q.as_of})" if q.as_of else ""
        prompt = JUDGE_PROMPT.format(
            as_of=as_of_str,
            question=q.text,
            gold_answer=gold_by_q.get(q.id, ""),
            answer=ans.text,
            context="\n\n".join(f"[{c}]\n{t}" for c, t, _ in ans.blocks[:6]),
        )
        try:
            raw_judge = judge_llm.generate(JUDGE_SYSTEM, prompt, max_tokens=400)
            judged = _parse_judge(raw_judge)
        except Exception as e:
            judged = {"error": str(e), "answer_correct": False}

        results.append({
            "id": q.id,
            "question": q.text,
            "as_of": str(q.as_of) if q.as_of else None,
            "answer": ans.text,
            "citations": ans.citations,
            "latency_s": lat,
            "judge": judged,
        })
        status = "CORRECT" if judged.get("answer_correct") else "WRONG"
        print(f"[{q.id}] {status} in {lat}s | Note: {judged.get('note', '')}")

    Path("outputs/opus_results.json").write_text(json.dumps(results, indent=2))

    correct_cnt = sum(1 for r in results if r["judge"].get("answer_correct"))
    faithfulness = [
        r["judge"]["faithfulness"] for r in results if "faithfulness" in r["judge"]
    ]
    citation_acc = [
        r["judge"]["citation_accuracy"] for r in results if "citation_accuracy" in r["judge"]
    ]

    print("\n=== SUMMARY: Claude Opus 4.6 Graph Path ===")
    print(f"Answer Correctness: {correct_cnt}/12 ({correct_cnt/12:.1%})")
    if faithfulness:
        print(f"Mean Faithfulness: {sum(faithfulness)/len(faithfulness):.3f}")
    if citation_acc:
        print(f"Mean Citation Accuracy: {sum(citation_acc)/len(citation_acc):.3f}")


if __name__ == "__main__":
    main()
