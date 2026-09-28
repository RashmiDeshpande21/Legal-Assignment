"""LLM comparison: identical gold context per question, judged by Bedrock (eval-only).

Candidates answer from fixed gold sections (assembled via graph as-of resolution) so only
the LLM varies. Judge: claude-sonnet-4-6 scores correctness + faithfulness 0-1 against the
gold answer. Latency measured per answer.
Run: CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.models.llm_comparison
Gated models (Llama, Mistral) need HF_TOKEN in the environment.
"""
from __future__ import annotations

import gc
import json
import time
from datetime import date
from pathlib import Path

CANDIDATES = [
    ("Equall/Saul-7B-Instruct-v1", "gguf", "models/Saul-7B-Instruct-v1.Q5_K_M.gguf"),
    ("meta-llama/Llama-3.1-8B-Instruct", "hf", None),
    # 14B slot (Phi-3-medium, phi-4, Qwen2.5-14B all attempted): unmeasurable on this
    # stack — transformers 5.3's loader materializes bf16 shards on GPU before bnb 4-bit
    # quantization, so the ~28GB transient OOMs a 22GB A10G. See _notes in llm_results.json.
    ("Qwen/Qwen2.5-14B-Instruct", "hf", None),
    ("mistralai/Mistral-7B-Instruct-v0.3", "hf", None),
]

PROMPT = """You are a legal analyst. Answer strictly from the provided credit agreement excerpts.
Cite section numbers and amendment status exactly as given in the excerpts.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER (with citations):"""

JUDGE_PROMPT = """Score this answer against the gold answer. Return JSON only:
{{"correctness": 0.0-1.0, "faithfulness": 0.0-1.0}}
correctness: does the substance match the gold answer (right sections, dates, conclusions)?
faithfulness: is every claim grounded in the excerpts (no hallucination)?

QUESTION: {question}
GOLD: {gold}
EXCERPTS: {context}
ANSWER: {answer}"""


def build_contexts() -> dict[str, str]:
    from graph.loader import load_graph
    from graph.traversal import find_effective
    G = load_graph(Path("artifacts/graph.json"))
    questions = json.loads(Path("experiments/mini_eval_set.json").read_text())
    ctxs = {}
    for q in questions:
        as_of = None
        if q["version_checks"]:
            as_of = date.fromisoformat(q["version_checks"][0]["as_of"])
        parts = []
        for nid in q["gold"][:8]:
            if G.nodes[nid]["node_type"] in ("Section", "Definition"):
                eff = find_effective(G, nid, as_of)
                parts.append(f"[{eff.citation}]\n{eff.text[:1800]}")
        ctxs[q["id"]] = "\n\n".join(parts)
    return ctxs


def gen_gguf(path: str, prompt: str) -> str:
    from llama_cpp import Llama
    if not hasattr(gen_gguf, "_m"):
        gen_gguf._m = Llama(path, n_gpu_layers=-1, n_ctx=8192, verbose=False)
    r = gen_gguf._m.create_chat_completion(
        messages=[{"role": "user", "content": prompt}], max_tokens=600, temperature=0.0)
    return r["choices"][0]["message"]["content"]


class HfGen:
    def __init__(self, model: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(model)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4")
        # trust_remote_code intentionally off: Phi-3's remote modeling code bypasses
        # the bitsandbytes quantization path and loads full bf16 (OOM on a 22GB A10G).
        self.m = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=quant, device_map="cuda")

    def __call__(self, prompt: str) -> str:
        import torch
        msgs = [{"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True,
                                              add_generation_prompt=True).to("cuda")
        with torch.no_grad():
            out = self.m.generate(**inputs, max_new_tokens=600, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        n_in = inputs["input_ids"].shape[1]
        return self.tok.decode(out[0][n_in:], skip_special_tokens=True)


def judge(question: str, gold: str, context: str, answer: str) -> dict:
    import boto3

    from config import settings
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    body = JUDGE_PROMPT.format(question=question, gold=gold, context=context[:6000],
                               answer=answer)
    resp = client.converse(
        modelId=settings.bedrock_model_judge,
        messages=[{"role": "user", "content": [{"text": body}]}],
        inferenceConfig={"maxTokens": 100, "temperature": 0},
    )
    txt = resp["output"]["message"]["content"][0]["text"]
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


def main() -> None:
    import sys
    import traceback

    import torch
    questions = json.loads(Path("experiments/mini_eval_set.json").read_text())
    ctxs = build_contexts()
    out_path = Path("experiments/models/results/llm_results.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for model, kind, gguf_path in CANDIDATES:
        if only and only not in model:
            continue
        gen = None
        try:
            gen = gen_gguf if kind == "gguf" else HfGen(model)
            corr, faith, times = [], [], []
            answers = {}
            for q in questions:
                prompt = PROMPT.format(context=ctxs[q["id"]], question=q["text"])
                t0 = time.time()
                ans = gen(gguf_path, prompt) if kind == "gguf" else gen(prompt)
                times.append(time.time() - t0)
                answers[q["id"]] = ans
                s = judge(q["text"], q["gold_answer"], ctxs[q["id"]], ans)
                corr.append(s["correctness"])
                faith.append(s["faithfulness"])
            results[model] = {
                "correctness": round(sum(corr) / len(corr), 3),
                "faithfulness": round(sum(faith) / len(faith), 3),
                "latency_s": round(sum(times) / len(times), 2),
            }
            Path(f"experiments/models/results/answers_{model.split('/')[-1]}.json").write_text(
                json.dumps(answers, indent=1, ensure_ascii=False))
            print(f"{model}: {results[model]}")
        except Exception as exc:  # noqa: BLE001 — record and continue
            results[model] = {"error": repr(exc)[:300]}
            print(f"{model}: FAILED {exc!r}")
            traceback.print_exc()
        finally:
            if kind == "gguf" and hasattr(gen_gguf, "_m"):
                del gen_gguf._m
            del gen
            gc.collect()
            torch.cuda.empty_cache()
    out_path.write_text(json.dumps(results, indent=1))
    ok = {m: r for m, r in results.items() if "correctness" in r}
    if ok:
        # faithfulness-first: for a citation-grounded legal system, hallucination is the
        # failure mode that matters; correctness deltas within ~0.02 on n=12 are noise.
        print("WINNER:", max(ok, key=lambda m: (ok[m]["faithfulness"], ok[m]["correctness"])))


if __name__ == "__main__":
    main()
