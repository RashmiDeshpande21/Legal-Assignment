"""Embedder comparison on mini_eval_set retrieval ground truth.

Corpus = every Section/Definition node text. Query = each of the 12 questions.
Relevant = the question's gold node ids. Metrics: Recall@10, MRR@10, encode seconds.
Run on GPU: CUDA_VISIBLE_DEVICES=1 uv run python -m experiments.models.embedder_comparison
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

CANDIDATES = [
    "minetta/nemotron-3-embed-8b-legal",      # #1 MTEB Law, SEC EDGAR trained
    "kmad00/legal-colbert-extractor",         # SEC EDGAR ColBERT (pylate)
    "Hanno-Labs/dinghy-law-0.6b-v1",          # Qwen3-based legal, 8k ctx
    "intfloat/e5-large-v2",                   # strong general
    "BAAI/bge-large-en-v1.5",                 # strong general
]
COLBERT = {"kmad00/legal-colbert-extractor"}


def load_corpus() -> tuple[list[str], list[str]]:
    from graph.loader import load_graph
    G = load_graph(Path("artifacts/graph.json"))
    ids, texts = [], []
    for n, d in G.nodes(data=True):
        if d["node_type"] in ("Section", "Definition") and d.get("text"):
            ids.append(n)
            texts.append(d["text"][:4000])
    return ids, texts


def eval_rankings(rankings: dict[str, list[str]], questions: list[dict]) -> dict:
    recalls, mrrs = [], []
    for q in questions:
        gold = set(q["gold"])
        if not gold:
            continue
        top = rankings[q["id"]][:10]
        recalls.append(sum(g in top for g in gold) / len(gold))
        rr = next((1 / (i + 1) for i, n in enumerate(top) if n in gold), 0.0)
        mrrs.append(rr)
    return {"recall@10": round(sum(recalls) / len(recalls), 3),
            "mrr@10": round(sum(mrrs) / len(mrrs), 3)}


def run_st(model: str, ids: list[str], texts: list[str], questions: list[dict]) -> dict:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    m = SentenceTransformer(model, trust_remote_code=True, model_kwargs={"torch_dtype": "float16"})
    doc_vecs = m.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=8)
    encode_s = round(time.time() - t0, 1)
    q_texts = [q["text"] for q in questions]
    q_vecs = m.encode(q_texts, normalize_embeddings=True, show_progress_bar=False)
    sims = np.asarray(q_vecs) @ np.asarray(doc_vecs).T
    rankings = {
        q["id"]: [ids[j] for j in np.argsort(-sims[i])[:30]] for i, q in enumerate(questions)
    }
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return {**eval_rankings(rankings, questions), "encode_s": encode_s, "rankings": rankings}


def run_colbert(model: str, ids: list[str], texts: list[str], questions: list[dict]) -> dict:
    import torch
    from pylate import models, rank
    t0 = time.time()
    m = models.ColBERT(model_name_or_path=model)
    doc_embs = m.encode(texts, is_query=False, show_progress_bar=False, batch_size=8)
    encode_s = round(time.time() - t0, 1)
    q_embs = m.encode([q["text"] for q in questions], is_query=True, show_progress_bar=False)
    rankings = {}
    for i, q in enumerate(questions):
        scored = rank.rerank(documents_ids=[ids], queries_embeddings=[q_embs[i]],
                             documents_embeddings=[doc_embs])[0]
        rankings[q["id"]] = [s["id"] for s in scored[:30]]
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return {**eval_rankings(rankings, questions), "encode_s": encode_s, "rankings": rankings}


def main() -> None:
    import sys

    import torch
    ids, texts = load_corpus()
    questions = json.loads(Path("experiments/mini_eval_set.json").read_text())
    print(f"corpus: {len(ids)} nodes")
    out = Path("experiments/models/results/embedder_results.json")
    results = json.loads(out.read_text()) if out.exists() else {}
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    for model in CANDIDATES:
        if only and only not in model:
            continue
        try:
            run = run_colbert if model in COLBERT else run_st
            r = run(model, ids, texts, questions)
            results[model] = {k: v for k, v in r.items() if k != "rankings"}
            Path(f"experiments/models/results/rankings_{model.split('/')[-1]}.json").write_text(
                json.dumps(r["rankings"]))
            print(f"{model}: {results[model]}")
        except Exception as exc:  # noqa: BLE001 — record and continue to next candidate
            results[model] = {"error": str(exc)[:300]}
            print(f"{model}: FAILED {exc}")
            gc.collect()
            torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    ok = {m: r for m, r in results.items() if "recall@10" in r}
    if ok:
        winner = max(ok, key=lambda m: (ok[m]["recall@10"], ok[m]["mrr@10"]))
        print("WINNER:", winner)


if __name__ == "__main__":
    main()
