"""Run all 12 questions through both paths; write outputs/*.md + answers.json (for eval).

PATHS=graph limits the run to the graph path (halves wall time while iterating).
RUN_TAG=<label> also archives the answers under outputs/runs/<timestamp>_<tag>/ so a
scorecard can always be traced back to the config that produced it.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime

from baseline.rag import baseline_answer, load_chunks
from config import settings
from eval.questions import QUESTIONS
from graph.loader import load_graph
from llm.client import answer_llm
from retrieval.embedder import Embedder, VectorIndex
from retrieval.pipeline import graph_answer
from retrieval.reranker import Reranker

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def run_provenance(tag: str = "") -> dict:
    """Config that decides answer content — embedded in every result file.

    Without this, two answers.json files with different scores are indistinguishable
    after the fact, which is exactly how the 11/12 evidence was lost to an overwrite.
    """
    return {
        "tag": tag,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "model_path": str(settings.model_path),
        "embedder_model": settings.embedder_model,
        "reranker_model": settings.reranker_model,
        "llm_context_length": settings.llm_context_length,
        "llm_max_tokens": settings.llm_max_tokens,
        "llm_enable_thinking": settings.llm_enable_thinking,
        "llm_think_budget": settings.llm_think_budget,
        "llm_repeat_penalty": settings.llm_repeat_penalty,
        "expand_stage2_top_n": settings.expand_stage2_top_n,
        "expand_stage2_keep": settings.expand_stage2_keep,
        "definition_closure_depth": settings.definition_closure_depth,
    }


def main() -> None:
    G = load_graph(settings.artifacts_dir / "graph.json")
    index = VectorIndex.load(settings.artifacts_dir / "index")
    chunks = load_chunks(settings.artifacts_dir / "chunks_baseline.json")
    embedder = Embedder(settings.embedder_model)
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)
    llm = answer_llm()

    wanted = {p for p in os.environ.get("PATHS", "graph,baseline").split(",") if p}
    provenance = run_provenance(os.environ.get("RUN_TAG", ""))
    results: dict[str, list] = {"graph": [], "baseline": []}
    for q in QUESTIONS:
        as_of_note = f" (as of {q.as_of})" if q.as_of else ""
        q_text = q.retrieval_text or q.text
        timings = {}
        if "graph" in wanted:
            t0 = time.time()
            g = graph_answer(G, q_text, q.as_of, reranker, llm,
                             index=index, embedder=embedder)
            timings["graph"] = time.time() - t0
            results["graph"].append({
                "id": q.id, "question": q.text,
                "as_of": str(q.as_of) if q.as_of else None,
                "answer": g.text, "citations": g.citations, "context_ids": g.context_ids,
                "context_blocks": [{"citation": c, "text": t} for c, t, _ in g.blocks],
                "latency_s": round(timings["graph"], 2),
            })
        if "baseline" in wanted:
            t0 = time.time()
            b = baseline_answer(q_text, q.as_of, index, chunks, embedder, reranker, llm)
            timings["baseline"] = time.time() - t0
            results["baseline"].append({
                "id": q.id, "question": q.text,
                "as_of": str(q.as_of) if q.as_of else None,
                "answer": b.text, "citations": b.citations, "context_ids": b.context_ids,
                "context_blocks": [{"citation": c, "text": t} for c, t, _ in b.blocks],
                "latency_s": round(timings["baseline"], 2),
            })
        lat = " | ".join(f"{k} {v:.1f}s" for k, v in timings.items())
        print(f"{q.id}{as_of_note}: {lat}", flush=True)

    settings.outputs_dir.mkdir(exist_ok=True)
    payload = {**results, "run": provenance}
    (settings.outputs_dir / "answers.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False)
    )
    if provenance["tag"]:
        stamp = provenance["timestamp"].replace(":", "").replace("-", "")
        archive = settings.outputs_dir / "runs" / f"{stamp}_{provenance['tag']}"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "answers.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False)
        )
        print(f"archived {archive}")
    for path_name, rows in results.items():
        if not rows:
            continue
        lines = [f"# {path_name.title()} path — all 12 questions",
                 f"\nModels: {settings.model_path.name} | {settings.embedder_model} | "
                 f"{settings.reranker_model}\n"]
        for r in rows:
            as_of = f"  \n*As of: {r['as_of']}*" if r["as_of"] else ""
            lines.append(f"\n## {r['id'].upper()}: {r['question']}{as_of}\n")
            lines.append(r["answer"])
            lines.append(f"\n*Latency: {r['latency_s']}s. Context provided:*")
            lines.extend(f"- {c}" for c in r["citations"])
        (settings.outputs_dir/f"{path_name}_answers.md").write_text("\n".join(lines))
    print("wrote outputs/answers.json, graph_answers.md, baseline_answers.md")
    print("eval/frozen/ is the submission snapshot and was not modified")


if __name__ == "__main__":
    main()
