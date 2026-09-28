"""Measured cost metrics: indexing cost per 100 pages + query latency.

Times a clean artifact build in a temp dir (real artifacts untouched), then summarizes
per-question latency from outputs/answers.json (written by eval.run_all).
Run: uv run python -m eval.metrics
"""
from __future__ import annotations

import json
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

# One A10G suffices for indexing; on-demand g5.xlarge us-west-2 (measured basis, not estimate
# of runtime — runtime is measured, this converts seconds to dollars).
GPU_USD_PER_HOUR = 1.006
WORDS_PER_PAGE = 500  # standard legal page convention


def measure_indexing(tmp: Path) -> dict:
    from config import settings
    from graph.loader import save_graph
    from ingestion.chunker import chunk_fixed
    from ingestion.graph_builder import build_graph
    from ingestion.parser import parse_all
    from retrieval.embedder import Embedder, VectorIndex

    t0 = time.perf_counter()
    G = build_graph(settings.data_dir)
    save_graph(G, tmp / "graph.json")
    t_graph = time.perf_counter() - t0

    t0 = time.perf_counter()
    parsed, docs = parse_all(settings.data_dir)
    chunks = [c for doc_id, secs in parsed.items() for c in chunk_fixed(secs, docs[doc_id])]
    (tmp / "chunks_baseline.json").write_text(json.dumps([asdict(c) for c in chunks]))
    index = VectorIndex.build([c.id for c in chunks], [c.text for c in chunks],
                              Embedder(settings.embedder_model))
    index.save(tmp / "index")
    t_index = time.perf_counter() - t0

    words = sum(len(s.text.split()) for secs in parsed.values() for s in secs)
    pages = words/WORDS_PER_PAGE
    total = t_graph + t_index
    return {
        "corpus_words": words,
        "corpus_pages": round(pages, 1),
        "words_per_page_convention": WORDS_PER_PAGE,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "baseline_chunks": len(chunks),
        "graph_build_s": round(t_graph, 2),
        "chunk_embed_index_s": round(t_index, 2),
        "total_s": round(total, 2),
        "s_per_100_pages": round(total / pages * 100, 2),
        "usd_per_100_pages": round(total / pages * 100 * GPU_USD_PER_HOUR/3600, 5),
        "gpu_usd_per_hour_basis": GPU_USD_PER_HOUR,
        "note": "graph build is rule-based (CPU, no LLM calls); GPU time is chunk embedding",
    }


def summarize_latency(answers_path: Path) -> dict:
    if not answers_path.exists():
        return {"note": "outputs/answers.json missing — run eval.run_all first"}
    results = json.loads(answers_path.read_text())
    out = {}
    for path_name, rows in results.items():
        if path_name == "run" or not rows:  # provenance block, not answer rows
            continue
        lat = [r["latency_s"] for r in rows]
        out[path_name] = {
            "n": len(lat),
            "mean_s": round(statistics.mean(lat), 2),
            "median_s": round(statistics.median(lat), 2),
            "min_s": round(min(lat), 2),
            "max_s": round(max(lat), 2),
        }
    return out


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="conqr_metrics_") as tmp:
        indexing = measure_indexing(Path(tmp))
    metrics = {"indexing": indexing,
               "query_latency": summarize_latency(Path("outputs/answers.json"))}
    Path("outputs/metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
