"""One-shot artifact build: graph.json + baseline chunks + FAISS index. Idempotent."""
import json
import logging
from dataclasses import asdict

from config import settings
from graph.loader import save_graph
from ingestion.chunker import chunk_sections
from ingestion.graph_builder import build_graph
from ingestion.parser import parse_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_graph_artifact() -> None:
    graph_path = settings.artifacts_dir / "graph.json"
    if graph_path.exists():
        logger.info("graph.json exists, skipping")
        return
    save_graph(build_graph(settings.data_dir), graph_path)
    logger.info("wrote %s", graph_path)


def build_baseline_artifacts() -> None:
    chunks_path = settings.artifacts_dir / "chunks_baseline.json"
    index_dir = settings.artifacts_dir / "index"
    if chunks_path.exists() and (index_dir / "chunks.faiss").exists():
        logger.info("baseline chunks + index exist, skipping")
        return
    from retrieval.embedder import Embedder, VectorIndex
    parsed, docs = parse_all(settings.data_dir)
    # section-aware chunking — winner of experiments/archive/baseline_chunking.py
    # (recall@10 0.484 vs 0.469 fixed-512/256, 0.369 fixed-256/64)
    chunks = [
        c for doc_id, sections in parsed.items()
        for c in chunk_sections(sections, docs[doc_id])
    ]
    chunks_path.write_text(json.dumps([asdict(c) for c in chunks], ensure_ascii=False))
    logger.info("wrote %d chunks (%s)", len(chunks), settings.embedder_model)
    index = VectorIndex.build(
        [c.id for c in chunks], [c.text for c in chunks], Embedder(settings.embedder_model)
    )
    index.save(index_dir)
    logger.info("wrote %s", index_dir)


if __name__ == "__main__":
    build_graph_artifact()
    build_baseline_artifacts()
