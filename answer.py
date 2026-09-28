"""Runner: python answer.py --question "..." --as-of 2020-06-01 [--path graph|baseline|both]"""
from datetime import date

import click
from rich.console import Console
from rich.panel import Panel

from config import settings

console = Console()

@click.command()
@click.option("--question", required=True)
@click.option("--as-of", "as_of", default=None, help="ISO date, e.g. 2020-06-01")
@click.option("--path", "path_", default="graph",
              type=click.Choice(["graph", "baseline", "both"]))
def main(question: str, as_of: str | None, path_: str) -> None:
    from graph.loader import load_graph
    from llm.client import answer_llm
    from retrieval.embedder import Embedder, VectorIndex
    from retrieval.reranker import Reranker

    as_of_date = date.fromisoformat(as_of) if as_of else None
    G = load_graph(settings.artifacts_dir / "graph.json")
    reranker = Reranker(settings.reranker_model, settings.reranker_batch_size)
    index = VectorIndex.load(settings.artifacts_dir / "index")
    embedder = Embedder(settings.embedder_model)
    llm = answer_llm()

    if path_ in ("graph", "both"):
        from retrieval.pipeline import graph_answer
        ans = graph_answer(G, question, as_of_date, reranker, llm,
                           index=index, embedder=embedder)
        console.print(Panel(ans.text, title=f"graph path · as-of {as_of or 'latest'}"))
    if path_ in ("baseline", "both"):
        from baseline.rag import baseline_answer, load_chunks
        chunks = load_chunks(settings.artifacts_dir / "chunks_baseline.json")
        ans = baseline_answer(question, as_of_date, index, chunks, embedder, reranker, llm)
        console.print(Panel(ans.text, title="baseline (flat RAG)"))


if __name__ == "__main__":
    main()
