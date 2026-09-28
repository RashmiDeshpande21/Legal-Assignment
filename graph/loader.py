"""Serialize the graph to artifacts/graph.json and load it back."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def save_graph(G: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G, edges="edges")
    path.write_text(json.dumps(data, ensure_ascii=False))


def load_graph(path: Path) -> nx.MultiDiGraph:
    data = json.loads(path.read_text())
    return nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")
