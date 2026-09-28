"""Schema progression experiment: v1 (minimal) -> v5 (full).

Each version allows a subset of edge types. Per question we measure:
- node_recall: fraction of gold nodes reachable from the entry nodes (undirected BFS,
  depth 3) using only that version's edges
- edge_recall: fraction of asserted semantic edges present in the version
- version_ok: as-of version checks pass (find_effective returns expected status)

score = mean of available components. Proves which questions each edge type unlocks.
CPU-only: isolates graph *structure* from model quality.
"""
from __future__ import annotations

import json
from collections import deque
from datetime import date
from pathlib import Path

import networkx as nx

from graph.loader import load_graph
from graph.traversal import find_effective

VERSIONS = {
    "v1_minimal": {"CONTAINS", "DEFINES"},
    "v2_amendments": {"CONTAINS", "DEFINES", "AMENDS"},
    "v3_references": {"CONTAINS", "DEFINES", "AMENDS", "REFERENCES", "PARTY_TO"},
    "v4_exceptions": {"CONTAINS", "DEFINES", "AMENDS", "REFERENCES", "PARTY_TO", "EXCEPTION_TO"},
    "v5_full": {"CONTAINS", "DEFINES", "AMENDS", "REFERENCES", "PARTY_TO", "EXCEPTION_TO",
                "DEPENDS_ON"},
}


def edge_type(k: str, d: dict) -> str:
    return d.get("edge_type") or k.split("::")[0]


def subgraph_for(G: nx.MultiDiGraph, allowed: set[str]) -> nx.MultiDiGraph:
    S = nx.MultiDiGraph()
    S.add_nodes_from(G.nodes(data=True))
    S.add_edges_from(
        (u, v, k, d) for u, v, k, d in G.edges(keys=True, data=True)
        if edge_type(k, d) in allowed
    )
    return S


def reachable(S: nx.MultiDiGraph, entries: list[str], depth: int = 3) -> set[str]:
    U = S.to_undirected(as_view=True)
    seen = set(entries)
    frontier = deque((e, 0) for e in entries)
    while frontier:
        n, d = frontier.popleft()
        if d >= depth:
            continue
        for nbr in U.neighbors(n):
            if nbr not in seen:
                seen.add(nbr)
                frontier.append((nbr, d + 1))
    return seen


def score_question(S: nx.MultiDiGraph, q: dict) -> dict:
    parts = {}
    if q["gold"]:
        got = reachable(S, q["entry"])
        parts["node_recall"] = sum(g in got for g in q["gold"]) / len(q["gold"])
    if q["edge_asserts"]:
        present = sum(
            S.has_edge(u, v)
            and any(edge_type(k, d) == et for k, d in S.get_edge_data(u, v).items())
            for u, et, v in q["edge_asserts"]
        )
        parts["edge_recall"] = present / len(q["edge_asserts"])
    if q["version_checks"]:
        ok = sum(
            find_effective(S, c["node"], date.fromisoformat(c["as_of"])).status == c["status"]
            for c in q["version_checks"]
        )
        parts["version_ok"] = ok / len(q["version_checks"])
    parts["score"] = round(sum(parts.values()) / len(parts), 3) if parts else 0.0
    return parts


def main() -> None:
    G = load_graph(Path("artifacts/graph.json"))
    questions = json.loads(Path("experiments/mini_eval_set.json").read_text())
    results: dict[str, dict] = {}
    for vname, allowed in VERSIONS.items():
        S = subgraph_for(G, allowed)
        results[vname] = {q["id"]: score_question(S, q) for q in questions}
    out = Path("experiments/schema/results/schema_progression.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"{'':14}" + "".join(f"{q['id']:>6}" for q in questions))
    for vname, scores in results.items():
        print(f"{vname:14}" + "".join(f"{scores[q['id']]['score']:>6.2f}" for q in questions))


if __name__ == "__main__":
    main()
