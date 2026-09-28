"""Graph visualizations -> artifacts/viz/. Static PNGs + interactive pyvis HTML."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

NODE_COLORS = {"Document": "#e63946", "Section": "#457b9d", "Definition": "#2a9d8f",
               "Party": "#e9c46a"}
EDGE_COLORS = {"CONTAINS": "#cccccc", "DEFINES": "#2a9d8f", "AMENDS": "#e63946",
               "REFERENCES": "#a8dadc", "PARTY_TO": "#e9c46a", "EXCEPTION_TO": "#f4a261",
               "DEPENDS_ON": "#9b5de5"}


def _etype(k: str, d: dict) -> str:
    return d.get("edge_type") or k.split("::")[0]


def _short(G: nx.MultiDiGraph, n: str) -> str:
    nd = G.nodes[n]
    if nd["node_type"] == "Definition":
        return nd["term"]
    if nd["node_type"] == "Document":
        return {"base": "Base\nAgreement", "amendment_1": "First\nAmendment",
                "amendment_2": "Second\nAmendment"}.get(n, n)
    if nd["node_type"] == "Party":
        return nd["name"].split(",")[0].title()
    return nd["number"]


def draw_schema_view(G: nx.MultiDiGraph, out: Path) -> None:
    """Meta-graph: node types as nodes, edge types as labelled edges with counts."""
    from collections import Counter
    counts = Counter(
        (G.nodes[u]["node_type"], _etype(k, d), G.nodes[v]["node_type"])
        for u, v, k, d in G.edges(keys=True, data=True)
    )
    M = nx.MultiDiGraph()
    node_n = Counter(d["node_type"] for _, d in G.nodes(data=True))
    for nt, n in node_n.items():
        M.add_node(nt, label=f"{nt}\n({n})")
    pos = {"Document": (0, 1), "Section": (1.4, 1), "Definition": (1.4, -0.2),
           "Party": (0, -0.2)}
    fig, ax = plt.subplots(figsize=(11, 7))
    nx.draw_networkx_nodes(M, pos, node_size=7000,
                           node_color=[NODE_COLORS[n] for n in M.nodes], alpha=0.9, ax=ax)
    nx.draw_networkx_labels(M, pos, {n: M.nodes[n]["label"] for n in M.nodes},
                            font_size=11, font_weight="bold", ax=ax)
    for i, ((src, et, dst), n) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
        rad = 0.12 + 0.13 * (i % 5)
        ax.annotate("", xy=pos[dst], xytext=pos[src],
                    arrowprops=dict(arrowstyle="-|>", color=EDGE_COLORS[et], lw=1.8,
                                    connectionstyle=f"arc3,rad={rad}", shrinkA=62, shrinkB=62))
        mx = (pos[src][0] + pos[dst][0]) / 2 + rad * (pos[dst][1] - pos[src][1]) * 0.9
        my = (pos[src][1] + pos[dst][1]) / 2 + rad * (pos[src][0] - pos[dst][0]) * 0.9
        ax.text(mx, my, f"{et} ({n})", fontsize=8.5, color=EDGE_COLORS[et],
                fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_title("Schema view: node/edge types with instance counts", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def draw_amendment_map(G: nx.MultiDiGraph, out: Path) -> None:
    """Which amendment touches which target — the Q11 picture."""
    edges = [(u, v, d) for u, v, k, d in G.edges(keys=True, data=True)
             if d.get("edge_type") == "AMENDS"]
    targets = sorted({v for _, v, _ in edges}, key=lambda v: G.nodes[v].get("number", "zz"))
    fig, ax = plt.subplots(figsize=(11, 12))
    pos = {"amendment_1": (0, len(targets) * 0.72), "amendment_2": (0, len(targets) * 0.28)}
    pos |= {t: (1, len(targets) - 1 - i) for i, t in enumerate(targets)}
    tcolors = [NODE_COLORS[G.nodes[t]["node_type"]] for t in targets]
    nx.draw_networkx_nodes(G, pos, nodelist=["amendment_1", "amendment_2"], node_size=5200,
                           node_color=NODE_COLORS["Document"], node_shape="s", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=targets, node_size=300, node_color=tcolors, ax=ax)
    style = {"restate": "-", "insert": "--", "replace": ":", "delete": ":"}
    for u, v, d in edges:
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, shrinkA=45, shrinkB=12,
                                    linestyle=style[d["amendment_type"]],
                                    color="#e63946" if u == "amendment_2" else "#f77f00"))
    nx.draw_networkx_labels(G, pos, {n: _short(G, n) for n in ["amendment_1", "amendment_2"]},
                            font_size=10, font_weight="bold", ax=ax)
    for t in targets:
        ax.text(pos[t][0] + 0.04, pos[t][1], _short(G, t), fontsize=9, va="center")
    ax.set_title("AMENDS edges: 4 ops (First Amendment, orange) + 22 ops (Second, red)\n"
                 "solid=restate  dashed=insert  dotted=replace/delete", fontsize=12)
    ax.set_xlim(-0.35, 1.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def draw_q12_chain(G: nx.MultiDiGraph, out: Path) -> None:
    """Consolidated Leverage Ratio neighborhood — the Q12 dependency/impact picture."""
    clr = "base::def::Consolidated Leverage Ratio"
    deps = [v for _, v, k in G.out_edges(clr, keys=True) if k == "DEPENDS_ON"]
    users = [u for u, _, k in G.in_edges(clr, keys=True) if k == "REFERENCES"]
    amender = [u for u, _, _, d in G.in_edges(clr, keys=True, data=True)
               if d.get("edge_type") == "AMENDS"]
    deps2 = [v for d_ in deps for _, v, k in G.out_edges(d_, keys=True) if k == "DEPENDS_ON"][:6]
    nodes = {clr, *deps, *users, *amender, *deps2}
    S = G.subgraph(nodes)
    pos = nx.spring_layout(S, seed=7, k=1.6)
    pos[clr] = (0, 0)
    fig, ax = plt.subplots(figsize=(13, 9))
    colors = [NODE_COLORS[S.nodes[n]["node_type"]] for n in S.nodes]
    sizes = [2600 if n == clr else 1100 for n in S.nodes]
    nx.draw_networkx_nodes(S, pos, node_color=colors, node_size=sizes, alpha=0.92, ax=ax)
    for u, v, k, d in S.edges(keys=True, data=True):
        et = _etype(k, d)
        nx.draw_networkx_edges(S, pos, [(u, v)], edge_color=EDGE_COLORS[et], width=1.6,
                               connectionstyle="arc3,rad=0.08", ax=ax)
    nx.draw_networkx_labels(S, pos, {n: _short(S, n) for n in S.nodes}, font_size=8, ax=ax)
    handles = [plt.Line2D([0], [0], color=EDGE_COLORS[e], lw=2, label=e)
               for e in ("DEPENDS_ON", "REFERENCES", "AMENDS", "DEFINES")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_title("Q12: \u201cConsolidated Leverage Ratio\u201d — depends-on chain, using sections, "
                 "amended-by", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def draw_full_graph(G: nx.MultiDiGraph, out: Path) -> None:
    H = nx.Graph(G)
    pos = nx.spring_layout(H, seed=42, k=0.06, iterations=60)
    fig, ax = plt.subplots(figsize=(14, 12))
    colors = [NODE_COLORS[H.nodes[n]["node_type"]] for n in H.nodes]
    sizes = [400 if H.nodes[n]["node_type"] == "Document" else 14 for n in H.nodes]
    nx.draw_networkx_edges(H, pos, alpha=0.04, width=0.4, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=colors, node_size=sizes, alpha=0.8, ax=ax)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10,
                          label=f"{t}") for t, c in NODE_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=10)
    ax.set_title("Full graph: 680 nodes, 3339 edges", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def export_pyvis(G: nx.MultiDiGraph, out: Path) -> None:
    from pyvis.network import Network
    net = Network(height="900px", width="100%", directed=True, notebook=False,
                  cdn_resources="in_line")
    for n, d in G.nodes(data=True):
        net.add_node(n, label=_short(G, n), color=NODE_COLORS[d["node_type"]],
                     title=f"{d['node_type']}: {n}",
                     size=26 if d["node_type"] == "Document" else 8)
    for u, v, k, d in G.edges(keys=True, data=True):
        et = _etype(k, d)
        net.add_edge(u, v, color=EDGE_COLORS[et], title=et, width=0.6)
    net.write_html(str(out), open_browser=False)


def main() -> None:
    from graph.loader import load_graph
    G = load_graph(Path("artifacts/graph.json"))
    viz = Path("artifacts/viz")
    viz.mkdir(parents=True, exist_ok=True)
    draw_schema_view(G, viz / "01_schema_view.png")
    draw_amendment_map(G, viz / "02_amendment_map.png")
    draw_q12_chain(G, viz / "03_q12_leverage_chain.png")
    draw_full_graph(G, viz / "04_full_graph.png")
    export_pyvis(G, viz / "graph_interactive.html")
    print("wrote", sorted(p.name for p in viz.iterdir()))


if __name__ == "__main__":
    main()
