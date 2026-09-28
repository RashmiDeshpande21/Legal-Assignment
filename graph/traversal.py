"""Graph traversal: as-of-date version resolution, DFS context collection, citations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import networkx as nx

from graph.schema import (
    EDGE_AMENDS,
    EDGE_CONTAINS,
    EDGE_DEPENDS_ON,
    EDGE_EXCEPTION_TO,
    EDGE_PARTY_TO,
    EDGE_REFERENCES,
)

FOLLOW_EDGES = (EDGE_REFERENCES, EDGE_EXCEPTION_TO, EDGE_DEPENDS_ON, EDGE_CONTAINS)


def edge_kind(key: str, data: dict | None = None) -> str:
    """Edge type for a MultiDiGraph edge, from the ``edge_type`` attribute.

    Keys carry a discriminator suffix (``AMENDS::amendment_2::1.5``) so parallel
    edges between the same pair stay distinct; the prefix is the fallback for
    graph.json artifacts built before ``edge_type`` was written on every edge.
    """
    if data:
        kind = data.get("edge_type")
        if kind:
            return kind
    return key.split("::")[0]


@dataclass
class Effective:
    node_id: str
    text: str
    citation: str               # "Credit Agreement §7.10(a), as amended by Second Amendment"
    status: str                 # "original" | "amended" | "added"
    source: str                 # "html" | "ocr"


def _amend_ops_before(G: nx.MultiDiGraph, node_id: str, as_of: date | None) -> list[dict]:
    ops = [
        d for _, _, d in G.in_edges(node_id, data=True)
        if d.get("edge_type") == EDGE_AMENDS
        and (as_of is None or date.fromisoformat(d["effective_date"]) <= as_of)
    ]
    return sorted(ops, key=lambda d: d["effective_date"])


def node_label(G: nx.MultiDiGraph, node_id: str) -> str:
    n = G.nodes[node_id]
    if n["node_type"] == "Definition":
        return f"definition of \u201c{n['term']}\u201d (\u00a71.01)"
    num = n["number"]
    return num if num.startswith("Exhibit") else f"\u00a7{num}"


def find_effective(G: nx.MultiDiGraph, node_id: str, as_of: date | None) -> Effective:
    """Operative text + citation for a Section/Definition node as of a date.

    restate: amended text replaces the original *when it is a real restatement*.
    Short / instruction-only "restates" (common in OCR of proviso surgeries —
    "by replacing the text in the proviso…") are treated like replace/delete:
    keep the base body and annotate the change. Otherwise a later amendment
    with as_of=None wipes clause-level carve-outs the model still needs.
    insert: appended. replace/delete: original kept with an explicit annotation.
    """
    n = G.nodes[node_id]
    text, status = n.get("text", ""), "original"
    notes = []
    for op in _amend_ops_before(G, node_id, as_of):
        stamp = f"{op['instrument']} (effective {op['effective_date']})"
        new = op.get("amended_text") or ""
        if op["amendment_type"] == "restate" and not _surgical_restate(text, new):
            text, status = new, "amended"
        elif op["amendment_type"] == "insert":
            text = f"{text}\n{new}" if text else new
            status = "amended" if n.get("text") else "added"
        else:
            # replace / delete / surgical "restate"
            kind = op["amendment_type"]
            if kind == "restate":
                kind = "proviso/restate"
            notes.append(f"[{kind} per {stamp}: {new}]")
            status = "amended"
        if status != "original" and not notes:
            notes = [f"[as amended by {stamp}]"]
    # Keep a single short stamp when that is all we have; never drop detailed
    # surgical/replace notes that follow an earlier stamp.
    if (
        len(notes) > 1
        and notes[0].startswith("[as amended")
        and all(n.startswith("[as amended") for n in notes)
    ):
        notes = notes[:1]
    doc_id = n.get("document_id", "")
    doc = G.nodes[doc_id] if G.has_node(doc_id) else {}
    doc_name = "Credit Agreement" if doc_id == "base" else doc.get("name", doc_id)
    last_ops = _amend_ops_before(G, node_id, as_of)
    if last_ops:
        last = last_ops[-1]
        suffix = f"as amended by {last['instrument']} (effective {last['effective_date']})"
    elif doc_id == "base":
        suffix = "as originally executed (2017-10-26)"
    else:
        suffix = f"dated {doc.get('date', '')}"
    citation = f"{doc_name} {node_label(G, node_id)}, {suffix}"
    if notes:
        text = text + "\n" + "\n".join(notes)
    return Effective(
        node_id=node_id, text=text, citation=citation, status=status,
        source=n.get("source", "html"),
    )


def _surgical_restate(current: str, new: str) -> bool:
    """True when 'restate' text is an amendment instruction, not a full substitute body."""
    if not (new or "").strip():
        return True
    if not (current or "").strip():
        return False
    head = new.lower()[:240]
    looks_instruction = any(
        m in head
        for m in (
            "by replacing", "by deleting", "by inserting", "is hereby amended",
            "amended by replacing", "replacing the text in the proviso",
        )
    )
    # Much shorter than the operative body → cannot be a full restatement.
    if len(new) < max(400, int(len(current) * 0.4)):
        return True
    return bool(looks_instruction and len(new) < len(current))


def dfs_context(
    G: nx.MultiDiGraph, start_ids: list[str], as_of: date | None,
    max_depth: int = 3, max_nodes: int = 25,
) -> list[Effective]:
    """Depth-limited DFS over legal edges, resolving each node as of the date."""
    seen: dict[str, int] = {}
    stack = [(nid, 0) for nid in reversed(start_ids)]
    order: list[str] = []
    while stack and len(order) < max_nodes:
        nid, depth = stack.pop()
        if nid in seen or not G.has_node(nid):
            continue
        seen[nid] = depth
        if G.nodes[nid]["node_type"] in ("Section", "Definition"):
            order.append(nid)
        if depth < max_depth:
            nbrs = [
                v for _, v, k, d in G.out_edges(nid, keys=True, data=True)
                if edge_kind(k, d) in FOLLOW_EDGES
            ]
            exceptions = [u for u, _, k in G.in_edges(nid, keys=True) if k == EDGE_EXCEPTION_TO]
            stack.extend((v, depth + 1) for v in reversed(exceptions + nbrs) if v not in seen)
    return [find_effective(G, nid, as_of) for nid in order]


def amendment_catalog(G: nx.MultiDiGraph) -> list[dict]:
    """Every AMENDS edge: target, instrument, date, type — the Q11 answer."""
    rows = [
        {
            "target": node_label(G, v),
            "target_id": v,
            "instrument": d["instrument"],
            "effective_date": d["effective_date"],
            "amendment_type": d["amendment_type"],
        }
        for _, v, d in G.edges(data=True)
        if d.get("edge_type") == EDGE_AMENDS
    ]
    return sorted(rows, key=lambda r: (r["effective_date"], r["target"]))


def hop_neighbors(
    G: nx.MultiDiGraph,
    nid: str,
    edge_types: tuple[str, ...] = (EDGE_DEPENDS_ON, EDGE_REFERENCES),
    limit: int | None = 8,
) -> list[str]:
    """1-hop Section/Definition neighbors in either direction over legal edges.

    DFS follows outbound edges only, so inbound dependents / reverse refs are
    invisible unless explicitly pulled in — this is the shared helper for that.
    """
    if not G.has_node(nid):
        return []
    nbrs: set[str] = set()
    edges = (
        list(G.out_edges(nid, keys=True, data=True))
        + list(G.in_edges(nid, keys=True, data=True))
    )
    for u, v, k, d in edges:
        if edge_kind(k, d) not in edge_types:
            continue
        other = v if u == nid else u
        if G.nodes[other].get("node_type") in ("Section", "Definition"):
            nbrs.add(other)
    out = sorted(nbrs)
    return out if limit is None else out[:limit]


def sibling_neighbors(
    G: nx.MultiDiGraph, nid: str, limit: int | None = 6,
) -> list[str]:
    """Sections sharing a CONTAINS parent with ``nid`` (itself excluded).

    DFS follows CONTAINS downward only, so from §8.01 it descends into §8.01(a)–(l)
    and can never reach §8.02 — the remedies section that answers "and then what
    happens". Article/section parents have few children (art8 -> 4, §7.10 -> 2), so
    one sibling hop is cheap and fixes the whole adjacent-provision class.
    """
    if not G.has_node(nid):
        return []
    parents = [
        u for u, _, k, d in G.in_edges(nid, keys=True, data=True)
        if edge_kind(k, d) == EDGE_CONTAINS
    ]
    sibs: set[str] = set()
    for p in parents:
        for _, v, k, d in G.out_edges(p, keys=True, data=True):
            if edge_kind(k, d) != EDGE_CONTAINS or v == nid:
                continue
            if G.nodes[v].get("node_type") == "Section":
                sibs.add(v)
    out = sorted(sibs)
    return out if limit is None else out[:limit]


def inbound_dependents(G: nx.MultiDiGraph, term_nid: str) -> list[str]:
    """Provisions that REFERENCE or DEPEND_ON a definition — the Q4-shaped answer."""
    if not G.has_node(term_nid):
        return []
    return sorted({
        u for u, _, k, d in G.in_edges(term_nid, keys=True, data=True)
        if edge_kind(k, d) in (EDGE_REFERENCES, EDGE_DEPENDS_ON)
    })


def amends_targets(G: nx.MultiDiGraph, section_id: str) -> list[str]:
    """Base provisions that an amendment's operative clause acts on.

    AMENDS edges hang off the amendment *document*, with the operative clause recorded
    as ``source_sec_id``, so there is no path from that clause to what it changes.
    It matters for retrieval because a section an amendment *adds* holds no text of its
    own in the base document — §1.08 and §10.23 are empty until the as-of resolver
    applies the amendment — so dense retrieval lands on the amendment clause
    (amendment_2::1.10) and the base node it created is never reached.
    """
    return sorted({
        v for _, v, k, d in G.edges(keys=True, data=True)
        if edge_kind(k, d) == EDGE_AMENDS and d.get("source_sec_id") == section_id
    })


def definition_closure(
    G: nx.MultiDiGraph, term_nid: str, depth: int = 1,
) -> list[str]:
    """Terms a definition is built out of, following DEPENDS_ON outward.

    A defined term cannot be read without the terms it incorporates: "Consolidated
    Leverage Ratio" *is* Consolidated Funded Indebtedness over Consolidated EBITDA.
    This is a structural relation, not a textual one, so a relevance reranker cannot
    recover it — asked whether the leverage ratio changed, the cross-encoder ranked
    "Consolidated EBITDA" 12th of 14 candidates because that definition never says
    "leverage ratio". Closures are small (median 1 term, max 12 at depth 1), so the
    read-through is included structurally instead of competing for a ranked slot.
    """
    if not G.has_node(term_nid):
        return []
    seen: set[str] = set()
    frontier = [term_nid]
    for _ in range(max(0, depth)):
        nxt = []
        for cur in frontier:
            for _, v, k, d in G.out_edges(cur, keys=True, data=True):
                if edge_kind(k, d) == EDGE_DEPENDS_ON and v != term_nid and v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return sorted(seen)


def party_roster(G: nx.MultiDiGraph) -> list[dict]:
    """Every PARTY_TO edge: party name, role, document — the Q3-shaped answer.

    Party nodes carry no text, so DFS (Section/Definition only) and the chunk index
    never surface them; the roles live on the edges. Same projection pattern as
    ``amendment_catalog``: structured graph facts the prose does not enumerate.
    """
    rows = [
        {
            "node_id": u,
            "name": G.nodes[u].get("name", u),
            "role": d.get("role", ""),
            "document_id": v,
        }
        for u, v, _, d in G.edges(keys=True, data=True)
        if d.get("edge_type") == EDGE_PARTY_TO
    ]
    return sorted(rows, key=lambda r: (r["document_id"], r["name"], r["role"]))
