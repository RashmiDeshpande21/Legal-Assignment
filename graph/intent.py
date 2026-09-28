"""Question -> graph seed nodes: section refs, defined-term match, dense retrieval."""
from __future__ import annotations

import re

import networkx as nx

from graph.traversal import amends_targets

# "§7.10", "Sec. 7.10" and "Section 7.10" are the same citation. Matching only the
# spelled-out word meant a question written the way a lawyer writes it got no section
# seed at all, which is how "§7.10" retrieved none of §7.10(a)-(c).
SEC_IN_Q_RE = re.compile(
    r"(?:[Ss]ections?|[Ss]ec\.?|\u00a7{1,2})\s*(\d{1,2}\.\d{1,2}(?:\([a-z0-9]+\))*)",
)
CHUNK_SUFFIX_RE = re.compile(r"::p\d+$")
# Curly or straight quotes around a term name, as questions cite defined terms.
QUOTED_TERM_RE = re.compile(r"[\u201c\"]\s*([^\u201d\"\n]{2,60}?)\s*[\u201d\"]")

# Ultra-generic defined terms: substring match fires on almost every natural-language
# question ("the agreement", "the Borrower") and hop-expands into the whole corpus.
_GENERIC_DEF_TERMS = {
    "agreement", "borrower", "lender", "person", "agent", "parent", "subsidiary",
    "section", "date", "code", "board", "officer", "affiliate",
}
# Qualifiers that carry no topic on their own: a sub-phrase made only of these would
# seed every "Consolidated ..." / "Permitted ..." term in §1.01 at once.
_WEAK_TOKENS = frozenset({
    "consolidated", "permitted", "applicable", "total", "net", "gross", "other",
    "related", "loan", "credit", "first", "second", "third", "new", "adjusted",
    "aggregate", "eligible", "excluded", "specified", "required", "restricted",
})


def _term_subphrases(term: str) -> list[str]:
    """Contiguous multi-word sub-phrases of a defined term, longest first.

    Questions cite defined terms colloquially — "the leverage ratio" for
    "Consolidated Leverage Ratio" — so exact substring matching alone seeds nothing
    and the graph never gets an entry point. Sub-phrases keep two or more words and
    at least one topical word, so this stays a phrase match rather than a keyword
    search: "leverage ratio" resolves, bare "consolidated" does not.
    """
    toks = term.lower().split()
    if len(toks) < 3:  # 2-word terms would degrade to single keywords
        return []
    out = []
    for n in range(len(toks) - 1, 1, -1):
        for i in range(len(toks) - n + 1):
            window = toks[i:i + n]
            if any(t not in _WEAK_TOKENS for t in window):
                out.append(" ".join(window))
    return out


def _section_seed(G: nx.MultiDiGraph, number: str) -> list[str]:
    """Resolve a cited section number to a node, falling back to its nearest ancestor.

    A question can cite deeper than the graph models a provision ("§2.15(a)(iv)" where
    the node is §2.15(a)); dropping trailing clause groups keeps the citation usable
    instead of silently yielding no seed.
    """
    while number:
        nid = f"base::{number}"
        if G.has_node(nid):
            return [nid]
        if "(" not in number:
            return []
        number = number.rsplit("(", 1)[0]
    return []


def _dense_seeds(G: nx.MultiDiGraph, question: str, index, embedder, k: int) -> list[str]:
    """Top-k graph nodes by dense retrieval over the chunk index (chunk id -> node id).

    Experiment-backed (experiments/hybrid_seeding.py): keyword-only entry points
    miss unseen phrasings; dense hits close that recall gap while DFS keeps precision.
    """
    hits = index.search(question, embedder, k=3 * k)  # over-fetch: chunks collapse to nodes
    out, seen = [], set()
    for cid, _, _ in hits:
        nid = CHUNK_SUFFIX_RE.sub("", cid)
        if nid not in seen and G.has_node(nid):
            out.append(nid)
            seen.add(nid)
        if len(out) >= k:
            break
    return out


def find_seeds(
    G: nx.MultiDiGraph, question: str, index=None, embedder=None, dense_k: int = 5,
) -> list[str]:
    """Seed the graph from the question string only — no per-question answer keys.

    Sources (in order): explicit § references in the question, defined-term substring
    matches, optional dense retrieval. Topic gazetteers were removed so seeding is not
    a hand-tuned map from assignment wording to gold nodes.
    """
    q = question.lower()
    seeds: list[str] = []
    for m in SEC_IN_Q_RE.finditer(question):
        seeds.extend(_section_seed(G, m.group(1)))
    # A term the question puts in quotes is being asked about by name, so it outranks
    # the generic-term guard below: 'the defined term "Code"' must seed base::def::Code
    # even though bare "code" is far too common to match on.
    quoted = {t.lower() for t in QUOTED_TERM_RE.findall(question)}
    terms = [
        (n, d["term"]) for n, d in G.nodes(data=True)
        if d["node_type"] == "Definition" and len(d["term"]) > 3
    ]
    for nid, term in sorted(terms, key=lambda x: -len(x[1])):
        low = term.lower()
        if low in quoted:
            seeds.append(nid)
            continue
        if low in _GENERIC_DEF_TERMS:
            continue
        if low in q or any(p in q for p in _term_subphrases(term)):
            seeds.append(nid)
    if index is not None and embedder is not None:
        seeds.extend(_dense_seeds(G, question, index, embedder, dense_k))
    # An amendment clause is only ever a route to the provision it changes: follow it
    # through so a question about a section the amendments *added* reaches the base node
    # that carries the effective text (see amends_targets).
    for nid in list(seeds):
        if G.nodes[nid].get("document_id", "").startswith("amendment"):
            seeds.extend(amends_targets(G, nid))
    out, seen = [], set()
    for s in seeds:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out
