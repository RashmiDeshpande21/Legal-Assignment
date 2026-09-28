"""Assemble the NetworkX MultiDiGraph: 4 node types, 7 edge types."""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from graph.schema import (
    DOC_BASE,
    EDGE_AMENDS,
    EDGE_CONTAINS,
    EDGE_DEFINES,
    EDGE_DEPENDS_ON,
    EDGE_EXCEPTION_TO,
    EDGE_PARTY_TO,
    EDGE_REFERENCES,
    Document,
    Section,
)
from ingestion.definitions import extract_definitions
from ingestion.extractor import (
    extract_amend_ops,
    extract_exceptions,
    extract_parties,
    extract_references,
    extract_section_term_refs,
    extract_term_deps,
)
from ingestion.parser import parse_all, read_html

logger = logging.getLogger(__name__)


def def_node_id(doc_id: str, term: str) -> str:
    return f"{doc_id}::def::{term}"


def _add_sections(G: nx.MultiDiGraph, doc: Document, sections: list[Section]) -> None:
    G.add_node(doc.id, node_type="Document", **{**asdict(doc), "date": doc.date.isoformat()})
    cur_article = cur_section = None
    for sec in sections:
        G.add_node(sec.id, node_type="Section", **asdict(sec))
        if sec.level == "article":
            G.add_edge(doc.id, sec.id, key=EDGE_CONTAINS, edge_type=EDGE_CONTAINS)
            cur_article, cur_section = sec.id, None
        elif sec.level == "section":
            G.add_edge(cur_article or doc.id, sec.id, key=EDGE_CONTAINS,
                       edge_type=EDGE_CONTAINS)
            cur_section = sec.id
        else:
            G.add_edge(cur_section, sec.id, key=EDGE_CONTAINS, edge_type=EDGE_CONTAINS)


def build_graph(data_dir: Path) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    parsed, docs = parse_all(data_dir)
    for doc_id, sections in parsed.items():
        _add_sections(G, docs[doc_id], sections)

    base_secs = parsed[DOC_BASE]
    sec_by_num = {s.number: s.id for s in base_secs}

    defs = extract_definitions(read_html(data_dir, DOC_BASE), DOC_BASE, "base::1.01")
    for d in defs:
        nid = def_node_id(DOC_BASE, d.term)
        G.add_node(nid, node_type="Definition", **asdict(d))
        G.add_edge("base::1.01", nid, key=EDGE_DEFINES, edge_type=EDGE_DEFINES)

    preamble = next(s for s in base_secs if s.id == "base::preamble")
    for party, roles in extract_parties(preamble.text):
        pid = f"party::{party.name}"
        if not G.has_node(pid):
            G.add_node(pid, node_type="Party", **asdict(party))
        for role in roles:
            G.add_edge(pid, DOC_BASE, key=f"{EDGE_PARTY_TO}::{role}", edge_type=EDGE_PARTY_TO,
                       role=role)

    for src_id, target_num, evidence in extract_references(base_secs):
        G.add_edge(src_id, sec_by_num[target_num], key=EDGE_REFERENCES,
                   edge_type=EDGE_REFERENCES, evidence=evidence)

    for sub_id, parent_id, evidence in extract_exceptions(base_secs):
        G.add_edge(sub_id, parent_id, key=EDGE_EXCEPTION_TO,
                   edge_type=EDGE_EXCEPTION_TO, evidence=evidence)

    term_ids = {d.term: def_node_id(DOC_BASE, d.term) for d in defs}
    for term, dep in extract_term_deps(defs):
        G.add_edge(term_ids[term], term_ids[dep], key=EDGE_DEPENDS_ON,
                   edge_type=EDGE_DEPENDS_ON)

    for sec_id, term in extract_section_term_refs(base_secs, defs):
        G.add_edge(sec_id, term_ids[term], key=EDGE_REFERENCES, edge_type=EDGE_REFERENCES,
                   via="defined_term")

    amendments = [d for d in docs.values() if d.type == "amendment"]
    for doc in amendments:
        for op in extract_amend_ops(parsed[doc.id], doc):
            if op.target_kind == "definition":
                target = term_ids.get(op.target)
                if target is None:  # term added by the amendment, not in base
                    target = def_node_id(doc.id, op.target)
                    G.add_node(target, node_type="Definition", term=op.target,
                               text=op.new_text, document_id=doc.id,
                               section_id=op.source_sec_id, aliases=[])
            elif op.target_kind == "exhibit":
                target = f"base::exhibit::{op.target}"
                if not G.has_node(target):
                    exh = f"Exhibit {op.target}"
                    G.add_node(target, node_type="Section", id=target, number=exh,
                               title=exh, text="", document_id=DOC_BASE,
                               level="section", source="html")
                    G.add_edge(DOC_BASE, target, key=EDGE_CONTAINS, edge_type=EDGE_CONTAINS)
            else:
                target = sec_by_num.get(op.target)
                if target is None:  # section added by the amendment, not in base
                    target = f"base::{op.target}"
                    G.add_node(target, node_type="Section", id=target, number=op.target,
                               title="", text="", document_id=DOC_BASE, level="section",
                               source="html")
                    G.add_edge(DOC_BASE, target, key=EDGE_CONTAINS, edge_type=EDGE_CONTAINS)
                    sec_by_num[op.target] = target
            G.add_edge(doc.id, target, key=f"{EDGE_AMENDS}::{op.source_sec_id}",
                       edge_type=EDGE_AMENDS, effective_date=op.eff_date.isoformat(),
                       instrument=op.instrument, amendment_type=op.op_type,
                       amended_text=op.new_text, source_sec_id=op.source_sec_id)

    logger.info("graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G
