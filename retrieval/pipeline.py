"""Graph answer path: intent -> DFS -> rerank -> LLM with citations."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

import networkx as nx

from config import settings
from graph.intent import find_seeds
from graph.schema import EDGE_DEPENDS_ON, EDGE_REFERENCES
from graph.traversal import (
    Effective,
    _amend_ops_before,
    amendment_catalog,
    definition_closure,
    dfs_context,
    find_effective,
    hop_neighbors,
    inbound_dependents,
    node_label,
    party_roster,
    sibling_neighbors,
)
from llm.prompts import ANSWER_SYSTEM, build_answer_prompt, build_context
from retrieval.reranker import Reranker


@dataclass
class Answer:
    text: str
    citations: list[str]
    context_ids: list[str]
    blocks: list[tuple[str, str, str]] = field(default_factory=list)  # (citation, text, source)


def _catalog_context(
    G: nx.MultiDiGraph,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Surface every AMENDS edge as a context block (same retrieval channel as sections).

    Used when the question mentions amendments; the LLM still writes the answer —
    this is not an answer bypass.

    Grouped by instrument so a complete answer can carry one citation per group
    instead of repeating the same label on all 61 rows: that repetition both burns
    the answer-token budget and trips llama.cpp's repeat_penalty window, which is
    how a row (§5.23) went missing from an otherwise correct catalog.

    Returns (blocks, amended_node_ids). The ids make the catalog count as retrieved
    context in Tier-1 recall, the same way ``_dependents_context`` reports its own —
    without them the graph path scored 0.0 on Q11 while its answer listed every row.
    """
    rows = amendment_catalog(G)
    by_instrument: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        key = (r["effective_date"], r["instrument"])
        by_instrument.setdefault(key, []).append(
            f"{r['target']} ({r['amendment_type']})"
        )
    parts = [
        f"Total amendment operations: {len(rows)}. "
        "Every row below is one AMENDS edge; the list is exhaustive.",
    ]
    for (eff_date, instrument), targets in sorted(by_instrument.items()):
        parts.append(
            f"\n{instrument} — effective {eff_date} — {len(targets)} operations:\n"
            + "\n".join(f"  {t}" for t in targets)
        )
    target_ids = list(dict.fromkeys(r["target_id"] for r in rows))
    blocks = [("Amendment catalog (all AMENDS edges)", "\n".join(parts), "html")]
    return blocks, target_ids


def _parties_context(G: nx.MultiDiGraph) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Party/role roster from PARTY_TO edges, for "who are the parties" questions.

    Party nodes carry no text, so they are invisible to both the chunk index and DFS
    (which collects Section/Definition only) — the roles live on the edges. Same
    projection pattern as the amendment catalog; the LLM still writes the answer.

    Returns (blocks, party_node_ids) so the roster counts as retrieved context.
    """
    rows = party_roster(G)
    if not rows:
        return [], []
    by_party: dict[str, list[str]] = {}
    ids: list[str] = []
    for r in rows:
        by_party.setdefault(r["name"], []).append(r["role"])
        if r["node_id"] not in ids:
            ids.append(r["node_id"])
    lines = [
        "Parties to the Credit Agreement and the role each holds, from the "
        "signature/preamble parties and their PARTY_TO edges. This list is exhaustive."
    ]
    lines += [
        f"- {name} — {', '.join(dict.fromkeys(roles))}"
        for name, roles in by_party.items()
    ]
    return [("Party roster (all PARTY_TO edges)", "\n".join(lines), "html")], ids


def _dependents_context(
    G: nx.MultiDiGraph, seeds: list[str],
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """For 'which sections depend on/rely on X' questions: enumerate inbound
    REFERENCES/DEPENDS_ON edges into the seed definitions. The DFS follows outbound
    edges, so dependents are invisible to it — this block surfaces them.

    Returns (blocks, dep_node_ids) so callers can treat the catalog as retrieved
    context (eval + citations), not just prompt text.
    """
    blocks: list[tuple[str, str, str]] = []
    dep_ids: list[str] = []
    for nid in seeds:
        if G.nodes[nid]["node_type"] != "Definition":
            continue
        deps = inbound_dependents(G, nid)
        if deps:
            term = G.nodes[nid]["term"]
            blocks.append((
                f"Provisions that depend on \u201c{term}\u201d (inbound graph edges)",
                "\n".join(f"{node_label(G, d)}  [{d}]" for d in deps),
                "html",
            ))
            dep_ids.extend(deps)
            if nid not in dep_ids:
                dep_ids.insert(0, nid)
    return blocks, dep_ids


def _expand_seeds_with_hops(
    G: nx.MultiDiGraph, seeds: list[str], per_seed: int = 6,
) -> tuple[list[str], list[str]]:
    """Pull 1-hop DEPENDS_ON/REFERENCES neighbors (both directions) into the seed list
    before DFS/rerank. Returns (expanded_for_dfs, force_include_ids).

    force_include_ids = original seeds + their hops — must survive rerank truncation
    so multi-hop gold on the inbound side of a definition is not dropped at top-k.
    """
    expanded: list[str] = []
    force: list[str] = []
    seen: set[str] = set()
    for s in seeds:
        if s not in seen:
            expanded.append(s)
            seen.add(s)
        if s not in force:
            force.append(s)
        for n in hop_neighbors(G, s, limit=per_seed):
            if n not in seen:
                expanded.append(n)
                seen.add(n)
            if n not in force:
                force.append(n)
    return expanded, force


def _stage2_candidates(
    G: nx.MultiDiGraph,
    evidence_ids: list[str],
    top_n: int,
    expand_from: list[str] | None = None,
) -> list[str]:
    """Stage 2 candidates: sibling + hub neighbours of the evidence already in hand.

    Stage 1 only expands the question's own seeds, so a provision the question never
    names stays out of reach: DFS descends CONTAINS, so from §8.01 (Events of Default)
    it reaches §8.01(a)–(l) but never its sibling §8.02 (Remedies) — the "and then what
    happens" half of a chain question is simply absent.

    Two query-agnostic hops:
      * sibling sections through the shared CONTAINS parent (§8.01 -> §8.02);
      * bidirectional REFERENCES/DEPENDS_ON, which routes through definition hubs
        (§8.01 -> "Event of Default" -> §8.02 and its other referrers, or
        "Consolidated Leverage Ratio" -> the components it is computed from).

    ``evidence_ids`` is everything already in context (never re-proposed);
    ``expand_from`` is the subset to expand around, defaulting to the first ``top_n``
    of the evidence. The caller passes seeds *and* top-ranked nodes: the question's own
    seeds are the highest-precision entry points, and when they sat at the tail of one
    combined list ``top_n`` cut them off — "Permitted Liens" (one hop from the §7.01
    seed) and "Consolidated EBITDA" (one hop from the leverage-ratio seed) were both
    unreachable no matter how many candidates were kept.

    Neighbours are collected unbounded; the caller cuts them by relevance. Capping
    per node here would cut alphabetically instead, which silently dropped
    "Permitted Liens" from the Q5 exception list.
    """
    already = set(evidence_ids)
    out: list[str] = []
    for nid in expand_from if expand_from is not None else evidence_ids[:top_n]:
        for cand in (
            sibling_neighbors(G, nid, limit=None) + hop_neighbors(G, nid, limit=None)
        ):
            if cand not in already:
                already.add(cand)
                out.append(cand)
    return out


# Force-include still lists the node in context_ids for recall, but prompt bodies
# must stay inside llama.cpp n_ctx. Parent dumps (§1.01) and long restatements
# (e.g. §10.22(b) after A3) otherwise blow past 16k when hops are force-included.
_FORCE_BODY_CHARS = 3500


def _clip_blocks(
    blocks: list[tuple[str, str, str]], max_chars: int,
) -> list[tuple[str, str, str]]:
    out = []
    for cit, text, src in blocks:
        if len(text) > max_chars:
            text = text[: max_chars - 20].rstrip() + "\n…[truncated]"
        out.append((cit, text, src))
    return out


def _fit_block(
    block: tuple[str, str, str], room: int,
) -> tuple[tuple[str, str, str], int]:
    """Clip one block to ``room`` characters. Always returns the block."""
    cit, text, src = block
    overhead = len(cit) + 24 + 10
    body_room = max(80, room - overhead)
    if len(text) > body_room:
        text = text[: body_room - 20].rstrip() + "\n…[truncated]"
    fitted = (cit, text, src)
    return fitted, overhead + len(text)


def _ids_of(tagged, packed) -> list[str]:
    ids: list[str] = []
    for (_, nids), _ in zip(tagged[: len(packed)], packed, strict=True):
        for nid in nids:
            if nid not in ids:
                ids.append(nid)
    return ids


def _pack_reserved(
    reserved: list[tuple[tuple[str, str, str], list[str]]],
    fill: list[tuple[tuple[str, str, str], list[str]]],
    max_chars: int,
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Every reserved block is in the prompt. Filler is what the window drops.

    Projection blocks (a catalog, a party roster) stay whole. Per-node blocks
    (seeds, definition closure, the sibling hop) are shortened to share whatever
    room is left, but none of them is omitted. Reranked filler is packed only
    into the characters that remain.
    """
    if max_chars <= 0:
        blocks = [b for b, _ in reserved]
        return blocks, _ids_of(reserved, blocks)

    projections = [(b, n) for b, n in reserved if len(n) != 1]
    nodes = [(b, n) for b, n in reserved if len(n) == 1]

    proj_blocks: list[tuple[str, str, str]] = []
    used = 0
    for block, _ in projections:
        fitted, cost = _fit_block(block, max(max_chars - used, 80))
        proj_blocks.append(fitted)
        used += cost

    node_blocks: list[tuple[str, str, str]] = []
    if nodes:
        need = sum(len(b[0]) + len(b[1]) + 34 for b, _ in nodes)
        room = max(0, max_chars - used)
        if need <= room:
            node_blocks = [b for b, _ in nodes]
            used += need
        else:
            share = max(160, room // len(nodes))
            for block, _ in nodes:
                fitted, cost = _fit_block(block, share)
                node_blocks.append(fitted)
                used += cost

    kept = proj_blocks + node_blocks
    ids = _ids_of(projections + nodes, kept)
    fill_room = max_chars - sum(len(c) + len(t) + 34 for c, t, _ in kept)
    fill_blocks = _pack_blocks([b for b, _ in fill], fill_room) if fill_room > 200 else []
    for nid in _ids_of(fill, fill_blocks):
        if nid not in ids:
            ids.append(nid)
    return kept + fill_blocks, ids


def _pack_blocks(
    blocks: list[tuple[str, str, str]], max_chars: int,
) -> list[tuple[str, str, str]]:
    """Keep block order (callers put priority blocks first); fit to a char budget.

    Budget is derived from settings.llm_context_length so generate() cannot request
    more tokens than n_ctx (prompt + max_tokens).
    """
    if max_chars <= 0 or not blocks:
        return blocks
    out: list[tuple[str, str, str]] = []
    used = 0
    sep = 10  # "\n\n---\n\n"
    for cit, text, src in blocks:
        header = len(cit) + 24
        room = max_chars - used - header - sep
        if room < 180:
            break
        if len(text) > room:
            text = text[: room - 20].rstrip() + "\n…[truncated]"
        out.append((cit, text, src))
        used += header + len(text) + sep
    return out


def _context_char_budget() -> int:
    # prompt + think_budget + answer max_tokens must fit n_ctx.
    think = settings.llm_think_budget if settings.llm_enable_thinking else 0
    reserve = settings.llm_max_tokens + think + 512
    return max(4000, (settings.llm_context_length - reserve) * 3)

_CHANGE_WORD = re.compile(r"\b(?:chang\w*|amend\w*|evolv\w*|modif\w*)\b", re.I)
_EFFECT_WORD = re.compile(r"\b(?:rely|relies|relied|depend\w*|affect\w*)\b", re.I)


def _component_effect_context(
    G: nx.MultiDiGraph, seeds: list[str], as_of: date | None,
) -> list[tuple[str, str, str]]:
    """Amendments to the inputs a definition is computed from, over DEPENDS_ON.

    A defined term can be changed two ways: by amending its own text, or by amending a
    term it incorporates. The second kind is invisible in the definition's body, so a
    model reading only the retrieved text reports the first and misses the second — and
    when the term's own text carries a proviso scoped to one section, it will cite that
    narrow scope as the whole answer and conclude nothing else is affected. An input
    restatement has the opposite reach: it applies wherever the term is used.

    Gated on questions that ask how a change affects what relies on it, which is the
    only shape where the distinction decides the answer. Narrower than
    ``_change_history_context``, which dumps every AMENDS op on the seeds and their
    components and is off by default (the holdout ablation in
    experiments/EXPERIMENT_SUMMARY.md §6 found the stacked blocks cost answers).
    """
    lines: list[str] = []
    for nid in seeds:
        if not G.has_node(nid) or G.nodes[nid].get("node_type") != "Definition":
            continue
        for comp in definition_closure(G, nid, depth=1):
            ops = _amend_ops_before(G, comp, as_of)
            if not ops:
                continue
            op = ops[-1]
            lines.append(
                f"- {node_label(G, nid)} is computed from {node_label(G, comp)}. "
                f"{op['instrument']} ({op['amendment_type']}, effective {op['effective_date']}) "
                f"changed that component. That is an input change: "
                f"it applies wherever {node_label(G, nid)} is used, including covenants "
                f"and pricing tied to it. A later proviso that limits one adjustment to "
                f"a single section does not undo this component restatement, and does "
                f"not mean {op['instrument']} amended {node_label(G, nid)} itself."
            )
    if not lines:
        return []
    return [(
        "Component changes (definitions this term is computed from)",
        "Report each line below in the answer, separately from any proviso inside "
        "the term's own definition.\n" + "\n".join(lines),
        "html",
    )]


def _change_history_context(
    G: nx.MultiDiGraph, seeds: list[str], as_of: date | None,
) -> list[tuple[str, str, str]]:
    """Evolution/aggregation questions ('how did X change', 'across both amendments')
    need per-instrument change attribution. The section blocks show the AMENDED text
    but not what changed, when, or by which instrument — and changes to a definition's
    COMPONENTS (e.g. Consolidated Leverage Ratio depends on Consolidated EBITDA, which
    the First Amendment restated) are invisible unless surfaced explicitly. This block
    enumerates every AMENDS op on each seed and on the definitions it depends on, and
    states the list is exhaustive so models can assert 'instrument Y made no change'.
    Experiment: experiments/change_history.py."""
    lines, seen = [], set()
    for nid in seeds[:3]:
        if G.nodes[nid]["node_type"] not in ("Definition", "Section"):
            continue
        components = [
            v for _, v, k in G.out_edges(nid, keys=True)
            if k.split("::")[0] in (EDGE_REFERENCES, EDGE_DEPENDS_ON)
            and G.nodes[v]["node_type"] == "Definition"
        ]
        for rid in [nid] + components:
            if rid in seen:
                continue
            seen.add(rid)
            rel = "" if rid == nid else (
                f" — a component that {node_label(G, nid)} relies on"
            )
            ops = _amend_ops_before(G, rid, as_of)
            if not ops:
                continue
            for op in ops:
                snippet = (op.get("amended_text") or "").strip().replace("\n", " ")[:400]
                lines.append(
                    f"- {node_label(G, rid)}{rel}: {op['amendment_type']} by "
                    f"{op['instrument']}, effective {op['effective_date']}. "
                    f"Change: {snippet}"
                )
    if not lines:
        return []
    header = (
        "Complete change history for the provisions relevant to this question, derived "
        "from the graph's AMENDS edges. This list is EXHAUSTIVE: any amendment "
        "instrument not shown for a provision made no change to that provision. "
        "Entries marked 'a component that X relies on' are INDIRECT: the instrument "
        "changed the component definition, not the provision itself. When the "
        "question asks how a definition's changes affect the provisions that rely on "
        "it, or how a ratio/regime evolved over time, you MUST also report these "
        "indirect changes, attributed precisely ('<instrument> restated <component>, "
        "which <provision> relies on') — but never say the instrument amended the "
        "provision itself when it only changed a component. "
        "Use this block for attribution (which instrument, when); quote substantive "
        "requirements from the full section texts below, not from these snippets."
    )
    return [("Change history (all AMENDS edges on relevant provisions)",
             header + "\n" + "\n".join(lines), "html")]


def _conflict_audit_context(G: nx.MultiDiGraph) -> list[tuple[str, str, str]]:
    """Q8-type questions ('find a conflict/a cross-reference that does not say what
    the referrer implies') cannot be answered from retrieved context — discovery
    requires comparing every referring clause against its target, which is an offline
    index-time computation (experiments/crossref_audit.py, local models only). This
    block injects the audit's findings: confirmed mismatches if any, the audit's
    provenance (so the model can honestly report a clean audit), and the explicit
    precedence clauses where the agreement itself acknowledges potential conflicts."""
    path = settings.artifacts_dir / "crossref_audit.json"
    if not path.exists():
        return []
    audit = json.loads(path.read_text())
    s = audit["summary"]
    lines = [
        f"An exhaustive offline audit checked all {s['section_crossrefs_checked']} "
        f"section cross-references: {s['dangling_found']} dangling references; "
        f"{s['proviso_claims_unverified']} of {s['proviso_claims_checked']} "
        f"proviso-claims unverifiable; {s['confirmed_mismatches']} of "
        f"{s['semantic_candidates_llm_checked']} semantic-mismatch candidates "
        f"confirmed (verdicts by {s['verdict_model']} with verbatim-quote "
        "verification)."
    ]
    for c in audit["mismatch"]:
        if c["is_mismatch"]:
            lines.append(
                f"- CONFIRMED MISMATCH: {c['from_label']} states \u201c{c['evidence'][:220]}\u201d "
                f"but {c['to_label']} actually says: \u201c{c['target_snippet'][:220]}\u201d. "
                f"Audit verdict: {c['verdict'][:180]}"
            )
    lines.append(
        "The audit being clean means no cross-reference is BROKEN — it does NOT mean "
        "the agreement has no conflicts. Genuine section-vs-section conflicts exist "
        "and are acknowledged and resolved by the explicit precedence clauses below. "
        "To answer a 'find a conflict' question: pick ONE pair below — prefer an "
        "entry whose overridden section is quoted — describe what each side "
        "requires, and cite the controlling clause. Do not claim that no conflicts "
        "exist, do not invent conflicts beyond these findings, and never pair a "
        "controlling clause with a section it does not name."
    )
    sec_re = re.compile(r"Section[s]?\s+(\d{1,2}\.\d{2}(?:\([a-z0-9]+\))?)")
    # clauses that NAME the section they override first: the model can quote both
    # sides. Unnamed 'notwithstanding anything herein' clauses invite fabricated
    # counterparts (mistral paired §2.03(j) with an unrelated §1.02(a)).
    clauses = sorted(audit["precedence_clauses"],
                     key=lambda p: 0 if sec_re.search(p["clause"]) else 1)
    for p in clauses[:8]:
        lines.append(f"- {p['label']}: \u201c{p['clause'][:240]}\u201d")
        # quote the other side of the conflict so the model can describe both
        for num in dict.fromkeys(sec_re.findall(p["clause"])):
            nid = f"base::{num}"
            if G.has_node(nid) and G.nodes[nid].get("text"):
                snip = " ".join(G.nodes[nid]["text"].split())[:200]
                lines.append(f"    (overridden \u00a7{num} provides: \u201c{snip}\u2026\u201d)")
    return [("Cross-reference audit (offline, all REFERENCES edges checked)",
             "\n".join(lines), "html")]


def _node_blocks(
    G: nx.MultiDiGraph, eff: Effective, as_of: date | None, split: bool = True,
) -> list[tuple[str, str, str]]:
    """Blocks for one node. Text inserted by an amendment into an existing section is
    emitted as its own labelled block BEFORE the section body: open-weight models
    ignore an appended proviso (primacy bias), but honor it when it leads with an
    explicit override label (experiments/proviso_prompting.py — Qwen-14B solves Q10
    under this treatment; inline seam markers had failed, see outputs/qa_notes.md).

    split=False returns the un-elevated single block (proviso still appended by
    find_effective) — the control arm for experiments/ablation.py."""
    inserts = [
        op for op in _amend_ops_before(G, eff.node_id, as_of)
        if op["amendment_type"] == "insert" and G.nodes[eff.node_id].get("text")
    ]
    if not split or not inserts:
        return [(eff.citation, eff.text, eff.source)]
    blocks, remaining = [], eff.text
    for op in inserts:
        blocks.append((
            f"{eff.citation} — PROVISO added by {op['instrument']} "
            f"(effective {op['effective_date']}), overrides the clauses of this "
            "section where they conflict",
            op["amended_text"], eff.source,
        ))
        remaining = remaining.replace("\n" + op["amended_text"], "")
    blocks.append((eff.citation, remaining, eff.source))
    return blocks


def graph_answer(
    G: nx.MultiDiGraph, question: str, as_of: date | None,
    reranker: Reranker, llm, top_k: int | None = None,
    index=None, embedder=None, change_history: bool = False,
    conflict_audit: bool = False, proviso_split: bool = False,
) -> Answer:
    """index/embedder enable hybrid dense seeding (experiments/hybrid_seeding.py:
    synthetic-set recall 0.833 -> 0.944; assignment set unchanged).

    The three context-shaping flags default OFF on the shipping path. Each earned
    its keep on Qwen2.5-14B (change_history.py, proviso_prompting.py,
    conflict_audit_ab.py) but a controlled holdout ablation on Qwen3.6-27B
    (experiments/ablation_holdout.py) found graph_only at 17/20 vs full at 15/20 —
    stacked together the blocks cost answers, with conflict_audit the clear
    offender on conflict_precedence. Pass True explicitly to re-enable a block
    (weaker generators, or A/B). experiments/ablation.py sets every flag
    explicitly so its CONDITIONS do not track this default."""
    ql = question.lower()
    seed_roots = find_seeds(G, question, index=index, embedder=embedder)
    # Multi-hop: expand seeds with 1-hop edge neighbors (inbound + outbound) so
    # dependents/refs of a definition enter the DFS pool before rerank — and
    # keep those hops in context_ids so top-k cannot drop them for recall.
    # Prompt bodies: ranked first, then seed roots, then hops (clipped). The
    # ablation-negative *context-shaping blocks* (change_history / conflict_audit /
    # proviso_split) stay off by default; this is retrieval packing, not those.
    seeds, force_ids = _expand_seeds_with_hops(G, seed_roots)
    # A defined-term seed drags in the terms it is computed from, unconditionally:
    # the read-through is what the definition *means*, and it is a graph relation no
    # relevance ranking recovers (see definition_closure).
    for root in seed_roots:
        if G.nodes[root].get("node_type") != "Definition":
            continue
        for dep in definition_closure(G, root, depth=settings.definition_closure_depth):
            if dep not in seeds:
                seeds.append(dep)
            if dep not in force_ids:
                force_ids.append(dep)
    root_set = set(seed_roots)
    effs = dfs_context(G, seeds, as_of, max_depth=2, max_nodes=40)
    by_id = {e.node_id: e for e in effs}
    ranked = reranker.top_k(
        question, [(e.node_id, e.text) for e in effs if e.text],
        k=top_k or settings.reranker_top_k,
    )
    ranked_ids = {nid for nid, _, _ in ranked}

    def _body(nid: str, *, skip_if_seen: bool) -> list[tuple[str, str, str]]:
        if not G.has_node(nid) or (skip_if_seen and nid in ranked_ids):
            return []
        if G.nodes[nid]["node_type"] not in ("Section", "Definition"):
            return []
        eff = by_id.get(nid) or find_effective(G, nid, as_of)
        if not eff.text:
            return []
        ranked_ids.add(nid)
        return _clip_blocks(
            _node_blocks(G, eff, as_of, split=proviso_split), _FORCE_BODY_CHARS,
        )

    # Stage 2: sibling/hub neighbours of the evidence already in the prompt. Expansion
    # runs from the seeds *and* the top-ranked nodes, because the reranker can leave a
    # seed out of the top-k while it is still the right anchor: on Q7 §8.01 is a seed,
    # ranks outside the top 10, and is the only route to §8.02. Candidates are then cut
    # by rerank score, not by list order.
    # The seeds and the ranked nodes get separate halves of the keep budget. Pooling
    # them lets one generic hub bury the precise hits: expanding from the top-6 ranked
    # nodes of Q12 pulls in the neighbours of "Loan"/"Indebtedness"/"Commitment" for
    # 347 candidates, and "Consolidated EBITDA" — one DEPENDS_ON hop from the
    # leverage-ratio seed, in a pool of 14 on its own — never survived the cut.
    top_n = settings.expand_stage2_top_n
    ranked_order = [nid for nid, _, _ in ranked]
    evidence_ids = list(dict.fromkeys(ranked_order + seed_roots))
    keep = settings.expand_stage2_keep
    stage2_ids: list[str] = []

    def _pick(centres: list[str], budget: int) -> None:
        if not centres or budget <= 0:
            return
        candidates = _stage2_candidates(
            G, evidence_ids + stage2_ids, top_n, expand_from=centres,
        )
        scored = [
            (nid, txt) for nid, txt in (
                (c, find_effective(G, c, as_of).text) for c in candidates
            ) if txt
        ]
        if scored:
            stage2_ids.extend(
                nid for nid, _, _ in reranker.top_k(question, scored, k=budget)
            )

    _pick(seed_roots[:top_n], (keep + 1) // 2)
    _pick(ranked_order[:top_n], keep - len(stage2_ids))

    # What the answer cannot do without, packed first so a long reranked definition
    # cannot push it out of n_ctx. Each entry is (block, node ids that block covers).
    # Order: question-specific projections, the question's own seeds, the terms those
    # seeds are computed from, then the reranked sibling/hub hop (the §8.02 case).
    reserved: list[tuple[tuple[str, str, str], list[str]]] = []
    if re.search(r"\bdepend|\breli(?:es|y) on|\breference or depend", ql):
        # Seed *roots*, not the hop-expanded list: expansion interleaves neighbours, so
        # slicing it dropped the very term the question named ("Eurodollar Rate" sat
        # past the window) and the dependents block enumerated the wrong definition.
        dep_blocks, dep_ids = _dependents_context(G, seed_roots)
        reserved.extend((b, dep_ids) for b in dep_blocks)
    if re.search(r"\bpart(?:y|ies)\b|\bwho (?:are|is)\b|\brole", ql):
        party_blocks, party_ids = _parties_context(G)
        reserved.extend((b, party_ids) for b in party_blocks)
    if "amend" in ql:
        catalog_blocks, catalog_ids = _catalog_context(G)
        reserved.extend((b, catalog_ids) for b in catalog_blocks)
    if _CHANGE_WORD.search(question) and _EFFECT_WORD.search(question):
        reserved.extend(
            (b, []) for b in _component_effect_context(G, seed_roots, as_of)
        )
    if change_history and re.search(
        r"chang|amend|trace|evolv|restructur|modif|history", ql
    ):
        reserved.extend((b, []) for b in _change_history_context(G, seeds, as_of))
    if conflict_audit and re.search(
        r"conflict|inconsisten|contradict|cross.?ref|does not say", ql
    ):
        reserved.extend((b, []) for b in _conflict_audit_context(G))

    closure_ids = {
        dep
        for root in seed_roots
        if G.nodes[root].get("node_type") == "Definition"
        for dep in definition_closure(G, root, depth=settings.definition_closure_depth)
    }
    for nid in force_ids:
        if nid not in root_set and nid not in closure_ids:
            continue
        for block in _body(nid, skip_if_seen=False):
            reserved.append((block, [nid]))
    for nid in stage2_ids:
        for block in _body(nid, skip_if_seen=False):
            reserved.append((block, [nid]))

    # Fill is whatever ranked/hop text still fits. It is not allowed to evict reserved.
    reserved_ids = {nid for _, nids in reserved for nid in nids}
    fill: list[tuple[tuple[str, str, str], list[str]]] = []
    for nid, _, _ in ranked:
        if nid in reserved_ids or nid not in by_id:
            continue
        for block in _node_blocks(G, by_id[nid], as_of, split=proviso_split):
            fill.append((block, [nid]))
    for nid in force_ids:
        if nid in reserved_ids:
            continue
        for block in _body(nid, skip_if_seen=True):
            fill.append((block, [nid]))

    blocks, ctx_ids = _pack_reserved(reserved, fill, _context_char_budget())
    prompt = build_answer_prompt(
        question, build_context(blocks), as_of.isoformat() if as_of else None
    )
    text = llm.generate(ANSWER_SYSTEM, prompt)
    return Answer(text=text, citations=[c for c, _, _ in blocks],
                  context_ids=ctx_ids, blocks=blocks)
