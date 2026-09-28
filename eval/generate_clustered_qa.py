"""Cluster-driven QnA set for Denny's legal KG — Stage 1 only, ~75–100 pairs.

Eval-breadth layer alongside assignment 12, synthetic-36, and holdout-hard:
grounded questions sampled from graph structure (semantic clusters + AMENDS /
DEPENDS_ON / REFERENCES), rubric-weighted for as-of, multi-hop, citations,
graph-vs-baseline, and unanswerables.

Deliberately NOT included (product-scale Stage 2):
  no Milvus, no complexity labels, no BGE reranker fine-tune, no 15k Qs.

Method (cluster → sample → LLM → dedup):
  1. Chunk = every Section/Definition with text (~650 nodes).
  2. Embed with the live-path embedder (dinghy); cluster with sklearn
     AgglomerativeClustering (cosine). Dedup uses the same embedder (≥0.92).
  3. Packs: amended nodes, graph multi-hop neighborhoods, edge-free cluster
     mates (contrast category), single-hop facts, mid-degree dependents
     (3–25 inbound), false-premise.
  4. Frontier Opus writes Q+gold offline only.
  5. Dedup; rewrite dependents gold from inbound edges; hard cap 100.

Target mix (~90, clamped 75–100):
  as_of_temporal / amendment_evolution  ~25
  multi_hop_graph                       ~22
  multi_hop_cluster                     ~10   (contrast: no required edge)
  fact_lookup                           ~15
  definition_dependents                 ~8
  conflict_precedence                   ~5
  false_premise                         ~7

Run:
  make generate-clustered
  uv run python -m eval.generate_clustered_qa --dry-run
  uv run python -m eval.generate_clustered_qa --smoke
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import boto3
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import settings
from graph.loader import load_graph
from graph.schema import EDGE_AMENDS
from graph.traversal import _amend_ops_before, hop_neighbors, inbound_dependents, node_label
from retrieval.embedder import Embedder

OUT_PATH = Path("eval/clustered_eval_set.json")
CLUSTER_META_PATH = Path("eval/clustered_qa_meta.json")

DEDUP_THRESHOLD = 0.92
MIN_Q, DEFAULT_Q, MAX_Q = 75, 90, 100
N_CLUSTERS = 48  # ~650 chunks → ~13/cluster; enough diversity, not HR-scale noise

AVOID = (
    "'How did the definition of Consolidated EBITDA change...' / "
    "'Which sections depend on the definition of Borrowing' / "
    "'find conflicting or inconsistent provisions' / "
    "'events of default relating to bankruptcy' / "
    "'maximum Consolidated Leverage Ratio as of 2020-06-01' / "
    "'maximum permitted restricted payment'"
)

PROMPT = """You are an expert financial-contracts lawyer designing a QA test set over
Denny's 2017 Credit Agreement and its First Amendment (eff. 2018-06-26) and Second
Amendment (eff. 2020-05-13). Original agreement date: 2017-10-26.

ARCHETYPE: {archetype}
{instructions}

GROUNDING (authentic; NODE_IDs are exact graph identifiers — cite only these):
{excerpts}

Generate exactly {n} question(s). Rules:
- Answerable ONLY from the grounding (except false_premise: state the premise is false).
- gold_answer precise and verifiable against the material.
- gold = NODE_IDs that support the answer (empty ONLY when nothing in the corpus
  answers — rare; prefer citing the disproving nodes for false_premise).
- Do NOT paraphrase these tuning questions: {avoid}

Return ONLY a JSON array:
[
  {{
    "category": "{archetype}",
    "question": "...",
    "as_of": <"YYYY-MM-DD" or null>,
    "gold": ["NODE_ID", ...],
    "gold_answer": "..."
  }}
]
"""

ARCHETYPE_INSTRUCTIONS = {
    "as_of_temporal": (
        "Pin as_of to a date that straddles amendments (before 2018-06-26, between "
        "amendments, or after 2020-05-13). Ask what the provision requires ON that date. "
        "gold_answer must reflect the version then in force and name the instrument if amended."
    ),
    "amendment_evolution": (
        "Trace what changed, when, and by which instrument. Require citing the amending "
        "instrument and effective date; substance of the change must be accurate."
    ),
    "multi_hop_graph": (
        "Question MUST require combining at least two linked nodes (definition→covenant, "
        "cross-reference chain, or exception). A single excerpt alone should be insufficient."
    ),
    "multi_hop_cluster": (
        "Nodes are semantic neighbors (may lack a direct graph edge). Ask a question that "
        "needs two or more of them. Prefer themes a credit analyst would actually ask."
    ),
    "fact_lookup": (
        "Single-hop factual / numerical / timeline lookup answerable from one primary node. "
        "Keep precise (thresholds, notice periods, parties, deadlines)."
    ),
    "definition_dependents": (
        "Ask which operative provisions rely on the defined term. The COMPLETE dependents "
        "list is in the grounding — gold_answer must enumerate it; completeness is the point."
    ),
    "conflict_precedence": (
        "Ask which provision controls under explicit 'notwithstanding' / precedence language. "
        "gold_answer must name both provisions and the controlling clause."
    ),
    "false_premise": (
        "Premise is false (nonexistent amendment, covenant, or defined term). gold_answer "
        "must reject the premise, state what actually exists, and cite disproving NODE_IDs."
    ),
}


# ---------------------------------------------------------------------------
# Chunking / clustering
# ---------------------------------------------------------------------------

def _chunk_nodes(G) -> list[str]:
    out = []
    for nid, d in G.nodes(data=True):
        if d.get("node_type") not in ("Section", "Definition"):
            continue
        if not (d.get("text") or "").strip():
            continue
        out.append(nid)
    return out


def _embed_texts(embedder: Embedder, texts: list[str]) -> np.ndarray:
    # Embedder.encode returns list/np depending on impl — normalize to 2d float32
    vecs = embedder.encode(texts)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def cluster_nodes(G, embedder: Embedder, n_clusters: int = N_CLUSTERS) -> dict[int, list[str]]:
    nids = _chunk_nodes(G)
    texts = []
    for nid in nids:
        d = G.nodes[nid]
        label = d.get("term") or d.get("title") or d.get("number", "")
        body = (d.get("text") or "")[:1200]
        texts.append(f"{label}\n{body}")
    emb = _embed_texts(embedder, texts)
    k = min(n_clusters, max(2, len(nids) // 8))
    labels = AgglomerativeClustering(
        n_clusters=k, metric="cosine", linkage="average"
    ).fit_predict(emb)
    clusters: dict[int, list[str]] = defaultdict(list)
    for nid, lab in zip(nids, labels, strict=True):
        clusters[int(lab)].append(nid)
    return dict(clusters)


# ---------------------------------------------------------------------------
# Excerpts / packs
# ---------------------------------------------------------------------------

def _excerpt(G, nid: str, max_chars: int = 1800) -> str:
    n = G.nodes[nid]
    label = n.get("term") or n.get("title") or n.get("number", "")
    parts = [f"--- NODE_ID: {nid} ({node_label(G, nid)} {label}) ---"]
    base = (n.get("text") or "")[:max_chars]
    parts.append(
        f"ORIGINAL TEXT (2017-10-26):\n{base}" if base
        else "(no original text — added by amendment)"
    )
    for op in _amend_ops_before(G, nid, None):
        parts.append(
            f"AMENDMENT OP: {op['instrument']} (effective {op['effective_date']}), "
            f"type={op['amendment_type']}:\n{(op.get('amended_text') or '')[:max_chars]}"
        )
    return "\n".join(parts)


def _dependents_excerpt(G, term_nid: str) -> str:
    n = G.nodes[term_nid]
    deps = inbound_dependents(G, term_nid)
    lines = [f"{node_label(G, d)}  [NODE_ID: {d}]" for d in deps]
    return (
        f"--- NODE_ID: {term_nid} (definition of \u201c{n['term']}\u201d) ---\n"
        f"DEFINITION TEXT:\n{(n.get('text') or '')[:1500]}\n"
        f"COMPLETE LIST OF PROVISIONS THAT REFERENCE OR DEPEND ON THIS TERM "
        f"(inbound graph edges — exhaustive):\n" + ("\n".join(lines) or "(none)")
    )


def _amended_nodes(G) -> list[str]:
    return sorted({
        v for _, v, _, d in G.edges(keys=True, data=True)
        if d.get("edge_type") == EDGE_AMENDS
        and G.nodes[v].get("node_type") in ("Section", "Definition")
    })


def _hop_neighbors(G, nid: str) -> list[str]:
    return hop_neighbors(G, nid, limit=None)


def _definition_terms_with_deps(
    G, min_deps: int = 3, max_deps: int = 25,
) -> list[str]:
    """Mid-degree definitions only — Loan/Borrower-scale inbound lists make
    top-k metrics meaningless and swamp the LLM; the live path answers these
    via the inbound-edge catalog, so gold should stay enumerable."""
    scored: list[tuple[int, str]] = []
    for nid, d in G.nodes(data=True):
        if d.get("node_type") != "Definition":
            continue
        n_deps = len(inbound_dependents(G, nid))
        if min_deps <= n_deps <= max_deps:
            scored.append((n_deps, nid))
    # Prefer richer-but-bounded terms; stable order for regen reproducibility.
    return [nid for _, nid in sorted(scored, key=lambda t: (-t[0], t[1]))]


def build_packs(G, clusters: dict[int, list[str]], budget: int) -> list[dict]:
    """Allocate packs so total requested questions ≈ budget (75–100)."""
    # Proportions of DEFAULT_Q=90; scale to budget
    plan = {
        "as_of_temporal": 15,
        "amendment_evolution": 10,
        "multi_hop_graph": 22,
        "multi_hop_cluster": 10,
        "fact_lookup": 15,
        "definition_dependents": 8,
        "conflict_precedence": 5,
        "false_premise": 7,
    }
    scale = budget / sum(plan.values())
    counts = {k: max(1, round(v * scale)) for k, v in plan.items()}
    # Fix rounding drift to hit budget exactly
    while sum(counts.values()) > budget:
        k = max(counts, key=counts.get)
        if counts[k] > 1:
            counts[k] -= 1
        else:
            break
    while sum(counts.values()) < budget:
        counts["multi_hop_graph"] += 1

    amended = _amended_nodes(G)
    def_terms = _definition_terms_with_deps(G)
    chunkable = _chunk_nodes(G)

    packs: list[dict] = []

    # --- temporal / evolution: one pack per amended node until quota ---
    for arch in ("as_of_temporal", "amendment_evolution"):
        need = counts[arch]
        for nid in amended:
            if need <= 0:
                break
            packs.append({
                "archetype": arch,
                "n": 1,
                "excerpts": [_excerpt(G, nid)],
                "source_nodes": [nid],
            })
            need -= 1
        # top up from high-degree defs if we run short of amended targets
        for nid in def_terms:
            if need <= 0:
                break
            if nid in amended:
                continue
            packs.append({
                "archetype": arch,
                "n": 1,
                "excerpts": [_excerpt(G, nid)],
                "source_nodes": [nid],
            })
            need -= 1

    # --- graph multi-hop: seed + 1–3 edge neighbors; diversify by article prefix ---
    need = counts["multi_hop_graph"]
    seen_seeds: set[str] = set()
    seen_prefixes: dict[str, int] = defaultdict(int)
    candidates = sorted(
        chunkable,
        key=lambda n: -len(_hop_neighbors(G, n)),
    )
    for seed in candidates:
        if need <= 0:
            break
        nbrs = _hop_neighbors(G, seed)[:3]
        if len(nbrs) < 1:
            continue
        if seed in seen_seeds:
            continue
        # Cap per article/def bucket so GAAP/lease variants don't dominate.
        prefix = seed.split("::")[1].split("(")[0].split(".")[0] if "::" in seed else seed
        if seen_prefixes[prefix] >= 3:
            continue
        seen_seeds.add(seed)
        seen_prefixes[prefix] += 1
        nodes = [seed] + nbrs
        packs.append({
            "archetype": "multi_hop_graph",
            "n": 1,
            "excerpts": [_excerpt(G, n) for n in nodes],
            "source_nodes": nodes,
        })
        need -= 1

    # --- cluster multi-hop: mates with NO legal edge (contrast category) ---
    need = counts["multi_hop_cluster"]
    sized = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    for _, members in sized:
        if need <= 0:
            break
        if len(members) < 2:
            continue
        # Prefer pairs/triples that do not share DEPENDS_ON/REFERENCES.
        pick: list[str] = []
        for cand in members:
            if not pick:
                pick.append(cand)
                continue
            linked = set()
            for p in pick:
                linked.update(_hop_neighbors(G, p))
            if cand in linked:
                continue
            pick.append(cand)
            if len(pick) >= 3:
                break
        if len(pick) < 2:
            continue
        packs.append({
            "archetype": "multi_hop_cluster",
            "n": 1,
            "excerpts": [_excerpt(G, n) for n in pick],
            "source_nodes": pick,
        })
        need -= 1

    # --- fact lookup: spread across clusters (single node) ---
    need = counts["fact_lookup"]
    for _, members in sized:
        if need <= 0:
            break
        nid = members[0]
        packs.append({
            "archetype": "fact_lookup",
            "n": 1,
            "excerpts": [_excerpt(G, nid)],
            "source_nodes": [nid],
        })
        need -= 1

    # --- definition dependents ---
    need = counts["definition_dependents"]
    for nid in def_terms:
        if need <= 0:
            break
        packs.append({
            "archetype": "definition_dependents",
            "n": 1,
            "excerpts": [_dependents_excerpt(G, nid)],
            "source_nodes": [nid],
        })
        need -= 1

    # --- conflict / precedence: nodes whose text mentions notwithstanding ---
    need = counts["conflict_precedence"]
    conflict_hits = [
        nid for nid in chunkable
        if re.search(r"notwithstanding|in the event of (any )?conflict",
                     (G.nodes[nid].get("text") or ""), re.I)
    ]
    # Always include a few known precedence-bearing sections if present
    for prefer in ("base::2.15(a)", "base::10.09", "base::10.03", "base::10.01(i)"):
        if prefer in G and prefer not in conflict_hits:
            conflict_hits.insert(0, prefer)
    for nid in conflict_hits:
        if need <= 0:
            break
        nbrs = _hop_neighbors(G, nid)[:2]
        nodes = [nid] + nbrs
        packs.append({
            "archetype": "conflict_precedence",
            "n": 1,
            "excerpts": [_excerpt(G, n) for n in nodes],
            "source_nodes": nodes,
        })
        need -= 1

    # --- false premise: fixed facts + financial-covenant grounding ---
    fp_excerpts = [
        "FACT: The only amendment instruments are the First Amendment (2018-06-26), "
        "the Second Amendment (2020-05-13), and the Third Amendment (2020-12-15). "
        "There is no Fourth Amendment.",
        "FACT: Terms NOT defined anywhere include: 'Consolidated Net Worth', "
        "'Minimum Liquidity Covenant Floor', 'Springing Maturity Date', "
        "'ESG Margin Adjustment'.",
    ]
    if G.has_node("base::7.10"):
        fp_excerpts.append(
            _excerpt(G, "base::7.10")
            + "\n\n" + _excerpt(G, "base::7.10(a)")
            + "\n\n" + _excerpt(G, "base::7.10(b)")
        )
    # Split false_premise into small batches of 2–3
    fp_n = counts["false_premise"]
    while fp_n > 0:
        take = min(3, fp_n)
        packs.append({
            "archetype": "false_premise",
            "n": take,
            "excerpts": fp_excerpts,
            "source_nodes": [],
        })
        fp_n -= take

    return packs


# ---------------------------------------------------------------------------
# Generation / dedup
# ---------------------------------------------------------------------------

def _parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw[raw.index("["): raw.rindex("]") + 1])


def generate_pack(client, pack: dict) -> list[dict]:
    arch = pack["archetype"]
    prompt = PROMPT.format(
        archetype=arch,
        instructions=ARCHETYPE_INSTRUCTIONS[arch],
        excerpts="\n\n".join(pack["excerpts"]),
        n=pack["n"],
        avoid=AVOID,
    )
    resp = client.converse(
        modelId=settings.bedrock_model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 3500, "temperature": 0.2},
    )
    raw = resp["output"]["message"]["content"][0]["text"]
    got = _parse_json_array(raw)
    for q in got:
        q["category"] = arch
        q.setdefault("as_of", None)
        q.setdefault("gold", [])
    return got


def dedup_questions(questions: list[dict], embedder: Embedder) -> list[dict]:
    if not questions:
        return []
    texts = [q["question"] for q in questions]
    emb = _embed_texts(embedder, texts)
    keep: list[dict] = []
    keep_idx: list[int] = []
    for i, q in enumerate(questions):
        if keep_idx:
            sims = emb[keep_idx] @ emb[i]
            if float(np.max(sims)) >= DEDUP_THRESHOLD:
                continue
        keep.append(q)
        keep_idx.append(i)
    return keep


def finalize_questions(G, questions: list[dict]) -> list[dict]:
    """Normalize gold + tags so the written set matches live-path design choices."""
    out = []
    for q in questions:
        q = dict(q)
        cat = q.get("category", "")
        q.pop("negative_control", None)
        if cat == "definition_dependents":
            term = next(
                (g for g in q.get("gold") or [] if str(g).startswith("base::def::")),
                None,
            )
            if term and G.has_node(term):
                q["gold"] = [term] + inbound_dependents(G, term)
        if cat == "multi_hop_cluster":
            q["contrast_set"] = True
        else:
            q.pop("contrast_set", None)
        out.append(q)
    return out


def validate_gold(G, questions: list[dict]) -> list[dict]:
    clean = []
    for q in questions:
        gold = [g for g in q.get("gold", []) if G.has_node(g)]
        # false_premise may legitimately have gold (disproving nodes) or empty
        if q.get("category") != "false_premise" and not gold:
            # drop ungrounded non-false-premise items — thoughtful filter, not product polish
            continue
        q = {**q, "gold": gold}
        clean.append(q)
    return clean


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max", type=int, default=DEFAULT_Q,
                    help=f"Target question count (clamped {MIN_Q}–{MAX_Q}; smoke may go lower)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Cluster + pack plan only; no Bedrock")
    ap.add_argument("--smoke", action="store_true",
                    help="Alias for --max 8 (cheap end-to-end check)")
    args = ap.parse_args()

    budget = 8 if args.smoke else args.max
    if not args.smoke:
        budget = max(MIN_Q, min(MAX_Q, budget))

    print(f"Loading graph + live embedder (target={budget})...")
    G = load_graph(settings.artifacts_dir / "graph.json")
    embedder = Embedder(settings.embedder_model)

    print(f"Clustering {len(_chunk_nodes(G))} section/definition chunks "
          f"→ ~{N_CLUSTERS} clusters...")
    clusters = cluster_nodes(G, embedder)
    print(f"  got {len(clusters)} clusters "
          f"(sizes {sorted(len(v) for v in clusters.values())[:5]}…)")

    packs = build_packs(G, clusters, budget)
    req = sum(p["n"] for p in packs)
    by_arch: dict[str, int] = defaultdict(int)
    for p in packs:
        by_arch[p["archetype"]] += p["n"]
    print(f"Packs: {len(packs)} batches requesting {req} questions")
    for k, v in sorted(by_arch.items()):
        print(f"  {k}: {v}")

    meta = {
        "target": budget,
        "n_clusters": len(clusters),
        "cluster_sizes": {str(k): len(v) for k, v in clusters.items()},
        "requested_by_archetype": dict(by_arch),
        "n_packs": len(packs),
        "dedup_threshold": DEDUP_THRESHOLD,
        "note": (
            "Stage-1 cluster QnA for eval breadth — experiment→implement with "
            "mid-degree dependents, hop multi-hop packs, and a semantic contrast category."
        ),
    }
    CLUSTER_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Wrote pack meta → {CLUSTER_META_PATH}")

    if args.dry_run:
        print("Dry-run complete (no Bedrock calls).")
        return

    print(f"Bedrock {settings.bedrock_model} — generating...")
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    questions: list[dict] = []
    for i, pack in enumerate(packs, 1):
        print(f"  [{i}/{len(packs)}] {pack['archetype']} n={pack['n']}...")
        try:
            got = generate_pack(client, pack)
        except Exception as e:  # noqa: BLE001 — keep going; assignment set > perfect run
            print(f"    WARN pack failed: {e}")
            continue
        questions.extend(got)
        print(f"    +{len(got)} (raw total {len(questions)})")
        if len(questions) >= budget + 15:
            # slight overshoot buffer for dedup losses, then stop early
            break

    before = len(questions)
    questions = validate_gold(G, questions)
    print(f"Gold validation: {before} → {len(questions)}")

    before = len(questions)
    questions = dedup_questions(questions, embedder)
    print(f"Dedup ≥{DEDUP_THRESHOLD}: {before} → {len(questions)}")

    questions = finalize_questions(G, questions)
    questions = questions[:budget]
    # If smoke, keep whatever we got; otherwise try to stay in 75–100 band
    if not args.smoke and len(questions) < MIN_Q:
        print(f"WARNING: only {len(questions)} after filters (wanted ≥{MIN_Q}). "
              "Re-run or lower dedup / add packs.")

    for i, q in enumerate(questions, 1):
        q["id"] = f"cl_{i:03d}"

    OUT_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    print(f"Wrote {len(questions)} questions → {OUT_PATH}")

    final_arch: dict[str, int] = defaultdict(int)
    for q in questions:
        final_arch[q.get("category", "?")] += 1
    print("Final mix:")
    for k, v in sorted(final_arch.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
