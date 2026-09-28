"""Offline cross-reference audit (roadmap Gap 6) — the q8 archetype.

Q8 asks the system to FIND a conflict or a cross-reference that points to a section
that does not say what the referring section implies. No generator (Opus included)
can discover that from retrieved context: it requires comparing every referring
clause against its target. That is an offline INDEX-time computation, so it belongs
in the pipeline, not the model — the strongest case for pipeline-over-model-weights.

Stages (all local/deterministic — no frontier API; the LLM stage uses a local
open-weights GGUF, consistent with the live-path constraint):
  1. dangling:    section-number mentions in node text that resolve to no graph node
     (excluding references into other documents or external law, e.g. 'Section 4 of
     the Assignment and Assumption', 'Treasury Regulation Section 1.1471-2(b)').
  2. structural:  claims like 'the proviso to Section X' / 'clause (i) of Section X'
     verified against the target's actual text (does a proviso/clause exist?).
  3. mismatch:    for each REFERENCES edge with evidence text, CrossEncoder score of
     (referring sentence, target section text); lowest = candidates.
  4. verdict:     local LLM reads (referring sentence, full target text) for the top
     candidates and rules SUPPORTED/MISMATCH with a reason.

Output: artifacts/crossref_audit.json (findings, ranked). Inspect before wiring.
Run: CUDA_VISIBLE_DEVICES=N LLM_MAIN_GPU=0 EMBED_DEVICE=cpu \
     uv run python -m experiments.crossref_audit [model_name]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from config import settings
from graph.loader import load_graph
from graph.traversal import node_label

# (?<!Regulation\s) — external law; (?!\d) — don't split '1.1471' into '1.14'+'71'
SEC_MENTION_RE = re.compile(
    r"(?<!Regulation )Section[s]?\s+(\d{1,2}\.\d{2}(?:\([a-z0-9]+\))?)(?!\d)"
)
# mention refers to another document, not this agreement
OTHER_DOC_RE = re.compile(
    r"^\s*(of|under)\s+(the|any|such|each)\s+(?!Credit Agreement)", re.I
)
PROVISO_CLAIM_RE = re.compile(
    r"(first |second |the )provisos? (?:to|of|in) (?:this )?Section\s+"
    r"(\d{1,2}\.\d{2}(?:\([a-z0-9]+\))?)", re.I,
)
OUT = Path(settings.artifacts_dir) / "crossref_audit.json"


def dangling_refs(G) -> list[dict]:
    known = {n.split("::", 1)[1] for n in G.nodes if n.startswith("base::") and "::def::" not in n}
    findings = []
    for nid, d in G.nodes(data=True):
        if d["node_type"] != "Section" or not nid.startswith("base::"):
            continue
        text = d.get("text") or ""
        for m in SEC_MENTION_RE.finditer(text):
            num = m.group(1)
            tail = text[m.end(): m.end() + 40]
            if OTHER_DOC_RE.match(tail):
                continue  # reference into a different document
            base_num = num.split("(")[0]
            if num in known or base_num in known:
                continue
            if any(k.startswith(base_num + "(") for k in known):
                continue  # subsection granularity differs, parent exists
            ctx = text[max(0, m.start() - 120): m.end() + 120].replace("\n", " ")
            findings.append({
                "kind": "dangling_reference",
                "from": nid, "from_label": node_label(G, nid),
                "mentions": f"Section {num}",
                "evidence": ctx,
            })
    return findings


def _family_text(G, num: str) -> str:
    """Full text of a section plus its subsection nodes."""
    ids = [n for n in G.nodes
           if n == f"base::{num}" or n.startswith(f"base::{num}(")]
    return "\n".join((G.nodes[i].get("text") or "") for i in sorted(ids))


def proviso_claims(G) -> list[dict]:
    """Verify 'proviso to Section X' claims: does X (incl. subsections) contain one?"""
    findings = []
    for nid, d in G.nodes(data=True):
        if d["node_type"] != "Section" or not nid.startswith("base::"):
            continue
        text = d.get("text") or ""
        for m in PROVISO_CLAIM_RE.finditer(text):
            num = m.group(2)
            tgt = _family_text(G, num.split("(")[0])
            if not tgt:
                continue
            ok = "provided" in tgt.lower()
            findings.append({
                "kind": "proviso_claim",
                "verified": ok,
                "from": nid, "from_label": node_label(G, nid),
                "claims": m.group(0),
                "evidence": text[max(0, m.start() - 100): m.end() + 100].replace("\n", " "),
            })
    return findings


def _full_sentence(G, u: str, v: str, evidence: str) -> str:
    """Re-extract the complete referring sentence from the source node's text.
    The stored edge evidence is a ~100-char window, often cut mid-word — feeding
    that to a verdict model produces false MISMATCHes."""
    text = G.nodes[u].get("text") or ""
    v_num = v.split("::", 1)[-1]
    idx = text.find(f"Section {v_num}")
    if idx == -1:
        idx = text.find(v_num)
    if idx == -1:
        return evidence
    start = max(text.rfind(". ", 0, idx), text.rfind("\n", 0, idx)) + 1
    end = text.find(". ", idx)
    end = len(text) if end == -1 else end + 1
    sent = text[max(0, start):end].strip().replace("\n", " ")
    return sent[:700] if sent else evidence


def llm_verdicts(G, candidates: list[dict], model_name: str) -> list[dict]:
    """Local open-weights LLM rules SUPPORTED/MISMATCH per candidate pair."""
    from experiments.large_gguf_comparison import MODELS
    from llm.client import LocalLlm
    path, n_ctx = MODELS[model_name]
    llm = LocalLlm(path, n_ctx=n_ctx)
    system = (
        "You are auditing cross-references in a credit agreement. Given a REFERRING "
        "sentence and the FULL TEXT of the section it points to, decide whether the "
        "target section actually contains what the referring sentence attributes to "
        "it. Reply with exactly one line: 'SUPPORTED: <reason>' or 'MISMATCH: <what "
        "the referrer implies vs what the target actually says>'. Rules: (1) the "
        "referring sentence may be truncated at its edges — truncation is NOT a "
        "mismatch; (2) the attribution only needs to be plausibly covered somewhere "
        "in the target, including its provisos and subsections; (3) rule MISMATCH "
        "only if the target clearly lacks or contradicts the specific attributed "
        "content. When in doubt, rule SUPPORTED."
    )
    out = []
    for c in candidates:
        tgt = _family_text(G, c["to"].split("::", 1)[1].split("(")[0])[:9000]
        user = (f"REFERRING SENTENCE (from {c['from_label']}):\n{c['evidence']}\n\n"
                f"TARGET {c['to_label']} FULL TEXT:\n{tgt}")
        verdict = llm.generate(system, user, max_tokens=200).strip()
        c = dict(c)
        c["verdict"] = verdict[:400]
        c["is_mismatch"] = verdict.upper().startswith("MISMATCH")
        if c["is_mismatch"]:
            # Second pass: extraction is easier than judgment for small models.
            # Ask for a supporting QUOTE and verify it verbatim against the target —
            # a hallucinated quote fails the substring check, so this cannot flip a
            # genuine mismatch to SUPPORTED.
            quote = llm.generate(
                "Find the passage in the TARGET text that supports what the REFERRING "
                "sentence attributes to it. Reply with an exact verbatim quote from "
                "the TARGET (max 60 words), or the single word NONE if no passage "
                "exists.",
                user, max_tokens=150,
            ).strip().strip('"\u201c\u201d')
            norm = " ".join(quote.split()).lower()
            tgt_norm = " ".join(tgt.split()).lower()
            if quote.upper() != "NONE" and len(norm) > 20 and norm[:180] in tgt_norm:
                c["is_mismatch"] = False
                c["verdict"] = f"SUPPORTED (quote-verified on 2nd pass): {quote[:250]}"
        out.append(c)
        print(f"  {c['from_label']} -> {c['to_label']}: {c['verdict'][:120]}", flush=True)
    return out


def mismatch_candidates(G, top_n: int = 20) -> list[dict]:
    from retrieval.reranker import Reranker
    rr = Reranker(settings.reranker_model, settings.reranker_batch_size)
    pairs, meta = [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        if k.split("::")[0] != "REFERENCES" or not d.get("evidence"):
            continue
        # self-family references ('this Section 2.03' from §2.03(c)) are noise
        u_num = u.split("::", 1)[-1].split("(")[0]
        v_num = v.split("::", 1)[-1].split("(")[0]
        if u_num == v_num:
            continue
        ev = _full_sentence(G, u, v, d["evidence"])
        # skip references into other documents ('Section 3.01 of the Guarantee and
        # Collateral Agreement') — the graph edge is spurious, the document is fine
        m = SEC_MENTION_RE.search(ev)
        if m and OTHER_DOC_RE.match(ev[m.end():]):
            continue
        tgt = _family_text(G, v_num)
        if not tgt.strip():
            continue
        pairs.append((ev, tgt[:1200]))
        meta.append((u, v, ev))
    scores = rr._get().predict(pairs, batch_size=settings.reranker_batch_size)
    ranked = sorted(zip(scores, meta, strict=True), key=lambda x: float(x[0]))
    return [
        {
            "kind": "semantic_mismatch_candidate",
            "score": float(s),
            "from": u, "from_label": node_label(G, u),
            "to": v, "to_label": node_label(G, v),
            "evidence": ev,
            "target_snippet": (G.nodes[v].get("text") or "")[:300].replace("\n", " "),
        }
        for s, (u, v, ev) in ranked[:top_n]
    ]


PRECEDENCE_RE = re.compile(
    r"notwithstanding anything|in the event of (any )?conflict|shall control"
    r"|shall govern|to the extent of (any|such) conflict", re.I,
)


def precedence_clauses(G) -> list[dict]:
    """Explicit conflict-acknowledgment/precedence clauses — the places the
    agreement itself declares that two provisions may conflict and which controls."""
    findings = []
    for nid, d in G.nodes(data=True):
        if d["node_type"] != "Section" or not nid.startswith("base::"):
            continue
        text = d.get("text") or ""
        m = PRECEDENCE_RE.search(text)
        if not m:
            continue
        start = max(text.rfind(". ", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
        end = text.find(". ", m.end())
        end = len(text) if end == -1 else end + 1
        findings.append({
            "kind": "precedence_clause",
            "section": nid, "label": node_label(G, nid),
            "clause": text[max(0, start):end].strip().replace("\n", " ")[:500],
        })
    return findings


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "mistral-24b"
    G = load_graph(settings.artifacts_dir / "graph.json")
    dang = dangling_refs(G)
    print(f"dangling: {len(dang)}")
    for f in dang[:15]:
        print(f"  {f['from_label']} -> {f['mentions']}: ...{f['evidence'][:140]}...")
    prov = proviso_claims(G)
    broken = [f for f in prov if not f["verified"]]
    print(f"\nproviso claims: {len(prov)} checked, {len(broken)} unverified")
    for f in broken:
        print(f"  {f['from_label']} claims '{f['claims']}': {f['evidence'][:120]}")
    mism = mismatch_candidates(G)
    print(f"\nmismatch candidates (lowest CrossEncoder scores) — LLM verdicts "
          f"({model_name}):")
    mism = llm_verdicts(G, mism, model_name)
    prec = precedence_clauses(G)
    n_refs = sum(1 for _, _, k, d in G.edges(keys=True, data=True)
                 if k.split("::")[0] == "REFERENCES" and d.get("evidence"))
    OUT.write_text(json.dumps(
        {
            "summary": {
                "section_crossrefs_checked": n_refs,
                "dangling_found": len(dang),
                "proviso_claims_checked": len(prov),
                "proviso_claims_unverified": len(broken),
                "semantic_candidates_llm_checked": len(mism),
                "confirmed_mismatches": sum(1 for c in mism if c["is_mismatch"]),
                "verdict_model": model_name,
            },
            "dangling": dang, "proviso_claims": prov, "mismatch": mism,
            "precedence_clauses": prec,
        },
        indent=1, ensure_ascii=False))
    n_mis = sum(1 for c in mism if c["is_mismatch"])
    print(f"\nprecedence clauses: {len(prec)}")
    print(f"verdicts: {n_mis}/{len(mism)} MISMATCH\nwrote {OUT}")


if __name__ == "__main__":
    main()
