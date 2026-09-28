"""Deterministic citation verification — no LLM judge in the loop.

Tier 2's `citation_accuracy` is scored by Bedrock Sonnet, which can be lenient or
hallucinate agreement. This module re-scores the same claim independently and
deterministically, so the headline citation number has a second, non-LLM witness.

Three checks per bracketed citation found in an answer:

  1. RESOLVABLE — does the cited label map to a node that exists in the graph?
     A citation to "§7.99" or a definition the agreement never defines is a
     fabrication and is counted as such.
  2. VERSION-CONSISTENT — the citation carries its own amendment claim ("as
     originally executed" / "as amended by <instrument>"). Recompute the true
     status with find_effective(node, as_of) and compare. This catches a model
     that retrieves the right section but mislabels which version it is quoting.
     A citation naming an instrument whose effective date is LATER than the
     question's as-of date is flagged separately as anachronistic: quoting a
     provision from the future is the specific failure this whole system exists
     to prevent, and it is checkable by date arithmetic alone.
  3. GROUNDED — token overlap between the sentence making the claim and the cited
     node's operative text. Reported as recall of the claim's content words
     (stopwords and the citation itself stripped) against the section text, so a
     citation pasted next to an unrelated assertion scores low.

Run: uv run python -m eval.citation_verify
Writes outputs/citation_verify.json (+ a short markdown table).
"""
from __future__ import annotations

import json
import re
import statistics
from datetime import date

import networkx as nx

from config import settings
from graph.loader import load_graph
from graph.traversal import find_effective

# Graph path (graph/traversal.py find_effective):
#   "[Credit Agreement §7.05(a), as amended by Second Amendment (effective 2020-05-13)]"
#   "[Credit Agreement definition of “Borrowing” (§1.01), as originally executed ...]"
#   "[Second Amendment to Third Amended and Restated Credit Agreement §1.16, dated ...]"
# Baseline path (baseline/rag.py): "[Third Amended ... Credit Agreement, near §7.05(a)]"
# — a chunk-level hint by construction, with no version qualifier at all.
BRACKET_RE = re.compile(r"\[([^\[\]]{4,300}?)\]")
# No trailing \b: ")" followed by "," is not a word boundary, which silently
# truncated "§7.05(a)" to "§7.05" and resolved it to the wrong (unamended) node.
# The bare-roman branch refuses "§X.XX" so the prompt's own format placeholder,
# when a model echoes it verbatim, does not resolve onto Article X.
SEC_RE = re.compile(
    r"\u00a7\s*("
    r"\d{1,2}\.\d{1,2}(?:\([a-z0-9ivx]{1,4}\))*"
    r"|\d{1,2}(?!\.)"
    r"|[IVX]{1,4}(?![.\w])"
    r")"
)
DEF_RE = re.compile(
    r"definition of\s+[\u201c\"']\s*([^\u201d\"']{2,60}?)\s*[\u201d\"']", re.I)
EXHIBIT_RE = re.compile(r"Exhibit\s+([A-Z])\b")

# Citations the pipeline emits for synthesised blocks rather than a single node.
# They are legitimate but have no single target to resolve against.
SYNTHETIC_PREFIXES = (
    "Amendment catalog",
    "Change history",
    "Cross-reference audit",
    "Provisions that depend on",
)


def is_citation_shaped(text: str) -> bool:
    """The answer prompt reserves square brackets for citations. A bracket that
    names no provision, definition or exhibit is a format violation (usually leaked
    scaffolding such as "[Output Generation]"), not a fabricated citation — the two
    are counted separately so neither inflates the other."""
    return bool(
        text.startswith(SYNTHETIC_PREFIXES)
        or SEC_RE.search(text) or DEF_RE.search(text) or EXHIBIT_RE.search(text)
    )

STOP = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "did",
    "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into",
    "is", "it", "it's", "its", "may", "must", "no", "not", "of", "on", "or", "shall",
    "should", "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "this", "to", "under", "was", "were", "what", "when", "which", "who", "will",
    "with", "would", "you", "your",
))


def _doc_of(citation: str) -> str:
    """Which document the citation names → graph document id.

    The document is named by the head of the citation, before the section symbol:
    "Credit Agreement §7.05(a), as amended by Second Amendment" is a BASE section
    (the trailing instrument is the amendment claim, not the host document).
    """
    head = citation.split("\u00a7")[0].lower()
    if head.startswith("first amendment"):
        return "amendment_1"
    if head.startswith("second amendment"):
        return "amendment_2"
    return "base"


def resolve(G: nx.MultiDiGraph, citation: str) -> str | None:
    """Citation string → graph node id, or None if it names nothing real."""
    if (m := DEF_RE.search(citation)):
        term = m.group(1).strip()
        nid = f"base::def::{term}"
        if G.has_node(nid):
            return nid
        # tolerate straight-vs-curly apostrophes and case
        low = term.lower().replace("\u2019", "'")
        for n, d in G.nodes(data=True):
            if (d.get("node_type") == "Definition"
                    and d.get("term", "").lower().replace("\u2019", "'") == low):
                return n
        return None
    if (m := EXHIBIT_RE.search(citation)):
        nid = f"base::exhibit::{m.group(1)}"
        return nid if G.has_node(nid) else None
    if (m := SEC_RE.search(citation)):
        num, doc = m.group(1), _doc_of(citation)
        for cand in (f"{doc}::{num}", f"base::{num}"):
            if G.has_node(cand):
                return cand
        if num == "0" and G.has_node(f"{doc}::preamble"):
            return f"{doc}::preamble"          # node_label renders the preamble as §0
        for n, d in G.nodes(data=True):
            if (d.get("node_type") == "Section" and d.get("number") == num
                    and d.get("document_id") == doc):
                return n
        return None
    return None


def claimed_status(citation: str) -> str | None:
    """The amendment claim the citation makes about itself, or None if it makes
    none. The baseline's "near §X" chunk hints are always None: a reader cannot
    tell from them which version of the provision is being quoted."""
    c = citation.lower()
    if "as originally executed" in c:
        return "original"
    if "as amended by" in c or "proviso added by" in c:
        return "amended"
    if "dated " in c:
        return "original"           # amendment instrument cited in its own right
    return None


EFFECTIVE_RE = re.compile(r"(?:effective|dated)\s+(\d{4}-\d{2}-\d{2})")


def anachronism(citation: str, as_of: date | None) -> str | None:
    """The effective date the citation claims, if it postdates the question date.

    Cheap, fully deterministic, and the highest-value check in the module: an
    answer to "as of 2018-01-01" that cites an instrument effective 2020-05-13 is
    quoting law that did not yet exist, no matter how fluent the prose is.
    """
    if as_of is None:
        return None
    for stamp in EFFECTIVE_RE.findall(citation):
        if date.fromisoformat(stamp) > as_of:
            return stamp
    return None


def _sentences_citing(answer: str, citation: str) -> list[str]:
    """Sentences in the answer that carry this citation (the claims it supports)."""
    out = []
    for chunk in re.split(r"(?<=[.;:])\s+|\n", answer):
        if citation[:60] in chunk:
            out.append(re.sub(re.escape(f"[{citation}]"), " ", chunk))
    return out


def _grounding(claim: str, node_text: str) -> float:
    """Recall of the claim's content words in the cited text."""
    words = {w for w in re.findall(r"[a-z0-9.$%()]+", claim.lower()) if w not in STOP}
    words = {w for w in words if len(w) > 2}
    if not words:
        return 1.0
    target = set(re.findall(r"[a-z0-9.$%()]+", node_text.lower()))
    return len(words & target) / len(words)


def verify_row(G: nx.MultiDiGraph, row: dict, grounding_floor: float = 0.30) -> dict:
    as_of = date.fromisoformat(row["as_of"]) if row.get("as_of") else None
    # Citations the answer actually asserts (bracketed in prose), not everything the
    # pipeline happened to retrieve — that is what a reader checks.
    brackets = [b.strip() for b in dict.fromkeys(BRACKET_RE.findall(row["answer"]))]
    asserted = [b for b in brackets if is_citation_shaped(b)]
    malformed = [b for b in brackets if not is_citation_shaped(b)]
    checks, unresolved, version_bad, ungrounded, anachronistic = [], [], [], [], []
    for cit in asserted:
        if cit.startswith(SYNTHETIC_PREFIXES):
            checks.append({"citation": cit, "kind": "synthetic_block", "ok": True})
            continue
        if (future := anachronism(cit, as_of)):
            anachronistic.append({"citation": cit, "claims_effective": future,
                                  "as_of": as_of.isoformat()})
        nid = resolve(G, cit)
        if nid is None:
            unresolved.append(cit)
            checks.append({"citation": cit, "kind": "unresolvable", "ok": False,
                           "version_qualified": False})
            continue
        eff = find_effective(G, nid, as_of)
        claim = claimed_status(cit)
        # "added" is how find_effective labels a provision that exists only because
        # an amendment inserted it — a citation calling that "amended" is correct.
        ver_ok = claim is not None and (
            claim == eff.status or (claim == "amended" and eff.status == "added")
        )
        claim_sents = _sentences_citing(row["answer"], cit)
        ground = max((_grounding(c, eff.text) for c in claim_sents), default=None)
        g_ok = ground is None or ground >= grounding_floor
        if claim is not None and not ver_ok:
            version_bad.append({"citation": cit, "claimed": claim,
                                "actual": eff.status, "node_id": nid})
        if not g_ok:
            ungrounded.append({"citation": cit, "grounding": round(ground, 3)})
        checks.append({
            "citation": cit, "node_id": nid,
            "kind": "resolved" if claim else "chunk_hint_unversioned",
            "version_qualified": claim is not None,
            "claimed_status": claim, "actual_status": eff.status,
            "version_ok": ver_ok,
            "grounding": round(ground, 3) if ground is not None else None,
            # verified = resolves to a real node, states which version it quotes,
            # and that version is what the graph says it is as of the question date
            "ok": ver_ok,
            "grounded": g_ok,
        })
    real = [c for c in checks if c["kind"] != "synthetic_block"]
    n = len(real) or 1
    versioned = [c for c in real if c.get("version_qualified")]
    grounds = [c["grounding"] for c in real if c.get("grounding") is not None]
    return {
        "id": row["id"],
        "asserted_citations": len(asserted),
        "malformed_brackets": malformed,
        "resolvable_rate": round(sum(c["kind"] != "unresolvable" for c in real) / n, 3),
        "version_qualified_rate": round(len(versioned) / n, 3),
        "version_consistent_rate": (
            round(sum(bool(c["version_ok"]) for c in versioned) / len(versioned), 3)
            if versioned else None),
        "mean_grounding": round(statistics.mean(grounds), 3) if grounds else None,
        "verified_citation_rate": round(sum(bool(c["ok"]) for c in real) / n, 3),
        "unresolvable": unresolved,
        "version_mismatches": version_bad,
        "anachronistic": anachronistic,
        "weakly_grounded": ungrounded,
        "checks": checks,
    }


def verify_path(G: nx.MultiDiGraph, rows: list[dict]) -> dict:
    per_q = {r["id"]: verify_row(G, r) for r in rows}

    def _mean(key: str) -> float | None:
        vals = [v[key] for v in per_q.values() if v[key] is not None]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "per_question": per_q,
        "mean_resolvable_rate": _mean("resolvable_rate"),
        "mean_version_qualified_rate": _mean("version_qualified_rate"),
        "mean_version_consistent_rate": _mean("version_consistent_rate"),
        "mean_grounding": _mean("mean_grounding"),
        "mean_verified_citation_rate": _mean("verified_citation_rate"),
        "total_asserted_citations": sum(v["asserted_citations"] for v in per_q.values()),
        "total_unresolvable": sum(len(v["unresolvable"]) for v in per_q.values()),
        "total_version_mismatches": sum(
            len(v["version_mismatches"]) for v in per_q.values()),
        "total_anachronistic": sum(len(v["anachronistic"]) for v in per_q.values()),
        "questions_with_as_of": sum(
            1 for r in rows if r.get("as_of")),
        "total_weakly_grounded": sum(len(v["weakly_grounded"]) for v in per_q.values()),
        "total_malformed_brackets": sum(
            len(v["malformed_brackets"]) for v in per_q.values()),
    }


def _report(results: dict, judge: dict | None) -> str:
    lines = [
        "# Deterministic citation verification",
        "",
        "Graph-resolved, no LLM in the loop. Every bracketed citation in the committed",
        "answers is resolved to a node, its amendment claim is recomputed with",
        "`find_effective`, and its claim sentence is scored for token grounding.",
        "",
        "| Metric | Graph | Baseline |",
        "|---|---|---|",
    ]
    for label, key in [
        ("Citations asserted in answers", "total_asserted_citations"),
        ("Resolvable to a real node", "mean_resolvable_rate"),
        ("States which version it quotes", "mean_version_qualified_rate"),
        ("That version claim is correct", "mean_version_consistent_rate"),
        ("VERIFIED (resolvable, versioned, version correct)",
         "mean_verified_citation_rate"),
        ("Citations naming a provision that does not exist", "total_unresolvable"),
        ("Amendment-status mismatches", "total_version_mismatches"),
        ("Anachronistic citations (instrument postdates the as-of date)",
         "total_anachronistic"),
        ("Token grounding of the supported claim", "mean_grounding"),
        ("Claims below the 0.30 grounding floor", "total_weakly_grounded"),
        ("Malformed brackets (non-citation text in [])", "total_malformed_brackets"),
    ]:
        lines.append(f"| {label} | {results['graph'][key]} | {results['baseline'][key]} |")
    lines += [
        "",
        "`States which version it quotes` is the load-bearing row, and it is the whole",
        "argument for the graph. The baseline emits `near \u00a7X` chunk hints by",
        "construction (baseline/rag.py), so a reader cannot tell which version of a",
        "provision it quoted \u2014 there is nothing to check, and a citation that cannot be",
        "checked cannot be trusted. The graph path carries an explicit `as originally",
        "executed` / `as amended by <instrument>` qualifier, and this module recomputes",
        "that qualifier independently from the AMENDS edges and the question's as-of",
        "date. Grounding is reported as a diagnostic rather than folded into the",
        "verified rate: the 0.30 floor is a heuristic, and legitimately synthesising",
        "prose across several provisions lowers per-sentence token overlap.",
    ]
    if judge:
        lines += [
            "",
            "## Deterministic check vs the LLM judge",
            "",
            "| Path | Judge citation_accuracy (Sonnet) | Verified rate (this module) | Delta |",
            "|---|---|---|---|",
        ]
        for p in ("graph", "baseline"):
            j = judge[p]["tier2"].get("mean_citation_accuracy")
            d = results[p]["mean_verified_citation_rate"]
            delta = round(d - j, 3) if (j is not None and d is not None) else None
            lines.append(f"| {p} | {j} | {d} | {delta} |")
        lines += [
            "",
            "A negative delta means the judge was more generous than a graph-resolved",
            "check. Reporting it is the point: the judge scores whether a citation looks",
            "apt, this module scores whether it resolves and whether its version claim",
            "survives recomputation.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    G = load_graph(settings.artifacts_dir / "graph.json")
    answers = json.loads((settings.outputs_dir / "answers.json").read_text())
    results = {p: verify_path(G, answers[p]) for p in ("graph", "baseline")}

    judge_path = settings.outputs_dir / "eval_results.json"
    judge = json.loads(judge_path.read_text()) if judge_path.exists() else None

    (settings.outputs_dir / "citation_verify.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False))
    (settings.outputs_dir / "citation_verify.md").write_text(_report(results, judge))

    for p in ("graph", "baseline"):
        r = results[p]
        print(f"{p}: asserted={r['total_asserted_citations']} "
              f"resolvable={r['mean_resolvable_rate']} "
              f"versioned={r['mean_version_qualified_rate']} "
              f"version_correct={r['mean_version_consistent_rate']} "
              f"verified={r['mean_verified_citation_rate']} "
              f"grounding={r['mean_grounding']} "
              f"nonexistent={r['total_unresolvable']} "
              f"anachronistic={r['total_anachronistic']} "
              f"malformed={r['total_malformed_brackets']}")
    print("wrote outputs/citation_verify.json, citation_verify.md")


if __name__ == "__main__":
    main()
