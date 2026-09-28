"""Fine-grained as-of boundary stress test — the gap the 12-question set leaves open.

The assignment's temporal questions use two dates: one before any amendment
(2018-01-01) and one after the Second (2020-06-01). Both are *endpoint-ish*
dates relative to A1/A2, and neither exercises the Third Amendment (2020-12-15)
or the window between Second and Third.

With three instruments the discriminating cases are:

  - 2018-06-26 <= d < 2020-05-13  — First in force, Second/Third not
  - 2020-05-13 <= d < 2020-12-15  — First+Second in force, Third not
  - d >= 2020-12-15               — all three in force

Nine nodes are touched by more than one instrument (A1+A3 or A2+A3), so a
"latest text wins" or "ignore amendments" policy is wrong on different subsets
in each window.

Three checks, all deterministic (no model, no judge, CPU only):

  1. LATTICE — for every amended node, walk each effective date's boundary triple
     (day before, the day itself, day after) plus window sentinels. Assert the ops
     find_effective applied are exactly those with effective_date <= as_of, that
     the inclusive boundary is honoured on the day itself, and that a version once
     adopted never silently reverts on a later date.
  2. WINDOW SENTINELS — at each mid date, assert the right subset is amended vs
     still original (and that future instruments do not leak into citations).
  3. MIXED VINTAGE — at the A1 mid window, definitions amended by First while
     DEPENDS_ON dependents stay original (flat "latest document" cannot express this).

Run: uv run python -m experiments.temporal_boundary
Writes experiments/results/temporal_boundary.json and a markdown summary.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import networkx as nx

from config import settings
from graph.loader import load_graph
from graph.schema import EDGE_AMENDS, EDGE_DEPENDS_ON
from graph.traversal import _amend_ops_before, find_effective, node_label

# Window sentinels between the three instruments.
BEFORE_ALL = date(2017, 12, 31)
MID_A1 = date(2019, 1, 1)       # First in force; Second + Third not
MID_A2 = date(2020, 8, 1)       # First + Second in force; Third not
AFTER_ALL = date(2021, 1, 1)    # all three in force

WINDOWS = (
    {"id": "mid_a1", "date": MID_A1, "label": "between First and Second"},
    {"id": "mid_a2", "date": MID_A2, "label": "between Second and Third"},
)


def amended_nodes(G: nx.MultiDiGraph) -> dict[str, list[dict]]:
    """node id -> its AMENDS ops, oldest first."""
    out: dict[str, list[dict]] = {}
    for _, v, d in G.edges(data=True):
        if d.get("edge_type") == EDGE_AMENDS:
            out.setdefault(v, []).append(d)
    return {k: sorted(v, key=lambda d: d["effective_date"]) for k, v in out.items()}


def _lattice(ops_by_node: dict[str, list[dict]]) -> list[date]:
    """Every effective date, its immediate neighbours, and the window sentinels."""
    stamps = {date.fromisoformat(o["effective_date"])
              for ops in ops_by_node.values() for o in ops}
    dates = {BEFORE_ALL, MID_A1, MID_A2, AFTER_ALL}
    for s in stamps:
        dates |= {s - timedelta(days=1), s, s + timedelta(days=1)}
    return sorted(dates)


def check_lattice(G: nx.MultiDiGraph, ops_by_node: dict[str, list[dict]]) -> dict:
    dates = _lattice(ops_by_node)
    failures, timeline = [], {}
    for nid, ops in sorted(ops_by_node.items()):
        seen_amended_on: date | None = None
        row = []
        for d in dates:
            expected = [o for o in ops if date.fromisoformat(o["effective_date"]) <= d]
            eff = find_effective(G, nid, d)
            applied = _amend_ops_before(G, nid, d)
            if len(applied) != len(expected):
                failures.append({
                    "node_id": nid, "as_of": d.isoformat(), "kind": "op_set_mismatch",
                    "expected_ops": len(expected), "applied_ops": len(applied),
                })
            # inclusive boundary: an amendment is in force ON its effective date
            want_status = "original" if not expected else "amended_or_added"
            got = "original" if eff.status == "original" else "amended_or_added"
            if got != want_status:
                failures.append({
                    "node_id": nid, "as_of": d.isoformat(), "kind": "status_mismatch",
                    "expected": want_status, "got": eff.status,
                })
            if got == "amended_or_added" and seen_amended_on is None:
                seen_amended_on = d
            if got == "original" and seen_amended_on is not None:
                failures.append({
                    "node_id": nid, "as_of": d.isoformat(), "kind": "version_reverted",
                    "first_amended_on": seen_amended_on.isoformat(),
                })
            # Future AMENDS instruments must not appear as "as amended by …" on base nodes.
            # Amendment-document node citations always name their own instrument in the
            # document title — that is not leakage.
            for o in ops:
                ed = date.fromisoformat(o["effective_date"])
                marker = f"as amended by {o['instrument']}"
                if ed > d and marker in eff.citation:
                    failures.append({
                        "node_id": nid, "as_of": d.isoformat(),
                        "kind": "future_instrument_in_citation",
                        "instrument": o["instrument"], "citation": eff.citation,
                    })
            row.append({"as_of": d.isoformat(), "status": eff.status,
                        "ops_applied": len(applied), "len_text": len(eff.text)})
        timeline[nid] = row
    return {"dates_tested": [d.isoformat() for d in dates],
            "nodes_tested": len(ops_by_node),
            "assertions": len(ops_by_node) * len(dates) * 3,
            "failures": failures, "timeline": timeline}


def check_window(
    G: nx.MultiDiGraph, ops_by_node: dict[str, list[dict]], as_of: date, window_id: str,
) -> dict:
    """At as_of, ops with effective_date <= as_of must apply; later ones must not leak."""
    rows, failures = [], []
    for nid, ops in sorted(ops_by_node.items()):
        expected = [o for o in ops if date.fromisoformat(o["effective_date"]) <= as_of]
        in_force = bool(expected)
        pre = find_effective(G, nid, BEFORE_ALL)
        cur = find_effective(G, nid, as_of)
        post = find_effective(G, nid, AFTER_ALL)
        applied = _amend_ops_before(G, nid, as_of)
        instruments_expected = sorted({o["instrument"] for o in expected})
        instruments_future = sorted({
            o["instrument"] for o in ops
            if date.fromisoformat(o["effective_date"]) > as_of
        })
        row = {
            "node_id": nid, "label": node_label(G, nid),
            "instruments": sorted({o["instrument"] for o in ops}),
            "instruments_in_force": instruments_expected,
            "in_force": in_force,
            "ops_expected": len(expected), "ops_applied": len(applied),
            "status": cur.status,
            "equals_pre": cur.text == pre.text,
            "equals_post": cur.text == post.text,
            "citation": cur.citation,
        }
        if len(applied) != len(expected):
            failures.append({
                "node_id": nid, "window": window_id, "kind": "op_count_mismatch",
                "expected": len(expected), "got": len(applied),
            })
        if in_force:
            if row["equals_pre"]:
                failures.append({
                    "node_id": nid, "window": window_id, "kind": "amendment_not_applied",
                    "detail": "ops in force but text identical to original",
                })
            if cur.status == "original":
                failures.append({
                    "node_id": nid, "window": window_id, "kind": "status_still_original",
                })
        else:
            if not row["equals_pre"]:
                failures.append({
                    "node_id": nid, "window": window_id, "kind": "future_amendment_leaked",
                    "detail": "no ops in force yet but text already changed",
                })
            if cur.status != "original":
                failures.append({
                    "node_id": nid, "window": window_id, "kind": "status_not_original",
                    "got": cur.status,
                })
            if "as amended" in cur.citation:
                failures.append({
                    "node_id": nid, "window": window_id, "kind": "future_citation_leaked",
                    "citation": cur.citation,
                })
        for inst in instruments_future:
            marker = f"as amended by {inst}"
            if marker in cur.citation:
                failures.append({
                    "node_id": nid, "window": window_id,
                    "kind": "future_instrument_in_citation",
                    "instrument": inst, "citation": cur.citation,
                })
        rows.append(row)
    return {
        "window_id": window_id,
        "as_of": as_of.isoformat(),
        "amended": sum(r["in_force"] for r in rows),
        "still_original": sum(not r["in_force"] for r in rows),
        "failures": failures,
        "rows": rows,
    }


def check_windows(G: nx.MultiDiGraph, ops_by_node: dict[str, list[dict]]) -> dict:
    windows = [
        check_window(G, ops_by_node, w["date"], w["id"]) for w in WINDOWS
    ]
    return {
        "windows": windows,
        "failures": [f for w in windows for f in w["failures"]],
    }


def check_mixed_vintage(G: nx.MultiDiGraph, ops_by_node: dict[str, list[dict]]) -> dict:
    """Chains where a node and the definitions it relies on have different vintages
    at the A1 mid date. Flat retrieval cannot express this state at all."""
    early = {n for n, ops in ops_by_node.items()
             if date.fromisoformat(ops[0]["effective_date"]) <= MID_A1}
    chains, failures = [], []
    for term in sorted(early):
        if G.nodes[term].get("node_type") != "Definition":
            continue
        dependents = sorted({u for u, _, k in G.in_edges(term, keys=True)
                             if k == EDGE_DEPENDS_ON})
        term_mid = find_effective(G, term, MID_A1)
        for dep in dependents:
            dep_mid = find_effective(G, dep, MID_A1)
            dep_post = find_effective(G, dep, AFTER_ALL)
            chains.append({
                "input": term, "input_status_at_mid": term_mid.status,
                "dependent": dep, "dependent_status_at_mid": dep_mid.status,
                "dependent_status_after_all": dep_post.status,
                "mixed_vintage_at_mid": (term_mid.status != "original"
                                         and dep_mid.status == "original"),
                "dependent_citation_at_mid": dep_mid.citation,
            })
            if term_mid.status == "original":
                failures.append({"input": term, "kind": "input_not_amended_at_mid"})
    return {
        "as_of": MID_A1.isoformat(),
        "chains": chains,
        "mixed_vintage_chains": sum(c["mixed_vintage_at_mid"] for c in chains),
        "failures": failures,
    }


def _report(res: dict) -> str:
    lat, wins, mix = res["lattice"], res["windows"], res["mixed_vintage"]
    n_fail = (len(lat["failures"]) + len(wins["failures"]) + len(mix["failures"]))
    ok = n_fail == 0
    lines = [
        "# As-of boundary stress test",
        "",
        f"**{'PASS' if ok else 'FAIL'}** — {lat['assertions']} lattice assertions over "
        f"{lat['nodes_tested']} amended nodes × {len(lat['dates_tested'])} dates, "
        f"{n_fail} failures.",
        "",
        "The 12-question set only asks for 2018-01-01 (before First) and 2020-06-01 "
        "(after Second, still before Third). This test covers every amendment boundary "
        "the graph actually has — day-before / day-of / day-after for First, Second, and "
        "Third — plus two mid-window sentinels.",
        "",
        "## Dates tested",
        "",
        "`" + "`, `".join(lat["dates_tested"]) + "`",
        "",
    ]
    for w in wins["windows"]:
        meta = next(x for x in WINDOWS if x["id"] == w["window_id"])
        lines += [
            f"## Window `{w['as_of']}` — {meta['label']}",
            "",
            f"{w['amended']} nodes must read as amended and {w['still_original']} must "
            "still read as originally executed (future instruments must not leak into "
            "citations).",
            "",
            "| Node | Instruments | In force at window | Status | Citation |",
            "|---|---|---|---|---|",
        ]
        for r in w["rows"]:
            if r["in_force"]:
                lines.append(
                    f"| {r['label']} | {', '.join(r['instruments'])} | "
                    f"{', '.join(r['instruments_in_force']) or '—'} | "
                    f"{r['status']} | {r['citation']} |"
                )
        lines += [
            "",
            f"({w['still_original']} not-yet-effective nodes verified as still "
            "`original` at this window.)",
            "",
        ]
    lines += [
        f"## Mixed vintage ({mix['as_of']})",
        "",
        f"{mix['mixed_vintage_chains']} of {len(mix['chains'])} dependency chains are "
        "mixed-vintage: the definition a provision relies on is amended while the "
        "provision's own text is not.",
        "",
        "| Amended input | Dependent | Dependent at mid | Dependent after all |",
        "|---|---|---|---|",
    ]
    for c in mix["chains"]:
        lines.append(
            f"| {c['input']} ({c['input_status_at_mid']}) | {c['dependent']} | "
            f"{c['dependent_status_at_mid']} | {c['dependent_status_after_all']} |"
        )
    if not ok:
        lines += [
            "", "## Failures", "", "```",
            json.dumps(
                lat["failures"] + wins["failures"] + mix["failures"], indent=1,
            ),
            "```",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    G = load_graph(settings.artifacts_dir / "graph.json")
    ops_by_node = amended_nodes(G)
    res = {
        "lattice": check_lattice(G, ops_by_node),
        "windows": check_windows(G, ops_by_node),
        "mixed_vintage": check_mixed_vintage(G, ops_by_node),
    }
    # Keep a mid_window alias pointing at the A1 window for older readers of the JSON.
    res["mid_window"] = {
        "mid_date": MID_A1.isoformat(),
        "amended_at_mid": res["windows"]["windows"][0]["amended"],
        "still_original_at_mid": res["windows"]["windows"][0]["still_original"],
        "failures": res["windows"]["windows"][0]["failures"],
        "rows": res["windows"]["windows"][0]["rows"],
    }
    out = settings.artifacts_dir.parent / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "temporal_boundary.json").write_text(json.dumps(res, indent=1))
    (out / "temporal_boundary.md").write_text(_report(res))

    n_fail = (
        len(res["lattice"]["failures"])
        + len(res["windows"]["failures"])
        + len(res["mixed_vintage"]["failures"])
    )
    print(f"dates tested : {len(res['lattice']['dates_tested'])}")
    print(f"  {res['lattice']['dates_tested']}")
    print(f"nodes tested : {res['lattice']['nodes_tested']}")
    print(f"assertions   : {res['lattice']['assertions']}")
    for w in res["windows"]["windows"]:
        print(f"window {w['as_of']}: {w['amended']} amended, "
              f"{w['still_original']} still original, "
              f"{len(w['failures'])} failures")
    print(f"mixed-vintage chains: {res['mixed_vintage']['mixed_vintage_chains']}"
          f"/{len(res['mixed_vintage']['chains'])}")
    print(f"{'PASS' if n_fail == 0 else f'FAIL ({n_fail} failures)'}")
    for bucket in (res["lattice"], res["windows"], res["mixed_vintage"]):
        for f in bucket["failures"][:12]:
            print("  ", f)
    print("wrote experiments/results/temporal_boundary.json, .md")


if __name__ == "__main__":
    main()
