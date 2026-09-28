"""Eval harness over outputs/answers.json.

Tier 1 — retrieval metrics (context recall/precision) computed deterministically
    against the gold node IDs in experiments/mini_eval_set.json. These are the RAGAS
    definitions computed directly; the ragas framework itself is not wired in because
    it would drag a langchain-aws dependency for no measurement benefit.
Tier 2 — legal-specific LLM-as-judge (Bedrock claude-sonnet-4-6; offline only, never
    in the live answer path): citation accuracy, amendment status, version correctness,
    chain completeness, answer correctness, faithfulness.
Tier 3 — structural: latency p50 and p95, subsection-level citation rate, amendment-status
    citation rate, Q9-vs-Q10 temporal consistency.

Writes outputs/eval_results.json and outputs/eval_report.md.

Tier 2 is a single LLM judge, which makes it a measurement instrument of unknown bias.
Four companion scripts check this harness rather than the pipeline, and two of them are
deterministic: eval/citation_verify.py (re-derives citation accuracy from the graph with
no LLM, and finds the judge generous), experiments/temporal_boundary.py (as-of resolution
at every amendment boundary, not just the two dates the question set happens to use),
experiments/judge_agreement.py (a second judge, and a judge with the gold answer
withheld), experiments/ablation.py (per-block contribution on the shipping model).
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from config import settings
from llm.client import BedrockLlm

JUDGE_SYSTEM = (
    "You are grading answers produced by a retrieval system over a credit agreement and "
    "its two amendments. Grade strictly against the gold answer and the retrieved context. "
    "Respond with ONLY a JSON object, no prose, no code fences."
)

JUDGE_PROMPT = """Question{as_of}: {question}

Gold answer (ground truth): {gold_answer}

System answer to grade:
{answer}

Context the system retrieved (the answer must be supported by this):
{context}

Grade the system answer. Return ONLY this JSON object:
{{
 "citation_accuracy": <0.0-1.0, do the bracketed citations point to the correct \
sections/definitions for each claim>,
 "amendment_status_correct": <true|false|null, does the answer label provisions \
correctly as original vs amended (null if not applicable)>,
 "version_correct": <true|false|null, is the operative text the right version for \
the as-of date (null if no as-of date)>,
 "chain_completeness": <0.0-1.0 or null, for chain/trace/list questions: fraction \
of required hops or items present (null if not a chain question)>,
 "answer_correct": <true|false, does the substance match the gold answer>,
 "faithfulness": <0.0-1.0, fraction of the answer's claims supported by the retrieved context>,
 "note": "<one sentence>"
}}"""


def _hier_match(a: str, b: str) -> bool:
    """base::7.05 matches base::7.05(a) in either direction; exact otherwise."""
    return a == b or a.startswith(b + "(") or b.startswith(a + "(")


def _baseline_units(context_ids: list[str], chunks: dict[str, dict]) -> list[dict]:
    return [chunks[cid] for cid in context_ids if cid in chunks]


def _gold_hit_graph(gold: str, context_ids: list[str]) -> bool:
    return any(_hier_match(gold, cid) for cid in context_ids)


def _gold_hit_baseline(gold: str, units: list[dict]) -> bool:
    if gold.startswith("base::def::"):
        term = gold.removeprefix("base::def::")
        return any(
            u["document_id"] == "base"
            and (f"\u201c{term}\u201d" in u["text"] or f"\u201c {term} \u201d" in u["text"])
            for u in units
        )
    if "::exhibit::" in gold:
        letter = gold.split("::")[-1]
        return any(f"Exhibit {letter}" in u["text"] for u in units)
    doc, sec = gold.split("::", 1)
    return any(
        u["document_id"] == doc and _hier_match(sec, u["section_hint"]) for u in units
    )


def _unit_relevant_baseline(unit: dict, golds: list[str]) -> bool:
    return any(_gold_hit_baseline(g, [unit]) for g in golds)


def tier1(rows: list[dict], gold_by_q: dict[str, list[str]], chunks: dict, path: str) -> dict:
    per_q, recalls, precisions = {}, [], []
    for r in rows:
        golds = gold_by_q.get(r["id"], [])
        if not golds:  # q8 has no gold retrieval set
            continue
        if path == "graph":
            hits = [g for g in golds if _gold_hit_graph(g, r["context_ids"])]
            relevant = [c for c in r["context_ids"] if any(_hier_match(g, c) for g in golds)]
            n_units = len(r["context_ids"])
        else:
            units = _baseline_units(r["context_ids"], chunks)
            hits = [g for g in golds if _gold_hit_baseline(g, units)]
            relevant = [u["id"] for u in units if _unit_relevant_baseline(u, golds)]
            n_units = len(units)
        recall = len(hits) / len(golds)
        precision = len(relevant)/n_units if n_units else 0.0
        per_q[r["id"]] = {"recall": round(recall, 3), "precision": round(precision, 3),
                          "missed": [g for g in golds if g not in hits]}
        recalls.append(recall)
        precisions.append(precision)
    return {
        "per_question": per_q,
        "mean_context_recall": round(statistics.mean(recalls), 3) if recalls else None,
        "mean_context_precision": round(statistics.mean(precisions), 3) if precisions else None,
    }


def _parse_judge(raw: str) -> dict:
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


def tier2(rows: list[dict], gold_by_answer: dict[str, str], judge: BedrockLlm) -> dict:
    per_q = {}
    for r in rows:
        context = "\n\n---\n\n".join(
            f"[{b['citation']}]\n{b['text']}" for b in r["context_blocks"]
        )[:12000] or "(the system retrieved no context)"
        prompt = JUDGE_PROMPT.format(
            as_of=f" (as of {r['as_of']})" if r["as_of"] else "",
            question=r["question"],
            gold_answer=gold_by_answer.get(r["id"], "(no gold answer)"),
            answer=r["answer"][:6000],
            context=context,
        )
        raw = judge.generate(JUDGE_SYSTEM, prompt, max_tokens=512)
        try:
            per_q[r["id"]] = _parse_judge(raw)
        except (ValueError, json.JSONDecodeError):
            per_q[r["id"]] = {"error": raw[:300]}
    scored = [v for v in per_q.values() if "error" not in v]

    def _mean(key):
        vals = [v[key] for v in scored if v.get(key) is not None]
        vals = [float(x) for x in vals]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "per_question": per_q,
        "mean_citation_accuracy": _mean("citation_accuracy"),
        "amendment_status_correct_rate": _mean("amendment_status_correct"),
        "version_correct_rate": _mean("version_correct"),
        "mean_chain_completeness": _mean("chain_completeness"),
        "answer_correct_rate": _mean("answer_correct"),
        "mean_faithfulness": _mean("faithfulness"),
    }


def tier3(rows: list[dict]) -> dict:
    if not rows:
        return {
            "latency_p50_s": None, "latency_p95_s": None,
            "subsection_citation_rate": None, "amendment_status_citation_count": 0,
            "q9_q10_answers_differ": None,
        }
    lat = sorted(r["latency_s"] for r in rows)
    cits = [c for r in rows for c in r["citations"]]
    subsec = [c for c in cits if re.search(r"\d+\.\d+\([a-z]+\)|definition of", c)]
    amended = [c for c in cits if "as amended by" in c]
    by_id = {r["id"]: r for r in rows}
    return {
        "latency_p50_s": lat[len(lat) // 2],
        "latency_p95_s": lat[min(len(lat) - 1, round(0.95 * len(lat)))],
        "subsection_citation_rate": round(len(subsec) / len(cits), 3) if cits else 0.0,
        "amendment_status_citation_count": len(amended),
        "q9_q10_answers_differ": (
            by_id["q9"]["answer"] != by_id["q10"]["answer"]
            if "q9" in by_id and "q10" in by_id else None
        ),
    }


def _report(results: dict) -> str:
    lines = ["# Evaluation: graph path vs flat RAG baseline", ""]
    lines.append("| Metric | Graph | Baseline |")
    lines.append("|---|---|---|")
    metric_rows = [
        ("Context recall (mean, T1)", "tier1", "mean_context_recall"),
        ("Context precision (mean, T1)", "tier1", "mean_context_precision"),
        ("Citation accuracy (judge, T2)", "tier2", "mean_citation_accuracy"),
        ("Amendment status correct (T2)", "tier2", "amendment_status_correct_rate"),
        ("Version correct for as-of (T2)", "tier2", "version_correct_rate"),
        ("Chain completeness (T2)", "tier2", "mean_chain_completeness"),
        ("Answer correct rate (T2)", "tier2", "answer_correct_rate"),
        ("Faithfulness (T2)", "tier2", "mean_faithfulness"),
        ("Latency p50 (s, T3)", "tier3", "latency_p50_s"),
        ("Latency p95 (s, T3)", "tier3", "latency_p95_s"),
        ("Subsection-level citation rate (T3)", "tier3", "subsection_citation_rate"),
        ("Citations carrying amendment status (T3)", "tier3",
         "amendment_status_citation_count"),
        ("Q9 vs Q10 answers differ (T3)", "tier3", "q9_q10_answers_differ"),
    ]
    for label, tier, key in metric_rows:
        g = results["graph"][tier].get(key, "skipped")
        b = results["baseline"][tier].get(key, "skipped")
        lines.append(f"| {label} | {g} | {b} |")
    run = results.get("run") or {}
    if run:
        lines.append("")
        lines.append(
            "Run: "
            + ", ".join(f"{k}={v}" for k, v in run.items() if k != "timestamp")
            + f" at {run.get('timestamp', '?')}"
        )
    lines.append("")
    lines.append("## Per-question answer correctness (judge)")
    lines.append("")
    lines.append("| Q | Graph correct | Baseline correct | Judge note (graph) |")
    lines.append("|---|---|---|---|")
    for qid in [f"q{i}" for i in range(1, 13)]:
        g = results["graph"]["tier2"]["per_question"].get(qid, {})
        b = results["baseline"]["tier2"]["per_question"].get(qid, {})
        lines.append(
            f"| {qid} | {g.get('answer_correct')} | {b.get('answer_correct')} "
            f"| {g.get('note', g.get('error', ''))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    answers = json.loads((settings.outputs_dir / "answers.json").read_text())
    eval_set = json.loads(Path("experiments/mini_eval_set.json").read_text())
    gold_by_q = {e["id"]: e["gold"] for e in eval_set}
    gold_answers = {e["id"]: e["gold_answer"] for e in eval_set}
    chunks = {c["id"]: c for c in
              json.loads((settings.artifacts_dir / "chunks_baseline.json").read_text())}
    judge = BedrockLlm(settings.bedrock_model_judge)

    results = {}
    for path in ("graph", "baseline"):
        rows = answers.get(path) or []
        if not rows:
            results[path] = {
                "tier1": {"per_question": {}, "mean_context_recall": None,
                          "mean_context_precision": None},
                "tier2": {"per_question": {}, "answer_correct_rate": None,
                          "mean_faithfulness": None},
                "tier3": tier3([]),
                "skipped": True,
            }
            print(f"{path}: skipped (no answers in this run)", flush=True)
            continue
        results[path] = {
            "tier1": tier1(rows, gold_by_q, chunks, path),
            "tier2": tier2(rows, gold_answers, judge),
            "tier3": tier3(rows),
        }
        print(f"{path}: recall={results[path]['tier1']['mean_context_recall']} "
              f"correct={results[path]['tier2']['answer_correct_rate']} "
              f"faithful={results[path]['tier2']['mean_faithfulness']}", flush=True)

    # Carry the answer run's provenance into the scorecard: a score with no record
    # of the config that produced it cannot be compared against another run.
    results["run"] = answers.get("run", {"tag": "unknown"})
    (settings.outputs_dir / "eval_results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False)
    )
    report = _report(results)
    (settings.outputs_dir / "eval_report.md").write_text(report)
    tag = results["run"].get("tag") or ""
    if tag:
        stamp = str(results["run"].get("timestamp", "")).replace(":", "").replace("-", "")
        archive = settings.outputs_dir / "runs" / f"{stamp}_{tag}"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "eval_results.json").write_text(
            json.dumps(results, indent=1, ensure_ascii=False)
        )
        (archive / "eval_report.md").write_text(report)
        print(f"archived {archive}")
    print("wrote outputs/eval_results.json, eval_report.md")


if __name__ == "__main__":
    main()
