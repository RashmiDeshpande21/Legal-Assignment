"""Is the 12/12 headline a property of the pipeline, or of the judge?

Tier 2 reports one number from one judge (Bedrock Sonnet 4.6). A single LLM judge is
a measurement instrument with unknown bias, so the headline score is only as
trustworthy as that instrument. This script re-grades the same committed answers
under three arms and reports how much the verdict moves.

  A  primary    — Sonnet 4.6, the exact Tier-2 configuration (the committed number)
  B  second     — Opus 4.6, same prompt, different model. Cross-model agreement.
  C  gold-blind — Sonnet 4.6 with the gold answer withheld. This is the validity
                  check that matters most: if C agrees with A, the judge is really
                  reading the answer and the context; if C diverges, arm A's verdict
                  was being carried by the gold answer rather than by legal
                  reasoning, and the rubric needs the gold to mean anything.

Reported per field: raw agreement, Cohen's kappa for the booleans, mean absolute
difference and Pearson r for the 0-1 scores. Both answer paths are graded (24
answers) because the graph path alone is near-ceiling on answer_correct, and a
degenerate marginal makes kappa uninformative — see the caveat in the report.

Honest limitation: arms A and B are both Anthropic models and will share some
inductive biases, so B is a weaker independence check than a judge from a different
family would be. Bedrock is the only provider configured here, and Claude never
touches the live answer path either way.

Run: uv run python -m experiments.judge_agreement          # all three arms
     JUDGE_ARMS=primary,second uv run python -m experiments.judge_agreement
     REPORT_ONLY=1 uv run python -m experiments.judge_agreement   # re-render only
Writes experiments/results/judge_agreement.json and a markdown summary.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path

from config import settings
from eval.harness import JUDGE_PROMPT, JUDGE_SYSTEM, _parse_judge
from llm.client import BedrockLlm

EVAL_SET = Path("experiments/mini_eval_set.json")

# Same rubric, but the gold answer is withheld so the judge must work from the
# question, the answer and the retrieved context alone.
GOLD_BLIND_PROMPT = JUDGE_PROMPT.replace(
    "Gold answer (ground truth): {gold_answer}",
    "(No reference answer is provided. Grade against the retrieved context and the "
    "credit agreement's own terms.)",
).replace(
    '"answer_correct": <true|false, does the substance match the gold answer>',
    '"answer_correct": <true|false, is the answer substantively correct and fully '
    'supported by the retrieved context>',
)

BOOL_FIELDS = ("answer_correct", "amendment_status_correct", "version_correct")
SCORE_FIELDS = ("citation_accuracy", "faithfulness", "chain_completeness")

ARMS = {
    # name: (model, prompt template, uses gold answer)
    "primary": (settings.bedrock_model_judge, JUDGE_PROMPT, True),
    "second": (settings.bedrock_model, JUDGE_PROMPT, True),
    "gold_blind": (settings.bedrock_model_judge, GOLD_BLIND_PROMPT, False),
}


def _cohen_kappa(a: list[bool], b: list[bool]) -> tuple[float | None, str]:
    """Kappa on paired binary labels, plus a note when the statistic degenerates.

    With n=24 and one label dominating, kappa can read near 0 while raw agreement is
    near 1 (the "kappa paradox"): chance agreement is already almost total, so there
    is no headroom left for the statistic to credit. Reporting kappa without that
    caveat would understate a genuinely consistent judge, so both are reported.
    """
    n = len(a)
    if n == 0:
        return None, "no paired observations"
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None, ("undefined: both arms assigned the same label to every item, "
                      f"so chance agreement is 100% (raw agreement {po:.3f})")
    kappa = (po - pe) / (1 - pe)
    note = ""
    if po > 0.9 and kappa < 0.5:
        note = ("kappa paradox: agreement is high but one label dominates the "
                "marginal, which deflates kappa")
    return round(kappa, 3), note


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return round(num / (da * db), 3) if da and db else None


def grade(arm: str, rows: list[dict], gold_by_answer: dict[str, str],
          path: str) -> dict:
    model, template, use_gold = ARMS[arm]
    judge = BedrockLlm(model)
    out = {}
    for r in rows:
        context = "\n\n---\n\n".join(
            f"[{b['citation']}]\n{b['text']}" for b in r["context_blocks"]
        )[:12000] or "(the system retrieved no context)"
        kwargs = dict(
            as_of=f" (as of {r['as_of']})" if r["as_of"] else "",
            question=r["question"], answer=r["answer"][:6000], context=context,
        )
        if use_gold:
            kwargs["gold_answer"] = gold_by_answer.get(r["id"], "(no gold answer)")
        raw = judge.generate(JUDGE_SYSTEM, template.format(**kwargs), max_tokens=512)
        try:
            out[r["id"]] = _parse_judge(raw)
        except (ValueError, json.JSONDecodeError):
            out[r["id"]] = {"error": raw[:300]}
        v = out[r["id"]]
        print(f"  [{arm}/{path}] {r['id']}: correct={v.get('answer_correct')} "
              f"cite={v.get('citation_accuracy')} faith={v.get('faithfulness')}",
              flush=True)
    return out


def compare(arm_a: str, arm_b: str, grades: dict) -> dict:
    """Field-by-field agreement between two arms, pooled over both answer paths."""
    fields: dict[str, dict] = {}
    disagreements = []
    for field in BOOL_FIELDS:
        xa, xb, ids = [], [], []
        for path in grades[arm_a]:
            for qid, va in grades[arm_a][path].items():
                vb = grades[arm_b][path].get(qid, {})
                if va.get(field) is None or vb.get(field) is None:
                    continue
                xa.append(bool(va[field]))
                xb.append(bool(vb[field]))
                ids.append(f"{path}/{qid}")
        if not xa:
            fields[field] = {"n": 0}
            continue
        agree = [x == y for x, y in zip(xa, xb, strict=True)]
        kappa, note = _cohen_kappa(xa, xb)
        fields[field] = {
            "n": len(xa), "agreement": round(sum(agree) / len(agree), 3),
            "cohen_kappa": kappa, "kappa_note": note,
            f"{arm_a}_true_rate": round(sum(xa) / len(xa), 3),
            f"{arm_b}_true_rate": round(sum(xb) / len(xb), 3),
        }
        disagreements += [
            {"item": i, "field": field, arm_a: x, arm_b: y}
            for i, x, y, ok in zip(ids, xa, xb, agree, strict=True) if not ok
        ]
    for field in SCORE_FIELDS:
        xa, xb, ids = [], [], []
        for path in grades[arm_a]:
            for qid, va in grades[arm_a][path].items():
                vb = grades[arm_b][path].get(qid, {})
                if va.get(field) is None or vb.get(field) is None:
                    continue
                xa.append(float(va[field]))
                xb.append(float(vb[field]))
                ids.append(f"{path}/{qid}")
        if not xa:
            fields[field] = {"n": 0}
            continue
        diffs = [abs(x - y) for x, y in zip(xa, xb, strict=True)]
        fields[field] = {
            "n": len(xa),
            "mean_abs_diff": round(statistics.mean(diffs), 3),
            "max_abs_diff": round(max(diffs), 3),
            "pearson_r": _pearson(xa, xb),
            f"{arm_a}_mean": round(statistics.mean(xa), 3),
            f"{arm_b}_mean": round(statistics.mean(xb), 3),
            "within_0.1": round(sum(d <= 0.1 for d in diffs) / len(diffs), 3),
        }
        disagreements += [
            {"item": i, "field": field, arm_a: x, arm_b: y, "diff": round(abs(x - y), 3)}
            for i, x, y in zip(ids, xa, xb, strict=True) if abs(x - y) > 0.25
        ]
    return {"fields": fields, "notable_disagreements": disagreements}


def _headline(grades: dict, arm: str, path: str) -> str:
    g = grades[arm][path]
    ok = sum(1 for v in g.values() if v.get("answer_correct"))
    return f"{ok}/{len(g)}"


def _asymmetry(cmp: dict, a: str, b: str) -> list[str]:
    """Whether the disagreements land on one answer path and run one way.

    A judge that disagrees with itself symmetrically across both paths is noisy. A
    judge whose disagreements all land on one path, all in one direction, is
    measuring something systematic — and that is worth naming in the report.
    """
    flips = [d for d in cmp["notable_disagreements"] if d["field"] == "answer_correct"]
    if not flips:
        return []
    by_path: dict[str, int] = {}
    for d in flips:
        by_path[d["item"].split("/")[0]] = by_path.get(d["item"].split("/")[0], 0) + 1
    up = sum(1 for d in flips if not d[a] and d[b])
    out = ["", f"All {len(flips)} `answer_correct` flips by path: "
               + ", ".join(f"{k} {v}" for k, v in sorted(by_path.items())) + "."]
    if up == len(flips) and len(by_path) == 1:
        path = next(iter(by_path))
        out.append(
            f"Every flip is on the **{path}** path and every one runs the same way "
            f"({a}=False, {b}=True). That is not judge noise. Withholding the gold "
            f"answer makes the {path} path look substantially better while leaving "
            "the other path's score untouched, which means the reference answer is "
            "doing the discriminating work: the failing answers are fluent and "
            "consistent with the chunks they retrieved, so a judge scoring only "
            "internal support accepts them. An LLM-as-judge RAG eval without ground "
            "truth would have reported these two systems as near-equivalent."
        )
    return out


def _report(grades: dict, comparisons: dict) -> str:
    arms = list(grades)
    lines = [
        "# Judge calibration and agreement",
        "",
        "The Tier-2 headline comes from one judge. These arms re-grade the same",
        "committed answers to show how much of that number is the pipeline and how",
        "much is the instrument.",
        "",
        "| Arm | Judge model | Gold shown | answer_correct (graph) | answer_correct (baseline) |",
        "|---|---|---|---|---|",
    ]
    for arm in arms:
        model, _, use_gold = ARMS[arm]
        lines.append(f"| {arm} | `{model}` | {'yes' if use_gold else 'no'} | "
                     f"{_headline(grades, arm, 'graph')} | "
                     f"{_headline(grades, arm, 'baseline')} |")
    for pair, cmp in comparisons.items():
        a, b = pair.split("__vs__")
        lines += [
            "",
            f"## {a} vs {b}",
            "",
            "| Field | n | Agreement / mean abs diff | Cohen's kappa / Pearson r |",
            "|---|---|---|---|",
        ]
        for field, f in cmp["fields"].items():
            if not f.get("n"):
                lines.append(f"| {field} | 0 | (no comparable observations) | |")
            elif "agreement" in f:
                k = f["cohen_kappa"]
                lines.append(f"| {field} | {f['n']} | {f['agreement']} agreement | "
                             f"kappa {k if k is not None else 'undefined'} |")
            else:
                lines.append(f"| {field} | {f['n']} | {f['mean_abs_diff']} mean abs "
                             f"diff (max {f['max_abs_diff']}) | r {f['pearson_r']} |")
        notes = {f["kappa_note"] for f in cmp["fields"].values() if f.get("kappa_note")}
        for n in sorted(notes):
            lines += ["", f"> {n}."]
        if cmp["notable_disagreements"]:
            lines += ["", f"{len(cmp['notable_disagreements'])} notable disagreements "
                          "(boolean flips, or scores more than 0.25 apart):", ""]
            for d in cmp["notable_disagreements"][:12]:
                lines.append(f"- `{d['item']}` **{d['field']}**: {a}={d[a]}, {b}={d[b]}")
            lines += _asymmetry(cmp, a, b)
        else:
            lines += ["", "No boolean flips and no score gap above 0.25."]
    lines += [
        "",
        "## How to read this",
        "",
        "`primary vs second` tests whether the verdict survives changing the judge",
        "model. `primary vs gold_blind` tests whether it survives removing the",
        "reference answer, which is the stronger validity check: a judge that only",
        "agrees with itself when handed the answer is pattern-matching, not grading.",
        "Both arms are Anthropic models, so cross-model agreement here is evidence of",
        "robustness within one family rather than across families \u2014 stated plainly",
        "because it bounds what the number can claim.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    arms = os.environ.get("JUDGE_ARMS", "primary,second,gold_blind").split(",")
    out = settings.artifacts_dir.parent / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "judge_agreement.json"

    if os.environ.get("REPORT_ONLY") == "1" and saved.exists():
        # re-render the write-up from stored grades without re-billing the judges
        prior = json.loads(saved.read_text())
        grades, comparisons = prior["grades"], prior["comparisons"]
        arms = list(grades)
    else:
        answers = json.loads((settings.outputs_dir / "answers.json").read_text())
        eval_set = json.loads(EVAL_SET.read_text())
        gold_by_answer = {q["id"]: q.get("gold_answer", "") for q in eval_set}
        grades = {arm: {p: grade(arm, answers[p], gold_by_answer, p)
                        for p in ("graph", "baseline")}
                  for arm in arms}
        comparisons = {f"{arms[0]}__vs__{other}": compare(arms[0], other, grades)
                       for other in arms[1:]}

    saved.write_text(json.dumps({"grades": grades, "comparisons": comparisons},
                                indent=1))
    (out / "judge_agreement.md").write_text(_report(grades, comparisons))

    print()
    for arm in arms:
        print(f"{arm}: graph {_headline(grades, arm, 'graph')}  "
              f"baseline {_headline(grades, arm, 'baseline')}")
    for pair, cmp in comparisons.items():
        ac = cmp["fields"].get("answer_correct", {})
        print(f"{pair}: answer_correct agreement={ac.get('agreement')} "
              f"kappa={ac.get('cohen_kappa')} "
              f"disagreements={len(cmp['notable_disagreements'])}")
    print("wrote experiments/results/judge_agreement.json, .md")


if __name__ == "__main__":
    main()
