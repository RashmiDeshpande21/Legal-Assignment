"""Generate a held-out HARD eval set covering the assignment's difficult archetypes.

Why this exists: the assignment's 12 questions are the tuning set — every pipeline
change (proviso-split, CoT, hybrid seeding, and the upcoming change-history /
conflict-validator blocks) risks overfitting to their exact wording. The existing
synthetic-36 set is retrieval-oriented (1/36 temporal, 0 evolution/conflict/false-
premise questions), so it cannot detect that. This set pre-registers ~20 unseen
questions at the same difficulty ACROSS the hard archetypes, generated BEFORE the
next round of pipeline changes and never used for tuning.

Archetypes (mirroring assignment q2/q4/q7/q8/q10/q12 shapes, different substance):
  amendment_evolution        trace a provision/definition across instruments
  as_of_temporal             which version is in force at a pinned date
  multi_amendment_aggregation changes spanning BOTH instruments on one theme
  definition_dependents      enumerate provisions relying on a defined term
  false_premise              question presupposes something that does not exist
  conflict_precedence        which provision controls under explicit conflict clauses

Frontier use: Opus 4.6 writes questions + gold answers (eval-label generation only —
never the live answer path). Grounding excerpts include base text, every amendment
op (instrument/type/date/amended text), and for dependents batches the exact inbound
edge lists, so gold answers are complete by construction.

Run: uv run python -m eval.generate_holdout_hard
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3

from config import settings
from graph.loader import load_graph
from graph.schema import EDGE_DEPENDS_ON, EDGE_REFERENCES
from graph.traversal import _amend_ops_before, node_label

OUT_PATH = Path("eval/holdout_hard_set.json")

PROMPT_TEMPLATE = """You are an expert financial-contracts lawyer designing a HARD QA test set
over Denny's 2017 Credit Agreement and its First Amendment (effective 2018-06-26) and
Second Amendment (effective 2020-05-13). Timeline: original agreement 2017-10-26.

ARCHETYPE FOR THIS BATCH: {archetype}
{archetype_instructions}

GROUNDING MATERIAL (authentic; NODE_IDs are exact graph identifiers):
{excerpts}

Generate {n_questions} questions of this archetype. Requirements:
- Questions must be answerable strictly from the grounding material.
- gold_answer must be precise, self-contained, and verifiable against the material.
- gold lists the NODE_IDs whose text supports the answer (may be empty ONLY for
  false_premise questions).
- Do NOT reuse the wording of these known tuning questions (paraphrases of the same
  fact are fine only if the target provision differs): {avoid}

Return ONLY a valid JSON array:
[
  {{
    "id": "placeholder",
    "category": "{archetype}",
    "question": "<the question>",
    "as_of": <"YYYY-MM-DD" or null>,
    "gold": ["<NODE_ID>", ...],
    "gold_answer": "<ground truth>"
  }}
]"""

AVOID = (
    "'How did the definition of Consolidated EBITDA change...' / "
    "'Which sections depend on the definition of Borrowing' / "
    "'find conflicting or inconsistent provisions' / "
    "'events of default relating to bankruptcy' / "
    "'maximum Consolidated Leverage Ratio as of 2020-06-01'"
)


def _excerpt(G, nid: str, max_chars: int = 2200) -> str:
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
    deps = sorted({
        u for u, _, k in G.in_edges(term_nid, keys=True)
        if k.split("::")[0] in (EDGE_REFERENCES, EDGE_DEPENDS_ON)
    })
    lines = [f"{node_label(G, d)}  [NODE_ID: {d}]" for d in deps]
    return (
        f"--- NODE_ID: {term_nid} (definition of \u201c{n['term']}\u201d) ---\n"
        f"DEFINITION TEXT:\n{(n.get('text') or '')[:1500]}\n"
        f"COMPLETE LIST OF PROVISIONS THAT REFERENCE OR DEPEND ON THIS TERM "
        f"(inbound graph edges — this list is exhaustive):\n" + "\n".join(lines)
    )


def _build_batches(G) -> list[dict]:
    ev = ("Ask the model to trace what changed, when, by which instrument, and the substance "
          "of the change.")
    return [
        {
            "archetype": "amendment_evolution",
            "instructions": ev + " One question per target below; questions must require citing "
                                 "the specific amending instrument and effective date.",
            "excerpts": [
                _excerpt(G, "base::def::Consolidated EBITDA"),
                _excerpt(G, "base::6.01"),
                _excerpt(G, "base::def::Eurodollar Rate") + "\n\n"
                + _excerpt(G, "base::def::Base Rate"),
                _excerpt(G, "base::7.05(a)"),
            ],
            "n": 4,
        },
        {
            "archetype": "as_of_temporal",
            "instructions": (
                "Pin each question to an as_of date and ask what the provision requires ON that "
                "date. Use dates that straddle the amendments: one before 2018-06-26, one between "
                "First and Second, one between Second and Third (2020-05-13..2020-12-15), and one "
                "after 2020-12-15. The gold answer must reflect the version in force at that date "
                "(original vs First- vs Second- vs Third-Amendment text)."
            ),
            "excerpts": [
                _excerpt(G, "base::def::Consolidated EBITDA"),
                _excerpt(G, "base::7.10(a)") + "\n\n" + _excerpt(G, "base::7.10(b)"),
                _excerpt(G, "base::def::Base Rate"),
                _excerpt(G, "base::2.15(a)"),
            ],
            "n": 4,
        },
        {
            "archetype": "multi_amendment_aggregation",
            "instructions": (
                "Each question must require aggregating changes made by the First, Second, AND "
                "Third Amendments (or verifying that a given instrument did not touch the theme). "
                "Themes: (1) the financial-covenant / leverage-ratio testing regime, "
                "(2) benchmark interest rate machinery (Eurodollar Rate, Base Rate, Eurodollar "
                "Rate Loan / Benchmark Replacement), (3) restricted payments and investments "
                "capacity. The gold answer must attribute each change to the correct instrument "
                "and date."
            ),
            "excerpts": [
                _excerpt(G, "base::def::Consolidated EBITDA")
                + "\n\n" + _excerpt(G, "base::7.10(a)")
                + "\n\n" + _excerpt(G, "base::def::Consolidated Leverage Ratio"),
                _excerpt(G, "base::def::Eurodollar Rate")
                + "\n\n" + _excerpt(G, "base::def::Base Rate")
                + "\n\n" + _excerpt(G, "base::def::Eurodollar Rate Loan")
                + "\n\n" + _excerpt(G, "base::1.08"),
                _excerpt(G, "base::7.05(a)") + "\n\n" + _excerpt(G, "base::7.03"),
            ],
            "n": 3,
        },
        {
            "archetype": "definition_dependents",
            "instructions": (
                "Ask which operative provisions rely on/are governed by the defined term. The "
                "COMPLETE dependents list is provided — the gold answer must enumerate it exactly "
                "(the pipeline under test must recover the full list, so completeness matters)."
            ),
            "excerpts": [
                _dependents_excerpt(G, "base::def::Threshold Amount"),
                _dependents_excerpt(G, "base::def::Default Rate"),
                _dependents_excerpt(G, "base::def::Letter of Credit Sublimit"),
            ],
            "n": 3,
        },
        {
            "archetype": "false_premise",
            "instructions": (
                "Write questions whose PREMISE is false — e.g. a 'Fourth Amendment' (only "
                "three amendments exist: First, Second, Third), a financial covenant the "
                "agreement does not contain (§7.10 below is the complete financial-covenants "
                "section), or a defined term that is not defined. The gold answer must state "
                "that the premise is false, explain what actually exists, and cite the "
                "disproving provision. gold should list the disproving NODE_IDs."
            ),
            "excerpts": [
                "FACT: The only amendment instruments are the First Amendment (2018-06-26), "
                "the Second Amendment (2020-05-13), and the Third Amendment (2020-12-15). "
                "There is no Fourth Amendment.",
                _excerpt(G, "base::7.10")
                + "\n\n" + _excerpt(G, "base::7.10(a)")
                + "\n\n" + _excerpt(G, "base::7.10(b)"),
                "FACT: Terms NOT defined anywhere in the agreement include: 'Consolidated Net "
                "Worth', 'Springing Maturity Date', 'ESG Margin Adjustment'.",
            ],
            "n": 3,
        },
        {
            "archetype": "conflict_precedence",
            "instructions": (
                "Ask which provision controls when two provisions could conflict, grounded in the "
                "EXPLICIT precedence / 'notwithstanding' language in the excerpts. The gold answer "
                "must quote or paraphrase the controlling clause and name both provisions."
            ),
            "excerpts": [
                _excerpt(G, "base::2.15(a)"),
                _excerpt(G, "base::10.09"),
                _excerpt(G, "base::10.03") + "\n\n" + _excerpt(G, "base::10.01(i)"),
            ],
            "n": 3,
        },
    ]


def generate_batch(client, batch: dict) -> list[dict]:
    prompt = PROMPT_TEMPLATE.format(
        archetype=batch["archetype"],
        archetype_instructions=batch["instructions"],
        excerpts="\n\n".join(batch["excerpts"]),
        n_questions=batch["n"],
        avoid=AVOID,
    )
    resp = client.converse(
        modelId=settings.bedrock_model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 4000, "temperature": 0.2},
    )
    raw = resp["output"]["message"]["content"][0]["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw[raw.index("["): raw.rindex("]") + 1])


def main() -> None:
    print(f"Bedrock {settings.bedrock_model} — generating held-out hard set...")
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    G = load_graph(settings.artifacts_dir / "graph.json")

    questions, cur = [], 1
    for batch in _build_batches(G):
        print(f"  batch {batch['archetype']} (n={batch['n']})...")
        got = generate_batch(client, batch)
        for q in got:
            q["id"] = f"hh_{cur:02d}"
            cur += 1
        questions.extend(got)
        print(f"    +{len(got)} (total {len(questions)})")

    # sanity: gold node ids must exist in the graph
    bad = [(q["id"], g) for q in questions for g in q.get("gold", []) if not G.has_node(g)]
    if bad:
        print(f"  WARNING dropping {len(bad)} nonexistent gold ids: {bad}")
        for q in questions:
            q["gold"] = [g for g in q["gold"] if G.has_node(g)]

    OUT_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    print(f"Wrote {len(questions)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
