"""Generate a 36-question synthetic legal test suite using Claude Opus 4.6.

Samples diverse sections across Articles II-X and the two amendments.
Outputs eval/synthetic_eval_set.json with question text, category, gold node IDs, and answers.
Run: uv run python -m eval.generate_synthetic_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import boto3

from config import settings
from graph.loader import load_graph

PROMPT_TEMPLATE = """You are an expert financial contracts lawyer and legal QA dataset designer.
You are given authentic excerpts from Denny's Credit Agreement from this topic:
{article_desc}

EXCERPTS:
{excerpts}

Generate {n_questions} high-quality, challenging test questions based strictly on these excerpts.
Cover these question archetypes:
- Specific numerical/timeline lookup (e.g. notice deadlines, percentages, thresholds)
- Multi-clause conditions or exceptions (e.g. carve-outs to covenants, conditions precedent)
- Defined-term dependencies (e.g. how a term functions within an operative covenant)
- Amendment/temporal difference (if excerpts contain amended text or amendment tags)

Return ONLY a valid JSON array of objects with this schema:
[
  {{
    "id": "syn_{start_id:02d}",
    "category": "<fact_lookup | definition_dependency | covenant_carveout | amendment_temporal>",
    "question": "<precise question a credit analyst or lawyer would ask>",
    "as_of": <ISO date string like "2018-01-01" or "2020-06-01" or null>,
    "gold": ["<exact node_id from excerpts that answers the question>"],
    "gold_answer": "<concise, accurate ground truth answer strictly supported by the text>"
  }}
]"""

BATCHES = [
    {
        "article_desc": "Article II — Credit Facilities, Borrowing Mechanics, Prepayments & Fees",
        "nodes": [
            "base::2.01", "base::2.02(a)", "base::2.03(a)", "base::2.05(a)",
            "base::2.06", "base::2.08(a)", "base::2.09(a)",
        ],
        "n": 6,
    },
    {
        "article_desc": "Article III & IV — Taxes, Yield Protection, and Conditions Precedent",
        "nodes": ["base::3.01(a)", "base::3.01(e)", "base::3.04(a)", "base::4.01", "base::4.02"],
        "n": 5,
    },
    {
        "article_desc": "Article V — Representations & Warranties (Solvency, ERISA, Sanctions)",
        "nodes": ["base::5.05(a)", "base::5.07", "base::5.12", "base::5.23"],
        "n": 5,
    },
    {
        "article_desc": "Article VI — Affirmative Covenants (Financial Reporting, Notices)",
        "nodes": ["base::6.01(a)", "base::6.01(b)", "base::6.02(a)", "base::6.03(a)", "base::6.10"],
        "n": 5,
    },
    {
        "article_desc": "Article VII — Negative Covenants (Debt, Investments, Mergers)",
        "nodes": ["base::7.02(a)", "base::7.02(b)", "base::7.03", "base::7.04(a)", "base::7.19"],
        "n": 6,
    },
    {
        "article_desc": "Article VIII — Events of Default, Grace Periods & Acceleration Remedies",
        "nodes": ["base::8.01(a)", "base::8.01(b)", "base::8.01(c)", "base::8.01(e)", "base::8.02"],
        "n": 5,
    },
    {
        "article_desc": "Article X & Amendments — Amendments voting threshold, Confidentiality",
        "nodes": ["base::10.01(a)", "base::10.07", "base::10.23", "base::1.08"],
        "n": 4,
    },
]


def generate_batch(client, batch: dict, G, start_id: int) -> list[dict]:
    excerpts_list = []
    for nid in batch["nodes"]:
        if G.has_node(nid):
            n = G.nodes[nid]
            label = n.get("number", nid)
            title = n.get("title", "")
            txt = n.get("text", "")[:1800]
            excerpts_list.append(f"--- NODE_ID: {nid} (§{label} {title}) ---\n{txt}")
    excerpts = "\n\n".join(excerpts_list)

    prompt = PROMPT_TEMPLATE.format(
        article_desc=batch["article_desc"],
        excerpts=excerpts,
        n_questions=batch["n"],
        start_id=start_id,
    )

    resp = client.converse(
        modelId=settings.bedrock_model,  # us.anthropic.claude-opus-4-6-v1
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 3000, "temperature": 0.2},
    )
    raw = resp["output"]["message"]["content"][0]["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start_idx = raw.index("[")
    end_idx = raw.rindex("]") + 1
    return json.loads(raw[start_idx:end_idx])


def main() -> None:
    print(f"Connecting to Bedrock Opus 4.6 ({settings.bedrock_model})...")
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    G = load_graph(settings.artifacts_dir / "graph.json")

    all_questions = []
    cur_id = 1
    for i, b in enumerate(BATCHES, 1):
        print(f"Generating batch {i}/{len(BATCHES)}: {b['article_desc']}...")
        questions = generate_batch(client, b, G, cur_id)
        # renumber sequentially just in case
        for q in questions:
            q["id"] = f"syn_{cur_id:02d}"
            cur_id += 1
        all_questions.extend(questions)
        print(f"  Generated {len(questions)} questions (total: {len(all_questions)})")

    out_path = Path("eval/synthetic_eval_set.json")
    out_path.write_text(json.dumps(all_questions, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(all_questions)} synthetic questions to {out_path}")


if __name__ == "__main__":
    main()
