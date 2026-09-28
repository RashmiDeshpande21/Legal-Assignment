# Experiment log — what was measured, and what it decided

Every design choice in the shipping path traces to one of the comparisons below.
Raw per-question JSON for the runs that are still reproducible lives in
`experiments/results/`; the earlier exploratory runs were summarised here and their
bulk artifacts removed, because they were superseded by the matrix runs and only the
conclusions still matter.

Scores are "answers the Bedrock judge marked correct / questions asked" unless the row
says recall. The assignment set is the 12 questions from the brief. The hold-out set is
20 questions of the same archetypes over different provisions, generated separately and
never used to tune retrieval.

---

## 1. Chunking (baseline RAG index)

Metric: gold context recall@10 after rerank, assignment set.

| Strategy | Chunks | Recall@10 |
|---|---|---|
| fixed 512 / overlap 256 | 308 | 0.469 |
| fixed 256 / overlap 64 | 413 | 0.369 |
| **section-aware 512** | 485 | **0.484** |

Decision: section-aware chunking for the flat-RAG baseline. The spread is small, which
is itself the finding — no chunking strategy recovers the questions that need an edge.
Both `q3` (parties) and `q10`/`q11`/`q12` (amendment-dependent) sit at or near 0.0 recall
under every strategy.

## 2. Embedder

Metric: recall@10 / MRR@10 on a synthetic question set.

| Model | Recall@10 | MRR@10 |
|---|---|---|
| **Hanno-Labs/dinghy-law-0.6b-v1** | **0.525** | **0.500** |
| nemotron-3-embed-8b-legal | 0.389 | 0.335 |

A legal-domain 0.6B embedder beat an 8B general one. Also evaluated: bge-large-en-v1.5,
e5-large-v2, legal-colbert-extractor.

## 3. Reranker

`cross-encoder/ms-marco-MiniLM-L-12-v2` ships. The domain candidate
`lxyuan/LegalBenchRAG-Ettin-150M-Reranker` is published without its classifier head
weights and cannot be loaded as a CrossEncoder at all, which is recorded so nobody
re-tries it.

## 4. Hybrid seeding (keyword entry points + dense retrieval)

Metric: mean gold recall.

| Set | Keyword only | Hybrid |
|---|---|---|
| assignment 12 | 0.634 | 0.654 |
| synthetic 36 | 0.833 | **0.944** |

Decision: keep dense seeding. The assignment set barely moves; the synthetic set shows
keyword-only entry points miss phrasings the reviewer is likely to use. This is the
clearest case in the project of the tuning set being blind to a real weakness.

## 5. Generator ladder

Identical retrieved context for every model, assignment set.

| Model | Assignment | Hold-out |
|---|---|---|
| Saul-7B-Instruct (legal-pretrained) | 2/12 | — |
| Llama-3.1-8B / Mistral-7B | ~4/12 | — |
| Qwen2.5-14B-Instruct | 5–7/12 | 8–9/20 |
| Mistral-Small-24B | 5–6/12 | 9–11/20 |
| gemma-2-27b-it | 6/12 | — |
| **Qwen3.6-27B Q4_K_M** | **12/12** | **16/20** |

The decisive result: legal pre-training did not help. Saul-7B, the only
legal-domain-pretrained candidate, scored worst. What separated models was multi-step
reasoning over the same context, and it only appears at 14B and above. Qwen3.6-27B is a
large jump above everything else in the bracket, not a marginal win.

An earlier Qwen3.6-27B run scored 9/12 on a pipeline revision several hours older than
the 12/12 run. Those two conditions differ in much more than one variable, so no
per-feature attribution was ever drawn from the gap.

## 6. Context-shaping blocks (proviso split, change history, conflict audit)

Each block was designed against a specific failure on Qwen2.5-14B and helped there:

| Block | Effect on Qwen2.5-14B |
|---|---|
| change history | 2/4 → 4/4 on the amendment-evolution subset |
| proviso elevation | fixed exception enumeration and amendment-proviso questions |
| conflict audit | fixed the conflict question |

Then a controlled ablation on the shipping model, on the hold-out set, reversed the
conclusion:

| Condition | Hold-out |
|---|---|
| all three blocks on | 15/20 |
| no proviso split | 15/20 |
| no change history | 15/20 |
| no conflict audit | 16/20 |
| **graph only (all three off)** | **17/20** |

Decision: all three default **off**. This is the most important negative result in the
project. Prompt scaffolding that a 14B model needs actively costs a 27B model answers —
`conflict_audit` was the clear offender on conflict questions. The same ablation on the
assignment set was useless for deciding this, because every condition scored 12/12: a
saturated tuning set cannot detect harm.

## 7. Generator thinking mode

Qwen3.6's chat template opens a `<think>` block unless `enable_thinking=false` is passed
into the Jinja template at render time. The documented `/no_think` prompt tag is a soft
switch and this GGUF ignores it, which is a known llama.cpp issue.

| Setting | Assignment | Hold-out | Wall clock, 12 questions |
|---|---|---|---|
| thinking on, uncapped | **12/12** | 16/20 | ~25 min graph only |
| thinking capped at 2048 CoT tokens | 7/12 | — | ~21 min |
| thinking off (hard Jinja switch) | 7/12 | 15/20 | ~7 min |

Decision: thinking on and uncapped. Turning it off is roughly four times faster and
costs five assignment answers; the model stops performing the premise checks and
enumeration checks that the harder questions need. A mid-way token cap got the worst of
both — it truncates reasoning without buying much time back.

## 8. Sampling

`repeat_penalty` was 1.15. llama.cpp applies it over a 64-token window, so it penalises
the citation label that a long enumeration legitimately has to repeat. The
68-row amendment catalog silently lost a row (§5.23) to this. Set to 1.0: greedy
decoding on a 27B instruct model does not need a loop guard. The catalog block was also
regrouped by instrument so one citation covers a whole group instead of repeating per row.

## 9. Retrieval structure

Findings that produced the graph-side design, all verified by the no-generator probe
(`make probe-all`, ~20 s per set):

- **Party roles live on edges, not text.** `Party` nodes carry no body text, so DFS
  (which collects `Section`/`Definition` only) and the chunk index can never surface
  them. The parties question was structurally unanswerable at 0.0 recall until
  `PARTY_TO` edges were projected into a roster block.
- **Downward-only DFS cannot reach a sibling.** `CONTAINS` is followed downward, so from
  §8.01 (Events of Default) traversal reaches §8.01(a)–(l) but never §8.02 (Remedies).
  The "and then what happens" half of every chain question was missing. Fixed with a
  sibling hop through the shared `CONTAINS` parent plus a bidirectional
  `REFERENCES`/`DEPENDS_ON` hop that routes through definition hubs.
- **A definition cannot be read without its inputs.** The leverage ratio *is* Consolidated
  Funded Indebtedness over Consolidated EBITDA. A relevance reranker cannot recover that:
  asked whether the leverage ratio changed, the cross-encoder ranked Consolidated EBITDA
  12th of 14 candidates, because that definition never says "leverage ratio". Definition
  closure over `DEPENDS_ON` is therefore included structurally, not ranked.
- **Neighbour lists must be cut by relevance, not by list order.** Capping neighbours per
  node truncated them alphabetically, which dropped "Permitted Liens" off the end of the
  lien question's exception list.
- **Retrieved is not the same as prompted.** Gold nodes were present in `context_ids`
  while their text had been dropped by the prompt packer at the 16k context limit, so
  recall read 1.0 on questions the model answered wrong from a prompt that never
  contained the decisive passage. Blocks the graph marks as necessary are now packed
  first and are shortened rather than omitted; `context_ids` reports only what was
  actually packed.

## 10. Eval-instrument checks

The judge is a single LLM, so it is a measurement instrument of unknown bias. Four
companion checks test the harness rather than the pipeline, two of them with no LLM at all:

| Check | Result |
|---|---|
| citation resolution against the graph (deterministic) | version-correct 1.0; the judge is more generous than this check |
| as-of resolution at every amendment boundary (deterministic) | PASS, 2223 assertions, 0 failures |
| second judge | 12/12 agreement with the primary judge on the shipping answers |
| gold-blind judge | scored higher than the gold-aware judge, i.e. gold wording is stricter than the documents require |

## 11. Cost

Rule-based graph build, no LLM calls: 0.79 s. Chunk embedding and index build: 21 s.
Together about 13.8 s and $0.004 per 100 pages on one A10G. Answer generation dominates
runtime, not indexing.
