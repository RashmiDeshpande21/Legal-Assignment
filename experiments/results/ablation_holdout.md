# Holdout component ablation (qwen36-27b)

Same five CONDITIONS as `experiments/ablation.py`, on the unsaturated held-out
hard set (`eval/holdout_hard_set.json`, 20 questions). The assignment-set
ablation was ceiling-limited at 12/12; this set sits at 15-16/20 and can detect
contribution as well as harm.

| Condition | Correct | vs full | amendment_evolution | as_of_temporal | conflict_precedence | definition_dependents | false_premise | multi_amendment_aggregation | Verified cites | Faithfulness | Latency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | 15/20 | — | 3/4 | 3/4 | 2/3 | 2/3 | 3/3 | 2/3 | 0.867 | 0.925 | 70.32s |
| `no_proviso_split` | 15/20 | +0 | 3/4 | 4/4 | 2/3 | 2/3 | 3/3 | 1/3 | 0.85 | 0.955 | 69.846s |
| `no_change_history` | 15/20 | +0 | 3/4 | 3/4 | 2/3 | 2/3 | 3/3 | 2/3 | 0.818 | 0.93 | 69.469s |
| `no_conflict_audit` | 16/20 | +1 | 3/4 | 3/4 | 3/3 | 2/3 | 3/3 | 2/3 | 0.867 | 0.922 | 69.598s |
| `graph_only` | 17/20 | +2 | 3/4 | 4/4 | 3/3 | 2/3 | 3/3 | 2/3 | 0.876 | 0.958 | 69.522s |

## Per-question verdicts

| Question | Category | `full` | `no_proviso_split` | `no_change_history` | `no_conflict_audit` | `graph_only` |
|---|---|---|---|---|---|---|
| hh_01 | amendment_evolution | ok | ok | ok | ok | ok |
| hh_02 | amendment_evolution | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| hh_03 | amendment_evolution | ok | ok | ok | ok | ok |
| hh_04 | amendment_evolution | ok | ok | ok | ok | ok |
| hh_05 | as_of_temporal | ok | ok | ok | ok | ok |
| hh_06 | as_of_temporal | ok | ok | ok | ok | ok |
| hh_07 | as_of_temporal | ok | ok | ok | ok | ok |
| hh_08 | as_of_temporal | **FAIL** | ok | **FAIL** | **FAIL** | ok |
| hh_09 | multi_amendment_aggregation | ok | **FAIL** | ok | ok | ok |
| hh_10 | multi_amendment_aggregation | ok | ok | ok | ok | ok |
| hh_11 | multi_amendment_aggregation | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| hh_12 | definition_dependents | ok | ok | ok | ok | ok |
| hh_13 | definition_dependents | **FAIL** | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| hh_14 | definition_dependents | ok | ok | ok | ok | ok |
| hh_15 | false_premise | ok | ok | ok | ok | ok |
| hh_16 | false_premise | ok | ok | ok | ok | ok |
| hh_17 | false_premise | ok | ok | ok | ok | ok |
| hh_18 | conflict_precedence | ok | ok | ok | ok | ok |
| hh_19 | conflict_precedence | ok | ok | ok | ok | ok |
| hh_20 | conflict_precedence | **FAIL** | **FAIL** | **FAIL** | ok | ok |

### Regressions and gains, per block

- Condition `no_proviso_split`: breaks hh_09 (multi_amendment_aggregation); fixes hh_08 (as_of_temporal).
- Condition `no_conflict_audit`: fixes hh_20 (conflict_precedence).
- Condition `graph_only`: fixes hh_08 (as_of_temporal), hh_20 (conflict_precedence).

### Reading this against the assignment-set ablation

Best condition: `graph_only` at 17/20 (shipping `full` is 15/20).

The unsaturated set does what the assignment set could not: it detects that the three context-shaping blocks are not merely unnecessary on this model — stacked together they *cost* answers. `graph_only` recovers the questions that `conflict_audit` and (with proviso elevation) the full stack lose, and it loses none that `full` gets right.

Per-block: `conflict_audit` is the clear offender (hh_20, a conflict_precedence question — the category the audit was built for). `proviso_split` is a wash (fixes one temporal, breaks one aggregation). `change_history` moves nothing. Three failures (hh_02, hh_11, hh_13) are untouched by any condition — residual model/retrieval limits, not missing scaffolding.

Implication for the shipping defaults: on Qwen3.6 the honest configuration is the graph with as-of version resolution and no context-shaping blocks. Keeping the blocks on because they once helped Qwen2.5-14B is overfitting the assignment set's model ladder, not the pipeline.

Prior holdout runs on an earlier pipeline revision (`holdout_qwen36-27b.json`) scored baseline 16/20 and change_history / combined 15/20 — suggestive, but not a controlled per-block ablation on one revision. This is.
