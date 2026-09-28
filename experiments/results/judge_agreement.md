# Judge calibration and agreement

The Tier-2 headline comes from one judge. These arms re-grade the same
committed answers to show how much of that number is the pipeline and how
much is the instrument.

| Arm | Judge model | Gold shown | answer_correct (graph) | answer_correct (baseline) |
|---|---|---|---|---|
| primary | `us.anthropic.claude-sonnet-4-6` | yes | 7/12 | 4/12 |
| second | `us.anthropic.claude-opus-4-6-v1` | yes | 6/12 | 5/12 |
| gold_blind | `us.anthropic.claude-sonnet-4-6` | no | 10/12 | 7/12 |

## primary vs second

| Field | n | Agreement / mean abs diff | Cohen's kappa / Pearson r |
|---|---|---|---|
| answer_correct | 24 | 0.833 agreement | kappa 0.664 |
| amendment_status_correct | 18 | 0.944 agreement | kappa 0.769 |
| version_correct | 10 | 1.0 agreement | kappa 1.0 |
| citation_accuracy | 24 | 0.045 mean abs diff (max 0.2) | r 0.934 |
| faithfulness | 24 | 0.035 mean abs diff (max 0.2) | r 0.907 |
| chain_completeness | 14 | 0.039 mean abs diff (max 0.15) | r 0.979 |

5 notable disagreements (boolean flips, or scores more than 0.25 apart):

- `graph/q8` **answer_correct**: primary=True, second=False
- `baseline/q2` **answer_correct**: primary=False, second=True
- `baseline/q8` **answer_correct**: primary=True, second=False
- `baseline/q12` **answer_correct**: primary=False, second=True
- `baseline/q11` **amendment_status_correct**: primary=False, second=True

All 4 `answer_correct` flips by path: baseline 3, graph 1.

## primary vs gold_blind

| Field | n | Agreement / mean abs diff | Cohen's kappa / Pearson r |
|---|---|---|---|
| answer_correct | 24 | 0.583 agreement | kappa 0.195 |
| amendment_status_correct | 17 | 0.882 agreement | kappa 0.0 |
| version_correct | 12 | 0.917 agreement | kappa 0.0 |
| citation_accuracy | 24 | 0.091 mean abs diff (max 0.4) | r 0.738 |
| faithfulness | 24 | 0.099 mean abs diff (max 0.3) | r 0.535 |
| chain_completeness | 13 | 0.16 mean abs diff (max 0.75) | r 0.727 |

> kappa paradox: agreement is high but one label dominates the marginal, which deflates kappa.

17 notable disagreements (boolean flips, or scores more than 0.25 apart):

- `graph/q2` **answer_correct**: primary=False, gold_blind=True
- `graph/q11` **answer_correct**: primary=False, gold_blind=True
- `graph/q12` **answer_correct**: primary=False, gold_blind=True
- `baseline/q2` **answer_correct**: primary=False, gold_blind=True
- `baseline/q5` **answer_correct**: primary=True, gold_blind=False
- `baseline/q6` **answer_correct**: primary=False, gold_blind=True
- `baseline/q8` **answer_correct**: primary=True, gold_blind=False
- `baseline/q9` **answer_correct**: primary=False, gold_blind=True
- `baseline/q11` **answer_correct**: primary=False, gold_blind=True
- `baseline/q12` **answer_correct**: primary=False, gold_blind=True
- `baseline/q9` **amendment_status_correct**: primary=False, gold_blind=True
- `baseline/q11` **amendment_status_correct**: primary=False, gold_blind=True

All 10 `answer_correct` flips by path: baseline 7, graph 3.

## How to read this

`primary vs second` tests whether the verdict survives changing the judge
model. `primary vs gold_blind` tests whether it survives removing the
reference answer, which is the stronger validity check: a judge that only
agrees with itself when handed the answer is pattern-matching, not grading.
Both arms are Anthropic models, so cross-model agreement here is evidence of
robustness within one family rather than across families — stated plainly
because it bounds what the number can claim.
