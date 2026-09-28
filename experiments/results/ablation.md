# Component ablation (qwen36-27b)

Each context-shaping block turned off in isolation on the shipping model and
the shipping question set. `graph_only` has all three off — still graph
retrieval with as-of version resolution, just no context shaping.

| Condition | Correct | vs full | Gold recall | Verified citations | Judge citation | Faithfulness | Blocks | Latency |
|---|---|---|---|---|---|---|---|---|
| `full` | 12/12 | — | 0.798 | 0.635 | 0.929 | 0.973 | 10.7 | 66.847s |
| `no_proviso_split` | 12/12 | +0 | 0.798 | 0.679 | 0.933 | 0.975 | 10.1 | 65.104s |
| `no_change_history` | 12/12 | +0 | 0.798 | 0.613 | 0.929 | 0.964 | 10.5 | 65.662s |
| `no_conflict_audit` | 12/12 | +0 | 0.798 | 0.635 | 0.946 | 0.977 | 10.6 | 65.729s |
| `graph_only` | 12/12 | +0 | 0.798 | 0.619 | 0.95 | 0.973 | 9.8 | 64.944s |

Gold recall is identical across conditions by construction: every block shapes
the context that is already retrieved, none of them changes retrieval. It is
reported anyway as a guard — a recall that moved would mean a flag was leaking
into the retrieval stage.

## Which questions each block actually carries

| Question | `full` | `no_proviso_split` | `no_change_history` | `no_conflict_audit` | `graph_only` |
|---|---|---|---|---|---|
| q1 | ok | ok | ok | ok | ok |
| q2 | ok | ok | ok | ok | ok |
| q3 | ok | ok | ok | ok | ok |
| q4 | ok | ok | ok | ok | ok |
| q5 | ok | ok | ok | ok | ok |
| q6 | ok | ok | ok | ok | ok |
| q7 | ok | ok | ok | ok | ok |
| q8 | ok | ok | ok | ok | ok |
| q9 | ok | ok | ok | ok | ok |
| q10 | ok | ok | ok | ok | ok |
| q11 | ok | ok | ok | ok | ok |
| q12 | ok | ok | ok | ok | ok |

### Regressions, per block

- No condition changed any per-question verdict, including `graph_only` with all three blocks off. On this model and this question set the context-shaping blocks are not load-bearing for correctness: graph retrieval and as-of version resolution carry the result on their own.

### What this test can and cannot conclude

The `full` condition is at ceiling, so read the null result with the asymmetry it implies. A saturated baseline leaves no headroom to detect a *positive* contribution — this design can only detect harm, and it found none. So the defensible claim is that removing any block does not break this question set, not that the blocks are worthless.

What it does refute is the stronger version of our own thesis. The three interventions were each justified on Qwen2.5-14B, where they demonstrably moved answers, and the write-up presented them as part of why the pipeline reaches 12/12. On the shipping generator they are not what produces that number. They were compensating for a weaker model's failure modes — primacy bias on appended provisos, inability to attribute changes across instruments — and a stronger model does not exhibit those failures. Scaffolding that a better generator no longer needs is worth naming as such.

The right place to detect residual value is a set that is not at ceiling: `eval/holdout_hard_set.json` sits at 15-16/20 for this model, so it has the headroom this one lacks. Running these same conditions there is the follow-up this result calls for.

Secondary metrics move within noise and without a consistent direction (verified citations 0.61-0.68, faithfulness 0.964-0.977 across all five conditions), so they do not rescue a contribution either. Latency and block count fall slightly as blocks are removed, which is the only monotonic effect in the table: the blocks cost context and time, and on this model they buy no correctness back.
