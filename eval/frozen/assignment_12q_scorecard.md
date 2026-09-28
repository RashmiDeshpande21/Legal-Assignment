# Evaluation: graph path vs flat RAG baseline

| Metric | Graph | Baseline |
|---|---|---|
| Context recall (mean, T1) | 1.0 | 0.484 |
| Context precision (mean, T1) | 0.211 | 0.227 |
| Citation accuracy (judge, T2) | 0.875 | 0.762 |
| Amendment status correct (T2) | 1.0 | 0.667 |
| Version correct for as-of (T2) | 1.0 | 0.667 |
| Chain completeness (T2) | 0.907 | 0.481 |
| Answer correct rate (T2) | 1.0 | 0.417 |
| Faithfulness (T2) | 0.897 | 0.833 |
| Latency p50 (s, T3) | 123.1 | 107.03 |
| Latency p95 (s, T3) | 199.26 | 186.74 |
| Subsection-level citation rate (T3) | 0.644 | 0.475 |
| Citations carrying amendment status (T3) | 29 | 0 |
| Q9 vs Q10 answers differ (T3) | True | True |

Run: tag=baseline-final, git_sha=unknown, model_path=models/Qwen_Qwen3.6-27B-Q4_K_M.gguf, embedder_model=Hanno-Labs/dinghy-law-0.6b-v1, reranker_model=cross-encoder/ms-marco-MiniLM-L-12-v2, llm_context_length=16384, llm_max_tokens=4096, llm_enable_thinking=True, llm_think_budget=0, llm_repeat_penalty=1.0, expand_stage2_top_n=6, expand_stage2_keep=8, definition_closure_depth=1, graph_tag=final, note=graph answers from tag=final; baseline generated after, same code at 2026-09-28T19:25:32+00:00

## Per-question answer correctness (judge)

| Q | Graph correct | Baseline correct | Judge note (graph) |
|---|---|---|---|
| q1 | True | True | The system answer correctly identifies New York law (§10.14(a)) and exclusive jurisdiction of NY state courts in NY County and SDNY (§10.14(b)), matching the gold answer, though §10.14(b) and §10.14(c) citations are not directly confirmed in the retrieved context snippets provided. |
| q2 | True | False | The system answer correctly identifies that no term loan facility exists, provides the correct original maturity date of October 26, 2022, and cites the appropriate sections (§1.01 and §2.06) from the originally executed Credit Agreement. |
| q3 | True | False | The system answer correctly identifies all parties and roles matching the gold answer; the only minor omission is not explicitly labeling Denny's Corporation as 'Parent/Guarantor' (only 'Parent'), but all other roles are accurate and complete. |
| q4 | True | False | The system answer exactly matches the gold answer, correctly defining 'Borrowing' per §1.01, listing all 8 dependent sections, and additionally listing all 4 dependent definitions, all supported by the retrieved context. |
| q5 | True | True | The system answer correctly identifies §7.01 as the governing covenant, 'Lien' and 'IP Rights' as operative defined terms, and lists valid exceptions including §7.01(d) via IP Security Agreement and Permitted Liens, but cites §7.01(a),(b),(d),(e),(f) as specific subsections when the retrieved context only shows the general §7.01 prohibition text without confirming all those lettered subsections exist as described, and §7.01(c) is acknowledged as missing yet the answer still references it implicitly through the full list. |
| q6 | True | False | The system correctly traces the chain from §7.05(a) prohibition → Restricted Payment definition → §7.05(a)(i) carve-out → Third Amendment proviso (suspending only clauses ii-iv), matching the gold answer's substance, but the retrieved context does not directly show §7.05(a)(i) text, making the carve-out claim partially unsupported by the provided excerpts; the Third Amendment reference is correctly attributed to the September 29, 2021 compliance date. |
| q7 | True | False | The system answer correctly traces the chain from §7.10 financial covenant breach → immediate Event of Default under §8.01(b) (no grace period, with §8.01(c)'s 30-day grace explicitly excluded) → §8.02 remedies, matching the gold answer; the additional details about default rate interest and setoff rights are supported by context but go beyond the gold answer's scope. |
| q8 | True | True | The system correctly identifies §2.13(g)'s 'Conflicting Provisions' superseding §2.12 and §10.01, which matches the gold answer's primary example, though it incorrectly characterizes §2.13(g) as lacking substantive rules; the §10.06(d) cross-reference claim is speculative and not fully supported by the retrieved context. |
| q9 | True | False | The system answer correctly identifies §7.05(a) as originally executed as the operative provision for 2018-01-01, accurately enumerates the carve-outs and caps, and notes the Second Amendment proviso does not apply (effective 2020), fully consistent with the gold answer and retrieved context. |
| q10 | True | True | The system answer correctly identifies the Second Amendment (2020-05-13) as the operative change, accurately describes the suspension of clauses (ii)-(iv) until financial covenant compliance for the quarter ending June 30, 2021, and contrasts with the pre-amendment state, all well-supported by the retrieved context. |
| q11 | True | False | The system answer accurately lists all three amendments with correct effective dates and covers nearly all 68 operations from the catalog, though it omits the operation types (restate/insert/replace) for most entries and slightly consolidates some individual definition entries. |
| q12 | True | True | The system answer correctly identifies all key changes (First Amendment restating Consolidated EBITDA, Second Amendment adding proviso, Third Amendment replacing it) and their effects on §7.10(a), Applicable Rate, §7.05(a), and other sections, though it adds a Third Amendment not mentioned in the gold answer and includes some sections (Incurrence Ratio, §2.09(b), §7.03(g/h)) not in the gold answer but plausibly supported by context. |
