# Denny's Credit Agreement – Graph vs Flat RAG

## My take on the data

I started with three documents – the 2017 credit agreement, the First Amendment (2018-06-26) and the Second (2020-05-13). I also ingested the optional Third (2020-12-15), so every count below is for four documents. Adding the fourth took no code change, which is basically the whole argument for storing amendments as edges instead of as text.

The documents are not the same kind of file. Two types:

What I mean by 'clean': the 2017 agreement and the First Amendment come as real HTML with section tags. Rules parse them.

What I mean by 'scanned': the Second and Third are images, but the filer left the OCR layer in the HTML in hidden white font. I pull that out and tag it `source: ocr`. Same parser, lower confidence.

The live answer is a local model – Qwen3.6-27B Q4, one A10G. Bedrock is only the offline judge. It never writes an answer that ships.

The flat RAG baseline uses the same embedder, same reranker, same prompt, same model, same judge. The only thing that changes is how the context gets built. That is deliberate – any gap in the numbers is the graph, not a better model.

Two calls the brief did not specify, written down here:

- Q9 and Q10 say "as of the date given in the question set" and no dates were given. I used 2018-01-01 and 2020-06-01 – one before any amendment, one after the Second.
- A `replace` or `delete` amendment is kept as the original text plus the instruction. I do not splice "delete the word X" into the body. A bad splice reads like the contract, which is worse than not answering.

---

## The Schema, and Why It Looks Like This

NetworkX multigraph. No database. Built by rules, no LLM in the build.

```
742 nodes                        3,420 edges
├─ 480 Section                   ├─ 2,029 REFERENCES
├─ 249 Definition                ├─   552 DEPENDS_ON
├─   9 Party                     ├─   480 CONTAINS
└─   4 Document                  ├─   227 DEFINES
                                 ├─    68 AMENDS
                                 ├─    52 EXCEPTION_TO
                                 └─    12 PARTY_TO
```

Main structure identified per edge type – what it stores.


| Edge           | What it stores                                                                              | The kind of question it helps                                                  |
| -------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `AMENDS`       | instrument, effective date, op type (`restate` / `insert` / `replace` / `delete`), new text | Which wording is in force on a date. Also the full amendment list.             |
| `DEPENDS_ON`   | definition → the terms it is computed from                                                  | "What is this ratio made of, and what uses it."                                |
| `REFERENCES`   | section → section or definition                                                             | Covenant → default → remedy.                                                   |
| `EXCEPTION_TO` | carve-out → the prohibition                                                                 | Liens, restricted payments.                                                    |
| `PARTY_TO`     | party → document, with a role                                                               | "Who are the parties." Party nodes have no text – the role is the edge.        |
| `CONTAINS`     | document → section → subsection                                                             | Hierarchy, and the sibling next door (Remedies sits beside Events of Default). |
| `DEFINES`      | §1.01 → definition                                                                          | Term lookup.                                                                   |


How an as-of date gets applied. `find_effective(node, as_of)` keeps only the `AMENDS` edges where `effective_date <= as_of`, then applies them:

- `restate` replaces the body.
- `insert` appends.
- `replace` / `delete` keeps the original and attaches the instruction.

This runs per section, not per document – which is the part that matters. At 2019-01-01, Consolidated EBITDA is already amended. Consolidated Leverage Ratio, which is computed from it, is not. "Just use the latest agreement" cannot say that.

Every citation carries the version it is quoting, e.g. `§7.05(a), as amended by Second Amendment (effective 2020-05-13)`.

### What I Considered and Dropped

- Putting the amendments into the same chunk index as the original. Similarity cannot tell the 2018 text of a section from the 2020 text of the same section.
- A big legal ontology, or a separate `SUPERSEDES` edge alongside `AMENDS`. One edge with a date already does versioning. The extra types did not make sense for now.
- Neo4j. 742 nodes. A Python walk is enough and the repo runs with no database to stand up.
- An LLM to extract the graph. The HTML is regular enough for rules, and rules are deterministic and cost nothing at query time.
- A hand-written list mapping assignment phrasing to gold sections. I removed it. Entry points are the citation in the question, the defined term, and dense retrieval – nothing keyed to a question id.

---

## How One Question Actually Runs

Q6: "Can a subsidiary distribute to its parent?"

```
Step 1: Seed – no section cite in the question, so defined terms ("Restricted Payment",
        "Subsidiary") plus dense retrieval over the chunk index
Step 2: Walk – 2 hops on CONTAINS / REFERENCES / DEPENDS_ON / EXCEPTION_TO,
        plus inbound EXCEPTION_TO, because the carve-out points back at the prohibition
Step 3: Resolve – find_effective() on every node for the as-of date
Step 4: Add what the walk cannot see – the sibling through the parent, the definition's
        inputs, the party roster, the amendment catalog
Step 5: Rerank – structural blocks pack first and get shortened if the window fills.
        They never get dropped. Ranked text fills whatever is left.
Step 6: Model writes the answer from that context
```

Answer: covenant → the defined term → the carve-out that permits it, each cited with its version.

Depth 2, not 4 because a deeper walk filled the reranker with loosely related definitions and buried the gold section. [Reranker tuning for a bigger corpus -> Future Scope]

---

## Findings – Where the Graph Beat the Baseline

Same 12 questions from the brief. Judge is Claude Sonnet, temperature 0.01, gold answer shown to the judge. Numbers are the frozen run in `eval/frozen/`.

```
12 assignment questions
├─ 5 answerable from one good passage   → graph 5/5,  baseline 5/5
└─ 7 that need an edge to answer        → graph 7/7,  baseline 0/7
                                           graph 12/12, baseline 5/12
```


| Q   | Question                                                                      | Graph | Baseline |
| --- | ----------------------------------------------------------------------------- | ----- | -------- |
| 1   | Governing law and forum                                                       | yes   | yes      |
| 2   | Maturity of the term loan. There is no term loan – it is a revolver.          | yes   | no       |
| 3   | Parties and roles                                                             | yes   | no       |
| 4   | Define Borrowing, list every section that depends on it                       | yes   | no       |
| 5   | Can the borrower lien its IP. Covenant plus exceptions.                       | yes   | yes      |
| 6   | Can a subsidiary distribute to its parent. Covenant → definition → carve-out. | yes   | no       |
| 7   | Financial covenant breach, through default, to remedies                       | yes   | no       |
| 8   | Find a conflict or a bad cross-reference                                      | yes   | yes      |
| 9   | Max restricted payment as of 2018-01-01                                       | yes   | no       |
| 10  | Same question as of 2020-06-01                                                | yes   | yes      |
| 11  | Every amended section, when, by which instrument                              | yes   | no       |
| 12  | How the leverage ratio changed                                                | yes   | yes      |


Q11 is the clean example of the failure mode. The baseline retrieved scattered amendment fragments, the model filled the gap from pretraining, and it named a different company's credit agreement. The graph puts the whole `AMENDS` list in the prompt and the model just lists from it.

Other numbers on the same 12:


|                                                     | Graph | Baseline |
| --------------------------------------------------- | ----- | -------- |
| Gold nodes actually in the prompt                   | 1.00  | 0.48     |
| Right version for the as-of date                    | 1.00  | 0.67     |
| Citations that say which amendment they are quoting | 29    | 0        |


There is also a check that uses no model at all – it resolves each citation against the graph and recomputes the version from the `AMENDS` edges. 0.812 of graph citations resolve and get the version claim right. 0.025 of baseline citations state a version at all, which is the real point: a baseline citation looks like `near §7.05(a)` and there is nothing there to check.

Held-out set, 20 questions, not used to tune retrieval, same model and same judge:

```
20 held-out questions
├─ graph    16/20
└─ baseline  9/20
   the gap is amendment-evolution 4/4 vs 1/4
             definition-dependents 3/3 vs 0/3
```

---

## Where It Lost, and Where It Cost Me Something For Nothing

Three things about that 12/12 before anything else.

The 12 questions were used during development. I ran them, read the failures, changed retrieval, ran again. 12/12 is the end state of that loop, not a held-out score. The 20-question set exists precisely because of this, and the graph gets 16/20 there.

One context block is gated on a question shape I found from a failure. `_component_effect_context` projects amendments onto the terms a definition is computed from, and fires only when a question mentions a change and asks what relies on it. There is no question id anywhere in that code – but I would not have written the gate without first watching the leverage-ratio question report the wrong thing.

One gold answer changed after a failure. The amendment-catalog gold said 61 operations. The graph has 68 once the Third Amendment is in, broken down 4/30/34 by instrument. The gold was stale, not the answer – but it is still a gold edit made after watching a model fail, so it goes here and not in a footnote.

On the brief's 12 the graph does not lose a question. It does lose on the held-out set.

Two held-out questions where the baseline was marked right and the graph was marked wrong:

- hh_08, as of 30 June 2021. The graph had the right amendment and the right leverage ratio, then said the fixed-charge covenant is measured over four fiscal quarters. The Third Amendment says that one quarter only. Flat RAG happened to say the single quarter.
- hh_09, changes across more than one amendment. Flat RAG was marked right. The graph answer got scored wrong because the judge returned no JSON and my harness scores a parse error as false. That is my scorer, not a better baseline answer – but it is still sitting in the 16/20.

Categories where the extra structure bought nothing, graph vs baseline:


| Held-out category             | Graph | Baseline |
| ----------------------------- | ----- | -------- |
| false premise                 | 3/3   | 3/3      |
| as-of                         | 2/4   | 2/4      |
| aggregation across amendments | 1/3   | 1/3      |


The as-of code does return the right vintage – verified separately with no model. On this set that did not turn into more correct answers, which is a different claim than "the resolution works".

On the three context blocks. Proviso split, change history and conflict audit each helped a 14B model. On this 27B I turned all three off. Held-out ablation, same revision – all three on: 15/20. Graph only: 17/20. The conflict block by itself cost a conflict question, which is the thing it was built for. An older note of mine said these blocks were why the score went 9/12 → 12/12. Those two runs were not the same code. The ablation does not support that, so I dropped the claim.

Other cost with no gain:

- Context precision on the 12 is 0.211 graph vs 0.227 baseline. Force-including structural nodes means some of them are not gold.
- Latency on the same 12 – graph p50 123 s, baseline p50 107 s. Thinking is on. See Cost.
- Neighbours that are semantically close with no edge between them. On that slice of an 88-question retrieval probe, baseline recall is 0.575 and graph recall is 0.50. Flat RAG should win there. If the graph had won, it would be matching on similarity and not on structure.

When I would not build this: one agreement, no amendments, and the question already names the section. The graph starts earning the latency the moment a question needs a date, a party role, a dependent term, or the full amendment list.

---

## Entity Resolution at Scale

What I have is one deal, four documents. Aliases cover `Dennys` vs `Denny's` in the OCR, and one bank appearing in two roles. That is string match, and it is fine at this size. It breaks at 10,000 documents from 40 firms, in three specific ways.

Same fund, six spellings. Matching on the name in the preamble splits Wells Fargo, WFB and the OCR variants into different parties, and merges two funds that share a short name. So stop keying the party on that string. A registry keyed by CIK or LEI, and the six surface forms become aliases on one record. Candidates come from blocking plus embeddings, and resolution happens inside the block – a person or a model decides within a block, never across 10,000 filings.

"Permitted Liens" means a different thing in every agreement. A global Definition node is wrong. Definitions get namespaced by agreement – Permitted Liens in this Denny's agreement is not Permitted Liens in some other credit, and the same goes for Consolidated EBITDA. A question that crosses deals is an alignment step, not a shared node.

Amendments. Today the parser reads "this amends the Existing Credit Agreement" out of the prose. Across 40 firms that sentence is ambiguous. `AMENDS` should come from filing metadata instead – this accession amends that accession. And a low-confidence OCR span does not get to support a hard answer without a person looking at it.

---

## What Else Breaks at 10,000 Documents

Identity is the first thing that breaks. It is not the only one.

Storage and serving. 742 nodes is a Python walk. 10,000 agreements is somewhere near seven million nodes, and one JSON file does not survive that. Edges go into a graph database, section bodies into a text store, and the vector index shards by agreement family so a question about one deal never scores against another. The as-of logic itself does not change – `find_effective` is a filter on edge dates and it is the same query at any size.

Retrieval quality – this is the piece I left on the table. The reranker shipping here is off-the-shelf `ms-marco-MiniLM-L-12-v2`. General web relevance, zero legal training. And the weak slice I already measured is exactly the one a trained reranker fixes: passages that are semantically close with no edge between them, baseline recall 0.575 vs graph 0.50. Those are hard negatives by definition. So mine them out of the index itself – top-K per question, grade each candidate 0–3 with the same frontier judge I already run offline, keep 2 and 3 as positives, 0 and 1 as hard negatives, roughly 4 positives and 5 negatives per row, split train/test doc-disjoint, fine-tune a domain cross-encoder. All the labelling is offline so none of it lands in query latency. I have built this shape of pipeline before, and the doc-disjoint split is the part people skip and then wonder why validation looks perfect.

The answer model. What is left failing is style, not knowledge. The version qualifier gets dropped on about one citation in five even though the context block carries it. 28 brackets came out malformed. hh_08 stated a measurement window that contradicts a sentence sitting in its own prompt. Those are preference pairs, not a retrieval problem, and not a reason to go buy a bigger model. Chosen = the citation with the instrument and the date, rejected = the one the model shortened, then DPO on the 27B. Cheaper than the next model up and aimed at the actual defect. The graph is what makes this affordable – it already knows the correct citation, so the preference data is generated and not annotated.

Throughput. At this size the 123 s median is the wall, and almost all of it is the model writing with thinking on and uncapped. Cap the thinking budget, batch the generations, serve the model once behind an endpoint instead of loading it per process, and shard across GPUs. I ran the held-out set that way already – one worker per GPU, embedder pinned to CPU to free VRAM – which is how a 20-question set finished in roughly the time one sequential pass would have taken.

Evaluation. 32 hand-written questions do not describe 10,000 documents, and the 12 in this brief were used during development, which is the same failure mode at any scale. Generate instead: cluster the sections, sample a context per cluster, have a frontier model write the question and the grounded answer, dedup near-duplicates at 0.92 cosine, and keep a deliberate unanswerable slice so "this agreement does not say" is a tested behaviour and not a hope. Hold the generated set doc-disjoint from anything the reranker trained on.

---

## Cost

Indexing, this corpus, about 158 pages, one A10G:

```
graph build                          0.79 s
embed chunks + build FAISS index    21    s
                                    ─────────
total                               21.8  s
per 100 pages (scaled by count)     13.8  s
API cost                            $0
```

No LLM call and no API call anywhere in the build, so the build is free.

Query latency – the frozen 12-question run, same GPU, same model, 16k context, thinking on:


|          | p50     | p95     |
| -------- | ------- | ------- |
| Graph    | 123.1 s | 199.3 s |
| Baseline | 107.0 s | 186.7 s |


Almost all of that is the model writing. Seeding and the walk are under 50 ms, rerank is a few hundred ms. So the graph adds about 16 s at the median, and that 16 s is what buys the seven questions the baseline gets wrong.

---

## If I Had One More Week

Three things. The code that measures each one already exists, so each is a day of work and a re-run, not a project.

1. Citation gate. Every context block already carries a full citation including the amendment and the date. The model drops the version on about one in five, and also emits brackets that are not citations at all. So after generation, resolve the citation with the checker I already have and repair it from the block label. But model the subsections below `(a)` first – otherwise the repair confidently claims a section the graph does not store.
2. Stop scoring a broken judge response as a wrong answer. hh_09 is a JSON parse failure. Retry it, or mark it unscored. Separately, if the answer stops mid-sentence because it hit the token limit, continue it – one held-out miss is just that cutoff.
3. hh_08. The model had the Third Amendment in context and still described the wrong measurement window. Check that a stated period matches the sentence `find_effective` already returned. No special case for that question id.

Item 1 moves citation verification off 0.812. Items 2 and 3 touch three of the four held-out misses – the parse error, the truncation, and hh_08 – and none of the three is a retrieval failure. The right text was in the prompt every time. Only one of the three is certain to flip, because hh_09 was never a wrong answer, just an unscored one.

Not in that week: a bigger ontology, a graph database, a trained reranker, or DPO. Those are the previous section and they need a corpus this size does not justify. What is actually left here is a scorer bug, a citation the model shortened, and one wrong sentence sitting in front of the right text.