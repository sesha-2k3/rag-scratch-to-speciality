# 04 · Evaluation

Code: [`retrieval_metrics_scratch.py`](../code/evaluation/retrieval_metrics_scratch.py) · [`retrieval_metrics_lib.py`](../code/evaluation/retrieval_metrics_lib.py) · [`compare_pipelines_scratch.py`](../code/evaluation/compare_pipelines_scratch.py) · [`compare_pipelines_lib.py`](../code/evaluation/compare_pipelines_lib.py)

## Evaluate retrieval and generation separately

A wrong answer has two possible causes:

- **Retrieval failure:** the right chunk never reached the LLM. No prompt change fixes this.
- **Generation failure:** the chunk was retrieved, but the LLM ignored or misread it, or trusted a conflicting chunk.

Grading only the final answer tells you *that* it failed, not *where*.

## Retrieval metrics

Running example: relevant = `{A, B, C}`, retrieved = `[X, A, Y, B, Z]`.

| Metric | Definition | Value |
|---|---|---|
| Precision@5 | relevant retrieved ÷ retrieved | 2/5 = **0.40** |
| Recall@5 | relevant retrieved ÷ all relevant | 2/3 = **0.67** |
| Hit rate@5 | at least one relevant retrieved? | **1** |
| Recall@1 | top result relevant? | **0** |

Use the k you actually pass to the LLM. If you send the top 3, @10 metrics flatter you. Recall matters most because a missed chunk can't be recovered later.

**Diagnostic:** recall@10 = 0.95 but recall@3 = 0.60 means retrieval *finds* the chunks but *ranks* them poorly. Add a reranker; don't replace the embedding model.

## MRR (Mean Reciprocal Rank)

"On average, how high is the first useful result?"

1. For each query, find the rank of the first relevant result.
2. Reciprocal rank = 1 ÷ rank (0 if not found in top k).
3. Average across queries.

| Query results | First relevant | RR |
|---|---|---|
| `[R, X, X, X, X]` | 1 | 1.000 |
| `[X, X, R, X, X]` | 3 | 0.333 |
| `[X, R, R, X, X]` | 2 | 0.500 (second R ignored) |
| `[X, X, X, X, X]` | none | 0 |

MRR = 1.833 ÷ 4 = **0.46**

Rank 1 → 1.00, rank 2 → 0.50, rank 3 → 0.33, rank 10 → 0.10. Steep at the top.

MRR ignores everything after the first hit, how many relevant chunks were found, and degrees of relevance. `[R, X, X, X, X]` and `[R, R, R, R, R]` both score 1.0. Use it when one chunk answers the question.

Pitfalls: incomplete labels hurt MRR heavily (an unlabeled correct chunk at rank 1 halves the score), and unanswerable questions should be excluded.

## nDCG (normalized Discounted Cumulative Gain)

"How good is the whole ordering, from 0 to 1?"

- **Gain:** relevance of each result (binary 0/1, or graded 0/1/2).
- **Cumulative:** add them up.
- **Discounted:** divide each by log₂(rank + 1), so lower ranks count less.

  | Rank | 1 | 2 | 3 | 4 | 5 | 10 |
  |---|---|---|---|---|---|---|
  | Discount | 1.000 | 0.631 | 0.500 | 0.431 | 0.387 | 0.289 |

- **Normalized:** divide by the **ideal DCG (IDCG)**.

### Computing the ideal DCG

1. List **every relevant chunk from your labels**, including ones you failed to retrieve.
2. Sort by relevance, highest first.
3. Place them at ranks 1, 2, 3, …; fill the rest with zeros up to k.
4. Compute DCG on that list.

Example: DCG of `[X, A, Y, B, Z]` = 0.631 + 0.431 = 1.062. Ideal `[A, B, C, -, -]` = 1 + 0.631 + 0.5 = 2.131. **nDCG@5 = 0.50**.

Common mistake: building the ideal only from retrieved relevant chunks (A, B) gives 1.062 ÷ 1.631 = 0.65, which stops penalizing the missed chunk C.

If more relevant chunks exist than k, the ideal fills all k slots. With no relevant chunks, IDCG = 0 and nDCG is undefined; skip those queries.

### Same recall, different nDCG

| Ranking | Recall@5 | nDCG@5 |
|---|---|---|
| `[A, B, X, Y, Z]` | 0.67 | 0.77 |
| `[X, A, Y, B, Z]` | 0.67 | 0.50 |
| `[X, Y, Z, A, B]` | 0.67 | 0.38 |

### Graded relevance

Labels A=2, C=2, B=1, D=1; retrieved `[B, X, A, Y, C]`. IDCG from `[2, 2, 1, 1, 0]` = 4.193. DCG = 1 + 2(0.5) + 2(0.387) = 2.774. **nDCG@5 = 0.66**. Recall and binary MRR can't see that a direct answer should outrank a side note.

### Why normalize

A query with 1 relevant chunk retrieved at rank 1 has DCG 1.0. A query with 5 relevant chunks, retrieved at ranks 2–5, has DCG 1.949, which is higher despite being worse. After normalizing: 1.00 vs 0.66.

### Why nDCG matters for rerankers

A reranker reorders the same candidates, so recall@k of the candidate set can't change. nDCG shows whether the reranker moved the right chunks up.

| Metric | Measures | Best for |
|---|---|---|
| Recall@k | Found all relevant chunks? | Multi-chunk questions |
| MRR | Rank of first relevant chunk | Factoid questions |
| nDCG@k | Quality of the whole ordering | Multiple or graded relevant chunks |

## Building a test set

1. **Real user questions** with hand-labeled chunks (most realistic, slowest).
2. **Expert-written questions.**
3. **LLM-generated questions** from sampled chunks (fast).

**Trap:** generated questions copy the chunk's wording ("How many days of paid time off do employees accrue per month?"), so retrieval looks far better than with real users ("how much vacation do I get?").

Mitigations: instruct paraphrasing, generate multi-chunk questions, include messy and ambiguous questions, and validate that synthetic scores rank pipelines the same way as 20–30 hand-labeled real questions.

## Generation metrics (RAGAS terminology)

| Metric | Question | Stage |
|---|---|---|
| Context recall | Did retrieval find everything needed? | Retrieval |
| Context precision | Is retrieved context relevant and ranked high? | Retrieval |
| Faithfulness | Is every claim supported by the context? | Generation |
| Answer relevance | Does the answer address the question? | Generation |

**Correct but unfaithful** is still a failure. If retrieval returns Sydney tourism chunks and the LLM answers "Canberra" from training knowledge, the answer is right but hides a retrieval bug, and the same behavior produces confident wrong answers on private data.

## LLM-as-judge biases and fixes

Biases: position, verbosity, self-preference, leniency/drift, factual blind spots.

Fixes: swap positions and count only consistent wins; use specific binary criteria; give the judge the reference answer or context; ask for reasoning before the verdict; use a different model family; calibrate against ~50 human-graded answers.

## Chunk size: recall up, answers down

Going from 300- to 1,000-token chunks can raise recall@5 while answers get worse: bigger chunks pass the "answer is somewhere inside" check more easily, the LLM gets 5,000 tokens of mostly noise, and embeddings blur. **Measure final answers, not just retrieval.** A common fix is small-to-big retrieval (see [06](06-advanced-retrieval.md)).

## Unanswerable questions

Include out-of-scope questions, **near misses** (contractor leave when the handbook only covers employees), and false premises ("Why was Atlas cancelled?").

Track both: abstention rate on unanswerable questions (want high) and false abstention rate on answerable ones (want low). Tightening thresholds raises both. For HR, legal, or medical content, accept more false abstentions.

## Is 82% vs 78% significant?

On 50 questions that's 41 vs 39 correct. Compare **paired**, using only disagreements: A-only = 6, B-only = 4.

McNemar: (6 − 4)² ÷ (6 + 4) = **0.4**, p ≈ **0.53** (exact binomial p ≈ 0.75). Not significant.

Instead: inspect the disagreeing questions, grow the set to hundreds, rerun non-deterministic steps, and break ties on cost, speed, and simplicity.

## Switching embedding models: evaluation plan

1. Build the new index alongside the old one, same chunking.
2. Compare retrieval (recall@k at your k, MRR/nDCG) on real and synthetic questions.
3. Compare end-to-end answers, including abstention.
4. Inspect per-question regressions (codes, non-English text, tables).
5. Check speed, cost, dimension, token limit, and data-location constraints.
6. Roll out gradually; watch thumbs-down rate, "I don't know" rate, rephrasing, citation clicks. Keep the old index for rollback.
7. Record model name/version in metadata and save evaluation results.

General rule: change one variable, measure on a fixed test set, inspect per-question differences, roll out with monitoring.
