# 06 · Advanced retrieval patterns

Code: [`rrf_scratch.py`](../code/retrieval/rrf_scratch.py) · [`rrf_lib.py`](../code/retrieval/rrf_lib.py)

## Hybrid search and Reciprocal Rank Fusion

Run vector search and BM25, then combine. **Don't add raw scores:** cosine sits roughly in 0.2–0.9, BM25 is unbounded, so 0.82 + 14.2 is just the BM25 order. Normalization is fragile because distributions shift per query.

**RRF uses ranks only:** `score(d) = Σ 1 / (k + rank of d in each list)`, with k = 60 (the original paper's default), which keeps rank 1 from dominating.

Vector `[A, B, C, D]`, BM25 `[C, A, E, F]`:

| Doc | Vector rank | BM25 rank | RRF |
|---|---|---|---|
| A | 1 | 2 | 1/61 + 1/62 = **0.03252** |
| C | 3 | 1 | 1/63 + 1/61 = **0.03227** |
| B | 2 | — | 0.01613 |
| E | — | 3 | 0.01587 |
| D, F | 4 | 4 | 0.01562 each |

Documents found by both methods rise to the top. Many databases (Weaviate, Qdrant, Elasticsearch, OpenSearch) support hybrid search natively. Send the fused top 20–30 to a reranker.

## Rerankers

Vector search uses a **bi-encoder** (query and chunk embedded separately: fast, less precise). A **cross-encoder** reads each query–chunk pair together: accurate, too slow for millions of chunks. Combine them:

```python
candidates = vector_db.search(query_vector, top_k=30)          # fast recall
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(user_query, c.text) for c in candidates])
top_chunks = [c for _, c in sorted(zip(scores, candidates), reverse=True)][:5]
```

Related: **MMR** (Maximal Marginal Relevance) picks chunks that are relevant and different from each other.

## Query transformation

- **Query rewriting:** turn "what about for contractors?" into a standalone question using chat history.
- **Multi-query:** generate several paraphrases, search each, fuse with RRF. Catches documents one phrasing misses.
- **HyDE:** embed an LLM-written *hypothetical answer* instead of the question, because answers resemble answer chunks. It hurts when the LLM doesn't know your domain (it invents a NASA "Atlas" and steers away from internal docs), for exact-term queries (it may drop "E-4012"), and when latency matters.

Treat all of these as optional and test them on your evaluation set.

## Query decomposition

| | Parallel | Sequential (multi-hop) |
|---|---|---|
| Example | "Compare the budgets of Atlas and Hermes" | "What was the previous project of the person who leads Atlas?" |
| Why single-shot fails | One vector for two topics; top 3 may be all Atlas | The answer chunk mentions "Priya Rao," not Atlas |
| Method | Search "Atlas budget" and "Hermes budget" separately, combine | Step 1 finds the lead; step 2 searches her previous projects |
| Latency | One round | Several rounds |
| Risk | Over-splitting simple questions | A wrong intermediate answer derails later steps |

## Self-query retrieval

"Show me Atlas status reports from last month" mixes a topic with filters. Embeddings have no concept of "last month." An LLM translates the question into a query plus filters (with the current date supplied):

```json
{"query": "project status report",
 "filter": {"project": "Atlas", "doc_type": "status_report",
            "date": {"gte": "2026-09-01", "lte": "2026-09-30"}}}
```

Give the LLM the schema and current date, validate its output, and fall back to looser filters if nothing matches. Implementations: LangChain `SelfQueryRetriever`, LlamaIndex auto-retrieval.

## Small-to-big (parent document) retrieval

Embed small **child** chunks (sentences or ~200 tokens) with a `parent_id`; return the **parent** section (~1,500 tokens) to the LLM. Precise matching plus enough context.

Costs: more tokens per query, duplicate parents to deduplicate, and choosing parent size (heading-based sections work well). Variants: **sentence window** retrieval and **auto-merging**.

## Contextual retrieval

"Revenue grew 3% over the previous quarter." Whose revenue? Which quarter? Chunking stripped the context.

Fix (Anthropic, 2024): for each chunk, give an LLM the whole document plus the chunk and ask for a short situating context, e.g. "This chunk is from ACME Corp's Q2 2026 earnings report, financial performance section." Prepend it, then embed **and** add to the BM25 index. Anthropic reported large reductions in failed retrievals, larger still combined with BM25 and reranking.

Cost: one LLM call per chunk, each including the full document. **Prompt caching** makes it practical. Re-run when documents change. The free version, prepending title and heading path, captures much of the benefit.

## GraphRAG

**Indexing:** an LLM extracts entities and relationships from every chunk → build a graph → cluster it into communities → an LLM summarizes each community at several levels.

**Strength: global and relationship questions.** "What are the main recurring risks across all our projects?" Vector RAG sees ~5 of 10,000 chunks; GraphRAG's global search map-reduces over community summaries built from the whole corpus. It can also trace connections (outage → payments service → Atlas dependency → delay) that no single document states.

**Costs:** expensive indexing (often far more than embedding), hard incremental updates, extraction errors that spread silently, and overkill for simple lookups.

For structured data ("How many projects went over budget in 2025?"), use **text-to-SQL**, not RAG.

## Agentic RAG

The LLM gets tools (`search_docs`, `query_database`, …) and loops: search → read → decide → search again → answer.

Worth it for: multi-hop questions with unknown paths, exploratory research, multiple heterogeneous sources, self-verification.

New failure modes: latency and cost (2 s → 20–40 s), unpredictable paths, loops or premature stops, compounding errors.

Mitigations: cap iterations and tokens, log traces, evaluate the path and the answer. Lighter alternative: a **router** that classifies questions and sends each type to a fixed pipeline.

## Choosing techniques

| Symptom | Cause | Technique |
|---|---|---|
| Part numbers and codes miss | Embeddings blur exact strings | Hybrid search |
| "Compare X and Y" only covers X | One vector for two topics | Parallel decomposition |
| Vague cross-corpus answers | Top-k sees few chunks | GraphRAG or map-reduce over summaries |
| Chunks say "this project" | Chunking removed context | Heading prefixes → contextual retrieval |
| Right chunks at ranks 4–10 | Ranking, not finding | Reranker |
| Pasted text breaks retrieval | Truncation/dilution | Query rewriting |

**Adoption order:** heading/title prefixes (almost free) → hybrid search + reranker (low risk, broad gains) → decomposition behind a router → global-question solutions only if those questions are frequent. **Add one technique at a time and measure each** against your evaluation set.
