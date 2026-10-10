# 03 · The query-time flow

Code: [`query_pipeline_scratch.py`](../code/retrieval/query_pipeline_scratch.py) · [`query_pipeline_lib.py`](../code/retrieval/query_pipeline_lib.py)

**The LLM usually doesn't do the search.** Your application code searches first; the LLM only sees the results, pasted into its prompt as text.

```
User question
   → Embedding model (same model as ingestion)
   → Vector database (nearest neighbors, top_k, optional metadata filter)   ← chunk vectors stored at ingestion
   → Prompt assembly (question + retrieved chunk TEXT)
   → LLM (answers from the given context)
   → Answer with citations
```

## Steps, using "What's the current status of Project Atlas?"

1. **Embed the question** with the same model used for the chunks → 384 numbers.
2. **Search.** The database finds the closest stored vectors (usually cosine similarity) using an approximate-nearest-neighbor index such as HNSW, so it doesn't compare against every vector. Apply filters like `project = "Atlas"` here. Each hit returns a score, its text, and metadata.
3. **Build the prompt** from the chunks' text, not their vectors:

   ```
   Answer the question using only the context below.
   If the answer isn't in the context, say you don't know.

   Context:
   [1] (Atlas_status_report.pdf, p.2) Project Atlas entered beta testing on Sept 12...
   [2] (Atlas_roadmap.docx) The public launch is scheduled for Q1 2027...
   [3] (Weekly_sync_notes.md) Atlas is blocked on the payments integration...

   Question: What's the current status of Project Atlas?
   ```

4. **Generate.** The LLM never sees embeddings or the database; it answers from the text.

```python
query_vec = embed_model.encode(question)
hits = vector_db.search(query_vec, top_k=3, filter={"project": "Atlas"})
context = "\n\n".join(h.text for h in hits)
answer = llm.generate(f"Context:\n{context}\n\nQuestion: {question}")
```

**Variation:** in agentic or tool-calling setups, the LLM is given a `search_documents` tool and decides when to call it. Your code still runs steps 1–2. See [06-advanced-retrieval.md](06-advanced-retrieval.md).

## "What if the query doesn't match the embedding size?"

| Situation | What happens |
|---|---|
| Question much shorter/longer than chunks | **Fine.** Output dimension is fixed. Only queries past the token limit are silently truncated. |
| Query vector has a different dimension | **Rejected** with a dimension-mismatch error. Similarity between different-length vectors is undefined. |
| Different model, same dimension | **No error, meaningless results.** |
| No relevant content exists | **Fails silently.** Vector search always returns the top_k nearest, even if barely related. |

## Designing for "no match"

- **Similarity threshold:** drop weak hits; if nothing survives, skip the LLM or say nothing was found.

  ```python
  hits = [h for h in hits if h.score >= 0.5]
  if not hits:
      return "I couldn't find anything about that in the project documents."
  ```

  The right cutoff depends on the embedding model. Calibrate it on real queries.

- **Prompt instruction:** "If the answer isn't in the context, say you don't know."
