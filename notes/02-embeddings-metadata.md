# 02 · Embeddings, token limits, and metadata

Code: [`token_aware_scratch.py`](../code/chunking/token_aware_scratch.py) · [`token_aware_lib.py`](../code/chunking/token_aware_lib.py)

## Fixed output size

An embedding model always outputs the same number of dimensions regardless of input length. "Atlas status?" and a 300-word chunk both become exactly 384 numbers with `all-MiniLM-L6-v2`. Length changes where the vector points, not its shape.

## Same model for chunks and queries

Each model learns its own coordinate system. Mixing models breaks in two ways:

- **Different dimensions** (MiniLM 384 vs `text-embedding-3-small` 1,536): the database rejects the search with a dimension-mismatch error.
- **Same dimension, different models:** no error, but results are meaningless.

Switching embedding models means **re-embedding the entire corpus**. Store raw chunk text and record the model name in each chunk's metadata so a migration is a batch job.

## Characters vs. tokens

Models read **tokens** (sub-word units), not characters. English prose is roughly 4 characters per token; code, URLs, numbers, and non-English text are often 2–3 or worse.

`all-MiniLM-L6-v2` has a 256-token limit (`model.max_seq_length`). Anything past it is **silently truncated**: no error, no log, and a normal-looking vector that only represents the first 256 tokens. Two texts with the same first 256 tokens produce identical vectors.

If you chunk at 2,000 characters (~500 tokens), about half of every chunk is never embedded. It's still stored and sent to the LLM, but retrieval only sees the first half.

**Fix:** measure chunks in tokens with the embedding model's tokenizer:

```python
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk=200, chunk_overlap=30)
```

For OpenAI-style models: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)`.

Providers differ: local sentence-transformers models truncate silently; OpenAI's embedding API (~8K-token limit) returns an error.

## Truncation on the query side

Short questions never hit the limit. Pasted content does. A user pastes an ~800-token email and ends with "Based on this, will the Atlas launch date slip?":

1. Only the first 256 tokens are embedded; the question is cut off.
2. The query vector means "opening of an email about budgets."
3. Retrieval returns budget chunks instead of the Atlas roadmap.
4. The LLM still sees the full message (its context window is larger), which makes the failure confusing to debug.

Even under the limit, a query that is mostly pasted text is dominated by that text (query-side contextual bleeding).

Fixes, best first:

- **Rewrite the query** with an LLM into a short search phrase ("Project Atlas launch date timeline risks"). Embed that for retrieval; send the original message plus retrieved chunks to the LLM for the answer.
- **Separate question and pasted content** in the UI.
- **Chunk long queries**, search with each piece, and merge results.
- **At minimum, detect it:** count tokens and log or warn above the limit.

## Metadata

Embeddings are good at meaning and bad at exact attributes (dates, IDs, names, versions). Metadata covers the exact attributes.

1. **Filtering:** `department = 'Finance' AND year >= 2024`.
2. **Citations:** "Source: Employee_Handbook.pdf, p. 14."
3. **Ranking boosts:** prefer newer or official documents; route numerical questions to tables.

Example: "What's the parental leave policy in the 2025 handbook?" Handbooks from 2021–2025 have near-identical leave sections, so pure vector search might return 2022. "2025" barely moves an embedding. A filter fixes it exactly:

```python
results = vector_db.search(query_vector, top_k=3,
                           filter={"doc_year": 2025, "doc_type": "handbook"})
```
