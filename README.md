# RAG: scratch to speciality

My notes and worked examples from learning Retrieval-Augmented Generation: chunking, embeddings, the query flow, evaluation, BM25, and advanced retrieval patterns.

## Layout

```
notes/      Concepts, numbered as a learning path
code/       Runnable examples; each reproduces a worked example from the notes
```

| # | Notes | Code | Quiz |
|---|---|---|---|
| 01 | [Chunking](notes/01-chunking.md) | `code/chunking/fixed_recursive_*` |
| 02 | [Embeddings, tokens, metadata](notes/02-embeddings-metadata.md) | `code/chunking/token_aware_*` |
| 03 | [Query-time flow](notes/03-query-flow.md) | `code/retrieval/query_pipeline_*` |
| 04 | [Evaluation](notes/04-evaluation.md) | `code/evaluation/*` |
| 05 | [BM25](notes/05-bm25.md) | `code/retrieval/bm25_*` |
| 06 | [Advanced retrieval](notes/06-advanced-retrieval.md) | `code/retrieval/rrf_*` |

## Code conventions

Every example comes in two versions:

- **`*_scratch.py`**: implemented from scratch with the Python standard library only. Runs offline, no installs. Read these to understand the mechanics.
- **`*_lib.py`**: the same idea with real libraries (sentence-transformers, LangChain, rank_bm25, ranx, statsmodels). Use these as templates for real projects.

Run everything from the repository root:

```bash
python code/evaluation/retrieval_metrics_scratch.py   # no installs needed

pip install -r requirements.txt                       # for *_lib.py files
python code/evaluation/retrieval_metrics_lib.py
```

The first `*_lib.py` run downloads the `all-MiniLM-L6-v2` model (~90 MB).

## Worked examples you can verify

| Example | Script | Expected output |
|---|---|---|
| Blog's "50-char" chunk is really 47 chars | `chunking/fixed_recursive_scratch.py` | `47 chars, not 50` |
| Question lost to truncation | `chunking/token_aware_scratch.py` | `Question survived truncation? False` |
| Dimension mismatch rejected | `retrieval/query_pipeline_scratch.py` | `ValueError` message |
| BM25 scores for "error code E-4012" | `retrieval/bm25_scratch.py` | `8.80, 3.87, 1.86` |
| RRF fusion | `retrieval/rrf_scratch.py` | `A, C, B, E, D, F` |
| Precision / recall / hit rate | `evaluation/retrieval_metrics_scratch.py` | `0.40, 0.67, 1, 0.00` |
| nDCG@5 (and the wrong-ideal mistake) | `evaluation/retrieval_metrics_scratch.py` | `0.498` (`0.65` if wrong) |
| MRR over 4 queries | `evaluation/retrieval_metrics_scratch.py` | `0.46` |
| 82% vs 78% significance | `evaluation/compare_pipelines_scratch.py` | `p = 0.53` |

## Source

Started from [From Zero to RAG: The Art of Document Chunking and Embedding for RAG](https://medium.com/@jesvinkjustin/from-zero-to-rag-the-art-of-document-chunking-and-embedding-for-rag-d9764695cc46), then extended with evaluation and advanced retrieval.
