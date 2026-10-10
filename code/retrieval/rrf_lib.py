"""Hybrid search (BM25 + vectors) fused with RRF using LangChain.

Concepts: notes/06-advanced-retrieval.md
Install:  uv pip install langchain langchain-community langchain-core
          langchain-huggingface sentence-transformers rank-bm25
Run:      python code/retrieval/rrf_lib.py

EnsembleRetriever combines retrievers with weighted Reciprocal Rank Fusion.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

try:  # LangChain < 1.0
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # LangChain >= 1.0 moved legacy retrievers
    from langchain_classic.retrievers import EnsembleRetriever

texts = [
    "Error code E-4012 means the payment token expired.",
    "Error code E-4021 is a timeout error; retry the code path.",
    "Employees accrue 1.5 days of paid time off per month.",
    "General error handling: log every error and alert on spikes.",
]

bm25 = BM25Retriever.from_texts(texts)
bm25.k = 3

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = InMemoryVectorStore.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})

hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])

for query in ["error code E-4012", "how much vacation do I get?"]:
    print(f"\nQ: {query}")
    for name, retriever in [("bm25", bm25), ("vector", vector), ("hybrid", hybrid)]:
        top = retriever.invoke(query)[0].page_content
        print(f"  {name:<7} top hit: {top}")
