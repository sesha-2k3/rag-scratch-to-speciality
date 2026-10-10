"""Hybrid search fusion: why adding raw scores fails, and how RRF works.

Concepts: notes/06-advanced-retrieval.md
Run:      python code/retrieval/rrf_scratch.py
"""
from collections import defaultdict


def rrf(ranked_lists, k=60):
    """Reciprocal Rank Fusion: uses only ranks, ignores raw scores."""
    scores = defaultdict(float)
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


if __name__ == "__main__":
    vector = ["A", "B", "C", "D"]
    bm25 = ["C", "A", "E", "F"]
    print("RRF (k=60):")
    for doc, score in rrf([vector, bm25]):
        print(f"  {doc}: {score:.5f}")

    # Why not add raw scores? Different scales: BM25 swamps cosine.
    cosine = {"A": 0.82, "B": 0.80, "C": 0.55}
    bm25_raw = {"C": 14.2, "A": 6.1, "B": 0.0}
    naive = {d: cosine[d] + bm25_raw[d] for d in cosine}
    print("\nNaive score addition:", sorted(naive, key=naive.get, reverse=True),
          "<- order is just the BM25 order")
