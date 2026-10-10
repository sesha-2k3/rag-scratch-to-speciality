"""Retrieval metrics from scratch: precision, recall, hit rate, MRR, nDCG.

Concepts: notes/04-evaluation.md
Run:      python code/evaluation/retrieval_metrics_scratch.py
"""
import math


def precision_at_k(retrieved, relevant, k):
    return sum(d in relevant for d in retrieved[:k]) / k


def recall_at_k(retrieved, relevant, k):
    return sum(d in relevant for d in retrieved[:k]) / len(relevant)


def hit_rate_at_k(retrieved, relevant, k):
    return float(any(d in relevant for d in retrieved[:k]))


def reciprocal_rank(retrieved, relevant, k=10):
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1 / rank  # only the FIRST relevant hit counts
    return 0.0


def mrr(all_retrieved, all_relevant, k=10):
    rrs = [reciprocal_rank(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
    return sum(rrs) / len(rrs)


def dcg(gains):
    # rank 1 -> divide by log2(2)=1, rank 2 -> log2(3), ...
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(retrieved, relevance, k):
    """relevance: {chunk_id: grade} from your ground-truth LABELS."""
    actual = [relevance.get(doc, 0) for doc in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]  # from labels, not results
    idcg = dcg(ideal)
    return dcg(actual) / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    retrieved, relevant = ["X", "A", "Y", "B", "Z"], {"A", "B", "C"}
    print("Retrieved [X, A, Y, B, Z], relevant {A, B, C}")
    print(f"  precision@5 = {precision_at_k(retrieved, relevant, 5):.2f}")
    print(f"  recall@5    = {recall_at_k(retrieved, relevant, 5):.2f}")
    print(f"  hit rate@5  = {hit_rate_at_k(retrieved, relevant, 5):.0f}")
    print(f"  recall@1    = {recall_at_k(retrieved, relevant, 1):.2f}")

    labels = {"A": 1, "B": 1, "C": 1}
    print(f"\n  nDCG@5      = {ndcg_at_k(retrieved, labels, 5):.3f}")
    wrong_ideal = dcg([0, 1, 0, 1, 0]) / dcg([1, 1])  # ideal built from retrieved hits only
    print(f"  nDCG@5 with the WRONG ideal (forgets C) = {wrong_ideal:.2f}")

    print("\nSame recall, different order:")
    for ranking in (["A", "B", "X", "Y", "Z"], ["X", "A", "Y", "B", "Z"], ["X", "Y", "Z", "A", "B"]):
        print(f"  {ranking}: recall@5={recall_at_k(ranking, relevant, 5):.2f} "
              f"nDCG@5={ndcg_at_k(ranking, labels, 5):.2f}")

    graded = {"A": 2, "C": 2, "B": 1, "D": 1}
    print(f"\nGraded nDCG@5 for [B, X, A, Y, C] = "
          f"{ndcg_at_k(['B', 'X', 'A', 'Y', 'C'], graded, 5):.2f}")

    queries = [["R1", "X", "X", "X", "X"], ["X", "X", "R2", "X", "X"],
               ["X", "R3", "R4", "X", "X"], ["X", "X", "X", "X", "X"]]
    labels_per_query = [{"R1"}, {"R2"}, {"R3", "R4"}, {"R5"}]
    print(f"\nMRR over 4 queries = {mrr(queries, labels_per_query, k=5):.2f}")
