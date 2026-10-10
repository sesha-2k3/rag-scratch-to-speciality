"""Retrieval metrics with the ranx library.

Concepts: notes/04-evaluation.md
Install:  uv pip install ranx
Run:      python code/evaluation/retrieval_metrics_lib.py

ranx takes ground-truth labels (Qrels) and system output with scores (Run).
Results should match retrieval_metrics_scratch.py.
"""
from ranx import Qrels, Run, evaluate

qrels = Qrels({
    "q1": {"A": 1, "B": 1, "C": 1},
    "q2": {"R2": 1},
})

# Scores only define the order: higher = ranked earlier.
run = Run({
    "q1": {"X": 5, "A": 4, "Y": 3, "B": 2, "Z": 1},   # [X, A, Y, B, Z]
    "q2": {"X1": 3, "X2": 2, "R2": 1},                # first relevant at rank 3
})

metrics = ["precision@5", "recall@5", "hit_rate@5", "mrr", "ndcg@5"]
print(evaluate(qrels, run, metrics))                     # averaged over queries
print(evaluate(qrels, run, metrics, return_mean=False))  # per query
