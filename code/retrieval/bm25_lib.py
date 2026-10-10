"""BM25 with the rank_bm25 library.

Concepts: notes/05-bm25.md
Install:  uv pip install rank-bm25
Run:      python code/retrieval/bm25_lib.py

rank_bm25 uses a slightly different IDF formula than Lucene, so scores
differ from bm25_scratch.py. The ranking is what matters.
"""
import re
from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


docs = [
    "Error code E-4012 means the payment token expired.",
    "Error code E-4021 is a timeout error; retry the code path.",
    "General error handling: log every error and alert on repeated error spikes.",
    "Project Atlas entered beta testing in September.",
    "Employees receive 12 paid sick days per year.",
]
bm25 = BM25Okapi([tokenize(d) for d in docs], k1=1.2, b=0.75)

scores = bm25.get_scores(tokenize("error code E-4012"))
for score, doc in sorted(zip(scores, docs), reverse=True):
    print(f"{score:6.2f}  {doc}")

print("\nTop 2:", bm25.get_top_n(tokenize("error code E-4012"), docs, n=2))
