"""BM25 from scratch: the hand-worked example, then a real inverted index.

Concepts: notes/05-bm25.md
Run:      python code/retrieval/bm25_scratch.py
"""
import math
import re
from collections import Counter, defaultdict

K1, B = 1.2, 0.75  # Lucene / Elasticsearch defaults


def idf(n_docs, doc_freq):
    """Rare terms get high weight; terms in almost every chunk get ~0."""
    return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


def tf_part(freq, doc_len, avg_len, k1=K1, b=B):
    """Saturating term frequency, normalized for chunk length."""
    return freq * (k1 + 1) / (freq + k1 * (1 - b + b * doc_len / avg_len))


def tokenize(text):
    # Keeps hyphenated identifiers like "e-4012" as one token.
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


class BM25Index:
    def __init__(self, docs):
        self.docs = docs
        self.tokens = [tokenize(d) for d in docs]
        self.avg_len = sum(map(len, self.tokens)) / len(docs)
        self.postings = defaultdict(dict)  # term -> {doc_id: frequency}
        for doc_id, toks in enumerate(self.tokens):
            for term, freq in Counter(toks).items():
                self.postings[term][doc_id] = freq

    def search(self, query, top_k=5):
        terms = tokenize(query)
        candidates = set().union(*(self.postings.get(t, {}).keys() for t in terms))
        scores = {}
        for doc_id in candidates:  # only chunks containing >= 1 query term
            scores[doc_id] = sum(
                idf(len(self.docs), len(self.postings[t]))
                * tf_part(self.postings[t][doc_id], len(self.tokens[doc_id]), self.avg_len)
                for t in terms if doc_id in self.postings.get(t, {})
            )
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(score, self.docs[doc_id]) for doc_id, score in ranked], len(candidates)


def worked_example():
    """Reproduces notes/05-bm25.md: N=1000 chunks, average length 100."""
    N, AVG = 1000, 100
    df = {"error": 300, "code": 200, "e-4012": 2}
    for term, n in df.items():
        print(f"  IDF({term}) = {idf(N, n):.2f}")
    chunks = {
        "A (E-4012)":         ({"error": 1, "code": 1, "e-4012": 1}, 100),
        "B (E-4021)":         ({"error": 2, "code": 2}, 100),
        "C (general errors)": ({"error": 5}, 200),
    }
    for name, (freqs, length) in chunks.items():
        score = sum(idf(N, df[t]) * tf_part(f, length, AVG) for t, f in freqs.items())
        print(f"  {name:<20} score = {score:.2f}")
    print("  Saturation (TF part at average length):",
          {f: round(tf_part(f, AVG, AVG), 2) for f in (1, 2, 5, 10, 100)})


if __name__ == "__main__":
    print("Worked example:")
    worked_example()

    docs = [
        "Error code E-4012 means the payment token expired.",
        "Error code E-4021 is a timeout error; retry the code path.",
        "General error handling: log every error and alert on repeated error spikes.",
        "Project Atlas entered beta testing in September.",
        "Employees receive 12 paid sick days per year.",
    ]
    results, n_candidates = BM25Index(docs).search("error code E-4012")
    print(f"\nInverted-index search scored {n_candidates} of {len(docs)} chunks:")
    for score, doc in results:
        print(f"  {score:5.2f}  {doc}")
