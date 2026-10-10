"""The query-time RAG flow with a toy embedding model, from scratch.

Concepts: notes/03-query-flow.md
Run:      python code/retrieval/query_pipeline_scratch.py

Shows: fixed vector size regardless of text length, top-k search,
dimension-mismatch rejection, similarity thresholds, and prompt assembly.
The "embedding" is a hashed bag of words, so it matches words, not meaning.
"""
import math
import re
import zlib

STOPWORDS = {"the", "a", "an", "is", "of", "in", "on", "what", "s", "to",
             "for", "and", "was", "our", "how", "do", "i", "it", "at"}


def embed(text, dim=64):
    """Any input length -> exactly `dim` numbers (normalized)."""
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        if word not in STOPWORDS:
            vec[zlib.crc32(word.encode()) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a, b):
    if len(a) != len(b):
        raise ValueError(f"query dimension {len(a)} does not match index dimension {len(b)}")
    return sum(x * y for x, y in zip(a, b))  # vectors are already normalized


class VectorStore:
    def __init__(self, dim):
        self.dim, self.rows = dim, []

    def add(self, text, metadata):
        self.rows.append((embed(text, self.dim), text, metadata))

    def search(self, query_vec, top_k=3, filter=None):
        """Always returns the top_k nearest rows, even if none are relevant."""
        if len(query_vec) != self.dim:
            raise ValueError(f"query dimension {len(query_vec)} does not match "
                             f"index dimension {self.dim}")
        rows = [r for r in self.rows
                if not filter or all(r[2].get(k) == v for k, v in filter.items())]
        scored = [(cosine(query_vec, vec), text, meta) for vec, text, meta in rows]
        return sorted(scored, key=lambda s: s[0], reverse=True)[:top_k]


def build_prompt(question, hits):
    context = "\n".join(f"[{i}] ({m['source']}) {t}" for i, (_, t, m) in enumerate(hits, 1))
    return ("Answer the question using only the context below.\n"
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}")


def answer(store, question, threshold=0.3, **search_kwargs):
    hits = store.search(embed(question, store.dim), **search_kwargs)
    print(f"\nQ: {question}")
    for score, text, meta in hits:
        print(f"   score={score:.2f}  {meta['source']}: {text[:50]}")
    hits = [h for h in hits if h[0] >= threshold]
    if not hits:
        print("-> No chunk passed the threshold: \"I couldn't find that in the documents.\"")
        return
    print("-> Prompt sent to the LLM:\n" + build_prompt(question, hits))


if __name__ == "__main__":
    store = VectorStore(dim=64)
    store.add("Project Atlas entered beta testing on Sept 12.",
              {"source": "atlas_status.pdf", "project": "Atlas"})
    store.add("The Atlas public launch is scheduled for Q1 2027.",
              {"source": "atlas_roadmap.docx", "project": "Atlas"})
    store.add("Atlas is blocked on the payments integration.",
              {"source": "sync_notes.md", "project": "Atlas"})
    store.add("Hermes migrated the billing database in March.",
              {"source": "hermes_notes.md", "project": "Hermes"})

    print("Vector size for a 2-word vs a 40-word text:",
          len(embed("Atlas status")), "vs", len(embed("word " * 40)))

    answer(store, "What is the status of Project Atlas launch?",
           top_k=3, filter={"project": "Atlas"})
    answer(store, "What is our parental leave policy?", top_k=3)

    try:
        store.search([0.1] * 1536)  # a query embedded with a different model
    except ValueError as err:
        print(f"\nDimension mismatch is rejected: {err}")
