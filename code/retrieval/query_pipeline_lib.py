"""The query-time RAG flow with a real embedding model.

Concepts: notes/03-query-flow.md
Install:  uv pip install sentence-transformers numpy
Run:      python code/retrieval/query_pipeline_lib.py

The final LLM call is left as a print, so this runs without an API key.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # use the SAME model for chunks and queries

chunks = [
    {"text": "Project Atlas entered beta testing on Sept 12.", "source": "atlas_status.pdf", "project": "Atlas"},
    {"text": "The Atlas public launch is scheduled for Q1 2027.", "source": "atlas_roadmap.docx", "project": "Atlas"},
    {"text": "Atlas is blocked on the payments integration.", "source": "sync_notes.md", "project": "Atlas"},
    {"text": "Hermes migrated the billing database in March.", "source": "hermes_notes.md", "project": "Hermes"},
]

# Ingestion (done once): embed every chunk. Normalized vectors -> dot product = cosine.
index = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
print(f"Index shape: {index.shape}")  # (4, 384)


def retrieve(question, top_k=3, threshold=0.3, project=None):
    q = model.encode(question, normalize_embeddings=True)            # 1. embed
    if q.shape[0] != index.shape[1]:
        raise ValueError("query/index dimension mismatch: re-embed with the same model")
    scores = index @ q                                                 # 2. similarity
    order = np.argsort(-scores)
    hits = [(float(scores[i]), chunks[i]) for i in order
            if project is None or chunks[i]["project"] == project][:top_k]
    return [h for h in hits if h[0] >= threshold]                     # 3. threshold (calibrate!)


def build_prompt(question, hits):                                      # 4. assemble
    context = "\n".join(f"[{i}] ({c['source']}) {c['text']}" for i, (_, c) in enumerate(hits, 1))
    return ("Answer using only the context below. If the answer isn't there, "
            f"say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}")


for question in ["What's the current status of Project Atlas?", "What is the parental leave policy?"]:
    hits = retrieve(question, project="Atlas" if "Atlas" in question else None)
    print(f"\nQ: {question}")
    for score, c in hits:
        print(f"   {score:.2f}  {c['text']}")
    if hits:
        print(build_prompt(question, hits))   # 5. send this to your LLM
    else:
        print("   Nothing above threshold -> answer \"I couldn't find that.\"")
