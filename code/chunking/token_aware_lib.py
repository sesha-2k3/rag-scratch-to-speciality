"""Token counting, silent truncation, and token-based splitters with real tools.

Concepts: notes/02-embeddings-metadata.md
Install:  uv pip install sentence-transformers langchain-text-splitters tiktoken
Run:      python code/chunking/token_aware_lib.py
"""
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model token limit: {model.max_seq_length}")

# 1. Count tokens with the model's own tokenizer
text = "Error code E-4012 means the payment token expired."
n_tokens = len(model.tokenizer(text)["input_ids"])
print(f"{len(text)} characters -> {n_tokens} tokens")

# 2. Silent truncation: same first 256+ tokens means the same vector
prefix = "This is filler text about quarterly planning. " * 60
a = model.encode(prefix + "The answer is about Atlas.", normalize_embeddings=True)
b = model.encode(prefix + "Completely different ending about bananas.",
                 normalize_embeddings=True)
print(f"Cosine similarity of two different long texts: {float(a @ b):.4f}")
print("(~1.0 because everything after the limit was dropped)")

# 3. Splitters that count tokens instead of characters
st_splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk=200,
    chunk_overlap=30,
)
tiktoken_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=200, chunk_overlap=30
)
long_text = "Project Atlas entered beta testing in September. " * 80
print(f"MiniLM-token chunks:  {len(st_splitter.split_text(long_text))}")
print(f"tiktoken chunks:      {len(tiktoken_splitter.split_text(long_text))}")
