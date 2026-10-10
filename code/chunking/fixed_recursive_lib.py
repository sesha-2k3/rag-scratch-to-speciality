"""Fixed-size and recursive chunking with LangChain.

Concepts: notes/01-chunking.md
Install:  uv pip install langchain-text-splitters
Run:      python code/chunking/fixed_recursive_lib.py

Compare the output with fixed_recursive_scratch.py. Expect small differences:
LangChain splits on the separator first, then merges pieces, and strips
whitespace from chunk edges, so chunk lengths vary.
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

TEXT = (
    "Artificial Intelligence is transforming industries. "
    "It has applications in healthcare, finance, and education.\n\n"
    "Machine learning models learn patterns from data.\n"
    "Deep learning uses neural networks with many layers."
)

fixed = CharacterTextSplitter(separator="", chunk_size=50, chunk_overlap=10)
recursive = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], chunk_size=60, chunk_overlap=10
)

for name, splitter in [("Fixed-size", fixed), ("Recursive", recursive)]:
    print(f"\n{name}")
    for i, chunk in enumerate(splitter.split_text(TEXT), start=1):
        print(f"  chunk {i} ({len(chunk):>2} chars): {chunk!r}")
