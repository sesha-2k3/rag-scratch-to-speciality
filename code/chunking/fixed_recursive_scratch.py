"""Fixed-size and recursive chunking, implemented from scratch.

Concepts: notes/01-chunking.md
Run:      python code/chunking/fixed_recursive_scratch.py
"""

TEXT = (
    "Artificial Intelligence is transforming industries. "
    "It has applications in healthcare, finance, and education.\n\n"
    "Machine learning models learn patterns from data.\n"
    "Deep learning uses neural networks with many layers."
)


def fixed_size_chunks(text, chunk_size, overlap):
    """Cut every `chunk_size` characters, repeating `overlap` characters."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def recursive_chunks(text, chunk_size, separators=("\n\n", "\n", " ", "")):
    """Split on the most natural separator first, then merge small pieces.

    Simplified version of LangChain's RecursiveCharacterTextSplitter
    (no overlap, to keep the logic easy to follow).
    """
    pieces = _split(text, chunk_size, list(separators))
    return _merge(pieces, chunk_size)


def _split(text, chunk_size, separators):
    if len(text) <= chunk_size:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep == "":  # last resort: hard cut by characters
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    parts = text.split(sep)
    out = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")  # keep the separator
        if len(piece) <= chunk_size:
            out.append(piece)
        else:
            out.extend(_split(piece, chunk_size, rest))  # recurse with next separator
    return out


def _merge(pieces, chunk_size):
    """Pack adjacent small pieces together until they approach chunk_size."""
    chunks, current = [], ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def show(title, chunks):
    print(f"\n{title}")
    for i, c in enumerate(chunks, start=1):
        print(f"  chunk {i} ({len(c):>2} chars): {c!r}")


if __name__ == "__main__":
    show("Fixed-size: chunk_size=50, overlap=10", fixed_size_chunks(TEXT, 50, 10))
    show("Recursive: chunk_size=60", recursive_chunks(TEXT, 60))

    # Lesson from the blog review: always count your chunks.
    blog_chunk_1 = "Artificial Intelligence is transforming industr"
    print(f"\nThe blog's 'Chunk 1' is {len(blog_chunk_1)} chars, not 50.")
    print(f"A real 50-char cut gives: {TEXT[:50]!r}")
