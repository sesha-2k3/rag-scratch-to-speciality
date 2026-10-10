"""Why chunk sizes should be measured in tokens, and what truncation does.

Concepts: notes/02-embeddings-metadata.md
Run:      python code/chunking/token_aware_scratch.py

Real models use sub-word tokenizers (see token_aware_lib.py). This toy
tokenizer splits words and punctuation, which is enough to show the effect.
"""
import re

MAX_TOKENS = 32  # stand-in for a model limit such as 256 for all-MiniLM-L6-v2


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)


def truncate(text, max_tokens=MAX_TOKENS):
    """What an embedding model does silently: keep the first max_tokens."""
    return tokenize(text)[:max_tokens]


def token_chunks(text, chunk_tokens, overlap_tokens):
    """Chunk by token count instead of character count."""
    tokens = tokenize(text)
    step = chunk_tokens - overlap_tokens
    return [" ".join(tokens[i:i + chunk_tokens])
            for i in range(0, max(len(tokens) - overlap_tokens, 1), step)]


if __name__ == "__main__":
    pasted_email = (
        "Hi team, thanks for the update on Q3 budgets. The finance review "
        "went well and spending is mostly on track, although travel costs "
        "rose slightly. We also discussed hiring plans for next year. "
    )
    question = "Based on this, will the Atlas launch date slip?"
    user_message = pasted_email + question

    kept = truncate(user_message)
    print(f"User message: {len(tokenize(user_message))} tokens, limit {MAX_TOKENS}")
    print("What actually gets embedded:")
    print("  " + " ".join(kept))
    print(f"Question survived truncation? {'Atlas' in kept}")

    print("\nToken-based chunks (12 tokens, 3 overlap):")
    for i, chunk in enumerate(token_chunks(pasted_email, 12, 3), start=1):
        print(f"  {i}: {chunk}")
