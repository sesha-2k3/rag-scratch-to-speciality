# 01 · Chunking

Code: [`fixed_recursive_scratch.py`](../code/chunking/fixed_recursive_scratch.py) · [`fixed_recursive_lib.py`](../code/chunking/fixed_recursive_lib.py)

## Why chunk at all

An embedding is a fixed-length vector (e.g. 384 or 1,536 numbers) no matter how much text goes in. Embedding a 100-page document as one vector averages every topic in it, so it is loosely similar to many queries and strongly similar to none.

Three more reasons to chunk:

- **Model limits.** Embedding models silently truncate input past their token limit, so most of a long document would never be embedded.
- **LLM cost and context.** Sending whole documents for every question is slow, expensive, and can overflow the context window.
- **Precision.** A question is usually answered by a few sentences. Burying them in 100 pages makes the LLM more likely to miss them, especially in the middle of a long context.

Chunking turns one blurry vector into many sharp ones, each about one topic.

## The five strategies

| Strategy | How it splits | Strength | Weakness |
|---|---|---|---|
| Fixed-size | Every N characters/tokens | Simple, predictable | Cuts mid-sentence, ignores meaning |
| Recursive | Tries `\n\n`, then `\n`, then `" "`, then `""` | Respects natural boundaries | Still size-driven, not meaning-driven |
| Document-based | By structure: headings, Markdown sections, code blocks, tables | Matches how the author organized content | Needs a layout-aware parser |
| Semantic | Embeds sentences, cuts where adjacent sentences' distance spikes | Splits at topic shifts | Extra embedding cost; only cuts contiguous text |
| Agentic | LLM extracts standalone propositions, then groups them | Can group related ideas from anywhere; resolves "it"/"this" | Many LLM calls; slow, costly, non-deterministic |

### Recursive splitting, step by step

With `chunk_size=100` and a 250-character paragraph:

1. Split on `\n\n`. Pieces that fit become chunks.
2. The oversized paragraph is split on `\n`.
3. Anything still too long is split on `" "`, so cuts land between words.
4. Only as a last resort, split on `""` (individual characters), e.g. for a 300-character URL.

Afterwards, small adjacent pieces are **merged** back together up to `chunk_size`, so three 30-character lines become one ~90-character chunk.

### Semantic chunking threshold

`SemanticSplitterNodeParser` (LlamaIndex) computes the distance between each pair of adjacent sentences and cuts wherever the distance exceeds the Xth percentile of all distances in that document.

- `breakpoint_percentile_threshold=95` (default): only the top 5% of distances become cuts → few, large chunks.
- Lower values (e.g. 70): more distances exceed the cutoff → **more, smaller chunks**.
- The threshold is relative to each document, so even a single-topic document gets split at its least-similar points.
- `buffer_size` embeds each sentence together with its neighbors, smoothing out noise.

## Overlap

`chunk_overlap` repeats the last N characters of one chunk at the start of the next, so a fact cut at a boundary still appears whole somewhere.

Without overlap: `"...employees are entitled to 12 days of"` | `"paid sick leave per calendar year..."`. Neither chunk answers "How many sick days do I get?" With overlap, the second chunk starts `"to 12 days of paid sick leave..."`.

Cost: redundant storage and embedding, and the same sentence may be retrieved twice. Rule of thumb: 10–20% of chunk size.

## Chunk size: the two failure modes

- **Too small → loss of context.** "It was increased to 20% in 2023." Increased what? The embedding can't capture meaning the text doesn't contain.
- **Too large → contextual bleeding.** A chunk covering tax rates, hiring, and office locations gets an averaged vector that matches every query weakly, and sends the LLM irrelevant text.

Typical starting point: 200–500 tokens with some overlap. Tune it against real queries using the evaluation methods in [04-evaluation.md](04-evaluation.md).

## Special content

- **Tables:** embed an LLM-written summary (natural language matches user questions), but store the raw table alongside it and pass the raw table to the LLM. Find with the summary, answer with the data. This is the **multi-vector retriever** pattern.
- **Images:** same pattern, using an image summary.

## Lesson: always print your chunks

The blog's fixed-size example claims a 50-character split produced `"Artificial Intelligence is transforming industr"`. That is 47 characters. A real 50-character cut gives `"Artificial Intelligence is transforming industries"`. Tutorial outputs are illustrations, so run splitters on your own text:

```python
for i, c in enumerate(chunks):
    print(i, len(c), repr(c))
```

## Worked design: HR policy chatbot

PDFs with headings, long paragraphs, and leave-entitlement tables.

1. **Extract** with a layout-aware parser (e.g. `unstructured`'s `partition_pdf`) so headings, paragraphs, and tables come out as typed elements.
2. **Split by heading first** (document-based); keep the heading path.
3. **Recursively split long sections** to ~300–400 tokens, ~50 tokens overlap, using a token-based splitter.
4. **Tables:** keep whole, embed a summary, return the raw table.
5. **Prepend context** before embedding: `"Employee Handbook 2025 > Leave Policies > Parental Leave: ..."`.

Metadata: `doc_name`, `page`, `section_path`, `content_type`, `effective_date`, `version`, `region`/`applies_to`, `embedding_model`. Retrieval: hybrid search + reranker, filter on current policies, cite sources, and say "I couldn't find this" rather than guess.
