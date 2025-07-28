# Approach Explanation – Round 1B

This solution identifies and prioritizes relevant sections from a document collection based on a persona's job-to-be-done using sentence embeddings.

## Methodology
1. Each PDF is parsed page-wise using PyMuPDF.
2. Text is chunked into paragraphs and encoded using a Sentence Transformer (MiniLM).
3. A semantic similarity score is computed between chunks and the combined persona+task query.
4. Top-ranked chunks are selected and organized in a structured JSON output.

## Tools and Models
- `PyMuPDF` for PDF parsing.
- `sentence-transformers` (`all-MiniLM-L6-v2`) for embedding.
- `cosine similarity` for ranking text relevance.

## Offline & Lightweight
- Entire model is <100MB.
- No internet calls.
- Executes within 60s limit on standard CPU.

## Output Format
JSON includes:
- Input metadata
- Ranked document sections
- Refined subsection analysis
