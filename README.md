# Simple RAG Pipeline from Scratch

A minimal, dependency-light Retrieval-Augmented Generation (RAG) pipeline that answers questions from a PDF document. Built from first principles — no LangChain, no vector database, no hidden abstractions — so every step of the pipeline is visible and understandable.

## How it works

RAG solves a simple problem: LLMs can't read your documents directly because of context window limits. The fix is to search the document first, then hand the LLM only the relevant pieces alongside your question.

The pipeline has two phases:

**Indexing (runs once)**
1. Extract raw text from a PDF
2. Split the text into overlapping chunks (~500 words, 50-word overlap)
3. Convert each chunk into a 384-dimensional vector using a sentence embedding model
4. Store the chunks and their vectors in memory

**Querying (runs per question)**
1. Embed the question using the same embedding model
2. Compute cosine similarity between the question vector and every stored chunk vector
3. Retrieve the top-k most similar chunks
4. Build a prompt containing those chunks as context, plus the question
5. Send the prompt to an LLM, which answers using only the provided context

## Project structure

```
rag_project/
├── pdf_loader.py      # extracts raw text from a PDF
├── chunker.py         # splits text into overlapping passages
├── embedder.py        # converts text into vectors (sentence-transformers)
├── vector_store.py    # stores vectors, performs cosine similarity search
├── rag.py             # orchestrates the full pipeline
└── your_document.pdf  # the PDF you want to query
```

## Setup

It's strongly recommended to use a virtual environment, since this project depends on PyTorch via `sentence-transformers`.

```bash
python -m venv rag-env

# Windows
rag-env\Scripts\activate

# Mac/Linux
source rag-env/bin/activate
```

Install dependencies:

```bash
pip install pdfplumber sentence-transformers numpy groq
```

> **Windows note:** if you hit a `c10.dll` / PyTorch DLL error, reinstall the CPU-only PyTorch build:
> ```bash
> pip uninstall torch torchvision torchaudio -y
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```
> Also make sure you're on Python 3.11, not 3.13+ — PyTorch wheels often lag behind the latest Python release.

## API key

This project uses [Groq](https://console.groq.com) for the LLM (free tier available).

Set your API key as an environment variable before running:

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_your-key-here"

# Mac/Linux
export GROQ_API_KEY="gsk_your-key-here"
```

## Usage

1. Drop your PDF into the project folder
2. Update the filename in `rag.py`:
   ```python
   store = build_index("your_document.pdf")
   ```
3. Run:
   ```bash
   python rag.py
   ```

The first run will download the embedding model (`all-MiniLM-L6-v2`, ~90MB). After indexing completes, ask questions interactively. Type `quit` to exit.

```
Ask a question (or 'quit'): What is this document about?
```

## What this project does NOT do

This is intentionally minimal, for learning purposes. It does not:

- Persist the index to disk (re-embeds on every run)
- Handle multiple documents
- Tune chunk size or retrieval count automatically
- Handle scanned/image-only PDFs (no OCR)

## Key things to understand

- **Embeddings** turn text into points in 384-dimensional space. Similar meanings end up near each other geometrically.
- **Cosine similarity** measures the angle between two vectors — this is the entire "search" mechanism.
- **The LLM never sees the whole document** — only the top-k retrieved chunks. If retrieval picks the wrong chunks, the answer will be wrong or incomplete, even though the LLM itself is working correctly.
- **Chunk size and overlap** are tuning knobs with real tradeoffs: too large and irrelevant text dilutes the context; too small and you lose surrounding meaning.

## License

MIT — use freely for learning and experimentation.
