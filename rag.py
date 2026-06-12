import os
import anthropic
from pdf_loader import load_pdf
from chunker import chunk_text
from embedder import embed
from vector_store import VectorStore
from groq import Groq


def build_index(pdf_path: str) -> VectorStore:
    """Run once: load PDF, chunk it, embed chunks, store."""
    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"  → {len(chunks)} chunks created")

    print("Embedding chunks...")
    embeddings = embed(chunks)
    print(f"  → embeddings shape: {embeddings.shape}")

    store = VectorStore()
    store.add(chunks, embeddings)
    return store


def answer_question(question: str, store: VectorStore) -> str:
    """Embed the question, retrieve relevant chunks, ask the LLM."""

    # 1. Embed the question using the SAME model used for indexing
    query_vec = embed([question])[0]  # shape: (384,)

    # 2. Find the top 3 most relevant chunks
    relevant_chunks = store.search(query_vec, top_k=3)

    # 3. Build the prompt — this is where RAG "happens"
    context = "\n\n---\n\n".join(relevant_chunks)
    prompt = f"""Answer the question using ONLY the context below.
    If the answer isn't in the context, say "I don't know based on the document."

    
    Context:    
    {context}

    Question: {question}
    Answer:"""

    # 4. Send to LLM
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    message = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.choices[0].message.content

if __name__ == "__main__":
    # --- INDEXING (run once) ---
    store = build_index("lecture6.pdf")

    # --- QUERYING (run as many times as you want) ---
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        answer = answer_question(q, store)
        print(f"\n{answer}")