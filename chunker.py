def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by word count.
    
    overlap ensures a sentence split across a boundary
    still appears whole in at least one chunk.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # move forward by (chunk_size - overlap) so chunks overlap
        start += chunk_size - overlap

    return chunks