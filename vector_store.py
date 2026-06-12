import numpy as np

class VectorStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: list[str], embeddings: np.ndarray):
        """Store chunks and their corresponding vectors."""
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[str]:
        """
        Find top_k most similar chunks to the query.
        
        Cosine similarity = dot(A, B) / (|A| * |B|)
        = 1.0 if identical direction, 0 if perpendicular, -1 if opposite
        """
        # Normalize all stored vectors (divide each by its magnitude)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normalized_store = self.embeddings / norms

        # Normalize the query vector
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Dot product of normalized vectors = cosine similarity
        similarities = normalized_store @ query_norm  # shape: (num_chunks,)

        # Get indices of top_k highest scores
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices]