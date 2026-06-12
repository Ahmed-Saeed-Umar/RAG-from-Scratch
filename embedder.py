from sentence_transformers import SentenceTransformer
import numpy as np

# Load once — this downloads ~90MB on first run
_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into a 2D array of vectors.
    Shape: (len(texts), 384)
    """
    return _model.encode(texts, convert_to_numpy=True)