import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from .config import MAX_TEXT_CHARS

def clip_text(t: str, n=MAX_TEXT_CHARS) -> str:
    t = (t or "").replace("\n", " ").strip()
    if n is None:
        return t
    return t if len(t) <= n else (t[:n] + "...")

class FaissRetriever:
    """
    Retrieval-only (Q1): embed query -> faiss.search -> return top-k docID/text.
    """
    def __init__(self, index_path, st_model_path):
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(str(st_model_path))

        # Many setups use cosine/IP with normalized vectors.
        # Keep this consistent with how the provided index expects vectors.
        self.normalize = True

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode([text], convert_to_numpy=True).astype(np.float32)
        if self.normalize:
            faiss.normalize_L2(v)
        return v

    def search(self, query_text: str, k: int):
        qv = self.embed(query_text)
        scores, idxs = self.index.search(qv, k)
        return scores[0].tolist(), idxs[0].tolist()
