from __future__ import annotations
from typing import List
import numpy as np

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

class Embeddings:
    """
    Uses Ollama for embeddings when backend starts with 'ollama:' (e.g., 'ollama:mxbai-embed-large').
    Fallback: sentence-transformers (optional) if you set a model name like 'bge-large-en-v1.5'.
    """
    def __init__(self, backend: str = "ollama:mxbai-embed-large", base_url: str | None = None):
        self.backend = backend
        self.base_url = base_url
        self._client = None
        self._st_model = None

    def _ensure(self):
        if self.backend.startswith("ollama:"):
            import ollama  # requires `pip install ollama`
            if self.base_url:
                # allow custom base URL if you use a proxy
                os.environ["OLLAMA_BASE_URL"] = self.base_url
            self._client = ollama
        else:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.backend)

    def embed(self, texts: List[str]) -> np.ndarray:
        self._ensure()
        if self.backend.startswith("ollama:"):
            model = self.backend.split(":", 1)[1]
            vecs = []
            for t in texts:
                out = self._client.embeddings(model=model, prompt=t)
                vecs.append(out["embedding"])
            arr = np.array(vecs, dtype=np.float32)
            return _normalize(arr)
        else:
            arr = self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False).astype("float32")
            return _normalize(arr)
