
from __future__ import annotations
from typing import List
import numpy as np

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v / n

class Embeddings:
    def __init__(self, backend: str = "ollama:mxbai-embed-large", base_url: str | None = None):
        self.backend = backend
        self.base_url = base_url
        self._model = None
        self._client = None

    def _ensure_model(self):
        if self.backend.startswith("ollama:"):
            if self._client is None:
                import ollama, os
                if self.base_url:
                    os.environ["OLLAMA_HOST"] = self.base_url
                self._client = ollama
        else:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.backend)

    def embed(self, texts: List[str]):
        self._ensure_model()
        if self.backend.startswith("ollama:"):
            model = self.backend.split(":",1)[1]
            vecs = []
            for t in texts:
                out = self._client.embeddings(model=model, prompt=t)
                vecs.append(out["embedding"])
            arr = np.array(vecs, dtype=np.float32)
        else:
            arr = np.array(self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False), dtype=np.float32)
        return _normalize(arr)
