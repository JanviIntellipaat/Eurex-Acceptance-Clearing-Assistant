
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json, numpy as np, hnswlib, duckdb

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: Optional[int] = None
    score: float = 0.0

class HNSWVectorStore:
    def __init__(self, dirpath: Path, duckdb_path: Path, dim: int = 1024):
        self.dir = Path(dirpath); self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_db = duckdb.connect(str(duckdb_path))
        self.dim = dim
        self.index_path = self.dir / "hnsw_index.bin"
        self.meta_db.execute("CREATE TABLE IF NOT EXISTS kb_chunks (id VARCHAR, source VARCHAR, page INTEGER, text VARCHAR)")
        self.meta_db.commit()
        self._index = None
        self._ensure_index()

    def _ensure_index(self):
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        if self.index_path.exists() and (self.dir / "hnsw_meta.json").exists():
            with open(self.dir / "hnsw_meta.json","r") as f:
                meta = json.load(f)
            self._index.load_index(str(self.index_path), max_elements=meta["max_elements"])
            self._index.set_ef(128)
        else:
            self._index.init_index(max_elements=1, ef_construction=200, M=48)
            self._index.set_ef(128)

    def _resize_if_needed(self, new_total: int):
        try:
            self._index.resize_index(new_total)
        except Exception:
            pass

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict]):
        for i, md in enumerate(metadatas):
            self.meta_db.execute("INSERT INTO kb_chunks VALUES (?, ?, ?, ?)", (ids[i], md.get("source",""), md.get("page"), md.get("text","") ))
        self.meta_db.commit()

        if self.index_path.exists():
            count = self._index.get_current_count()
            self._resize_if_needed(count + len(ids))
            self._index.add_items(embeddings, ids)
        else:
            self._index.add_items(embeddings, ids)

        with open(self.dir / "hnsw_meta.json","w") as f:
            json.dump({"max_elements": int(self._index.get_max_elements())}, f)
        self._index.save_index(str(self.index_path))

    def query(self, qvec, k: int = 5) -> List[Chunk]:
        labels, dists = self._index.knn_query(qvec, k=k)
        labels = labels[0]; dists = dists[0]
        res = []
        for i, lab in enumerate(labels):
            cur = self.meta_db.execute("SELECT source, page, text FROM kb_chunks WHERE id=?", [str(lab)]).fetchone()
            if cur:
                res.append(Chunk(id=str(lab), text=cur[2], source=cur[0], page=cur[1], score=float(1.0 - dists[i])))
        return res

    def purge(self):
        if self.index_path.exists():
            self.index_path.unlink()
        (self.dir / "hnsw_meta.json").unlink(missing_ok=True)
        self.meta_db.execute("DELETE FROM kb_chunks")
        self.meta_db.commit()
        self._ensure_index()
