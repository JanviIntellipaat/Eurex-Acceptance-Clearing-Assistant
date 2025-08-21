from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import duckdb
import faiss

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: Optional[int] = None
    score: float = 0.0   # cosine similarity (because embeddings are normalized)

class FaissVectorStore:
    """
    FAISS-based vector store (cosine similarity via inner product).
    - Index persisted to faiss_index.bin
    - ID mapping persisted to faiss_ids.json (since FAISS IDs are int64)
    - Metadata (source/page/text) kept in DuckDB table kb_chunks
    """
    def __init__(self, dirpath: Path, duckdb_path: Path, dim: int = 1024):
        self.dir = Path(dirpath); self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_db = duckdb.connect(str(duckdb_path))
        self.dim = dim
        self.index_path = self.dir / "faiss_index.bin"
        self.map_path = self.dir / "faiss_ids.json"
        self.meta_db.execute("CREATE TABLE IF NOT EXISTS kb_chunks (id VARCHAR, source VARCHAR, page INTEGER, text VARCHAR)")
        self.meta_db.commit()

        self._idmap: dict[str, int] = {}   # string id -> int id
        self._revmap: dict[int, str] = {}  # int id -> string id
        self._next = 1
        self._index = None
        self._ensure_index()

    def _ensure_index(self):
        base = faiss.IndexFlatIP(self.dim)  # inner product (cosine if vectors normalized)
        self._index = faiss.IndexIDMap2(base)
        if self.index_path.exists() and self.map_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
            except Exception:
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
            try:
                data = json.loads(self.map_path.read_text())
                self._idmap = data.get("str2int", {})
                self._revmap = {int(k): v for k, v in data.get("int2str", {}).items()}
                self._next = int(data.get("next", 1))
            except Exception:
                self._idmap = {}; self._revmap = {}; self._next = 1

    def _save(self):
        faiss.write_index(self._index, str(self.index_path))
        self.map_path.write_text(json.dumps({
            "str2int": self._idmap,
            "int2str": {str(k): v for k, v in self._revmap.items()},
            "next": self._next
        }))

    def _get_or_assign_int_ids(self, ids: List[str]) -> np.ndarray:
        int_ids = []
        for sid in ids:
            if sid not in self._idmap:
                iid = self._next
                self._next += 1
                self._idmap[sid] = iid
                self._revmap[iid] = sid
            int_ids.append(self._idmap[sid])
        return np.array(int_ids, dtype="int64")

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict]):
        # store metadata
        for i, md in enumerate(metadatas):
            self.meta_db.execute(
                "INSERT INTO kb_chunks VALUES (?, ?, ?, ?)",
                (ids[i], md.get("source",""), md.get("page"), md.get("text",""))
            )
        self.meta_db.commit()

        int_ids = self._get_or_assign_int_ids(ids)
        vecs = np.ascontiguousarray(embeddings.astype("float32"))
        self._index.add_with_ids(vecs, int_ids)
        self._save()

    def query(self, qvec: np.ndarray, k: int = 5) -> List[Chunk]:
        vec = np.ascontiguousarray(qvec.astype("float32"))
        sims, id_arr = self._index.search(vec, k)
        sims = sims[0]; labels = id_arr[0]
        res: List[Chunk] = []
        for score, iid in zip(sims, labels):
            if iid == -1:  # no result slot
                continue
            sid = self._revmap.get(int(iid))
            if not sid:
                continue
            row = self.meta_db.execute("SELECT source, page, text FROM kb_chunks WHERE id=?", [sid]).fetchone()
            if row:
                res.append(Chunk(id=sid, text=row[2], source=row[0], page=row[1], score=float(score)))
        return res

    def purge(self):
        if self.index_path.exists(): self.index_path.unlink()
        if self.map_path.exists(): self.map_path.unlink()
        self.meta_db.execute("DELETE FROM kb_chunks")
        self.meta_db.commit()
        self._ensure_index()
