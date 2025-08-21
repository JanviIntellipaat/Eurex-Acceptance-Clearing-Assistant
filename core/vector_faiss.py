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
    score: float = 0.0  # cosine similarity (we normalize embeddings)

class FaissVectorStore:
    def __init__(self, index_dir: Path, duckdb_path: Path, dim: int):
        self.dir = Path(index_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "faiss_index.bin"
        self.map_path = self.dir / "faiss_ids.json"
        self.dim = dim

        self.meta_db = duckdb.connect(str(duckdb_path))
        self.meta_db.execute("""
            CREATE TABLE IF NOT EXISTS kb_chunks(
                id TEXT PRIMARY KEY,
                source TEXT,
                page INTEGER,
                text TEXT
            )
        """)
        self.meta_db.commit()

        self._index = None
        self._idmap = {}
        self._revmap = {}
        self._ensure_index()

    def _ensure_index(self):
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
        else:
            self._index = faiss.IndexFlatIP(self.dim)
        if self.map_path.exists():
            self._idmap = json.loads(self.map_path.read_text())
            self._revmap = {int(v): k for k, v in self._idmap.items()}

    def _save(self):
        faiss.write_index(self._index, str(self.index_path))
        self.map_path.write_text(json.dumps(self._idmap))

    def _get_or_assign_int_ids(self, ids: List[str]) -> np.ndarray:
        next_id = max(self._idmap.values(), default=-1)
        got = []
        for sid in ids:
            if sid in self._idmap:
                got.append(self._idmap[sid])
            else:
                next_id += 1
                self._idmap[sid] = next_id
                self._revmap[next_id] = sid
                got.append(next_id)
        return np.array(got, dtype="int64")

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict]):
        # store metadata
        for i, md in enumerate(metadatas):
            self.meta_db.execute(
                "INSERT INTO kb_chunks(id, source, page, text) VALUES (?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
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
            if iid == -1:
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
