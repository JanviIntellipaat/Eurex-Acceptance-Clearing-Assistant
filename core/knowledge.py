
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

from .embeddings import Embeddings
from .vector_hnsw import HNSWVectorStore, Chunk
from .structured import StructuredStore

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

@dataclass
class IndexTextResult:
    id: str
    source: str
    page: Optional[int]
    chars: int
    preview: str

class KnowledgeBase:
    def __init__(self, settings):
        self.dir = Path(settings.kb_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.embed = Embeddings(backend=settings.embed_backend, base_url=settings.base_url)
        self.vs = HNSWVectorStore(self.dir / "index", Path(settings.duckdb_path), dim=settings.embed_dim)
        self.structured = StructuredStore(settings.duckdb_path)

    def _chunk_text(self, text: str, size: int = 300) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

    def add_files(self, files, parse_tables: bool = True) -> Dict[str, Any]:
        text_results: List[IndexTextResult] = []
        tables_added = 0
        for f in files:
            name = getattr(f, "name", "upload")
            content = f.read()
            path = self.dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as out:
                out.write(content)

            if name.lower().endswith(".pdf") and pdfplumber:
                with pdfplumber.open(path) as pdf:
                    if parse_tables:
                        tables_added += self.structured.add_pdf_tables(name, pdf).tables_added
                    for i, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        for idx, chunk in enumerate(self._chunk_text(text)):
                            uid = hashlib.sha1(f"{name}-{i}-{idx}-{len(chunk)}".encode()).hexdigest()
                            text_results.append(IndexTextResult(id=uid, source=name, page=i, chars=len(chunk), preview=chunk[:160]))
            elif name.lower().endswith(".docx") and docx:
                document = docx.Document(str(path))
                try:
                    res = self.structured.add_docx_tables(name, document)
                    tables_added += res.tables_added
                except Exception:
                    pass
                buf = [p.text for p in document.paragraphs if p.text and p.text.strip()]
                text_results.append(IndexTextResult(id=hashlib.sha1(f"{name}-full".encode()).hexdigest(), source=name, page=1, chars=len(" ".join(buf)), preview=" ".join(buf)[:160]))
            elif name.lower().endswith(".csv"):
                tables_added += self.structured.add_csv(name, content).tables_added
                try:
                    import pandas as pd, io
                    df = pd.read_csv(io.BytesIO(content))
                    text = df.head(50).to_csv(index=False)
                    for idx, chunk in enumerate(self._chunk_text(text, size=200)):
                        uid = hashlib.sha1(f"{name}-csv-{idx}-{len(chunk)}".encode()).hexdigest()
                        text_results.append(IndexTextResult(id=uid, source=name, page=1, chars=len(chunk), preview=chunk[:160]))
                except Exception:
                    pass
            else:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = ""
                for idx, chunk in enumerate(self._chunk_text(text)):
                    uid = hashlib.sha1(f"{name}-raw-{idx}-{len(chunk)}".encode()).hexdigest()
                    text_results.append(IndexTextResult(id=uid, source=name, page=1, chars=len(chunk), preview=chunk[:160]))

        if text_results:
            docs = [t.preview for t in text_results]
            vecs = self.embed.embed(docs)
            ids = [t.id for t in text_results]
            metadatas = [{"source":t.source, "page":t.page, "text":t.preview} for t in text_results]
            self.vs.add(ids, vecs, metadatas)

        return {"text_chunks":[t.__dict__ for t in text_results], "tables_added": tables_added}

    def search_text(self, query: str, k: int = 5) -> List[Chunk]:
        if not query.strip(): return []
        vec = self.embed.embed([query])
        return self.vs.query(vec, k=k)

    def search_structured_context(self, query: str, top_k: int = 5) -> str:
        return self.structured.search_context(query, top_k=top_k)

    def purge_index(self):
        self.vs.purge()

    def purge_structured(self):
        self.structured.purge()
