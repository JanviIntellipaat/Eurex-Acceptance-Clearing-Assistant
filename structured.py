
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import duckdb, pandas as pd
import io

@dataclass
class StructuredIngestResult:
    tables_added: int
    table_names: List[str]

class StructuredStore:
    """
    Stores structured data in DuckDB:
    - CSV files are imported as tables.
    - PDF tables (via pdfplumber) are ingested as tables per page.
    - DOCX tables are ingested when available.
    Type inference ONLY (no constraints).
    """
    def __init__(self, duckdb_path: str):
        self.db = duckdb.connect(duckdb_path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS kb_tables (
            table_name VARCHAR, source VARCHAR, page INTEGER, created_at TIMESTAMP DEFAULT now()
        )""")
        self.db.commit()

    def _register(self, name: str, source: str, page: Optional[int]):
        self.db.execute("INSERT INTO kb_tables(table_name, source, page) VALUES (?, ?, ?)", (name, source, page))
        self.db.commit()

    def _infer_duckdb_type(self, s: pd.Series) -> str:
        nonnull = s.dropna()
        if nonnull.empty:
            return "VARCHAR"
        low = nonnull.astype(str).str.strip().str.lower()
        if (low.isin(["true","false","0","1","yes","no"]).mean() > 0.9):
            return "BOOLEAN"
        try:
            ints = pd.to_numeric(nonnull, errors="raise", downcast="integer")
            if (ints % 1 == 0).all():
                return "BIGINT"
        except Exception:
            pass
        try:
            pd.to_numeric(nonnull, errors="raise")
            return "DOUBLE"
        except Exception:
            pass
        try:
            sample = nonnull.astype(str).str.strip()
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.7:
                return "DATE"
        except Exception:
            pass
        return "VARCHAR"

    def _create_table_with_types(self, table_name: str, df: pd.DataFrame):
        cols = []
        for col in df.columns:
            duck_type = self._infer_duckdb_type(df[col])
            cols.append(f'"{col}" {duck_type}')
        ddl = f"CREATE OR REPLACE TABLE {table_name} ({', '.join(cols)});"
        self.db.execute(ddl)
        self.db.register("df_tmp", df)
        self.db.execute(f'INSERT INTO {table_name} SELECT * FROM df_tmp')
        self.db.unregister("df_tmp")
        self.db.commit()

    def add_csv(self, name: str, content: bytes) -> StructuredIngestResult:
        df = pd.read_csv(io.BytesIO(content))
        tname = self._normalize_name(name.replace(".csv",""))
        self._create_table_with_types(tname, df)
        self._register(tname, name, None)
        return StructuredIngestResult(tables_added=1, table_names=[tname])

    def add_docx_tables(self, name: str, doc) -> StructuredIngestResult:
        tables = getattr(doc, "tables", [])
        count = 0; names = []
        for idx, tbl in enumerate(tables, start=1):
            rows = []
            for r in tbl.rows:
                rows.append([c.text for c in r.cells])
            if not rows: continue
            df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None)
            tname = self._normalize_name(f"{name.replace('.docx','')}_table_{idx}")
            self._create_table_with_types(tname, df)
            self._register(tname, name, None)
            count += 1; names.append(tname)
        return StructuredIngestResult(tables_added=count, table_names=names)

    def add_pdf_tables(self, name: str, pdf) -> StructuredIngestResult:
        count = 0; names = []
        for pidx, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for tidx, tbl in enumerate(tables, start=1):
                if not tbl: continue
                header = tbl[0]
                body = tbl[1:] if len(tbl) > 1 else []
                has_header = all(h is not None and str(h).strip() != "" for h in header)
                if has_header:
                    df = pd.DataFrame(body, columns=header)
                else:
                    df = pd.DataFrame(tbl)
                tname = self._normalize_name(f"{name.replace('.pdf','')}_p{pidx}_t{tidx}")
                self._create_table_with_types(tname, df)
                self._register(tname, name, pidx)
                count += 1; names.append(tname)
        return StructuredIngestResult(tables_added=count, table_names=names)

    def search_context(self, query: str, top_k: int = 5) -> str:
        q = query.lower().split()
        tables = self.db.execute("SELECT table_name, source, coalesce(cast(page as int), -1) FROM kb_tables").fetchall()
        snippets = []
        for tname, source, page in tables:
            cols = [c[0] for c in self.db.execute(f"PRAGMA table_info('{tname}')").fetchall()]
            text = f"{tname} (cols: {', '.join(cols[:10])}) from {source}{f' p{page}' if page and page>0 else ''}"
            score = sum(1 for tok in q if tok in tname.lower() or any(tok in c.lower() for c in cols))
            if score > 0:
                sample = self.db.execute(f"SELECT * FROM {tname} LIMIT 3").fetchdf()
                snippets.append(text + "\n" + sample.to_markdown(index=False))
        return "\n\n".join(snippets[:top_k])

    def purge(self):
        tbls = [r[0] for r in self.db.execute("SELECT table_name FROM kb_tables").fetchall()]
        for t in tbls:
            self.db.execute(f"DROP TABLE IF EXISTS {t}")
        self.db.execute("DELETE FROM kb_tables")
        self.db.commit()

    def _normalize_name(self, s: str) -> str:
        return "".join(ch if ch.isalnum() or ch=='_' else '_' for ch in s.lower())
