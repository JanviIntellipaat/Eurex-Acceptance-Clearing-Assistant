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
    Stores structured data in DuckDB (TEXT-only columns):
    - CSV files → tables
    - XLSX/XLS files → each sheet as a table
    - PDF tables (pdfplumber) → per-page tables
    - DOCX tables → tables

    IMPORTANT: All columns are stored as VARCHAR (TEXT). No type inference, no constraints.
    """

    def __init__(self, duckdb_path: str):
        self.db = duckdb.connect(duckdb_path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS kb_tables (
                table_name VARCHAR,
                source VARCHAR,
                page INTEGER,
                created_at TIMESTAMP DEFAULT now()
            )
        """)
        self.db.commit()

    def _register(self, name: str, source: str, page: Optional[int]):
        self.db.execute(
            "INSERT INTO kb_tables(table_name, source, page) VALUES (?, ?, ?)",
            (name, source, page),
        )
        self.db.commit()

    def _create_table_all_text(self, table_name: str, df: pd.DataFrame):
        # Handle empty inputs
        if df is None or df.shape[0] == 0:
            if df is None or df.shape[1] == 0:
                ddl = f'CREATE OR REPLACE TABLE {table_name} ("_empty" VARCHAR);'
                self.db.execute(ddl)
                self.db.commit()
                return
        # Column names as strings
        df.columns = [str(c) if c is not None else "_col" for c in df.columns]
        # Coerce all values to string (keep empty strings for NaNs)
        df = df.astype(str)

        # Build DDL with VARCHAR for all columns
        cols = [f'"{c}" VARCHAR' for c in df.columns]
        ddl = f"CREATE OR REPLACE TABLE {table_name} ({', '.join(cols)});"
        self.db.execute(ddl)

        # Insert the data
        self.db.register("df_tmp", df)
        self.db.execute(f"INSERT INTO {table_name} SELECT * FROM df_tmp")
        self.db.unregister("df_tmp")
        self.db.commit()

    # -------- Ingestors --------
    def add_csv(self, name: str, content: bytes) -> StructuredIngestResult:
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, low_memory=False)
        tname = self._normalize_name(name.replace(".csv",""))
        self._create_table_all_text(tname, df)
        self._register(tname, name, None)
        return StructuredIngestResult(tables_added=1, table_names=[tname])

    def add_xlsx(self, name: str, content: bytes) -> StructuredIngestResult:
        sheets = pd.read_excel(io.BytesIO(content), dtype=str, sheet_name=None, engine="openpyxl")
        count = 0; names = []
        base = name.replace(".xlsx","")
        for sheet_name, df in sheets.items():
            tname = self._normalize_name(f"{base}_{sheet_name}")
            self._create_table_all_text(tname, df)
            self._register(tname, name, None)
            count += 1; names.append(tname)
        return StructuredIngestResult(tables_added=count, table_names=names)

    def add_xls(self, name: str, content: bytes) -> StructuredIngestResult:
        # Requires: pip install xlrd
        sheets = pd.read_excel(io.BytesIO(content), dtype=str, sheet_name=None, engine="xlrd")
        count = 0; names = []
        base = name.replace(".xls","")
        for sheet_name, df in sheets.items():
            tname = self._normalize_name(f"{base}_{sheet_name}")
            self._create_table_all_text(tname, df)
            self._register(tname, name, None)
            count += 1; names.append(tname)
        return StructuredIngestResult(tables_added=count, table_names=names)

    def add_docx_tables(self, name: str, doc) -> StructuredIngestResult:
        tables = getattr(doc, "tables", [])
        count = 0; names = []
        for idx, tbl in enumerate(tables, start=1):
            rows = []
            for r in tbl.rows:
                rows.append([c.text for c in r.cells])
            if not rows:
                continue
            df = pd.DataFrame(rows[1:], columns=rows[0] if rows else None).astype(str)
            tname = self._normalize_name(f"{name.replace('.docx','')}_table_{idx}")
            self._create_table_all_text(tname, df)
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
                if not tbl:
                    continue
                header = tbl[0]
                body = tbl[1:] if len(tbl) > 1 else []
                has_header = all(h is not None and str(h).strip() != "" for h in header)
                if has_header:
                    df = pd.DataFrame(body, columns=header).astype(str)
                else:
                    df = pd.DataFrame(tbl).astype(str)
                tname = self._normalize_name(f"{name.replace('.pdf','')}_p{pidx}_t{tidx}")
                self._create_table_all_text(tname, df)
                self._register(tname, name, pidx)
                count += 1; names.append(tname)
        return StructuredIngestResult(tables_added=count, table_names=names)

    # -------- Search & Admin --------
    def search_context(self, query: str, top_k: int = 5) -> str:
        q = query.lower().split()
        tables = self.db.execute(
            "SELECT table_name, source, coalesce(cast(page as int), -1) FROM kb_tables"
        ).fetchall()
    
        snippets = []
        for tname, source, page in tables:
            # FIX: pick column_name at index 1 (not column_id at index 0) and cast to str
            info_rows = self.db.execute(f"PRAGMA table_info('{tname}')").fetchall()
            cols = [str(r[1]) for r in info_rows]  # r[1] == column_name in DuckDB
    
            text = f"{tname} (cols: {', '.join(cols[:10])}) from {source}{f' p{page}' if page and page > 0 else ''}"
    
            # quick relevance score: token matches in table name or column names
            score = sum(1 for tok in q if tok in tname.lower() or any(tok in (c.lower() if c else "") for c in cols))
            if score > 0:
                sample = self.db.execute(f"SELECT * FROM {tname} LIMIT 3").fetchdf()
                # ensure markdown-friendly string rendering
                sample = sample.astype(str)
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
