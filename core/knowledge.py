from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import duckdb, pandas as pd
import io

@dataclass
class StructuredIngestResult:
    tables_added: int
    table_names: List[str]

class StructuredStore:
    """
    Stores structured data in DuckDB (TEXT-only columns) + rich metadata:
    - CSV → one table
    - XLSX/XLS → each sheet → table (sheet recorded)
      * Multi-row headers: detect & record parent→child (sub-columns)
    - PDF tables (pdfplumber) → per-page table (page/sheet recorded)
    - DOCX tables → tables (sheet recorded as docx_table_X)
    - Headings for DOCX/PDF recorded
    """

    def __init__(self, duckdb_path: str):
        self.db = duckdb.connect(duckdb_path)
        # Tables
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS kb_tables (
                table_name VARCHAR,
                source VARCHAR,
                page INTEGER,
                sheet VARCHAR,
                created_at TIMESTAMP DEFAULT now()
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS kb_documents (
                source VARCHAR PRIMARY KEY,
                kind   VARCHAR,            -- csv, xlsx, xls, pdf, docx
                bytes  BIGINT,
                sheet_count INTEGER DEFAULT 0,
                table_count INTEGER DEFAULT 0,
                ingested_at TIMESTAMP DEFAULT now()
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS kb_sheets (
                source VARCHAR,
                sheet  VARCHAR,
                n_cols INTEGER,
                n_rows INTEGER,
                PRIMARY KEY (source, sheet)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS kb_columns (
                source VARCHAR,
                sheet  VARCHAR,
                col_ordinal INTEGER,
                column_name VARCHAR,
                parent VARCHAR,    -- parent header if detected
                PRIMARY KEY (source, sheet, col_ordinal)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS kb_headings (
                source VARCHAR,
                page   INTEGER,
                level  INTEGER,
                heading VARCHAR
            )
        """)
        self.db.commit()

    # ---------- metadata helpers ----------
    def record_document(self, source: str, kind: str, size_bytes: int, sheet_count: int, table_count: int):
        self.db.execute("""
            INSERT INTO kb_documents(source, kind, bytes, sheet_count, table_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (source) DO UPDATE SET
                kind=excluded.kind,
                bytes=excluded.bytes,
                sheet_count=excluded.sheet_count,
                table_count=excluded.table_count,
                ingested_at=now()
        """, (source, kind, int(size_bytes), int(sheet_count), int(table_count)))
        self.db.commit()

    def add_sheet_meta(self, source: str, sheet: str, n_cols: int, n_rows: int):
        self.db.execute("""
            INSERT INTO kb_sheets(source, sheet, n_cols, n_rows)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (source, sheet) DO UPDATE SET
                n_cols=excluded.n_cols,
                n_rows=excluded.n_rows
        """, (source, sheet, int(n_cols), int(n_rows)))
        self.db.commit()

    def add_columns_meta(self, source: str, sheet: str, columns: List[str], parents: Optional[List[Optional[str]]] = None):
        if parents is None:
            parents = [None] * len(columns)
        for i, (col, par) in enumerate(zip(columns, parents), start=1):
            self.db.execute("""
                INSERT INTO kb_columns(source, sheet, col_ordinal, column_name, parent)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (source, sheet, col_ordinal) DO UPDATE SET
                    column_name=excluded.column_name,
                    parent=excluded.parent
            """, (source, sheet, i, str(col), (str(par) if par else None)))
        self.db.commit()

    def add_headings(self, source: str, headings: List[Tuple[int,int,str]]):
        # headings: List[(page, level, text)]
        for page, level, text in headings:
            self.db.execute("INSERT INTO kb_headings(source, page, level, heading) VALUES (?,?,?,?)",
                            (source, page, level, text))
        self.db.commit()

    # ---------- internal utilities ----------
    def _normalize_name(self, s: str) -> str:
        return "".join(ch if ch.isalnum() or ch=='_' else '_' for ch in s.lower())

    def _repair_headers_if_unnamed(self, df: pd.DataFrame) -> pd.DataFrame:
        """If many headers are 'Unnamed:*', synthesize from first rows."""
        if df is None or df.shape[1] == 0:
            return df
        cols = [str(c) for c in df.columns]
        unnamed_ratio = sum(c.lower().startswith("unnamed") for c in cols) / max(1, len(cols))
        if unnamed_ratio < 0.6:
            return df
        head = df.head(2).fillna("")
        def pick(i):
            v0 = str(head.iat[0, i]) if head.shape[0] > 0 and i < head.shape[1] else ""
            v1 = str(head.iat[1, i]) if head.shape[0] > 1 and i < head.shape[1] else ""
            cand = (v1 or v0 or cols[i]).strip()
            return cand if cand else f"col_{i+1}"
        new_cols = [pick(i) for i in range(len(cols))]
        df = df.iloc[2:, :].copy() if head.shape[0] >= 2 else df.iloc[1:, :].copy()
        df.columns = new_cols
        return df

    def _create_table_all_text(self, table_name: str, df: pd.DataFrame):
        # handle empty
        if df is None or df.shape[1] == 0:
            self.db.execute(f'CREATE OR REPLACE TABLE {table_name} ("_empty" VARCHAR)')
            self.db.commit()
            return
        # clean headers if needed
        df = self._repair_headers_if_unnamed(df)
        df.columns = [str(c) if c is not None else "_col" for c in df.columns]
        df = df.astype(str)
        cols = [f'"{c}" VARCHAR' for c in df.columns]
        ddl = f"CREATE OR REPLACE TABLE {table_name} ({', '.join(cols)});"
        self.db.execute(ddl)
        self.db.register("df_tmp", df)
        self.db.execute(f"INSERT INTO {table_name} SELECT * FROM df_tmp")
        self.db.unregister("df_tmp")
        self.db.commit()

    # Try to detect parent→child from first 2 header rows (best effort)
    def _header_hierarchy(self, content: bytes, sheet_name: str, engine: str) -> Tuple[List[str], List[Optional[str]]]:
        try:
            raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=None, nrows=2, dtype=str, engine=engine)
        except Exception:
            return [], []
        if raw is None or raw.shape[1] == 0:
            return [], []
        row0 = [str(x) if x is not None else "" for x in (raw.iloc[0].tolist() if raw.shape[0] > 0 else [])]
        row1 = [str(x) if x is not None else "" for x in (raw.iloc[1].tolist() if raw.shape[0] > 1 else [])]
        cols = []
        parents = []
        width = max(len(row0), len(row1))
        for i in range(width):
            p = row0[i] if i < len(row0) else ""
            c = row1[i] if i < len(row1) else ""
            name = (c or p or f"col_{i+1}").strip()
            cols.append(name)
            parents.append(p.strip() or None)
        return cols, parents

    # ---------- ingestors ----------
    def add_csv(self, name: str, content: bytes) -> StructuredIngestResult:
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, low_memory=False)
        tname = self._normalize_name(name.replace(".csv",""))
        self._create_table_all_text(tname, df)
        self.db.execute("INSERT INTO kb_tables(table_name, source, page, sheet) VALUES (?,?,?,?)",
                        (tname, name, None, "csv"))
        self.add_sheet_meta(name, "csv", n_cols=df.shape[1], n_rows=df.shape[0])
        self.add_columns_meta(name, "csv", list(df.columns))
        self.db.commit()
        # record doc (1 sheet, 1 table)
        self.record_document(name, "csv", size_bytes=len(content), sheet_count=1, table_count=1)
        return StructuredIngestResult(tables_added=1, table_names=[tname])

    def add_xlsx(self, name: str, content: bytes) -> StructuredIngestResult:
        sheets = pd.read_excel(io.BytesIO(content), dtype=str, sheet_name=None, engine="openpyxl")
        count = 0; names = []
        base = name.replace(".xlsx","")
        for sheet_name, df in sheets.items():
            tname = self._normalize_name(f"{base}_{sheet_name}")
            self._create_table_all_text(tname, df)
            self.db.execute("INSERT INTO kb_tables(table_name, source, page, sheet) VALUES (?,?,?,?)",
                            (tname, name, None, sheet_name))
            self.add_sheet_meta(name, sheet_name, n_cols=df.shape[1], n_rows=df.shape[0])
            # parent/child from first two header rows (best-effort)
            cols, parents = self._header_hierarchy(content, sheet_name, engine="openpyxl")
            if cols:
                self.add_columns_meta(name, sheet_name, cols, parents)
            else:
                self.add_columns_meta(name, sheet_name, list(df.columns))
            count += 1; names.append(tname)
        self.db.commit()
        self.record_document(name, "xlsx", size_bytes=len(content), sheet_count=len(sheets), table_count=count)
        return StructuredIngestResult(tables_added=count, table_names=names)

    def add_xls(self, name: str, content: bytes) -> StructuredIngestResult:
        sheets = pd.read_excel(io.BytesIO(content), dtype=str, sheet_name=None, engine="xlrd")
        count = 0; names = []
        base = name.replace(".xls","")
        for sheet_name, df in sheets.items():
            tname = self._normalize_name(f"{base}_{sheet_name}")
            self._create_table_all_text(tname, df)
            self.db.execute("INSERT INTO kb_tables(table_name, source, page, sheet) VALUES (?,?,?,?)",
                            (tname, name, None, sheet_name))
            self.add_sheet_meta(name, sheet_name, n_cols=df.shape[1], n_rows=df.shape[0])
            cols, parents = self._header_hierarchy(content, sheet_name, engine="xlrd")
            if cols:
                self.add_columns_meta(name, sheet_name, cols, parents)
            else:
                self.add_columns_meta(name, sheet_name, list(df.columns))
            count += 1; names.append(tname)
        self.db.commit()
        self.record_document(name, "xls", size_bytes=len(content), sheet_count=len(sheets), table_count=count)
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
            self.db.execute("INSERT INTO kb_tables(table_name, source, page, sheet) VALUES (?,?,?,?)",
                            (tname, name, None, f"docx_table_{idx}"))
            self.add_sheet_meta(name, f"docx_table_{idx}", n_cols=df.shape[1], n_rows=df.shape[0])
            self.add_columns_meta(name, f"docx_table_{idx}", list(df.columns))
            count += 1; names.append(tname)
        self.db.commit()
        self.record_document(name, "docx", size_bytes=0, sheet_count=len(tables), table_count=count)
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
                self.db.execute("INSERT INTO kb_tables(table_name, source, page, sheet) VALUES (?,?,?,?)",
                                (tname, name, pidx, f"pdf_p{pidx}_t{tidx}"))
                self.add_sheet_meta(name, f"pdf_p{pidx}_t{tidx}", n_cols=df.shape[1], n_rows=df.shape[0])
                self.add_columns_meta(name, f"pdf_p{pidx}_t{tidx}", list(df.columns))
                count += 1; names.append(tname)
        self.db.commit()
        # table_count increments with found tables; sheet_count equals it here
        self.record_document(name, "pdf", size_bytes=0, sheet_count=count, table_count=count)
        return StructuredIngestResult(tables_added=count, table_names=names)

    # ---------- queries / search ----------
    def list_sheets(self, source: Optional[str] = None) -> str:
        if source:
            rows = self.db.execute(
                "SELECT sheet, n_cols, n_rows FROM kb_sheets WHERE source = ? ORDER BY sheet", [source]
            ).fetchall()
            if not rows:
                return f"{source} — sheets: (none)"
            lines = [f"{source} — sheets:"]
            for sh, nc, nr in rows:
                lines.append(f"- {sh} ({nc} cols × {nr} rows)")
            return "\n".join(lines)
        else:
            rows = self.db.execute(
                "SELECT source, sheet, n_cols, n_rows FROM kb_sheets ORDER BY source, sheet"
            ).fetchall()
            if not rows: return "(no sheets recorded)"
            lines = []
            cur = None
            for src, sh, nc, nr in rows:
                if src != cur:
                    lines.append(f"\n{src} — sheets:")
                    cur = src
                lines.append(f"- {sh} ({nc} cols × {nr} rows)")
            return "\n".join(lines).strip()

    def sheet_schema(self, source: str, sheet: str) -> str:
        cols = self.db.execute("""
            SELECT col_ordinal, column_name, parent
            FROM kb_columns
            WHERE source=? AND sheet=?
            ORDER BY col_ordinal
        """, (source, sheet)).fetchall()
        if not cols:
            return f"{source} / {sheet}: (no columns recorded)"
        lines = [f"{source} / {sheet}: {len(cols)} columns"]
        for ord_, name, parent in cols:
            if parent:
                lines.append(f"{ord_:>2}. {parent} ▸ {name}")
            else:
                lines.append(f"{ord_:>2}. {name}")
        return "\n".join(lines)

    def search_context(self, query: str, top_k: int = 5) -> str:
        q_tokens = [t.strip().lower() for t in query.split() if t.strip()]
        tables = self.db.execute(
            "SELECT table_name, source, coalesce(cast(page as int), -1) FROM kb_tables"
        ).fetchall()
        snippets = []
        for tname, source, page in tables:
            info_rows = self.db.execute(f"PRAGMA table_info('{tname}')").fetchall()
            cols = [str(r[1]) for r in info_rows]
            base_score = sum(
                1 for tok in q_tokens
                if tok in tname.lower() or any(tok in (c.lower() if c else "") for c in cols)
            )
            if base_score <= 0:
                continue
            where = ""
            if q_tokens and cols:
                per_tok = []
                for tok in q_tokens:
                    esc = tok.replace("'", "''")
                    per_tok.append(" OR ".join([f"lower(\"{c}\") LIKE '%{esc}%'" for c in cols]))
                where = " WHERE " + " OR ".join([f"({cl})" for cl in per_tok])
            try:
                df = self.db.execute(f"SELECT * FROM {tname}{where} LIMIT 3").fetchdf()
                if df.empty:
                    df = self.db.execute(f"SELECT * FROM {tname} LIMIT 3").fetchdf()
            except Exception:
                df = self.db.execute(f"SELECT * FROM {tname} LIMIT 3").fetchdf()
            df = df.astype(str)
            sample_txt = df.to_string(index=False)
            header = f"{tname} (cols: {', '.join(cols[:10])}) from {source}{f' p{page}' if page and page>0 else ''}"
            snippets.append(header + "\n" + sample_txt)
            if len(snippets) >= top_k:
                break
        return "\n\n".join(snippets)

    def purge(self):
        tbls = [r[0] for r in self.db.execute("SELECT table_name FROM kb_tables").fetchall()]
        for t in tbls:
            self.db.execute(f"DROP TABLE IF EXISTS {t}")
        for t in ["kb_tables","kb_sheets","kb_columns","kb_headings","kb_documents"]:
            self.db.execute(f"DELETE FROM {t}")
        self.db.commit()
