
# Eurex Assistant — RAG + Table‑Aware Testing (Final, per your settings)

**What’s in this build**
- **No Docker** files.
- **Embeddings**: default `ollama:mxbai-embed-large` (local). Switch in sidebar or `EMBED_BACKEND=` env.
- **Vectors**: `hnswlib` (fast, local). Metadata in DuckDB.
- **Structured**: **DuckDB** tables for CSV/PDF/DOCX tables with **type inference (no constraints)**.
- **Unstructured**: text chunks from PDF/DOCX/MD/TXT/CSV headers → vectors.
- **Admin SQL runner**: read‑only (SELECT‑only) console in the Admin tab.
- **Chat memory**: SQLite conversation history with rolling summaries.
- **Gherkin generator**: Eurex/DBAG acceptance‑testing context included.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Embeddings (recommended local):
export EMBED_BACKEND=ollama:mxbai-embed-large
# or:
# export EMBED_BACKEND=bge-large-en-v1.5
# export EMBED_BACKEND=bge-m3

# If using OpenAI chat models instead of Ollama:
# export OPENAI_API_KEY=sk-...

streamlit run app.py
```

## Notes
- **Type inference only**: DuckDB column types are inferred (BOOLEAN/INTEGER/DOUBLE/DATE/VARCHAR). No NOT NULL / CHECK constraints are added.
- **SQL runner** is **read‑only** and rejects non‑SELECT statements automatically.
- **Strict mode** is **OFF by default** (toggle in sidebar). With strict ON, the bot refuses to guess and will say “I don’t know…” when context is missing.
- PDFs that are images need OCR if text extraction is empty; we can add that on request.
