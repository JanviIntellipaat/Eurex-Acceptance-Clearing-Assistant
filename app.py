
import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import streamlit as st

from core.config import Settings, get_settings, save_settings_to_env
from core.llm import LLMRouter, ChatMessage
from core.knowledge import KnowledgeBase
from core.memory import ConversationStore
from core.gherkin import GherkinGenerator
from core.prompts import SYSTEM_ASSISTANT, SYSTEM_STRICT_SUFFIX, SYSTEM_CONTEXT_SUFFIX, SYSTEM_ORG_CONTEXT, GHERKIN_SYSTEM
from core.utils import ensure_dirs, readable_bytes

APP_TITLE = "Eurex Assistant — Chat + Table‑Aware Gherkin Generator"

def inject_css():
    css_path = Path("static/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

def sidebar(settings: Settings) -> Settings:
    with st.sidebar:
        st.header("⚙️ Settings")

        provider = st.selectbox("Model provider", ["ollama", "openai"], index=(0 if settings.provider=="ollama" else 1))
        model = st.text_input("Model name", value=settings.model)
        temperature = st.slider("Temperature", 0.0, 1.5, settings.temperature, 0.05)
        strict = st.toggle("Strict mode (no guessing, no fabrication)", value=settings.strict)
        max_tokens = st.number_input("Max tokens (hint)", min_value=256, max_value=8192, value=settings.max_tokens, step=128)

        st.divider()
        st.caption("Storage & Embeddings")
        kb_dir = st.text_input("KB dir", value=settings.kb_dir)
        db_path = st.text_input("Conversations DB (SQLite)", value=settings.db_path)
        duckdb_path = st.text_input("Structured DB (DuckDB)", value=settings.duckdb_path)
        embed_backend = st.selectbox("Embedding backend", ["ollama:mxbai-embed-large", "bge-large-en-v1.5", "bge-m3"], index=0)
        ef_dim_map = {"ollama:mxbai-embed-large":1024, "bge-large-en-v1.5":1024, "bge-m3":1024}
        ef_dim = ef_dim_map.get(embed_backend, settings.embed_dim)

        settings.show_resources = st.sidebar.checkbox(
        "Show resources (JSON)", value=settings.show_resources,
        help="When enabled, the sidebar shows a JSON of the documents/sheets/columns and headings used for the current answer."
        )
        
        st.divider()
        st.caption("Provider endpoints")
        base_url = st.text_input("Base URL (optional)", value=settings.base_url or "")
        if provider == "openai":
            openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY",""))
        else:
            openai_key = ""

        if st.button("Save settings"):
            new = Settings(
                provider=provider, model=model, temperature=temperature,
                kb_dir=kb_dir, db_path=db_path, duckdb_path=duckdb_path,
                base_url=base_url or None, strict=strict, max_tokens=max_tokens,
                embed_backend=embed_backend, embed_dim=ef_dim
            )
            save_settings_to_env(new, openai_key=openai_key)
            st.success("Saved ✅")
            return new

        return settings

def tab_chat(settings: Settings, kb: KnowledgeBase, mem: ConversationStore, router: LLMRouter):
    """
    Chat tab:
    - Composer appears UNDER the latest answer (form-based).
    - Streams safely into a single placeholder (no flicker).
    - Auto-includes sheet/schema inventory in context when relevant.
    - No citations by default; optional Resources JSON shown in sidebar when enabled.
    """
    import json, re
    from core.prompts import (
        SYSTEM_ASSISTANT, SYSTEM_ORG_CONTEXT, SYSTEM_CONTEXT_SUFFIX, SYSTEM_STRICT_SUFFIX
    )
    from core.llm import ChatMessage

    st.subheader("💬 Team Chatbot (Retrieval-Augmented)")
    st.caption("Answers use retrieved context from documents and structured tables. Sources are hidden by default; ask “source?” to see them in plain English.")

    # Ensure a session exists
    if "session_id" not in st.session_state:
        st.session_state.session_id = mem.start_session(user="user")
    session_id = st.session_state.session_id

    # Header controls
    cols = st.columns([3,1,1,1])
    with cols[0]:
        conv_name = st.text_input("Conversation name", value=mem.get_title(session_id) or "Untitled chat")
    with cols[1]:
        if st.button("💾 Save"):
            mem.rename_session(session_id, conv_name)
            st.toast("Saved")
    with cols[2]:
        if st.button("🗑️ Clear"):
            mem.clear_session(session_id)
            st.toast("Cleared conversation")
    with cols[3]:
        # Always offer current export
        export_data = mem.export_all_for_session(session_id)
        st.download_button(
            "⬇️ Export", data=json.dumps(export_data, indent=2),
            file_name=f"chat_{session_id}.json", mime="application/json"
        )

    # Render history (kept above the composer so the composer sits under last answer)
    history = mem.get_messages(session_id)
    for m in history[-200:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Composer BELOW the last assistant message
    with st.form("ask_form", clear_on_submit=True):
        query = st.text_area("Ask a question", height=90, placeholder="Type your question…")
        submitted = st.form_submit_button("Send")

    if not submitted or not query.strip():
        return

    # Echo user message immediately
    with st.chat_message("user"):
        st.markdown(query)
    mem.append_message(session_id, "user", query)

    # -------- Retrieve context (text + structured) --------
    with st.spinner("Retrieving context…"):
        # Unstructured
        ctx_chunks = kb.search_text(query, k=6)  # FAISS top-k
        ctx_text = (
            "\n\n".join(
                [f"[{i+1}] {c.source} p{c.page or '-'} — {c.text}" for i, c in enumerate(ctx_chunks, 1)]
            )
            if ctx_chunks else "(no text context)"
        )

        # Structured summaries (keyword-aware table snippets)
        struct_ctx = kb.search_structured_context(query, top_k=3)

        # If the user asks about sheets/schema/columns/headers, include precise sheet inventory
        ql = query.lower()
        if any(t in ql for t in ["sheet", "worksheet", "schema", "columns", "headers", "header", "fields"]):
            m = re.search(r'([A-Za-z0-9._ ()-]+\.xlsx?)', query)
            if m:
                inv = kb.structured.list_sheets(source=m.group(1))
            else:
                inv = kb.structured.list_sheets()
            struct_ctx = (struct_ctx + "\n\n" if struct_ctx else "") + "Sheet inventory:\n" + inv

        # Optional: build Resources JSON for the sidebar
        resources_json = None
        if getattr(settings, "show_resources", False):
            try:
                # Structured resources likely relevant to the query
                struct_resources = kb.structured.resources_for_query(query, top_k=6)
            except Exception:
                struct_resources = []

            # Text resources: doc + page + nearest headings (PDF page, DOCX headings)
            text_resources = []
            try:
                con = kb.structured.db  # reuse DuckDB connection
                for c in ctx_chunks[:6]:
                    src = c.source.split("#", 1)[0] if "#" in c.source else c.source
                    page_num = c.page if c.page is not None else None
                    headings = []
                    if isinstance(src, str) and src.lower().endswith(".pdf"):
                        try:
                            rows = con.execute(
                                "SELECT heading FROM kb_headings WHERE source=? AND page=? ORDER BY level ASC LIMIT 3",
                                [src, int(page_num) if page_num else 1]
                            ).fetchall()
                            headings = [r[0] for r in rows]
                        except Exception:
                            pass
                    elif isinstance(src, str) and src.lower().endswith(".docx"):
                        try:
                            rows = con.execute(
                                "SELECT heading FROM kb_headings WHERE source=? ORDER BY level ASC LIMIT 5",
                                [src]
                            ).fetchall()
                            headings = [r[0] for r in rows]
                        except Exception:
                            pass
                    text_resources.append({
                        "document": src,
                        "page": page_num,
                        "headings": headings
                    })
            except Exception:
                pass

            resources_json = {"structured": struct_resources, "text": text_resources}
            with st.sidebar.expander("Resources used (JSON)", expanded=True):
                st.json(resources_json)

    # -------- Build messages (no citations by default) --------
    system = SYSTEM_ASSISTANT + SYSTEM_ORG_CONTEXT + SYSTEM_CONTEXT_SUFFIX
    if getattr(settings, "strict", False):
        system += SYSTEM_STRICT_SUFFIX

    # Short rolling summary to keep context small
    summary = ""
    try:
        summary = mem.summarize(session_id, router, max_turns=12)
    except Exception:
        pass

    sys_msgs = [ChatMessage(role="system", content=system)]
    if summary:
        sys_msgs.append(ChatMessage(role="system", content=f"Prior summary:\n{summary}"))
    if ctx_text:
        sys_msgs.append(ChatMessage(role="system", content=f"Retrieved text context:\n{ctx_text}"))
    if struct_ctx:
        sys_msgs.append(ChatMessage(role="system", content=f"Structured data context (read-only):\n{struct_ctx}"))

    msgs = sys_msgs + [ChatMessage(role=m["role"], content=m["content"]) for m in mem.get_messages(session_id)]

    # -------- Stream assistant reply safely --------
    with st.chat_message("assistant"):
        ph = st.empty()
        out = []
        try:
            for ch in router.stream_chat(msgs):
                if ch:
                    out.append(ch)
                # update UI every few tokens to reduce rerenders
                if len(out) % 8 == 0:
                    ph.markdown("".join(out))
        except Exception:
            # fall back to non-stream completion if streaming fails
            pass

        final_resp = "".join(out).strip()
        if not final_resp:
            try:
                final_resp = router.complete("", system=system)
            except Exception:
                final_resp = "Sorry—something went wrong generating a response."

        ph.markdown(final_resp)
        mem.append_message(session_id, "assistant", final_resp)


def tab_testcases(settings: Settings, kb: KnowledgeBase, mem: ConversationStore, router: LLMRouter):
    st.subheader("🧪 Acceptance Test Generator (Eurex / Deutsche Börse)")
    st.caption("Generates Gherkin using Eurex Clearing acceptance‑testing conventions.")

    with st.form("gherkin"):
        feature_name = st.text_input("Feature/Epic")
        component = st.text_input("Component/Module (optional)")
        constraints = st.text_input("Non‑functional constraints (comma‑separated, e.g., WCAG AA, GDPR, latency)")
        scenario_count = st.slider("Scenarios", 1, 20, 7)
        use_kb = st.checkbox("Use knowledge base context", value=True)
        submitted = st.form_submit_button("Generate")

    if submitted:
        if not feature_name.strip():
            st.error("Feature name is required."); return
        retrieved = kb.search_text(f"{feature_name} {component}", k=12) if use_kb else []
        text, meta = GherkinGenerator(router).generate(
            feature_name=feature_name,
            component=component or None,
            constraints=[c.strip() for c in constraints.split(",") if c.strip()] if constraints else [],
            retrieved=retrieved,
            scenario_count=scenario_count
        )
        st.code(text, language="gherkin")
        st.download_button("⬇️ Download .feature", data=text, file_name=f"{feature_name.replace(' ','_').lower()}.feature", mime="text/plain")
        st.json(meta)

def tab_kb(settings: Settings, kb: KnowledgeBase):
    st.subheader("📚 Knowledge Base")
    st.caption("Add documents. Structured tables go to DuckDB; text chunks go to the vector index.")

    uploaded = st.file_uploader("Upload files", accept_multiple_files=True, type=["pdf","txt","md","docx","csv","xlsx","xls"])

    table_aware = st.toggle("Table-aware parsing for PDFs", value=True)
    if uploaded:
        results = kb.add_files(uploaded, parse_tables=table_aware)
        st.success(f"Indexed {len(results['text_chunks'])} text chunks. Added {results['tables_added']} tables to DuckDB.")
        with st.expander("Details"):
            st.write("Text chunk examples:")
            for ch in results["text_chunks"][:5]:
                st.write(ch)

    q = st.text_input("🔎 Try a search")
    if q:
        chunks = kb.search_text(q, k=8)
        for i,ch in enumerate(chunks,1):
            st.markdown(f"**[{i}]** {ch.source} · p{ch.page or '-'} · score={ch.score:.4f}")
            st.write(ch.text)

        st.markdown("**Structured data hits (by table/column match):**")
        st.code(kb.search_structured_context(q, top_k=5))
        
            # --- Inventory panel ---
    import duckdb
    st.divider()
    st.subheader("📦 Documents in Knowledge Base")

    try:
        con = duckdb.connect(settings.duckdb_path, read_only=True)
        docs = con.execute("""
            SELECT source, kind, bytes, sheet_count, table_count, strftime(ingested_at, '%Y-%m-%d %H:%M') AS ingested
            FROM kb_documents
            ORDER BY ingested_at DESC
        """).fetchdf()
        st.dataframe(docs, use_container_width=True) if not docs.empty else st.write("(no documents)")

        st.subheader("📑 Sheets")
        sheets = con.execute("""
            SELECT source, sheet, n_cols, n_rows
            FROM kb_sheets
            ORDER BY source, sheet
        """).fetchdf()
        st.dataframe(sheets, use_container_width=True) if not sheets.empty else st.write("(no sheets)")

        st.subheader("🧭 Columns (select a file+sheet to preview)")
        if not sheets.empty:
            src = st.selectbox("Source file", sorted(sheets["source"].unique()))
            shs = sheets[sheets["source"] == src]["sheet"].tolist()
            sh  = st.selectbox("Sheet", shs)
            cols = con.execute("""
                SELECT col_ordinal, column_name, parent
                FROM kb_columns WHERE source=? AND sheet=? ORDER BY col_ordinal
            """, [src, sh]).fetchdf()
            if not cols.empty:
                cols["display"] = cols.apply(lambda r: (f"{int(r['col_ordinal']):02d}. " + (f"{r['parent']} ▸ " if r['parent'] else "") + f"{r['column_name']}"), axis=1)
                st.dataframe(cols[["display"]], hide_index=True, use_container_width=True)
            else:
                st.write("(no columns recorded)")
        con.close()
    except Exception as e:
        st.error(f"Inventory error: {e}")



def tab_admin(settings: Settings, kb: KnowledgeBase, mem: ConversationStore):
    st.subheader("🔐 Admin")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Purge vector index"):
            kb.purge_index(); st.success("Cleared vector index.")
        if st.button("Purge structured DB"):
            kb.purge_structured(); st.success("Cleared structured DB.")

    with col2:
        st.write("Stats")
        st.json({
            "kb_dir": settings.kb_dir,
            "duckdb_path": settings.duckdb_path,
            "db_path": settings.db_path
        })

    st.divider()
    st.subheader("🧪 SQL runner (read‑only)")
    enable_sql = st.toggle("Enable SQL runner", value=False, help="SELECT-only, read-only against DuckDB")
    if enable_sql:
        default_q = "SELECT table_name, source, page FROM kb_tables LIMIT 100;"
        q = st.text_area("SQL (SELECT-only)", value=default_q, height=140)
        if st.button("Run query"):
            q_low = q.strip().lower()
            if not q_low.startswith("select"):
                st.error("Read-only: only SELECT statements are allowed.")
            else:
                import duckdb
                if " limit " not in q_low:
                    q = q.rstrip(";") + " LIMIT 1000;"
                try:
                    con = duckdb.connect(settings.duckdb_path, read_only=True)
                    df = con.execute(q).fetchdf()
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Query error: {e}")
                finally:
                    try: con.close()
                    except: pass

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🛡️", layout="wide")
    inject_css()
    st.title(APP_TITLE)

    settings = get_settings()
    settings = sidebar(settings)

    ensure_dirs(settings.kb_dir, Path(settings.db_path).parent, Path(settings.duckdb_path).parent)

    kb = KnowledgeBase(settings)
    mem = ConversationStore(settings.db_path)
    router = LLMRouter(settings)

    tabs = st.tabs(["Chat", "Test Cases", "Knowledge Base", "Admin", "Agent Instructions"])
    with tabs[0]:
        tab_chat(settings, kb, mem, router)
    with tabs[1]:
        tab_testcases(settings, kb, mem, router)
    with tabs[2]:
        tab_kb(settings, kb)
    with tabs[3]:
        tab_admin(settings, kb, mem)
    with tabs[4]:
        st.subheader("📜 Active System Instructions")
        st.code(SYSTEM_ASSISTANT + SYSTEM_ORG_CONTEXT + SYSTEM_CONTEXT_SUFFIX + (SYSTEM_STRICT_SUFFIX if settings.strict else ""), language="markdown")
        st.code(GHERKIN_SYSTEM, language="markdown")

if __name__ == "__main__":
    main()
