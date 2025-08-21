import os, json
from pathlib import Path
import streamlit as st

from core.config import Settings, load_settings, save_settings, save_settings_to_env
from core.llm import LLMRouter, ChatMessage
from core.knowledge import KnowledgeBase
from core.memory import ConversationStore
from core.gherkin import GherkinGenerator
from core.prompts import SYSTEM_ASSISTANT, SYSTEM_STRICT_SUFFIX, SYSTEM_CONTEXT_SUFFIX, SYSTEM_ORG_CONTEXT, GHERKIN_SYSTEM

APP_TITLE = "Eurex Assistant — Chat + Gherkin Test Case Generator"

def sidebar(settings: Settings):
    st.sidebar.title("⚙️ Settings")
    provider = st.sidebar.selectbox("Model provider", ["ollama","openai"], index=0 if settings.provider=="ollama" else 1)
    model = st.sidebar.text_input("Model name", value=settings.model)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, settings.temperature, 0.05)
    strict = st.sidebar.toggle("Strict mode (no guessing, no fabrication)", value=settings.strict)
    max_tokens = st.sidebar.number_input("Max tokens (hint)", 256, 8192, settings.max_tokens, 128)

    st.sidebar.divider()
    st.sidebar.caption("Storage & Embeddings")
    kb_dir = st.sidebar.text_input("KB dir", value=settings.kb_dir)
    db_path = st.sidebar.text_input("Conversations DB (SQLite)", value=settings.db_path)
    duckdb_path = st.sidebar.text_input("Structured DB (DuckDB)", value=settings.duckdb_path)
    embed_backend = st.sidebar.selectbox("Embedding backend", ["ollama:mxbai-embed-large", "bge-large-en-v1.5", "bge-m3"],
                                         index=0 if settings.embed_backend.startswith("ollama:") else 1)
    embed_dim = st.sidebar.number_input("Embedding dim", 256, 2048, settings.embed_dim, 64)

    settings.show_resources = st.sidebar.checkbox(
        "Show resources (JSON)", value=settings.show_resources,
        help="When enabled, the sidebar shows a JSON of the documents/sheets/columns and headings used for the current answer."
    )

    st.sidebar.divider()
    st.sidebar.caption("Provider endpoints")
    base_url = st.sidebar.text_input("Base URL (optional)", value=settings.base_url or "")
    openai_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY","")) if provider=="openai" else ""

    if st.sidebar.button("Save settings"):
        new = Settings(
            provider=provider, model=model, temperature=temperature, strict=strict,
            max_output_tokens=max_tokens, kb_dir=kb_dir, duckdb_path=duckdb_path, conv_db_path=db_path,
            embed_backend=embed_backend, embed_dim=int(embed_dim),
            base_url=(base_url or None), openai_api_key=(openai_key or None),
            show_resources=settings.show_resources
        )
        save_settings(new)
        save_settings_to_env(new)
        st.sidebar.success("Saved.")
        st.session_state["_settings"] = new

def init_objects(settings: Settings):
    kb = KnowledgeBase(settings)
    mem = ConversationStore(settings.conv_db_path)
    router = LLMRouter(settings)
    return kb, mem, router

# ---------------- Chat tab ----------------
def tab_chat(settings: Settings, kb: KnowledgeBase, mem: ConversationStore, router: LLMRouter):
    """
    Composer appears UNDER the latest answer (form-based).
    Streams safely into a single placeholder. Auto sheet/schema inventory when relevant.
    No citations by default; optional Resources JSON in sidebar.
    """
    import re
    st.subheader("💬 Team Chatbot (Retrieval-Augmented)")
    st.caption("Answers use retrieved context from documents and structured tables. Sources are hidden by default; ask “source?” to see them in plain English.")

    # Ensure a session exists
    if "session_id" not in st.session_state:
        st.session_state.session_id = mem.start_session(user="user")
    session_id = st.session_state.session_id

    # Header
    cols = st.columns([3,1,1,1])
    with cols[0]:
        conv_name = st.text_input("Conversation name", value=mem.get_title(session_id) or "Untitled chat")
    with cols[1]:
        if st.button("💾 Save"): mem.rename_session(session_id, conv_name); st.toast("Saved")
    with cols[2]:
        if st.button("🗑️ Clear"): mem.clear_session(session_id); st.toast("Cleared")
    with cols[3]:
        export_data = mem.export_all_for_session(session_id)
        st.download_button("⬇️ Export", data=json.dumps(export_data, indent=2), file_name=f"chat_{session_id}.json", mime="application/json")

    # History
    history = mem.get_messages(session_id)
    for m in history[-200:]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Composer BELOW history
    with st.form("ask_form", clear_on_submit=True):
        query = st.text_area("Ask a question", height=90, placeholder="Type your question…")
        submitted = st.form_submit_button("Send")
    if not submitted or not query.strip():
        return

    # echo user
    with st.chat_message("user"):
        st.markdown(query)
    mem.append_message(session_id, "user", query)

    # Retrieve context
    with st.spinner("Retrieving context…"):
        ctx_chunks = kb.search_text(query, k=6)
        ctx_text = ("\n\n".join([f"[{i+1}] {c.source} p{c.page or '-'} — {c.text}" for i,c in enumerate(ctx_chunks,1)]) if ctx_chunks else "(no text context)")
        struct_ctx = kb.search_structured_context(query, top_k=3)

        ql = query.lower()
        if any(t in ql for t in ["sheet", "worksheet", "schema", "columns", "headers", "fields"]):
            m = re.search(r'([A-Za-z0-9._ ()-]+\.xlsx?)', query)
            inv = kb.structured.list_sheets(source=m.group(1)) if m else kb.structured.list_sheets()
            struct_ctx = (struct_ctx + "\n\n" if struct_ctx else "") + "Sheet inventory:\n" + inv

        # Optional resources JSON in sidebar
        if settings.show_resources:
            try:
                struct_resources = kb.structured.resources_for_query(query, top_k=6)
            except Exception:
                struct_resources = []
            text_resources = []
            try:
                con = kb.structured.db
                for c in ctx_chunks[:6]:
                    src = c.source.split("#",1)[0] if "#" in c.source else c.source
                    page_num = c.page if c.page is not None else None
                    headings = []
                    if isinstance(src,str) and src.lower().endswith(".pdf"):
                        rows = con.execute("SELECT heading FROM kb_headings WHERE source=? AND page=? ORDER BY level ASC LIMIT 3",
                                           [src, int(page_num) if page_num else 1]).fetchall()
                        headings = [r[0] for r in rows]
                    elif isinstance(src,str) and src.lower().endswith(".docx"):
                        rows = con.execute("SELECT heading FROM kb_headings WHERE source=? ORDER BY level ASC LIMIT 5",
                                           [src]).fetchall()
                        headings = [r[0] for r in rows]
                    text_resources.append({"document": src, "page": page_num, "headings": headings})
            except Exception:
                pass
            with st.sidebar.expander("Resources used (JSON)", expanded=True):
                st.json({"structured": struct_resources, "text": text_resources})

    # Build messages
    system = SYSTEM_ASSISTANT + "\n" + SYSTEM_ORG_CONTEXT + "\n" + SYSTEM_CONTEXT_SUFFIX
    if settings.strict:
        system += "\n" + SYSTEM_STRICT_SUFFIX

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

    # Stream reply
    with st.chat_message("assistant"):
        ph = st.empty()
        out = []
        try:
            for ch in router.stream_chat(msgs):
                if ch: out.append(ch)
                if len(out) % 8 == 0:
                    ph.markdown("".join(out))
        except Exception:
            pass
        final_resp = "".join(out).strip() or router.complete("", system=system)
        ph.markdown(final_resp)
        mem.append_message(session_id, "assistant", final_resp)

# ---------------- Testcases tab ----------------
def tab_testcases(settings: Settings, kb: KnowledgeBase, mem: ConversationStore, router: LLMRouter):
    st.subheader("🧪 Gherkin Test Case Generator")
    feature = st.text_input("Feature/Epic")
    component = st.text_input("Component/Module (optional)")
    scenarios = st.slider("Number of scenarios", 1, 15, 6)
    constraints = st.text_input("Non-functional constraints (comma-separated)", value="GDPR, P95 < 250ms")
    use_ctx = st.toggle("Use knowledge base context", value=True)

    if st.button("Generate"):
        retrieved = kb.search_text(feature, k=6) if use_ctx else []
        gen = GherkinGenerator(router)
        text, meta = gen.generate(feature, component or None, scenarios, [s.strip() for s in constraints.split(",") if s.strip()], retrieved)
        st.success("Generated.")
        st.download_button("Download .feature", data=text, file_name=f"{feature.replace(' ','_')}.feature", mime="text/plain")
        st.code(text, language="gherkin")

# ---------------- Knowledge Base tab ----------------
def tab_kb(settings: Settings, kb: KnowledgeBase):
    st.subheader("📚 Knowledge Base")
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
        for i, ch in enumerate(chunks, 1):
            st.write(f"[{i}] {ch.source} p{ch.page or '-'} · score={ch.score:.4f}")
            st.caption(ch.text)
        st.write("Structured data hits (by table/column match):")
        try:
            st.code(kb.search_structured_context(q, top_k=5))
        except Exception as e:
            st.error(f"Structured preview error: {e}")

    # Inventory panel (documents/sheets/columns)
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

        st.subheader("🧭 Columns (pick a file & sheet)")
        if not sheets.empty:
            src = st.selectbox("Source file", sorted(sheets["source"].unique()))
            shs = sheets[sheets["source"] == src]["sheet"].tolist()
            sh = st.selectbox("Sheet", shs)
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

# ---------------- Admin tab ----------------
def tab_admin(settings: Settings, kb: KnowledgeBase, mem: ConversationStore):
    st.subheader("🛠️ Admin")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Purge vector index"):
            kb.purge_index(); st.success("Purged FAISS index.")
    with cols[1]:
        if st.button("Purge structured DB"):
            kb.purge_structured(); st.success("Purged DuckDB tables & metadata.")

    st.caption("Stats")
    st.json({
        "kb_dir": settings.kb_dir,
        "duckdb_path": settings.duckdb_path,
        "conv_db_path": settings.conv_db_path,
        "embed_backend": settings.embed_backend
    })

    st.subheader("🧪 SQL runner (SELECT-only, read-only)")
    enable_sql = st.toggle("Enable SQL runner")
    if enable_sql:
        default_q = "SELECT table_name, source, page, sheet FROM kb_tables LIMIT 100;"
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
                    con.close()
                except Exception as e:
                    st.error(str(e))

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
    settings = st.session_state.get("_settings") or load_settings()
    sidebar(settings)
    settings = st.session_state.get("_settings") or load_settings()

    kb, mem, router = init_objects(settings)

    tabs = st.tabs(["Chat", "Test Cases", "Knowledge Base", "Admin", "Agent Instructions"])
    with tabs[0]: tab_chat(settings, kb, mem, router)
    with tabs[1]: tab_testcases(settings, kb, mem, router)
    with tabs[2]: tab_kb(settings, kb)
    with tabs[3]: tab_admin(settings, kb, mem)
    with tabs[4]:
        st.subheader("📜 Active System Instructions")
        st.code(SYSTEM_ASSISTANT + "\n\n" + SYSTEM_ORG_CONTEXT + "\n\n" + (SYSTEM_STRICT_SUFFIX if settings.strict else ""), language="markdown")
        st.code(GHERKIN_SYSTEM, language="markdown")

if __name__ == "__main__":
    main()
