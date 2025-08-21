
from __future__ import annotations
import sqlite3, time
from typing import List, Dict, Any, Optional
from pathlib import Path
from .llm import LLMRouter, ChatMessage

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    user TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT,
    created_at REAL
);
"""

class ConversationStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA)
            con.commit()
        finally:
            con.close()

    def start_session(self, user: str) -> str:
        sid = str(int(time.time() * 1000))
        con = sqlite3.connect(self.db_path)
        with con:
            con.execute("INSERT INTO sessions(id,title,user,created_at) VALUES(?,?,?,?)",
                        (sid, "New conversation", user, time.time()))
        return sid

    def rename_session(self, session_id: str, title: str):
        con = sqlite3.connect(self.db_path)
        with con:
            con.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))

    def get_title(self, session_id: str) -> Optional[str]:
        con = sqlite3.connect(self.db_path)
        cur = con.execute("SELECT title FROM sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
        con.close()
        if row: return row[0]

    def clear_session(self, session_id: str):
        con = sqlite3.connect(self.db_path)
        with con:
            con.execute("DELETE FROM messages WHERE session_id=?", (session_id,))

    def append_message(self, session_id: str, role: str, content: str):
        con = sqlite3.connect(self.db_path)
        with con:
            con.execute("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
                        (session_id, role, content, time.time()))

    def get_messages(self, session_id: str, limit: int = 300) -> List[Dict[str,Any]]:
        con = sqlite3.connect(self.db_path)
        cur = con.execute("SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id ASC LIMIT ?",
                          (session_id, limit))
        rows = [{"role":r[0], "content":r[1], "created_at":r[2]} for r in cur.fetchall()]
        con.close()
        return rows

    def summarize(self, session_id: str, router: LLMRouter, max_turns: int = 12) -> str:
        history = self.get_messages(session_id, limit=2*max_turns)
        if not history: return ""
        transcript = "\n".join([f"{m['role'].title()}: {m['content']}" for m in history[-2*max_turns:]])
        prompt = "Summarize the following chat succinctly for future context. Focus on user goals, constraints, definitions, and decisions.\n\n" + transcript + "\n\nSummary:"
        return router.complete(prompt)

    def export_all_for_session(self, session_id: str):
        con = sqlite3.connect(self.db_path)
        data = {"session": None, "messages": []}
        cur = con.execute("SELECT id, title, user, created_at FROM sessions WHERE id=?", (session_id,))
        row = cur.fetchone()
        if row:
            data["session"] = {"id":row[0],"title":row[1],"user":row[2],"created_at":row[3]}
        cur = con.execute("SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id", (session_id,))
        data["messages"] = [{"role":r[0],"content":r[1],"created_at":r[2]} for r in cur.fetchall()]
        con.close()
        return data
