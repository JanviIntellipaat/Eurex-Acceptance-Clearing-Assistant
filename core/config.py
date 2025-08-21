
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field
import os

class Settings(BaseModel):
    provider: str = Field(default="ollama")
    model: str = Field(default="llama3.2")
    base_url: str | None = Field(default=None)
    temperature: float = 0.1
    strict: bool = False  # default OFF per your request
    max_tokens: int = 2048
    kb_dir: str = "data/kb"
    db_path: str = "data/conversations.sqlite"
    duckdb_path: str = "data/structured.duckdb"
    embed_backend: str = "ollama:mxbai-embed-large"
    embed_dim: int = 1024

def get_settings() -> "Settings":
    cfg = Path(".settings.json")
    if cfg.exists():
        try:
            return Settings.model_validate_json(cfg.read_text())
        except Exception:
            pass
    return Settings(
        provider=os.getenv("LLM_PROVIDER","ollama"),
        model=os.getenv("LLM_MODEL","llama3.2"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=float(os.getenv("LLM_TEMPERATURE","0.1")),
        strict=os.getenv("STRICT_MODE","false").lower() in ("1","true","yes"),
        max_tokens=int(os.getenv("MAX_TOKENS","2048")),
        kb_dir=os.getenv("KB_DIR","data/kb"),
        db_path=os.getenv("DB_PATH","data/conversations.sqlite"),
        duckdb_path=os.getenv("DUCKDB_PATH","data/structured.duckdb"),
        embed_backend=os.getenv("EMBED_BACKEND","ollama:mxbai-embed-large"),
        embed_dim=int(os.getenv("EMBED_DIM","1024"))
    )

def save_settings_to_env(s: Settings, openai_key: str | None = None):
    Path(".settings.json").write_text(s.model_dump_json())
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
