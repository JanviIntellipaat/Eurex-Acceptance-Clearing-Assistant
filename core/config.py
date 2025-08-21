from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import os
from pydantic import BaseModel, Field

# where we persist settings
SETTINGS_PATH = Path(".settings.json")

class Settings(BaseModel):
    # Provider / model
    provider: str = Field(default="ollama")          # "ollama" or "openai"
    model: str = Field(default="llama3.2")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Behavior flags
    strict: bool = False
    show_resources: bool = False                      # <— NEW toggle for sidebar JSON
    max_output_tokens: int = 1024

    # Storage
    kb_dir: str = "data/kb"
    duckdb_path: str = "data/structured.duckdb"
    conv_db_path: str = "data/conversations.sqlite"

    # Embeddings
    embed_backend: str = Field(default_factory=lambda: os.environ.get("EMBED_BACKEND", "bge-large-en-v1.5"))
    embed_dim: int = 1024

    # Optional base URL override (e.g., custom Ollama/OpenAI endpoint)
    base_url: Optional[str] = None

def _ensure_dirs(s: Settings) -> None:
    Path(s.kb_dir).mkdir(parents=True, exist_ok=True)
    Path(s.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    Path(s.conv_db_path).parent.mkdir(parents=True, exist_ok=True)

def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Primary loader used by the app."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            s = Settings(**data)
            _ensure_dirs(s)
            return s
        except Exception:
            pass
    s = Settings()
    _ensure_dirs(s)
    return s

def save_settings(s: Settings, path: Path = SETTINGS_PATH) -> None:
    _ensure_dirs(s)
    path.write_text(s.model_dump_json(indent=2), encoding="utf-8")

# --- Compatibility shim for older imports ---
def get_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Backwards-compatible alias so existing code that imports get_settings keeps working."""
    return load_settings(path)
