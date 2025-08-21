from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import os
from pydantic import BaseModel, Field

# Persisted settings path
SETTINGS_PATH = Path(".settings.json")

class Settings(BaseModel):
    # Provider / model
    provider: str = Field(default="ollama")          # "ollama" or "openai"
    model: str = Field(default="llama3.2")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Behavior flags
    strict: bool = False
    show_resources: bool = False                     # sidebar toggle to show resources JSON
    max_output_tokens: int = 1024

    # Storage
    kb_dir: str = "data/kb"
    duckdb_path: str = "data/structured.duckdb"
    conv_db_path: str = "data/conversations.sqlite"

    # Embeddings
    # default from env so you can set EMBED_BACKEND in ~/.bashrc
    embed_backend: str = Field(default_factory=lambda: os.environ.get("EMBED_BACKEND", "bge-large-en-v1.5"))
    embed_dim: int = 1024

    # Optional endpoints / keys
    base_url: Optional[str] = None                   # custom provider base URL (Ollama/OpenAI proxies, etc.)
    openai_api_key: Optional[str] = None             # optional: store key here if you set it in the UI

def _ensure_dirs(s: Settings) -> None:
    Path(s.kb_dir).mkdir(parents=True, exist_ok=True)
    Path(s.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    Path(s.conv_db_path).parent.mkdir(parents=True, exist_ok=True)

def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Primary loader used by the app; falls back to defaults."""
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
    """Persist settings to .settings.json"""
    _ensure_dirs(s)
    path.write_text(s.model_dump_json(indent=2), encoding="utf-8")

# --- Compatibility shims for older imports in the app ---

def get_settings(path: Path = SETTINGS_PATH) -> Settings:
    """Backwards-compatible alias to load settings."""
    return load_settings(path)

def save_settings_to_env(s: Settings) -> None:
    """
    Backwards-compatible helper: push selected settings into environment vars.
    Safe no-op if values are None. This does NOT write ~/.bashrc; it only
    updates the current process env so downstream clients can read them.
    """
    # Embeddings backend (used by our embedding wrapper)
    if s.embed_backend:
        os.environ["EMBED_BACKEND"] = s.embed_backend

    # OpenAI key (used if provider == "openai")
    if s.openai_api_key:
        os.environ["OPENAI_API_KEY"] = s.openai_api_key

    # Optional custom base URL for either provider
    if s.base_url:
        os.environ["PROVIDER_BASE_URL"] = s.base_url
        # Provide common aliases some SDKs look for (harmless if unused)
        os.environ["OPENAI_BASE_URL"] = s.base_url
        os.environ["OLLAMA_BASE_URL"] = s.base_url

def load_settings_from_env(s: Settings | None = None) -> Settings:
    """
    Optional helper: overlay current environment into a Settings object.
    Useful if you're launching with env vars and want the UI to reflect them.
    """
    s = s or load_settings()
    s.embed_backend = os.environ.get("EMBED_BACKEND", s.embed_backend)
    s.base_url = os.environ.get("PROVIDER_BASE_URL", s.base_url) or os.environ.get("OPENAI_BASE_URL", s.base_url) or os.environ.get("OLLAMA_BASE_URL", s.base_url)
    s.openai_api_key = os.environ.get("OPENAI_API_KEY", s.openai_api_key)
    return s
