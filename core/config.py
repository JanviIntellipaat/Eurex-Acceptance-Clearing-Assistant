from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import os
from pydantic import BaseModel, Field

SETTINGS_PATH = Path(".settings.json")

class Settings(BaseModel):
    # Provider / model
    provider: str = Field(default="ollama")     # "ollama" or "openai"
    model: str = Field(default="llama3.2")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Behavior flags
    strict: bool = False
    show_resources: bool = False
    max_output_tokens: int = 1024

    # Storage
    kb_dir: str = "data/kb"
    duckdb_path: str = "data/structured.duckdb"
    conv_db_path: str = "data/conversations.sqlite"

    # Embeddings
    embed_backend: str = Field(default_factory=lambda: os.environ.get("EMBED_BACKEND", "ollama:mxbai-embed-large"))
    embed_dim: int = 1024

    # Optional endpoints / keys
    base_url: Optional[str] = None
    openai_api_key: Optional[str] = None

    # ---------- Back-compat shims ----------
    def __init__(self, **data):
        # legacy incoming keys -> new keys
        if "max_tokens" in data and "max_output_tokens" not in data:
            data["max_output_tokens"] = data["max_tokens"]
        if "db_path" in data and "duckdb_path" not in data:
            data["duckdb_path"] = data["db_path"]
        if "sqlite_path" in data and "conv_db_path" not in data:
            data["conv_db_path"] = data["sqlite_path"]
        if "history_db_path" in data and "conv_db_path" not in data:
            data["conv_db_path"] = data["history_db_path"]
        if "kb_path" in data and "kb_dir" not in data:
            data["kb_dir"] = data["kb_path"]
        if "kb_folder" in data and "kb_dir" not in data:
            data["kb_dir"] = data["kb_folder"]
        if "embedding_dim" in data and "embed_dim" not in data:
            data["embed_dim"] = data["embedding_dim"]
        if "vector_dim" in data and "embed_dim" not in data:
            data["embed_dim"] = data["vector_dim"]
        if "openai_key" in data and "openai_api_key" not in data:
            data["openai_api_key"] = data["openai_key"]
        if "api_base" in data and "base_url" not in data:
            data["base_url"] = data["api_base"]
        if "openai_api_base" in data and "base_url" not in data:
            data["base_url"] = data["openai_api_base"]
        if "provider_base_url" in data and "base_url" not in data:
            data["base_url"] = data["provider_base_url"]
        if "ollama_base_url" in data and "base_url" not in data:
            data["base_url"] = data["ollama_base_url"]
        super().__init__(**data)

    # Legacy attribute aliases
    @property
    def max_tokens(self) -> int:  # old -> new
        return self.max_output_tokens
    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self.max_output_tokens = int(value)

    @property
    def db_path(self) -> str:     # old -> new
        return self.duckdb_path
    @db_path.setter
    def db_path(self, value: str) -> None:
        self.duckdb_path = value

    @property
    def sqlite_path(self) -> str:
        return self.conv_db_path
    @sqlite_path.setter
    def sqlite_path(self, value: str) -> None:
        self.conv_db_path = value

    @property
    def history_db_path(self) -> str:
        return self.conv_db_path
    @history_db_path.setter
    def history_db_path(self, value: str) -> None:
        self.conv_db_path = value

    @property
    def kb_path(self) -> str:
        return self.kb_dir
    @kb_path.setter
    def kb_path(self, value: str) -> None:
        self.kb_dir = value

    @property
    def kb_folder(self) -> str:
        return self.kb_dir
    @kb_folder.setter
    def kb_folder(self, value: str) -> None:
        self.kb_dir = value

    @property
    def embedding_dim(self) -> int:
        return self.embed_dim
    @embedding_dim.setter
    def embedding_dim(self, value: int) -> None:
        self.embed_dim = int(value)

    @property
    def vector_dim(self) -> int:
        return self.embed_dim
    @vector_dim.setter
    def vector_dim(self, value: int) -> None:
        self.embed_dim = int(value)

    @property
    def openai_key(self) -> Optional[str]:
        return self.openai_api_key
    @openai_key.setter
    def openai_key(self, value: Optional[str]) -> None:
        self.openai_api_key = value

    @property
    def api_base(self) -> Optional[str]:
        return self.base_url
    @api_base.setter
    def api_base(self, value: Optional[str]) -> None:
        self.base_url = value

    @property
    def openai_api_base(self) -> Optional[str]:
        return self.base_url
    @openai_api_base.setter
    def openai_api_base(self, value: Optional[str]) -> None:
        self.base_url = value

    @property
    def provider_base_url(self) -> Optional[str]:
        return self.base_url
    @provider_base_url.setter
    def provider_base_url(self, value: Optional[str]) -> None:
        self.base_url = value

    @property
    def ollama_base_url(self) -> Optional[str]:
        return self.base_url
    @ollama_base_url.setter
    def ollama_base_url(self, value: Optional[str]) -> None:
        self.base_url = value


def _ensure_dirs(s: Settings) -> None:
    Path(s.kb_dir).mkdir(parents=True, exist_ok=True)
    Path(s.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    Path(s.conv_db_path).parent.mkdir(parents=True, exist_ok=True)

def load_settings(path: Path = SETTINGS_PATH) -> Settings:
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
    data = s.model_dump()
    # also write legacy keys for older code that reads JSON
    data["max_tokens"] = data.get("max_output_tokens", s.max_output_tokens)
    data["db_path"] = data.get("duckdb_path", s.duckdb_path)
    data["sqlite_path"] = data.get("conv_db_path", s.conv_db_path)
    data["history_db_path"] = data.get("conv_db_path", s.conv_db_path)
    data["kb_path"] = data.get("kb_dir", s.kb_dir)
    data["kb_folder"] = data.get("kb_dir", s.kb_dir)
    data["embedding_dim"] = data.get("embed_dim", s.embed_dim)
    data["vector_dim"] = data.get("embed_dim", s.embed_dim)
    data["openai_key"] = data.get("openai_api_key", s.openai_api_key)
    data["api_base"] = data.get("base_url", s.base_url)
    data["openai_api_base"] = data.get("base_url", s.base_url)
    data["provider_base_url"] = data.get("base_url", s.base_url)
    data["ollama_base_url"] = data.get("base_url", s.base_url)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

# Back-compat alias
def get_settings(path: Path = SETTINGS_PATH) -> Settings:
    return load_settings(path)

def save_settings_to_env(s: Settings) -> None:
    """Push selected settings into current process env (not permanent)."""
    if s.embed_backend:
        os.environ["EMBED_BACKEND"] = s.embed_backend
    if s.openai_api_key:
        os.environ["OPENAI_API_KEY"] = s.openai_api_key
    if s.base_url:
        os.environ["PROVIDER_BASE_URL"] = s.base_url
        os.environ["OPENAI_BASE_URL"] = s.base_url
        os.environ["OLLAMA_BASE_URL"] = s.base_url

def load_settings_from_env(s: Settings | None = None) -> Settings:
    s = s or load_settings()
    s.embed_backend = os.environ.get("EMBED_BACKEND", s.embed_backend)
    s.openai_api_key = os.environ.get("OPENAI_API_KEY", s.openai_api_key)
    s.base_url = (
        os.environ.get("PROVIDER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OLLAMA_BASE_URL")
        or s.base_url
    )
    return s
