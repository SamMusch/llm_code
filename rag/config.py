# config.py
# Loads in settings from config/rag.yaml

from pydantic import BaseModel
from pathlib import Path
import os
import yaml

class Settings(BaseModel):
    # paths
    data_dir: Path
    docs_dir: Path      # directory for documents
    faiss_dir: Path     # directory for FAISS index
    logs_dir: Path      # directory for logs

    # llm
    llm_provider: str
    llm_model: str  # LLM model
    llm_base_url: str | None = None  # optional, for providers like Ollama

    # embedding
    k: int                  # top k docs
    chunk_size: int
    chunk_overlap: int
    embedding_model: str    # embed model from env (e.g., HuggingFace model name)

    # system runtime settings
    max_retries: int
    hallucination_guard: bool
    max_context_chars: int

    # postgres (runtime SQL tools)
    postgres_uri: str | None = None
    postgres_schema: str | None = None

    @classmethod
    def load(cls):
        # 1. Load YAML configuration first
        yaml_path = Path(__file__).resolve().parents[1] / "config" / "rag.yaml"
        yaml_data = {}
        if yaml_path.exists():
            with yaml_path.open("r") as f:
                raw = yaml.safe_load(f) or {}
                yaml_data = raw.get("rag", {})

        cfg_vals: dict[str, object] = {}  # merged config values (YAML first, then env overrides)
        unified_mapping = {
            # paths
            "paths.docs_dir": ("docs_dir", Path),
            "paths.data_dir": ("data_dir", Path),
            "paths.index_dir": ("faiss_dir", Path),
            "paths.logs_dir": ("logs_dir", Path),

            # llm
            "llm.provider": ("llm_provider", str),
            "llm.model": ("llm_model", str),
            "llm.base_url": ("llm_base_url", str),

            # embedding
            "embedding.model": ("embedding_model", str),
            "embedding.chunk_size": ("chunk_size", int),
            "embedding.chunk_overlap": ("chunk_overlap", int),
            "embedding.top_k": ("k", int),

            # runtime
            "runtime.max_retries": ("max_retries", int),
            "runtime.hallucination_guard": ("hallucination_guard", bool),
            "runtime.max_context_chars": ("max_context_chars", int),

            # postgres
            "postgres.uri": ("postgres_uri", str),
            "postgres.schema": ("postgres_schema", str),
        }
        for dotted_key, (env_key, cast) in unified_mapping.items():
            section, key = dotted_key.split(".")
            section_data = yaml_data.get(section, {})
            if key in section_data:
                value = section_data[key]
                if value is None:
                    continue
                cfg_vals[env_key] = cast(value) if cast else value

        env_override_mapping = {
            "DATA_DIR":        ("data_dir", Path),
            "DOCS_DIR":        ("docs_dir", Path),
            "INDEX_DIR":       ("faiss_dir", Path),
            "LOGS_DIR":        ("logs_dir", Path),
            "LLM_PROVIDER":    ("llm_provider", str),  # 260102 for aws 
            "LLM_MODEL":       ("llm_model", str),     # 260102 for aws 
            "LLM_BASE_URL":    ("llm_base_url", str),
            "EMBEDDING_MODEL": ("embedding_model", str),
            "POSTGRES_URI":    ("postgres_uri", str),
            "POSTGRES_SCHEMA": ("postgres_schema", str),
        }
        for env_var, (env_key, cast) in env_override_mapping.items():
            if env_var not in os.environ:
                continue
            raw_value = os.environ.get(env_var, "")
            if raw_value is None:
                continue
            raw_value = raw_value.strip()
            # Do not let empty env vars override YAML defaults.
            if raw_value == "":
                continue
            cfg_vals[env_key] = cast(raw_value) if cast else raw_value

        return cls(**cfg_vals)

# Singleton config instance for convenient import
# Return a singleton Settings instance loaded from rag.yaml + env vars
from functools import lru_cache
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()

cfg = get_settings()