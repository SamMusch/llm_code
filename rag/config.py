# config.py
# Loads in settings from config/rag.yaml

from pydantic import BaseModel
from pathlib import Path
import os, yaml

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

    @classmethod
    def load(cls):
        # 1. Load YAML configuration first
        yaml_path = Path(__file__).resolve().parents[1] / "config" / "rag.yaml"
        yaml_data = {}
        if yaml_path.exists():
            with yaml_path.open("r") as f:
                raw = yaml.safe_load(f) or {}
                yaml_data = raw.get("rag", {})

        env = {}    # Map YAML → env dict
        unified_mapping = {
            "paths.docs_dir":      ("docs_dir", Path),      # paths
            "paths.data_dir":      ("data_dir", Path),
            "paths.index_dir":     ("faiss_dir", Path),
            "paths.logs_dir":      ("logs_dir", Path),

            "llm.provider":        ("llm_provider", None),  # llm
            "llm.model":           ("llm_model", None),
            "llm.base_url":        ("llm_base_url", None),
            
            "embedding.model":        ("embedding_model", None),  # embedding
            "embedding.chunk_size":   ("chunk_size", None),
            "embedding.chunk_overlap":("chunk_overlap", None),
            "embedding.top_k":        ("k", None),

            "runtime.max_retries":      ("max_retries", None),      # runtime
            "runtime.hallucination_guard": ("hallucination_guard", None),
            "runtime.max_context_chars":   ("max_context_chars", None),}
        for dotted_key, (env_key, cast) in unified_mapping.items():
            section, key = dotted_key.split(".")
            section_data = yaml_data.get(section, {})
            if key in section_data:
                value = section_data[key]
                env[env_key] = cast(value) if cast else value

        env_override_mapping = {
            "DATA_DIR":        ("data_dir", Path),
            "DOCS_DIR":        ("docs_dir", Path),
            "INDEX_DIR":       ("faiss_dir", Path),
            "LOGS_DIR":        ("logs_dir", Path),
            "LLM_PROVIDER":    ("llm_provider", str),  # 260102 for aws 
            "LLM_MODEL":       ("llm_model", str),     # 260102 for aws 
            "LLM_BASE_URL":    ("llm_base_url", str),
            "EMBEDDING_MODEL": ("embedding_model", str),
        }
        for env_var, (env_key, cast) in env_override_mapping.items():
            if env_var in os.environ:
                raw_value = os.environ[env_var]
                env[env_key] = cast(raw_value) if cast else raw_value

        return cls(**env)

# Singleton config instance for convenient import
# Return a singleton Settings instance loaded from rag.yaml + env vars
from functools import lru_cache
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()

cfg = get_settings()