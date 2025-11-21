"""
config.py

class Settings(BaseModel) ---> defines a data model using Pydantic.
- validates data types / provides default values / allows serialization
- class attributes becomes fields w validation & metadata.

def load() ---> builds a Settings instance
"""

from pydantic import BaseModel, Field
from pathlib import Path
import os, yaml

class Settings(BaseModel):
    # paths
    data_dir: Path = Field(default=Path("Data"))
    docs_dir: Path = Field(default=Path("Data/docs"))               # docs_dir
    faiss_dir: Path = Field(default=Path("Data/faiss_index"))
    
    # llm
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")                   # LLM model
    
    # embedding
    k: int = Field(default=4)  # top k docs
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)
    embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2")  # embed model from env (huggingface)

    # system runtime settings
    max_retries: int = Field(default=3)
    hallucination_guard: bool = Field(default=True)     # graph.py
    max_context_chars: int = Field(default=60_000)      # graph.py

    @classmethod
    def load(cls):
        # 1. Load YAML first
        yaml_path = Path(__file__).resolve().parents[1] / "config" / "rag.yaml"
        yaml_data = {}
        if yaml_path.exists():
            with yaml_path.open("r") as f:
                raw = yaml.safe_load(f) or {}
                yaml_data = raw.get("rag", {})

        # Map YAML → env dict
        env = {}
        paths = yaml_data.get("paths", {})
        if "docs_dir" in paths:
            env["docs_dir"] = Path(paths["docs_dir"])
        if "data_dir" in paths:
            env["data_dir"] = Path(paths["data_dir"])
        if "index_dir" in paths:
            env["faiss_dir"] = Path(paths["index_dir"])

        llm = yaml_data.get("llm", {})
        if "provider" in llm:
            env["llm_provider"] = llm["provider"]
        if "model" in llm:
            env["llm_model"] = llm["model"]

        embedding = yaml_data.get("embedding", {})
        if "model" in embedding:
            env["embedding_model"] = embedding["model"]
        if "chunk_size" in embedding:
            env["chunk_size"] = embedding["chunk_size"]
        if "chunk_overlap" in embedding:
            env["chunk_overlap"] = embedding["chunk_overlap"]
        if "top_k" in embedding:
            env["k"] = embedding["top_k"]

        runtime = yaml_data.get("runtime", {})
        if "max_retries" in runtime:
            env["max_retries"] = runtime["max_retries"]
        if "hallucination_guard" in runtime:
            env["hallucination_guard"] = runtime["hallucination_guard"]
        if "max_context_chars" in runtime:
            env["max_context_chars"] = runtime["max_context_chars"]

        if "DATA_DIR" in os.environ: env["data_dir"] = Path(os.environ["DATA_DIR"])
        if "DOCS_DIR" in os.environ: env["docs_dir"] = Path(os.environ["DOCS_DIR"])                 # docs_dir
        if "INDEX_DIR" in os.environ: env["faiss_dir"] = Path(os.environ["INDEX_DIR"])
        if "LLM_PROVIDER" in os.environ: env["llm_provider"] = os.environ["LLM_PROVIDER"]
        if "LLM_MODEL" in os.environ: env["llm_model"] = os.environ["LLM_MODEL"]
        if "K" in os.environ: env["k"] = int(os.environ["K"])
        if "EMBEDDING_MODEL" in os.environ: env["embedding_model"] = os.environ["EMBEDDING_MODEL"]
        return cls(**env)
    


from functools import lru_cache
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a singleton Settings instance loaded from rag.yaml + env vars.
    """
    return Settings.load()
cfg = get_settings()       # Canonical config object for convenience imports