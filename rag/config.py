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
        env = {}
        if "DATA_DIR" in os.environ: env["data_dir"] = Path(os.environ["DATA_DIR"])
        if "DOCS_DIR" in os.environ: env["docs_dir"] = Path(os.environ["DOCS_DIR"])                 # docs_dir
        if "LLM_PROVIDER" in os.environ: env["llm_provider"] = os.environ["LLM_PROVIDER"]
        if "LLM_MODEL" in os.environ: env["llm_model"] = os.environ["LLM_MODEL"]
        if "K" in os.environ: env["k"] = int(os.environ["K"])
        if "EMBEDDING_MODEL" in os.environ: env["embedding_model"] = os.environ["EMBEDDING_MODEL"]
        return cls(**env)