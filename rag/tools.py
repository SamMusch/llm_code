"""
tools.py
- includes: a collection of app-level ops. Can be invoked by other parts of the system.
- each function: a discrete, well-defined task
"""

from pathlib import Path
from langchain_core.tools import tool
from .config import Settings

@tool
def read_doc_by_name(name: str, cfg: dict | None = None) -> str:
    """flow: load config ---> scan docs files ---> match filename --> return text of 1st first matching file"""
    _cfg = Settings.load() if cfg is None else Settings(**cfg)
    root = Path(_cfg.docs_dir)
    candidates = list(root.rglob("*"))
    name_l = name.lower()
    for p in candidates:
        if p.is_file() and name_l in p.name.lower():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return ""