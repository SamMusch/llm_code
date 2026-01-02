"""
tools.py
- includes: a collection of app-level ops. Can be invoked by other parts of the system.
- each function: a discrete, well-defined task
"""

from pathlib import Path
from .config import Settings
from langchain.tools import tool
from .retriever import load_retriever, build_index


@tool
def read_doc_by_name(
    name: str | None = None,
    cfg: dict | None = None,
    properties: dict | None = None,
    type: str | None = None,  # noqa: A002  (shadows built-in, but matches incoming key)
) -> str:
    """
    Read a document by filename substring.

    Expected args (normal):
      {"name": "<substring>", "cfg": {...}}

    Some local models may incorrectly send a JSON-schema-shaped payload:
      {"type":"object","properties":{"name":"...","cfg":{...}}}
    This function accepts both.
    """
    if name is None and properties:
        # Handle schema-shaped tool call args
        name = properties.get("name")
        cfg = properties.get("cfg") if cfg is None else cfg

    if not name:
        return ""

    _cfg = Settings.load() if cfg is None else Settings(**cfg)
    root = Path(_cfg.docs_dir)
    candidates = list(root.rglob("*"))
    name_l = str(name).lower()

    for p in candidates:
        if p.is_file() and name_l in p.name.lower():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    return ""



@tool
def search_docs(query: str, k: int | None = None) -> str:
    """
    Semantic search over FAISS index; returns top-k doc content.
    Use this to look up information in the local KB.
    """
    retriever = load_retriever(k=k)
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    results: list[str] = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        source = meta.get("source", "unknown")
        results.append(f"[{i}] Source: {source}\n{d.page_content}")
    return "\n\n".join(results)


@tool
def rebuild_index(max_docs: int | None = None) -> str:
    """
    Rebuild the FAISS index over the configured docs_dir.
    Use if index is missing or out-of-date.
    """
    cfg = Settings.load()
    build_index(docs_dir=cfg.docs_dir, max_docs=max_docs)
    return (
        f"Index rebuilt in {cfg.faiss_dir} "
        f"from docs in {cfg.docs_dir} "
        f"(max_docs={max_docs})."
    )