"""
tools.py
- includes: a collection of app-level ops. Can be invoked by other parts of the system.
- each function: a discrete, well-defined task
"""

from pathlib import Path
import logging
import os

from langchain.tools import tool

from .config import Settings
from .retriever import load_retriever, build_index


logger = logging.getLogger(__name__)


def _get_postgres_uri(cfg: Settings | None = None) -> str | None:
    """Return a SQLAlchemy-compatible Postgres URI, if configured."""
    # Prefer explicit env var so local/dev/prod can share the same code.
    uri = os.environ.get("POSTGRES_URI") or os.environ.get("DATABASE_URL")
    if uri:
        return uri

    # Optional: allow passing via Settings if you later add a field.
    if cfg is not None and hasattr(cfg, "postgres_uri"):
        v = getattr(cfg, "postgres_uri")
        return v if isinstance(v, str) and v else None

    return None


def get_sql_database_tools(llm, cfg: Settings | None = None, enabled: bool = False):
    """Build LangChain SQLDatabaseToolkit tools for the configured Postgres DB.

    This follows LangChain's SQLDatabase toolkit pattern.
    If Postgres isn't configured, returns an empty list.

    IMPORTANT:
    - This must be fail-open (return []) if Postgres is temporarily unreachable,
      so chat streaming does not crash.
    """
    if not enabled:
        return []

    uri = _get_postgres_uri(cfg)
    if not uri:
        return []

    # Local import so Postgres deps are optional unless enabled.
    from sqlalchemy.exc import OperationalError
    from langchain_community.utilities.sql_database import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit

    # Schema scoping (preferred). DB-level enforcement should also be applied.
    schema = os.environ.get("POSTGRES_SCHEMA")
    if not schema and cfg is not None and hasattr(cfg, "postgres_schema"):
        v = getattr(cfg, "postgres_schema")
        schema = v if isinstance(v, str) and v else None

    # Avoid LangChain/psycopg issue when `schema=` triggers `SET search_path TO %s`.
    # Instead, set search_path at connection time via libpq options.
    engine_args = {}
    if schema:
        engine_args = {"connect_args": {"options": f"-csearch_path={schema}"}}

    try:
        db = SQLDatabase.from_uri(uri, engine_args=engine_args)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        return toolkit.get_tools()
    except OperationalError as e:
        logger.exception("Postgres unavailable; disabling SQL tools for this request", exc_info=e)
        return []
    except Exception as e:
        # Defensive: do not let SQL tool init crash /chat/stream.
        logger.exception("Failed to initialize SQL tools; disabling SQL tools for this request", exc_info=e)
        return []


@tool
def run_sql_query(query: str) -> str:
    """Execute a read-only SQL query against Postgres and return rows.

    Guardrails:
    - Only allows SELECT / WITH queries.
    - Blocks multiple statements.
    - Applies a hard LIMIT if none is present.

    This tool exists to provide a stable, explicit interface for local models that
    sometimes fail to call `sql_db_query` correctly.
    """
    q = (query or "").strip().rstrip(";")
    if not q:
        return ""

    q_low = q.lower().strip()
    if not (q_low.startswith("select") or q_low.startswith("with")):
        return "Only read-only SELECT/WITH queries are allowed."

    # block multiple statements
    if ";" in q:
        return "Only a single SQL statement is allowed."

    # Add a conservative LIMIT if missing
    if " limit " not in f" {q_low} ":
        q = f"{q} LIMIT 100"

    cfg = Settings.load()
    uri = _get_postgres_uri(cfg)
    if not uri:
        return "Postgres is not configured (missing POSTGRES_URI)."

    schema = os.environ.get("POSTGRES_SCHEMA") or getattr(cfg, "postgres_schema", None)

    from sqlalchemy import create_engine, text

    connect_args = {}
    if schema:
        connect_args = {"options": f"-csearch_path={schema}"}

    engine = create_engine(uri, connect_args=connect_args, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(q)).fetchall()
            if not rows:
                return "(no rows)"
            # Render as TSV for readability
            out_lines = []
            for r in rows:
                out_lines.append("\t".join(str(x) for x in r))
            return "\n".join(out_lines)
    finally:
        engine.dispose()


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