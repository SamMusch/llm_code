"""
tools.py
- includes: a collection of app-level ops. Can be invoked by other parts of the system.
- each function: a discrete, well-defined task
"""

from pathlib import Path
import logging
import os

from langchain.tools import tool
from sqlalchemy.engine import Engine

from .config import Settings
from .retriever import load_retriever, build_index

from functools import lru_cache
from typing import Optional


logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _cached_pg_engine(uri: str, schema: Optional[str]) -> Engine:
    """Create (and cache) a small SQLAlchemy engine.

    We avoid LangChain SQLDatabaseToolkit here because toolkit/database initialization can
    trigger reflection/introspection that spikes memory in small ECS tasks.
    """
    from sqlalchemy import create_engine

    # Normalize common DSN formats.
    # - SQLAlchemy expects an explicit driver when using psycopg v3.
    # - Our app may also pass a plain libpq DSN (postgresql:// / postgres://) that works with psycopg.connect().
    if isinstance(uri, str):
        if uri.startswith("postgresql://"):
            uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
        elif uri.startswith("postgres://"):
            uri = uri.replace("postgres://", "postgresql+psycopg://", 1)

    # Fast fail + server-side statement timeout.
    # Note: connect_timeout is libpq seconds; statement_timeout is ms.
    opts = "-c statement_timeout=15000"
    if schema:
        opts = f"-c search_path={schema} -c statement_timeout=15000"

    return create_engine(
        uri,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 5,
            "options": opts,
        },
    )


def _invalidate_pg_engine_cache() -> None:
    """Clear cached engines (useful for local dev / hot reload)."""
    try:
        _cached_pg_engine.cache_clear()
    except Exception:
        pass


def _get_postgres_uri(cfg: Settings | None = None) -> str | None:
    """Return a Postgres URI (SQLAlchemy or psycopg DSN), if configured."""
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
    """Return lightweight Postgres tools.

    IMPORTANT: Do NOT use SQLDatabaseToolkit/SQLDatabase here; their init can trigger
    reflection-heavy work that OOMs small ECS tasks.

    We expose only `sql_db_query` (and an alias `sql_db_query_checker`). Listing schemas/tables
    should be done via information_schema queries through sql_db_query.
    """
    if not enabled:
        return []

    uri = _get_postgres_uri(cfg)
    if not uri:
        return []

    schema = os.environ.get("POSTGRES_SCHEMA")
    if not schema and cfg is not None and hasattr(cfg, "postgres_schema"):
        v = getattr(cfg, "postgres_schema")
        schema = v if isinstance(v, str) and v else None

    try:
        engine = _cached_pg_engine(uri, schema)
    except Exception as e:
        logger.exception("Failed to create Postgres engine; disabling SQL tools for this request", exc_info=e)
        _invalidate_pg_engine_cache()
        return []

    @tool("sql_db_query")
    def sql_db_query(query: str) -> str:
        """Run a READ-ONLY SQL query against Postgres and return a small result set."""
        q = (query or "").strip().rstrip(";")
        if not q:
            return "Empty query."

        # Block common write/ddl verbs.
        lowered = q.lower()
        blocked = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "truncate ")
        if any(b in lowered for b in blocked):
            return "Blocked: write/DDL statements are not allowed."

        # Only allow single statement.
        if ";" in q:
            return "Only a single SQL statement is allowed."

        from sqlalchemy import text

        try:
            with engine.connect() as conn:
                res = conn.execute(text(q))
                if not res.returns_rows:
                    return "Query executed. No rows returned."
                cols = list(res.keys())
                rows = res.fetchmany(200)
        except Exception as e:
            return f"SQL error: {e}"

        if not rows:
            return "(no rows)"

        out = [" | ".join(cols), " | ".join(["---"] * len(cols))]
        for r in rows:
            out.append(" | ".join([str(v) if v is not None else "" for v in r]))
        if len(rows) == 200:
            out.append("(truncated to 200 rows)")
        return "\n".join(out)

    @tool("sql_db_query_checker")
    def sql_db_query_checker(query: str) -> str:
        """Lightweight checker: returns the query back (no-op).

        Kept for compatibility with prompts/tooling that expect this tool.
        """
        return (query or "").strip()

    return [sql_db_query, sql_db_query_checker]


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

    from sqlalchemy import text

    engine = _cached_pg_engine(uri, schema)
    with engine.connect() as conn:
        rows = conn.execute(text(q)).fetchall()
        if not rows:
            return "(no rows)"
        out_lines = []
        for r in rows:
            out_lines.append("\t".join(str(x) for x in r))
        return "\n".join(out_lines)


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