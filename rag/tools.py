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
from typing import Any, Optional



logger = logging.getLogger(__name__)

# ----
# Tool reliability helpers
# LangChain/LangGraph agents work best when tools return a value (even on failure)
# instead of raising exceptions (which triggers retries and opaque "tool error" UX).


def _tool_error(tool_name: str, msg: str, hint: str | None = None) -> str:
    out = f"ERROR[{tool_name}]: {msg}"
    if hint:
        out += f"\nHINT: {hint}"
    return out


def _safe_settings(cfg: Any | None) -> Settings:
    """Best-effort Settings loader.

    Accepts:
      - None: Settings.load()
      - Settings: returned as-is
      - dict: Settings(**cfg) with fallback to Settings.load() if invalid

    Never raises.
    """
    if cfg is None:
        try:
            return Settings.load()
        except Exception:
            # Absolute fallback: instantiate with defaults
            return Settings()

    if isinstance(cfg, Settings):
        return cfg

    if isinstance(cfg, dict):
        try:
            return Settings(**cfg)
        except Exception:
            try:
                return Settings.load()
            except Exception:
                return Settings()

    # Unknown type
    try:
        return Settings.load()
    except Exception:
        return Settings()


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

    This tool never raises: on failure it returns an ERROR[...] string.
    """
    tool_name = "run_sql_query"

    try:
        q = (query or "").strip().rstrip(";")
        if not q:
            return _tool_error(tool_name, "Empty query.")

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

        try:
            engine = _cached_pg_engine(uri, schema)
            with engine.connect() as conn:
                rows = conn.execute(text(q)).fetchall()
        except Exception as e:
            logger.exception("run_sql_query failed", exc_info=e)
            return _tool_error(tool_name, f"SQL error: {e}")

        if not rows:
            return "(no rows)"

        out_lines = []
        for r in rows:
            out_lines.append("\t".join(str(x) for x in r))
        return "\n".join(out_lines)

    except Exception as e:
        logger.exception("run_sql_query unexpected failure", exc_info=e)
        return _tool_error(tool_name, str(e))



@tool
def read_doc_by_name(
    name: str | None = None,
    cfg: dict | None = None,
    properties: dict | None = None,
    type: str | None = None,  # noqa: A002  (shadows built-in, but matches incoming key)
) -> str:
    """Read a document by filename substring.

    Returns the file contents as text.

    Notes:
    - This tool is intentionally permissive about argument shapes because some local models
      send JSON-schema-shaped payloads.
    - This tool never raises: on failure it returns an ERROR[...] string.

    Expected args (normal):
      {"name": "<substring>", "cfg": {...}}

    Some local models may incorrectly send a JSON-schema-shaped payload:
      {"type":"object","properties":{"name":"...","cfg":{...}}}
    """
    tool_name = "read_doc_by_name"

    try:
        if name is None and properties:
            # Handle schema-shaped tool call args
            name = properties.get("name")
            cfg = properties.get("cfg") if cfg is None else cfg

        if not name or not str(name).strip():
            return _tool_error(tool_name, "Missing 'name' argument.", hint="Provide a filename substring.")

        settings = _safe_settings(cfg)
        docs_dir = getattr(settings, "docs_dir", None)
        if not docs_dir:
            return _tool_error(tool_name, "docs_dir is not configured.")

        root = Path(docs_dir)
        if not root.exists() or not root.is_dir():
            return _tool_error(
                tool_name,
                f"docs_dir does not exist or is not a directory: {root}",
                hint="Check Settings.docs_dir / DOCS_DIR and ensure documents are mounted in this path.",
            )

        name_l = str(name).lower().strip()

        # Avoid scanning enormous trees blindly. Cap candidates.
        max_candidates = int(os.getenv("READ_DOC_MAX_CANDIDATES", "20000"))
        checked = 0

        for p in root.rglob("*"):
            if checked >= max_candidates:
                break
            if not p.is_file():
                continue
            checked += 1
            if name_l in p.name.lower():
                try:
                    return p.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    logger.exception("read_doc_by_name failed to read %s", p, exc_info=e)
                    return _tool_error(tool_name, f"Failed to read file: {p.name}")

        if checked >= max_candidates:
            return _tool_error(
                tool_name,
                f"No match found after scanning {max_candidates} files.",
                hint="Try a more specific filename substring.",
            )

        return ""

    except Exception as e:
        logger.exception("read_doc_by_name unexpected failure", exc_info=e)
        return _tool_error(tool_name, str(e))



@tool
def search_docs(query: str, k: int | None = None, filters: dict | None = None) -> str:
    """Semantic search over the FAISS index; returns top-k doc content.

    This tool never raises: on failure it returns an ERROR[...] string.

    Args:
      query: search query string
      k: number of results (default 6)
      filters: optional metadata filters. Supported (post-filters):
        - title: substring match (case-insensitive)
        - tags: list of tags; all must be present

    Returns:
      A formatted string with sources + snippets, or "No relevant documents found.".
    """
    tool_name = "search_docs"

    try:
        q = (query or "").strip()
        if not q:
            return _tool_error(tool_name, "Empty query.", hint="Provide a non-empty search query.")

        # FAISS metadata filters are exact-match oriented. For UX like "Title contains",
        # we fetch a larger candidate set and apply deterministic post-filters here.
        k_eff = int(k) if isinstance(k, int) and k > 0 else 6

        faiss_filters: dict | None = None
        post_title: str | None = None
        post_tags: list[str] | None = None

        if filters and isinstance(filters, dict):
            # Title is treated as substring match (post-filter).
            v = filters.get("title")
            if isinstance(v, str) and v.strip():
                post_title = v.strip().lower()

            # Tags are also enforced deterministically here (so we don't rely on FAISS filter semantics for lists).
            t = filters.get("tags")
            if isinstance(t, list) and t:
                post_tags = [str(x).strip() for x in t if str(x).strip()]

            # If you later add exact-match fields you want FAISS to handle, put them into faiss_filters.
            faiss_filters = None

        # When post-filtering, fetch more candidates so we can still return k_eff results after filtering.
        fetch_k = max(k_eff * 50, 200) if (post_title or post_tags) else None

        try:
            retriever = load_retriever(k=k_eff, filters=faiss_filters, fetch_k=fetch_k)
        except Exception as e:
            logger.exception("search_docs failed to load retriever", exc_info=e)
            return _tool_error(
                tool_name,
                "Retriever/index is unavailable.",
                hint="If this is first run or index is missing, run rebuild_index. Also verify docs_dir/faiss_dir paths.",
            )

        try:
            docs = retriever.invoke(q)
        except Exception as e:
            logger.exception("search_docs retriever.invoke failed", exc_info=e)
            return _tool_error(tool_name, "Retriever failed during search.", hint=str(e))

        if docs and (post_title or post_tags):
            filtered = []
            for d in docs:
                meta = getattr(d, "metadata", {}) or {}

                if post_title:
                    title = str(meta.get("title", "")).lower()
                    if post_title not in title:
                        continue

                if post_tags:
                    tags = meta.get("tags")
                    if isinstance(tags, str):
                        tags_list = [tags]
                    elif isinstance(tags, list):
                        tags_list = [str(x) for x in tags]
                    else:
                        tags_list = []

                    tags_set = {t.lower() for t in tags_list}
                    if any(t.lower() not in tags_set for t in post_tags):
                        continue

                filtered.append(d)

            docs = filtered[:k_eff]

        if not docs:
            return "No relevant documents found."

        results: list[str] = []
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            source = meta.get("source", "unknown")
            results.append(f"[{i}] Source: {source}\n{d.page_content}")

        return "\n\n".join(results)

    except Exception as e:
        logger.exception("search_docs unexpected failure", exc_info=e)
        return _tool_error(tool_name, str(e))



@tool
def rebuild_index(max_docs: int | None = None) -> str:
    """Rebuild the FAISS index over the configured docs_dir.

    This tool never raises: on failure it returns an ERROR[...] string.
    """
    tool_name = "rebuild_index"

    try:
        cfg = Settings.load()
        build_index(docs_dir=cfg.docs_dir, max_docs=max_docs)
        return (
            f"Index rebuilt in {cfg.faiss_dir} "
            f"from docs in {cfg.docs_dir} "
            f"(max_docs={max_docs})."
        )
    except Exception as e:
        logger.exception("rebuild_index failed", exc_info=e)
        return _tool_error(tool_name, str(e), hint="Verify docs_dir exists and is readable; verify faiss_dir is writable.")