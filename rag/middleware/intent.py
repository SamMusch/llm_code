

from __future__ import annotations

from typing import Any, Dict, List, Literal

from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import BaseMessage, HumanMessage


Intent = Literal["sql", "docs", "mixed", "unknown"]


def last_human_text(messages: List[BaseMessage]) -> str:
    """Return the content of the most recent HumanMessage, or ""."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def classify_intent(q: str) -> Intent:
    """Heuristic intent classification for routing tools.

    Keep this conservative: prefer docs-first when uncertain.
    """
    ql = q.lower()

    sql_terms = [
        "postgres",
        "pgadmin",
        "sql",
        "schema",
        "schemas",
        "table",
        "tables",
        "column",
        "columns",
        "database",
        "query",
        "select ",
        "join ",
        "group by",
        "where ",
        "llm_code.",
        "spend",
        "click",
        "clicks",
        "metric",
        "metrics",
        "campaign",
    ]
    doc_terms = [
        "doc",
        "docs",
        "document",
        "documents",
        "note",
        "notes",
        "markdown",
        "pdf",
        "ppt",
        "pptx",
        "slide",
        "slides",
        "xlsx",
        "sheet",
        "file",
        "folder",
        "read",
        "find",
        "search",
        "in the docs",
        "in my notes",
        "dataset",
        "data set",
        "what did it cover",
        "coverage",
        "project",
    ]

    has_sql = any(t in ql for t in sql_terms)
    has_doc = any(t in ql for t in doc_terms)

    if has_sql and has_doc:
        return "mixed"
    if has_sql:
        return "sql"
    if has_doc:
        return "docs"
    return "unknown"


def intent_instructions(intent: Intent) -> str:
    if intent == "sql":
        return (
            "Intent=SQL. Prefer LIGHTWEIGHT SQL queries via sql_db_query. "
            "For listing schemas/tables/columns, query information_schema/pg_catalog (e.g., information_schema.schemata, information_schema.tables, information_schema.columns). "
            "Avoid sql_db_schema unless the user explicitly asks for full schema details. "
            "You may also use sql_db_list_tables. "
            "Do NOT use document tools unless the user explicitly asks about documents. "
            "Never answer database questions from memory; compute via tools."
        )
    if intent == "docs":
        return (
            "Intent=DOCS. Use ONLY document tools (search_docs, read_doc_by_name, rebuild_index). "
            "Do NOT use SQL tools unless the user explicitly asks for database facts/metrics."
        )
    if intent == "mixed":
        return (
            "Intent=MIXED. You may use BOTH document tools and SQL tools. "
            "When you need DB facts, prefer sql_db_query with lightweight information_schema/pg_catalog queries; avoid sql_db_schema unless explicitly requested. "
            "First gather evidence (docs search/read and/or DB queries), then synthesize a single final answer."
        )
    return (
        "Intent=UNKNOWN. Default docs-first: call search_docs for relevant notes/files before asking clarifying questions. "
        "If the user clearly asks for database facts/metrics, switch to SQL tools."
    )


@before_model
def intent_router_hints(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Inject per-turn tool-use guidance based on the most recent HumanMessage."""
    messages = state.get("messages", [])
    if not messages:
        return None

    q = last_human_text(messages)
    if not q:
        return None

    intent = classify_intent(q)
    guidance = intent_instructions(intent)

    return {
        "messages": messages
        + [
            {
                "role": "system",
                "content": guidance,
            }
        ]
    }