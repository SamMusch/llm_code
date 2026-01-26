from __future__ import annotations

from typing import Any, Dict

from langchain.agents.middleware import AgentState, before_model

from .context import has_retrieved_sources
from .intent import Intent, classify_intent, last_human_text


# autosearch.py ---> Auto-retrieval logic. Triggers search when user intent is unclear or context is missing.
#   - docs_first_autosearch() ---> auto-call search_docs for docs/unknown intent when no retrieval yet
#   - force_list_tables() ---> for "list tables" requests, force a lightweight SQL query


@before_model
def docs_first_autosearch(state: AgentState, runtime) -> Dict[str, Any] | None:
    """If user intent is docs/unknown and nothing has been retrieved yet, auto-call search_docs once.

    Controlled via env var:
      - DOCS_FIRST_AUTOSEARCH=true|false
    """
    import os

    messages = state.get("messages", [])
    if not messages:
        return None

    enabled = os.environ.get("DOCS_FIRST_AUTOSEARCH", "true").lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return None

    # If the Database toolset is enabled for this request, avoid auto-calling search_docs
    # for ambiguous/unknown intent. This prevents unnecessary doc retrieval on DB questions.
    try:
        cfg = getattr(runtime, "config", None) or {}
        configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
        selected = configurable.get("selected_tools") or []
        if any(str(t).lower() == "database" for t in selected):
            intent_now: Intent = classify_intent(last_human_text(messages) or "")
            if intent_now in {"unknown", "mixed", "sql"}:
                return None
    except Exception:
        pass

    human_q = last_human_text(messages)
    if not human_q:
        return None

    if has_retrieved_sources(messages):
        return None

    intent: Intent = classify_intent(human_q)
    if intent not in {"docs", "unknown"}:
        return None

    return {
        "messages": messages
        + [
            {
                "role": "assistant",
                "content": "Calling search_docs now.",
                "tool_calls": [
                    {"id": "docs_first_autosearch_0", "name": "search_docs", "args": {"query": human_q}},
                ],
            }
        ]
    }


@before_model
def force_list_tables(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Force a lightweight information_schema query for common 'list tables' requests.

    We intentionally avoid sql_db_list_tables/sql_db_schema because they can trigger
    heavy reflection and memory spikes. Listing tables via sql_db_query is cheaper.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    human = last_human_text(messages) or ""
    text = human.lower().strip()

    triggers = [
        "list available tables",
        "list tables",
        "show tables",
        "what tables",
        "available tables",
    ]

    if any(t in text for t in triggers):
        q = (
            "SELECT table_schema, table_name "
            "FROM information_schema.tables "
            "WHERE table_type='BASE TABLE' "
            "AND table_schema NOT IN ('pg_catalog','information_schema') "
            "ORDER BY table_schema, table_name;"
        )
        return {
            "messages": messages
            + [
                {
                    "role": "assistant",
                    "content": "Calling sql_db_query to list tables via information_schema.",
                    "tool_calls": [
                        {"id": "force_list_tables_0", "name": "sql_db_query", "args": {"query": q}},
                    ],
                }
            ]
        }

    return None


@before_model
def force_list_schemas(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Force a lightweight information_schema query for common 'list schemas' requests."""
    messages = state.get("messages", [])
    if not messages:
        return None

    human = last_human_text(messages) or ""
    text = human.lower().strip()

    triggers = [
        "list available schemas",
        "list schemas",
        "show schemas",
        "what schemas",
        "available schemas",
        "list schema",
        "show schema",
    ]

    if any(t in text for t in triggers):
        q = (
            "SELECT schema_name "
            "FROM information_schema.schemata "
            "WHERE schema_name NOT IN ('pg_catalog','information_schema') "
            "ORDER BY schema_name;"
        )
        return {
            "messages": messages
            + [
                {
                    "role": "assistant",
                    "content": "Calling sql_db_query to list schemas via information_schema.",
                    "tool_calls": [
                        {"id": "force_list_schemas_0", "name": "sql_db_query", "args": {"query": q}},
                    ],
                }
            ]
        }

    return None
