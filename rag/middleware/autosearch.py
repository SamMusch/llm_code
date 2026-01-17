from __future__ import annotations

from typing import Any, Dict

from langchain.agents.middleware import AgentState, before_model

from .context import has_retrieved_sources
from .intent import Intent, classify_intent, last_human_text


@before_model
def docs_first_autosearch(state: AgentState, runtime) -> Dict[str, Any] | None:
    """If user intent is docs/unknown and nothing has been retrieved yet, auto-call search_docs once.

    Controlled via env var:
      - DOCS_FIRST_AUTOSEARCH=true|false

    Notes:
    - We intentionally keep this simple and deterministic.
    - We avoid repeated auto-search after a tool result is already present.
    """
    import os

    messages = state.get("messages", [])
    if not messages:
        return None

    enabled = os.environ.get("DOCS_FIRST_AUTOSEARCH", "true").lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return None

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
    """Force initial sql_db_list_tables tool call for common 'list tables' requests."""
    messages = state.get("messages", [])
    if not messages:
        return None

    text = last_human_text(messages).lower().strip()

    triggers = [
        "list available tables",
        "list tables",
        "show tables",
        "what tables",
        "available tables",
    ]
    if any(t in text for t in triggers):
        return {
            "messages": messages
            + [
                {
                    "role": "assistant",
                    "content": "Calling sql_db_list_tables now.",
                    "tool_calls": [
                        {"id": "force_list_tables_0", "name": "sql_db_list_tables", "args": {}}
                    ],
                }
            ]
        }

    return None
