

from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import BaseMessage

from .intent import last_human_text


def _is_sql_write_request(q: str) -> bool:
    """Block anything that sounds like DDL/DML or permission changes."""
    write_terms = ["vacuum","truncate "]
    ql = q.lower()
    return any(t in ql for t in write_terms)


def _concat_text(messages: List[BaseMessage]) -> str:
    parts: List[str] = []
    for m in messages:
        c = getattr(m, "content", "")
        parts.append(c if isinstance(c, str) else str(c))
    return "\n".join(parts)


def _has_retrieved_sources(messages: List[BaseMessage]) -> bool:
    # Our retriever formats sources like: "Source: Data/docs/<name>" and/or "[1] Source: ..."
    return "source:" in _concat_text(messages).lower()


@before_model
def sql_write_guard(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Refuse any request that appears to modify the database."""
    messages = state.get("messages", [])
    if not messages:
        return None

    text = last_human_text(messages)
    if not text:
        return None

    if _is_sql_write_request(text):
        return {
            "messages": messages
            + [
                {
                    "role": "assistant",
                    "content": (
                        "I can’t help modify the database (DDL/DML/privileges). "
                        "If you want, ask for a read-only query/report and I’ll compute it via SQL tools."
                    ),
                }
            ]
        }

    return None


@before_model
def hallucination_guard_hints(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Lightweight guardrails to reduce answers without evidence.

    Controlled via env var:
      - HALLUCINATION_GUARD=true|false
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    enabled = os.environ.get("HALLUCINATION_GUARD", "true").lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return None

    human_q = last_human_text(messages)
    if not human_q:
        return None

    if not _has_retrieved_sources(messages):
        return {
            "messages": messages
            + [
                {
                    "role": "system",
                    "content": (
                        "HallucinationGuard=ON. Do not answer from memory. "
                        "If this is about docs, call search_docs/read_doc_by_name first. "
                        "If this is about DB facts, call SQL tools first."
                    ),
                }
            ]
        }

    return {
        "messages": messages
        + [
            {
                "role": "system",
                "content": "HallucinationGuard=ON. Answer only using evidence retrieved via tools; include a Sources: line.",
            }
        ]
    }