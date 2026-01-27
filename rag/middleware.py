from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.agents.middleware import AgentState, before_model

Intent = Literal["sql", "docs", "mixed", "unknown"]
def last_human_text(messages: List[BaseMessage]) -> str:
    """Return the content of the most recent HumanMessage, or ""."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


@before_model
def trim_history(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Keep the newest messages within a rough token budget.

    Notes:
    - We avoid LLM-based summarization here to keep middleware side-effect free.
    - If you want true summarization, add a separate summarizer node in LangGraph later.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    # Prefer explicit config if present; otherwise default to a safe cap.
    max_chars = int(os.environ.get("MAX_CONTEXT_CHARS", "60000"))
    max_tokens = max(2000, max_chars // 4)

    def _tokens(msg: BaseMessage) -> int:
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        return max(1, len(text) // 4)

    total = sum(_tokens(m) for m in messages)
    if total <= max_tokens:
        return None

    kept: List[BaseMessage] = []
    running = 0
    for msg in reversed(messages):
        t = _tokens(msg)
        if kept and running + t > max_tokens:
            break
        kept.append(msg)
        running += t
    kept.reverse()

    if len(kept) < len(messages):
        kept = [
            {
                "role": "system",
                "content": "Note: earlier conversation history was trimmed for context-length limits.",
            }
        ] + kept
    return {"messages": kept}


def concat_text(messages: List[BaseMessage]) -> str:
    parts: List[str] = []
    for m in messages:
        c = getattr(m, "content", "")
        parts.append(c if isinstance(c, str) else str(c))
    return "\n".join(parts)

def has_retrieved_sources(messages: List[BaseMessage]) -> bool:
    # Our retriever formats sources like: "Source: Data/docs/<name>" and/or "[1] Source: ..."
    return "source:" in concat_text(messages).lower()


def keyword_overlap_score(a: str, b: str) -> float:
    def toks(s: str) -> set[str]:
        s = s.lower()
        s = re.sub(r"[^a-z0-9_\s]", " ", s)
        raw = [t for t in s.split() if len(t) >= 3]
        stop = {
            "the","and","for","with",
            "that","this","from","into",
            "what","did","does","how",
            "your","you","our","about",
            "when","where","which","who",
        }
        return {t for t in raw if t not in stop}

    ta = toks(a)
    tb = toks(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union

@before_model
def context_relevance_hint(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Heuristic relevance check between the last user question and retrieved context.

    Controlled via env var:
      - CONTEXT_RELEVANCE_HINT=true|false
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    enabled = os.environ.get("CONTEXT_RELEVANCE_HINT", "true").lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return None

    human_q = last_human_text(messages)
    if not human_q:
        return None

    if not has_retrieved_sources(messages):
        return None

    score = keyword_overlap_score(human_q, concat_text(messages))
    if score < 0.03:
        return {
            "messages": messages
            + [
                {
                    "role": "system",
                    "content": (
                        f"ContextRelevance=LOW (heuristic score={score:.3f}). "
                        "Prefer another search_docs call with a refined query or ask one clarifying question "
                        "before giving a definitive answer."
                    ),
                }
            ]
        }

    return {
        "messages": messages
        + [
            {
                "role": "system",
                "content": f"ContextRelevance=OK (heuristic score={score:.3f}).",
            }
        ]
    }

@before_model
def stop_after_final_answer(state: AgentState, runtime):
    messages = state.get("messages", [])
    if not messages:
        return None

    last = messages[-1]
    if getattr(last, "role", None) == "assistant":
        # Final answer already produced → stop graph
        runtime.stop()
    return None


def classify_intent(q: str) -> Intent:
    """Heuristic intent classification for routing tools.

    Keep this conservative: prefer docs-first when uncertain.
    """
    ql = q.lower()

    sql_terms = [
        "postgres","pgadmin","sql","schema",
        "schemas","table","tables","column",
        "columns","database","query","select ",
        "join ","group by","where ","llm_code.",
        "spend","click","clicks","metric",
        "metrics","campaign",
    ]
    doc_terms = [
        "doc","docs","document","documents",
        "note","notes","markdown","pdf",
        "ppt","pptx","slide","slides",
        "xlsx","sheet","file","folder",
        "read","find","search","in the docs",
        "in my notes","dataset","data set","what did it cover",
        "coverage","project",
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
