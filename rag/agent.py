# rag/agent.py
# https://docs.langchain.com/oss/python/langchain/rag
from __future__ import annotations

import os
from typing import Any, Dict, List, Literal

from langchain.agents import create_agent as _lc_create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from .config import Settings, get_settings
from .tools import search_docs, rebuild_index, get_sql_database_tools

try:
    from retrieval_graph.tools import read_doc_by_name as READ_DOC_TOOL
except ImportError:
    from .tools import read_doc_by_name as READ_DOC_TOOL


SYSTEM_PROMPT = (
    "You are an assistant with TWO data sources: (1) documents (md, pptx, xlsx, pdf) and (2) a Postgres database. "
    "For database questions you MUST use tools and you MUST NOT answer from memory. "
    "When the user asks about campaigns, metrics, spend, clicks, performance, tables, schema, or pgAdmin: "
    "(a) call sql_db_list_tables if the question is about what tables exist, "
    "(b) call sql_db_schema(table_names=...) before any query if you need columns, and "
    "(c) call sql_db_query(query=...) to compute the answer. "
    "Return the final answer only AFTER tool results. "
    "Use document tools (search_docs, read_doc_by_name, rebuild_index) ONLY for questions about documents."
)


# ----
# Helper types and functions for intent classification and SQL write blocking
Intent = Literal["sql", "docs", "mixed", "unknown"]


def _is_sql_write_request(q: str) -> bool:
    # block anything that sounds like DDL/DML or permission changes
    write_terms = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "create ",
        "truncate ",
        "grant ",
        "revoke ",
        "vacuum",
        "analyze",
        "set role",
        "owner",
        "privilege",
    ]
    ql = q.lower()
    return any(t in ql for t in write_terms)


def _classify_intent(q: str) -> Intent:
    ql = q.lower()

    sql_terms = [
        "postgres",
        "pgadmin",
        "sql",
        "schema",
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


def _intent_instructions(intent: Intent) -> str:
    if intent == "sql":
        return (
            "Intent=SQL. Use ONLY database tools (sql_db_list_tables, sql_db_schema, sql_db_query). "
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
            "First gather evidence (docs search/read and/or DB queries), then synthesize a single final answer."
        )
    return (
        "Intent=UNKNOWN. Default docs-first: call search_docs for relevant notes/files before asking clarifying questions. "
        "If the user clearly asks for database facts/metrics, switch to SQL tools."
    )


def _last_human_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


# ----
# Middleware: trim conversation history based on a token budget.
@before_model
def trim_history(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Keep the newest messages within a rough token budget.

    Notes:
    - We avoid LLM-based summarization here to keep middleware side-effect free.
    - If you want true summarization, we can add a separate summarizer node in LangGraph later.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    cfg = get_settings()
    max_chars = getattr(cfg, "max_context_chars", 60000)
    max_tokens = max(2000, (max_chars // 4) if max_chars else 15000)

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


# ----
@before_model
def sql_write_guard(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Refuse any request that appears to modify the database."""
    messages = state.get("messages", [])
    if not messages:
        return None

    text = _last_human_text(messages)
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
def intent_router_hints(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Classify intent from the most recent HumanMessage and inject per-turn tool-use guidance."""
    messages = state.get("messages", [])
    if not messages:
        return None

    text = _last_human_text(messages)
    if not text:
        return None

    intent: Intent = _classify_intent(text)
    guidance = _intent_instructions(intent)

    return {
        "messages": messages
        + [
            {
                "role": "system",
                "content": guidance,
            }
        ]
    }


@before_model
def force_list_tables(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Force initial sql_db_list_tables tool call"""
    messages = state.get("messages", [])
    if not messages:
        return None

    text = _last_human_text(messages).lower().strip()

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


# ----
def get_agent(cfg: Settings | None = None):
    cfg = cfg or get_settings()
    provider = cfg.llm_provider
    model_name = cfg.llm_model

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model_name)
    elif provider in {"google", "gemini", "google-genai", "google_genai"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=model_name)
    elif provider == "bedrock":
        import boto3
        from langchain_aws import ChatBedrock

        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        llm = ChatBedrock(
            client=bedrock_runtime,
            model_id=model_name,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=model_name,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
            temperature=0,
        )
    else:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model=model_name, base_url=os.environ.get("OPENAI_API_BASE"))
        except Exception:
            raise ValueError(f"Unsupported LLM provider in config: {provider!r}")

    tools = [search_docs, rebuild_index, READ_DOC_TOOL]
    tools += get_sql_database_tools(llm, cfg)
    llm = llm.bind_tools(tools)

    agent = _lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[trim_history, sql_write_guard, intent_router_hints, force_list_tables],
    )
    return agent


# ----
# for langgraph
def create_agent(config: RunnableConfig):
    return get_agent()


# ----
def run_agent(question: str, cfg: Settings | None = None) -> dict:
    agent = get_agent(cfg)
    state = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return {"messages": state.get("messages", [])}
