# rag/agent.py
# https://docs.langchain.com/oss/python/langchain/rag
from __future__ import annotations

import os
from typing import Any, Dict, List

from langchain.agents import create_agent as _lc_create_agent
from langchain.agents.middleware import (
    AgentState,
    before_model,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from .config import Settings, get_settings
from .tools import search_docs, rebuild_index, get_sql_database_tools
from .observability import setup_observability

from .middleware.intent import Intent, classify_intent, intent_router_hints, last_human_text
from .middleware.guards import hallucination_guard_hints, sql_write_guard
from .middleware.context import context_relevance_hint, has_retrieved_sources, trim_history
from .middleware.autosearch import docs_first_autosearch, force_list_tables
from .middleware.termination import stop_after_final_answer

try:
    from retrieval_graph.tools import read_doc_by_name as READ_DOC_TOOL
except ImportError:
    from .tools import read_doc_by_name as READ_DOC_TOOL


SYSTEM_PROMPT = (
    "You are an assistant with TWO data sources: (1) documents (md, pptx, xlsx, pdf) and (2) a Postgres database. "
    "For database questions you MUST use tools and you MUST NOT answer from memory. "
    "For document questions you MUST use document tools to find/read relevant passages; do not guess. "
    "When you answer using documents, include a short 'Sources:' line listing file names you relied on. "
    "When you answer using SQL, compute via tools and briefly describe what you queried (no need to show raw SQL unless asked). "
    "If you do not have enough evidence from tools to answer, say you don't know and ask one clarifying question."
)


# ----
def get_agent(cfg: Settings | None = None):
    cfg = cfg or get_settings()

    # Bootstrap OpenTelemetry as early as possible so it applies to CLI, API, and langgraph.
    # Fully config-driven via OTEL_* env vars; safe no-op if disabled/misconfigured.
    setup_observability(service_name="llm_code")

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

    # Hard stop on runaway loops (per single user turn). Use run_limit only so we don't require a checkpointer.
    model_call_limiter = ModelCallLimitMiddleware(run_limit=6, exit_behavior="end")

    # Global tool-call limiter + tool-specific caps for the most expensive/loop-prone tools.
    tool_call_limiter = ToolCallLimitMiddleware(run_limit=10, exit_behavior="continue")
    search_docs_limiter = ToolCallLimitMiddleware(tool_name="search_docs", run_limit=3, exit_behavior="continue")
    sql_query_limiter = ToolCallLimitMiddleware(tool_name="sql_db_query", run_limit=3, exit_behavior="continue")

    agent = _lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            model_call_limiter,
            tool_call_limiter,
            search_docs_limiter,
            sql_query_limiter,
            trim_history,
            sql_write_guard,
            intent_router_hints,
            docs_first_autosearch,
            hallucination_guard_hints,
            context_relevance_hint,
            force_list_tables,
            stop_after_final_answer # last
        ],
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
