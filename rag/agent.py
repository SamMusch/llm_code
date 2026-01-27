# rag/agent.py
# https://docs.langchain.com/oss/python/langchain/rag
from __future__ import annotations

import os
import inspect
import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain.agents import create_agent as _lc_create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.runnables import RunnableConfig

from .config import Settings, get_settings
from .tools import search_docs, rebuild_index, get_sql_database_tools
from .observability import setup_observability
from .middleware import *

try:
    from retrieval_graph.tools import read_doc_by_name as READ_DOC_TOOL
except ImportError:
    from .tools import read_doc_by_name as READ_DOC_TOOL


log = logging.getLogger("uvicorn.error")


SYSTEM_PROMPT = (
    "You are an assistant with TWO data sources: (1) documents (md, pptx, xlsx, pdf) and (2) a Postgres database. "
    "For database questions you MUST use tools and you MUST NOT answer from memory. "
    "For document questions you MUST use document tools to find/read relevant passages; do not guess. "
    "When you answer using documents, include a short 'Sources:' line listing file names you relied on. "
    "When you answer using SQL, compute via tools and briefly describe what you queried (no need to show raw SQL unless asked). "
    "If you do not have enough evidence from tools to answer, say you don't know and ask one clarifying question."
)


def _env_truthy(key: str, default: str) -> bool:
    return os.getenv(key, default).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _validate_llm_config(provider: str, model_name: str) -> None:
    # Fail fast with a readable error instead of silently producing start/meta/end only.
    if not provider:
        raise ValueError(
            "cfg.llm_provider is empty. Set LLM_PROVIDER (e.g., 'bedrock') or llm.provider in config/rag.yaml."
        )
    if not model_name:
        raise ValueError(
            "cfg.llm_model is empty. Set LLM_MODEL (e.g., 'anthropic.claude-3-sonnet-20240229-v1:0') or llm.model in config/rag.yaml."
        )


def _build_llms(provider: str, model_name: str, fallback_model_name: str | None):
    """Return (llm, fallback_llm) for the configured provider."""
    if provider == "bedrock":
        import boto3
        from langchain_aws import ChatBedrock, ChatBedrockConverse
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        client = boto3.client("bedrock-runtime", region_name=region)

        try:
            llm = ChatBedrockConverse(client=client, model_id=model_name)
        except Exception as e:
            log.warning(f"[agent] Converse unavailable; using ChatBedrock. err={e}")
            llm = ChatBedrock(client=client, model_id=model_name)

        fallback_llm = None
        if fallback_model_name:
            try:
                fallback_llm = ChatBedrockConverse(client=client, model_id=fallback_model_name)
            except Exception:
                fallback_llm = ChatBedrock(client=client, model_id=fallback_model_name)
        return llm, fallback_llm

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        llm = ChatOllama(model=model_name, base_url=base_url, temperature=0)
        fallback_llm = (
            ChatOllama(model=fallback_model_name, base_url=base_url, temperature=0)
            if fallback_model_name
            else None
        )
        return llm, fallback_llm
    raise ValueError(f"Unsupported LLM provider: {provider!r}")


# ----
def get_agent(cfg: Settings | None = None, selected_tools: Optional[Sequence[str]] = None):
    cfg = cfg or get_settings()

    # Bootstrap OpenTelemetry as early as possible so it applies to CLI, API, and langgraph.
    # Fully config-driven via OTEL_* env vars; safe no-op if disabled/misconfigured.
    setup_observability(service_name="llm_code")

    provider = cfg.llm_provider
    model_name = cfg.llm_model

    # Middleware knobs (env-only for now)
    enable_model_retry = _env_truthy("MIDDLEWARE_MODEL_RETRY", "true")
    enable_tool_retry = _env_truthy("MIDDLEWARE_TOOL_RETRY", "true")
    enable_model_fallback = _env_truthy("MIDDLEWARE_MODEL_FALLBACK", "false")

    model_retries = int(os.getenv("MIDDLEWARE_MODEL_RETRY_MAX_RETRIES", "2"))
    tool_retries = int(os.getenv("MIDDLEWARE_TOOL_RETRY_MAX_RETRIES", "2"))

    fallback_model_name = os.getenv("LLM_FALLBACK_MODEL", "").strip() or None

    _validate_llm_config(provider, model_name)
    log.info(f"[agent] llm_provider={provider} llm_model={model_name}")

    llm, fallback_llm = _build_llms(provider, model_name, fallback_model_name)

    tools = [search_docs, rebuild_index, READ_DOC_TOOL]

    # Postgres tools are opt-in (UI-controlled). Only enable when explicitly selected.
    selected = set(t.lower() for t in (selected_tools or []))
    db_enabled = "database" in selected
    if db_enabled:
        tools += get_sql_database_tools(llm, cfg, enabled=True)

    # Debug: confirm what the agent can actually call.
    try:
        tool_names = [getattr(t, "name", type(t).__name__) for t in tools]
    except Exception:
        tool_names = [type(t).__name__ for t in tools]
    log.info(f"[agent] db_enabled={db_enabled} selected_tools={list(selected_tools or [])} tools={tool_names}")

    llm = llm.bind_tools(tools)
    if fallback_llm is not None:
        fallback_llm = fallback_llm.bind_tools(tools)

    # Global tool-call limiter + tool-specific caps.
    search_docs_limiter = ToolCallLimitMiddleware(tool_name="search_docs", run_limit=3, exit_behavior="continue")
    sql_query_limiter = ToolCallLimitMiddleware(tool_name="sql_db_query", run_limit=8, exit_behavior="continue")

    middleware = [
        search_docs_limiter,
        sql_query_limiter,
        # Stop the agent loop once a final assistant answer has been produced.
        stop_after_final_answer,
    ]

    if db_enabled:
        middleware += [
            intent_router_hints,
            hallucination_guard_hints,
            force_list_tables,
            force_list_schemas,
        ]

    if enable_tool_retry:
        middleware.append(
            ToolRetryMiddleware(
                max_retries=tool_retries,
                backoff_factor=2.0,
                initial_delay=1.0,
                tools=["search_docs", "sql_db_query"],
            )
        )

    if enable_model_retry:
        middleware.append(
            ModelRetryMiddleware(
                max_retries=model_retries,
                backoff_factor=2.0,
                initial_delay=1.0,
            )
        )

    if enable_model_fallback and fallback_llm is not None:
        middleware.append(ModelFallbackMiddleware(fallback_llm))

    # Debug: middleware list
    try:
        mw_names: list[str] = []
        for m in middleware:
            tn = getattr(m, "tool_name", None)
            mw_names.append(f"{type(m).__name__}({tn})" if tn else type(m).__name__)
    except Exception:
        mw_names = [type(m).__name__ for m in middleware]
    log.info(f"[agent] middleware={mw_names}")

    agent = _lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
    )
    return agent


def create_agent(config: RunnableConfig):
    cfg = get_settings()
    configurable = (config or {}).get("configurable", {})
    selected_tools = configurable.get("selected_tools") or []
    return get_agent(cfg, selected_tools=selected_tools)


def run_agent(question: str, cfg: Settings | None = None, selected_tools: Optional[Sequence[str]] = None) -> dict:
    agent = get_agent(cfg, selected_tools=selected_tools or [])
    state = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return {"messages": state.get("messages", [])}
