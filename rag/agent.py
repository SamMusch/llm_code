# rag/agent.py
# https://docs.langchain.com/oss/python/langchain/rag
from __future__ import annotations
from typing import Any, Dict, List
import tiktoken
import os
from langchain.agents import create_agent as _lc_create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import BaseMessage
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
# Middleware: trim conversation history based on a token budget.
@before_model
def trim_history(state: AgentState, runtime) -> Dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    # Load config to get trimming settings if available, otherwise default
    cfg = get_settings()

    # hard limit or read from a new config field if it existed.
    max_chars = getattr(cfg, "max_context_chars", 60000)
    MAX_TOKENS = max_chars // 4 if max_chars else 15000
    def _get_tokens(msg: BaseMessage) -> int:
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        return max(1, len(text) // 4)

    total_tokens = sum(_get_tokens(m) for m in messages)
    if total_tokens <= MAX_TOKENS:
        return None

    trimmed: List[BaseMessage] = []
    current_tokens = 0
    # Keep messages from the end until we hit the limit
    for msg in reversed(messages):
        tokens = _get_tokens(msg)
        if trimmed and current_tokens + tokens > MAX_TOKENS:
            # Always keep at least the last message if possible
            break
        trimmed.append(msg)
        current_tokens += tokens
    
    trimmed.reverse()
    return {"messages": trimmed}

@before_model
def force_list_tables(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Force initial sql_db_list_tables tool call"""
    messages = state.get("messages", [])
    if not messages:
        return None
    last = messages[-1]
    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)
    q = text.lower().strip()

    triggers = [
        "list available tables",
        "list tables",
        "show tables",
        "what tables",
        "available tables",
    ]
    if any(t in q for t in triggers):
        # Return a tool call instruction in the message stream.
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
    provider = cfg.llm_provider # Access settings directly from the typed Settings object
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
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        llm = ChatBedrock(client=bedrock_runtime,model_id=model_name,)
    
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=model_name,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
            temperature=0,)
    else:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model_name, base_url=os.environ.get("OPENAI_API_BASE"))
        except Exception:
            raise ValueError(f"Unsupported LLM provider in config: {provider!r}")

    # Build tool list (docs + runtime SQL) and bind them to the model
    tools = [search_docs, rebuild_index, READ_DOC_TOOL]
    tools += get_sql_database_tools(llm, cfg)
    llm = llm.bind_tools(tools)

    agent = _lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[trim_history, force_list_tables],
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

    # Always return state — LangGraph requires dicts
    return {"messages": state.get("messages", [])}
