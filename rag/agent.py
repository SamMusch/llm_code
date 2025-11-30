# rag/agent.py
from __future__ import annotations
from typing import Any, Dict, List
from langchain.agents import create_agent as _lc_create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from .config import Settings
from .tools import search_docs, rebuild_index

try:
    from retrieval_graph.tools import read_doc_by_name as READ_DOC_TOOL
except Exception:
    from .tools import read_doc_by_name as READ_DOC_TOOL


class HistoryTrimMiddleware(AgentMiddleware):
    """Middleware to trim conversation history based on a character budget."""

    def __init__(self, max_context_chars: int) -> None:
        self.max_context_chars = max_context_chars

    def before_model(self, state: AgentState, runtime) -> Dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages or self.max_context_chars is None:
            return None

        def _content_str(msg: BaseMessage) -> str:
            c = getattr(msg, "content", "")
            return c if isinstance(c, str) else str(c)

        total_chars = sum(len(_content_str(m)) for m in messages)
        if total_chars <= self.max_context_chars:
            return None

        trimmed: List[BaseMessage] = []
        running = 0
        for msg in reversed(messages):
            text = _content_str(msg)
            length = len(text)
            # Always keep at least one message
            if trimmed and running + length > self.max_context_chars:
                break
            trimmed.append(msg)
            running += length
        trimmed.reverse()

        # Return partial state update
        return {"messages": trimmed}

# --- LLM construction --------------------------------------------------------


def build_llm(cfg: Settings | None = None):
    """
    Build base chat model
    """
    cfg = cfg or Settings.load()

    # Try to navigate nested rag.llm settings; fall back gracefully
    rag_cfg = getattr(cfg, "rag", cfg)
    llm_cfg = getattr(rag_cfg, "llm", rag_cfg)
    provider = getattr(llm_cfg, "provider", "openai")
    model_name = getattr(llm_cfg, "model", "gpt-4o-mini")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name)

    elif provider in {"google", "gemini", "google-genai", "google_genai"}:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name)

    else:
        raise ValueError(f"Unsupported LLM provider in config: {provider!r}")


# --- Agent factory -----------------------------------------------------------


def get_agent(cfg: Settings | None = None):
    """Create agent over local RAG tools."""
    cfg = cfg or Settings.load()
    llm = build_llm(cfg)

    rag_cfg = getattr(cfg, "rag", cfg)
    runtime_cfg = getattr(rag_cfg, "runtime", rag_cfg)
    max_context_chars = getattr(runtime_cfg, "max_context_chars", 60000)
    middleware = [HistoryTrimMiddleware(max_context_chars=max_context_chars)]


    tools = [
        search_docs,     # semantic search over FAISS index
        rebuild_index,   # rebuild index if stale/missing
        READ_DOC_TOOL,   # read doc by name (LRAT or local)
    ]

    system_prompt = (
        "You are a retrieval-augmented assistant over the user's local documents. "
        "Use the tools to search, read, or reindex documents as needed. "
        "Prefer `search_docs` for normal lookups."
    )

    agent = _lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
    )
    return agent


# --- Convenience wrapper for simple Q&A -------------------------------------

def run_agent(question: str, cfg: Settings | None = None) -> str:
    """Helper: input question ---> output answer"""
    agent = get_agent(cfg)

    state: Dict[str, Any] = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages: List[BaseMessage] = state.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    # BaseMessage has .content; fall back to str as a guard
    content = getattr(last, "content", None)
    return content if isinstance(content, str) else str(last)


def create_agent(config: RunnableConfig):
    """LangGraph factory: return the runnable agent for this graph.

    The RunnableConfig is currently unused; configuration is loaded via Settings.
    """
    return get_agent()