# rag/agent.py
# https://docs.langchain.com/oss/python/langchain/rag
from __future__ import annotations
from typing import Any, Dict, List
from langchain.agents import create_agent as _lc_create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from .config import Settings
from .tools import search_docs, rebuild_index

try:
    from retrieval_graph.tools import read_doc_by_name as READ_DOC_TOOL
except Exception:
    from .tools import read_doc_by_name as READ_DOC_TOOL



# ----
# Middleware: trim conversation history based on a character budget.
# node-style
@before_model
def trim_history(state: AgentState, runtime) -> Dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    MAX_CONTEXT_CHARS = 60000  # char budget for all message content

    def _content_str(msg: BaseMessage) -> str:
        c = getattr(msg, "content", "")
        return c if isinstance(c, str) else str(c)

    total_chars = sum(len(_content_str(m)) for m in messages)
    if total_chars <= MAX_CONTEXT_CHARS:
        return None

    trimmed: List[BaseMessage] = []
    running = 0
    for msg in reversed(messages):
        text = _content_str(msg)
        length = len(text)
        # Always keep at least one message
        if trimmed and running + length > MAX_CONTEXT_CHARS:
            break
        trimmed.append(msg)
        running += length
    trimmed.reverse()
    return {"messages": trimmed} # partial state update + pruned message list


# ----
def get_agent(cfg: Settings | None = None):

    # from config.py --> from rag.yaml
    cfg = cfg or Settings.load()
    rag_cfg = getattr(cfg, "rag", cfg)
    llm_cfg = getattr(rag_cfg, "llm", rag_cfg)
    provider = getattr(llm_cfg, "provider", "openai")
    model_name = getattr(llm_cfg, "model", "gpt-4o-mini")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model_name)
    elif provider in {"google", "gemini", "google-genai", "google_genai"}:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model_name)
    else:
        raise ValueError(f"Unsupported LLM provider in config: {provider!r}")

    tools = [search_docs, rebuild_index, READ_DOC_TOOL]

    system_prompt = (
        "You are a retrieval-augmented assistant over the user's local documents. "
        "Use the tools to search, read, or reindex documents as needed. "
        "Prefer `search_docs` for normal lookups."
    )

    agent = _lc_create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[trim_history],
    )
    return agent

# ----
# for langgraph
def create_agent(config: RunnableConfig):
    return get_agent()

# ----
def run_agent(question: str, cfg: Settings | None = None) -> str:
    agent = get_agent(cfg)
    state = agent.invoke({"messages": [{"role": "user", "content": question}]})
    messages = state.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)
    return content if isinstance(content, str) else str(last)