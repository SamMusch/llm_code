# rag/agent.py
# https://docs.langchain.com/oss/python/langchain/rag
from __future__ import annotations

import os
import re
import json
import time
import logging
import contextvars
from typing import Any, Optional, Sequence, Literal

from langchain.agents import create_agent as _lc_create_agent
from langchain.agents.middleware import (
    AgentState,
    before_model,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
try:
    # Prefer LangChain's conversion utility when available.
    from langchain_core.messages.utils import convert_to_messages as _lc_convert_to_messages
except Exception:  # pragma: no cover
    _lc_convert_to_messages = None
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from .config import Settings, get_settings
from .observability import setup_observability
from .tools import search_docs as _search_docs, rebuild_index, get_sql_database_tools
from .tools import read_doc_by_name as READ_DOC_TOOL


log = logging.getLogger("uvicorn.error")

# LangGraph checkpointer for stateful threads (dev/local). For multi-replica/prod,
# swap to a durable saver (e.g., SqliteSaver/PostgresSaver).
_CHECKPOINTER = InMemorySaver()

# --- begin rag/middleware.py --- #
# middleware.py
# There are 2 message "dialects"
    # 1. LangChain: BaseMessage objects
    # 2. Bedrock/Nova: {"role","content"} dicts.
# Use #1 within LangGraph/LangChain ---> convert to #2 at the edges (UI payloads / logging / external APIs)


Intent = Literal["sql", "docs", "mixed", "unknown"]
def last_human_text(messages: list[BaseMessage]) -> str:
    """Return the content of the most recent HumanMessage, or ""."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""



@before_model
def stop_after_final_answer(state: AgentState, runtime):
    messages = state.get("messages", [])
    if not messages:
        return None

    emit_step("middleware", "stop_after_final_answer", "start", {})
    try:
        last = messages[-1]
        if isinstance(last, AIMessage):
            tool_calls = getattr(last, "tool_calls", None) or []
            content = getattr(last, "content", "")
            if (not tool_calls) and str(content).strip():
                # Final answer already produced → stop graph
                runtime.stop()
        emit_step("middleware", "stop_after_final_answer", "ok", {})
        return None
    except Exception as e:
        emit_step("middleware", "stop_after_final_answer", "error", {"error": str(e)})
        raise


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
            "Intent=SQL. Use SQL tools for DB facts/metrics. Prefer sql_db_query with information_schema/pg_catalog. "
            "Do not answer DB questions from memory."
        )
    if intent == "docs":
        return "Intent=DOCS. Use document tools: search_docs then read_doc_by_name. Do not use SQL tools."
    if intent == "mixed":
        return "Intent=MIXED. Use tools to gather evidence (docs and/or SQL), then synthesize one final answer."
    return "Intent=UNKNOWN. Default docs-first: call search_docs before asking clarifying questions."

@before_model
def intent_router_hints(state: AgentState, runtime) -> dict[str, Any] | None:
    """Inject per-turn tool-use guidance based on the most recent HumanMessage."""
    messages = state.get("messages", [])
    if not messages:
        return None

    q = last_human_text(messages)
    if not q:
        return None

    emit_step("middleware", "intent_router_hints", "start", {})
    try:
        intent = classify_intent(q)
        guidance = intent_instructions(intent)
        emit_step("middleware", "intent_router_hints", "ok", {"intent": intent})
        return {
            "messages": messages + [SystemMessage(content=f"<analysis>{guidance}</analysis>")]
        }
    except Exception as e:
        emit_step("middleware", "intent_router_hints", "error", {"error": str(e)})
        raise


@before_model
def hallucination_guard_hints(state: AgentState, runtime) -> dict[str, Any] | None:
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

    emit_step("middleware", "hallucination_guard_hints", "start", {})
    try:
        # Heuristic: if we have no tool messages yet, nudge the model to call a tool.
        has_tool = any(getattr(m, "type", None) == "tool" for m in messages)
        if not has_tool:
            emit_step("middleware", "hallucination_guard_hints", "ok", {"has_tool": False})
            return {
                "messages": messages
                + [
                    SystemMessage(
                        content="<analysis>HallucinationGuard=ON. Use a tool first (docs: search_docs/read_doc_by_name; DB: sql_db_query).</analysis>"
                    )
                ]
            }

        emit_step("middleware", "hallucination_guard_hints", "ok", {"has_tool": True})
        return {
            "messages": messages
            + [
                SystemMessage(
                    content="<analysis>HallucinationGuard=ON. Answer using retrieved evidence; include Sources: filenames.</analysis>"
                )
            ]
        }
    except Exception as e:
        emit_step("middleware", "hallucination_guard_hints", "error", {"error": str(e)})
        raise







# --- end rag/middleware.py --- #

 # --- Step emission (migrated from rag/steps.py) ---

try:
    from opentelemetry import trace  # type: ignore
except Exception:  # pragma: no cover
    trace = None  # type: ignore

_STEP_SINK: contextvars.ContextVar[Optional[list[dict[str, Any]]]] = contextvars.ContextVar(
    "STEP_SINK", default=None
)


def attach_step_sink(sink: list[dict[str, Any]]):
    return _STEP_SINK.set(sink)

def detach_step_sink(token) -> None:
    try:
        _STEP_SINK.reset(token)
    except Exception:
        pass

def emit_step(kind: str, name: str, status: str, payload: Any | None = None) -> None:
    sink = _STEP_SINK.get()
    if sink is None:
        return
    if payload is None:
        out = ""
    elif isinstance(payload, str):
        out = payload
    else:
        try:
            out = json.dumps(payload, default=str)
        except Exception:
            out = str(payload)

    sink.append(
        {"ts": int(time.time() * 1000), "step_type": kind, "name": name, "status": status, "output": out}
    )

def drain_step_sink() -> list[dict[str, Any]]:
    sink = _STEP_SINK.get()
    if sink is None or not sink:
        return []
    out = list(sink)
    sink.clear()
    return out

def _trunc(v: Any, n: int) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        s = v
    else:
        try:
            s = json.dumps(v, default=str)
        except Exception:
            s = str(v)
    return s if len(s) <= n else s[:n] + "...(truncated)"

class StepSpanCallbackHandler(BaseCallbackHandler):
    def __init__(self, *, max_chars: int = 2000, emit_logs: bool = True):
        self.max_chars = max_chars
        self.emit_logs = emit_logs
        self._t0: dict[str, float] = {}
        self._name: dict[str, str] = {}

    def _span(self):
        if trace is None:
            return None
        span = trace.get_current_span()
        if not span or not span.is_recording():
            return None
        return span

    def _event(self, name: str, attrs: dict[str, Any]):
        span = self._span()
        if span is None:
            return
        span.add_event(name, {k: str(v) for k, v in attrs.items() if v is not None})
        if self.emit_logs:
            logging.getLogger("rag.steps").info(name, extra={"step": attrs})

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: str, **kwargs: Any):
        self._t0[run_id] = time.perf_counter()
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or ""
        self._name[run_id] = name
        self._event("lc.tool.start", {"step_type": "tool", "name": name, "run_id": run_id, "input": _trunc(input_str, self.max_chars)})
        emit_step("tool", name, "start", {"input": _trunc(input_str, self.max_chars)})

    def on_tool_end(self, output: str, *, run_id: str, **kwargs: Any):
        dt = int((time.perf_counter() - self._t0.pop(run_id, time.perf_counter())) * 1000)
        name = self._name.pop(run_id, "")
        self._event("lc.tool.end", {"step_type": "tool", "name": name, "run_id": run_id, "duration_ms": dt, "status": "ok", "output": _trunc(output, self.max_chars)})
        emit_step("tool", name, "ok", {"duration_ms": dt, "output": _trunc(output, self.max_chars)})

    def on_tool_error(self, error: BaseException, *, run_id: str, **kwargs: Any):
        dt = int((time.perf_counter() - self._t0.pop(run_id, time.perf_counter())) * 1000)
        name = self._name.pop(run_id, "")
        self._event("lc.tool.end", {"step_type": "tool", "name": name, "run_id": run_id, "duration_ms": dt, "status": "error", "error": repr(error)})
        emit_step("tool", name, "error", {"duration_ms": dt, "error": repr(error)})

# --- rag/steps.py --- #

SYSTEM_PROMPT = (
    "You are an assistant with TWO data sources: (1) documents (md, pptx, xlsx, pdf) and (2) a Postgres database. "
    "You MUST use tools for any question that depends on documents or database data; do not answer from memory. "
    "Routing rules: "
    "(A) If the question is about the database (tables, metrics, counts, aggregations, joins), use the SQL tools. "
    "(B) Otherwise, treat it as a document question and FIRST call the document search tool `search_docs` with a NON-EMPTY query equal to the user's question text (or a short keyword summary). "
    "(C) If the user asks what documents/files are available, still call `search_docs` with a NON-EMPTY query like 'what documents are available' (do not send an empty string). "
    "After you call `search_docs`, use the returned passages to answer. If `search_docs` returns an error or no results, say you don't know and ask one clarifying question. "
    "When you answer using documents, include a short 'Sources:' line listing file names you relied on. "
    "When you answer using SQL, compute via tools and briefly describe what you queried (no need to show raw SQL unless asked)."
    "Never paste raw tool output. Always summarize in your own words in at most 6 sentences. "
    "If a tool returns large passages, extract only the specific facts needed to answer. "
)


# ----
# Streaming + config helpers
# These are intentionally FastAPI-agnostic so the web layer can be thin.

_TAG_BLOCKS = ("thinking", "analysis")


class _TaggedBlockStripper:
    """Remove <thinking>...</thinking> and <analysis>...</analysis> blocks from streamed text."""

    def __init__(self, tags: Sequence[str] = _TAG_BLOCKS):
        self.tags = tuple(tags)

    def strip(self, text: str) -> str:
        if not text:
            return ""
        out = text
        for tag in self.tags:
            out = self._strip_tag(out, tag)
        return out

    @staticmethod
    def _strip_tag(text: str, tag: str) -> str:
        # Simple, non-regex stripping to avoid catastrophic backtracking.
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        out = text
        while True:
            i = out.find(open_tag)
            if i == -1:
                break
            j = out.find(close_tag, i + len(open_tag))
            if j == -1:
                # If the close tag never arrives, drop everything from the open tag onward.
                out = out[:i]
                break
            out = out[:i] + out[j + len(close_tag) :]
        return out


def _to_basemessages(items: Sequence[BaseMessage | dict[str, Any]]) -> list[BaseMessage]:
    """Edge-normalization.

    Prefer LangChain's `convert_to_messages` when available. Fall back to a minimal
    role/content adapter used by this app's UI/storage.
    """
    if _lc_convert_to_messages is not None:
        try:
            return list(_lc_convert_to_messages(items))
        except Exception:
            # Fall through to local adapter if the representation isn't supported.
            pass

    out: list[BaseMessage] = []
    for m in items:
        if isinstance(m, BaseMessage):
            out.append(m)
            continue
        if not isinstance(m, dict):
            raise TypeError(f"Unsupported message type: {type(m)!r}")
        role = (m.get("role") or "").lower()
        content = m.get("content")
        if role in {"user", "human"}:
            out.append(HumanMessage(content=content))
        elif role in {"assistant", "ai"}:
            out.append(AIMessage(content=content))
        elif role in {"system"}:
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


# --- Streaming payload text extraction helper

def _text_from_stream_payload(payload: Any) -> str:
    """Best-effort text extraction for LangGraph/LangChain streaming.

    `stream_mode=["messages","updates"]` may yield:
      - BaseMessage / BaseMessageChunk (has `.content`)
      - (message, metadata) tuple
      - dict-like payloads
      - lists of message chunks

    Return a string (possibly empty). Never raises.
    """
    try:
        obj = payload
        if isinstance(obj, (tuple, list)) and obj:
            # Common shape: (message, metadata)
            obj = obj[0]

        # Only stream assistant text. Drop tool/system/user messages.
        # LangChain message objects expose a `type` attribute (e.g., "ai", "tool", "human", "system").
        msg_type = getattr(obj, "type", None)
        if msg_type is not None and msg_type != "ai":
            return ""

        # Some chunk classes don't set `.type`; fall back to class-name check.
        cls_name = type(obj).__name__
        if cls_name and ("Tool" in cls_name or "System" in cls_name or "Human" in cls_name):
            return ""

        if isinstance(obj, dict):
            # Common LangGraph update shapes:
            #   {"model": {"messages": [...]}}
            #   {"tools": {"messages": [...]}}
            # Also sometimes the message itself is dict-like with a "content" field.

            # 1) Direct content
            if "content" in obj:
                val = obj.get("content", "")
                return "" if val is None else str(val)

            # 2) Nested model/tools messages
            for key in ("model", "tools"):
                inner = obj.get(key)
                if isinstance(inner, dict) and isinstance(inner.get("messages"), list):
                    msgs = inner.get("messages") or []
                    # Prefer the newest message
                    last = msgs[-1] if msgs else None
                    # If it's a message object, extract `.content`.
                    if last is not None and not isinstance(last, str):
                        mtype = getattr(last, "type", None)
                        if mtype is not None and mtype != "ai":
                            return ""
                        val = getattr(last, "content", "")
                        if isinstance(val, list):
                            return "".join("" if v is None else str(v) for v in val)
                        return "" if val is None else str(val)

                    # If it's a string repr (common in updates), parse out content='...'
                    if isinstance(last, str):
                        # Match content='...'
                        m = re.search(r"content=(?:'([^']*)'|\"([^\"]*)\")", last)
                        if m:
                            txt = m.group(1) if m.group(1) is not None else (m.group(2) or "")
                            return txt

            # Unknown dict shape
            return ""

        # If this is a LangChain message-like object but not AI, drop it.
        msg_type = getattr(obj, "type", None)
        if msg_type is not None and msg_type != "ai":
            return ""

        val = getattr(obj, "content", "")
        if isinstance(val, list):
            # Some providers represent content as a list of parts
            return "".join("" if v is None else str(v) for v in val)
        return "" if val is None else str(val)
    except Exception:
        return ""




async def llm_stream(
    *,
    messages: Sequence[BaseMessage | dict[str, Any]],
    cfg: Settings | None = None,
    selected_tools: Sequence[str] | None = None,
    doc_filters: dict[str, Any] | None = None,
    recursion_limit: int | None = None,
    callbacks: Sequence[BaseCallbackHandler] | None = None,
    thread_id: str | None = None,
) -> Any:
    """Async generator yielding normalized events for SSE (without SSE formatting).

    The web layer should:
      1) pass messages (BaseMessage or {role,content} dicts)
      2) forward yielded events as SSE

    """
    agent = get_agent(cfg, selected_tools=selected_tools)

    runnable_cfg: RunnableConfig = {
        "configurable": {
            "selected_tools": list(selected_tools or []),
            "doc_filters": doc_filters or {},
        }
    }
    if thread_id:
        runnable_cfg["configurable"]["thread_id"] = thread_id
    if recursion_limit is not None:
        runnable_cfg["recursion_limit"] = recursion_limit
    if callbacks is not None:
        runnable_cfg["callbacks"] = list(callbacks)

    stripper = _TaggedBlockStripper()

    # Prefer LangGraph-native streaming modes when available.
    try:
        # When using a checkpointer, the graph will load prior state from `thread_id`.
        # In that mode, only pass the *new* user message for this turn to avoid duplicating history.
        in_msgs = _to_basemessages(messages)
        if thread_id and in_msgs:
            in_msgs = [in_msgs[-1]]

        stream = agent.astream(
            {"messages": in_msgs},
            config=runnable_cfg,
            stream_mode=["messages", "updates"],
        )
        async for mode, payload in stream:
            if mode == "messages":
                text = _text_from_stream_payload(payload)
                text = stripper.strip(text)
                if not text:
                    continue
                yield {"type": "token", "text": text}
                continue

            if mode == "updates":
                # Some runtimes emit model text only via updates (e.g., {"model": {"messages": [...]}}).
                text = _text_from_stream_payload(payload)
                text = stripper.strip(text)
                if text:
                    yield {"type": "token", "text": text}
                else:
                    yield {"type": "step", "data": payload}
                continue

            # Unknown mode: forward for debugging.
            yield {"type": "step", "data": {"mode": mode, "payload": payload}}

    except TypeError:
        # Some runtimes don't accept stream_mode; fall back to default streaming.
        in_msgs = _to_basemessages(messages)
        if thread_id and in_msgs:
            in_msgs = [in_msgs[-1]]
        stream = agent.astream({"messages": in_msgs}, config=runnable_cfg)
        async for item in stream:
            # Best-effort fallback: treat message-like objects as tokens, otherwise as steps.
            text = _text_from_stream_payload(item)
            text = stripper.strip(text)
            if text:
                yield {"type": "token", "text": text}
            else:
                yield {"type": "step", "data": item}


# Wrapper tool to inject doc_filters from config into search_docs
@tool("search_docs")
def search_docs(query: str, k: int | None = None, config: RunnableConfig | None = None) -> str:
    """Semantic search over docs, with optional deterministic metadata filtering from config.

    If UI supplies doc_filters, they are applied at retrieval time (the model does not need to pass them).
    """
    cfg = (config or {}).get("configurable", {}) if isinstance((config or {}), dict) else {}
    doc_filters = cfg.get("doc_filters") or None
    try:
        # Prefer LangChain tool interfaces first. Some tool objects can appear "callable"
        # but still raise TypeError when invoked like a plain function.
        if hasattr(_search_docs, "invoke"):
            return _search_docs.invoke({"query": query, "k": k, "filters": doc_filters})
        if hasattr(_search_docs, "run"):
            return _search_docs.run(query=query, k=k, filters=doc_filters)
        if callable(_search_docs):
            return _search_docs(query=query, k=k, filters=doc_filters)
        raise TypeError(f"Unsupported tool type for _search_docs: {type(_search_docs)!r}")
    except Exception as e:
        log.exception("[search_docs wrapper] underlying tool failed", exc_info=e)
        return f"ERROR[search_docs]: {e}"


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

    middleware = [
        search_docs_limiter,
        # Stop the agent loop once a final assistant answer has been produced.
        stop_after_final_answer,
    ]

    if db_enabled:
        sql_query_limiter = ToolCallLimitMiddleware(tool_name="sql_db_query", run_limit=8, exit_behavior="continue")
        middleware.insert(1, sql_query_limiter)

    # Routing / guardrail hints
    # Always include lightweight routing hints so doc questions reliably trigger retrieval.
    middleware += [
        intent_router_hints,
        hallucination_guard_hints,
    ]

    if enable_tool_retry:
        middleware.append(
            ToolRetryMiddleware(
                max_retries=tool_retries,
                backoff_factor=2.0,
                initial_delay=1.0,
                tools=(
                    ["search_docs", "sql_db_query"]
                    if db_enabled
                    else ["search_docs"]
                ),
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
        checkpointer=_CHECKPOINTER,
    )
    return agent


def create_agent(config: RunnableConfig):
    cfg = get_settings()
    configurable = (config or {}).get("configurable", {})
    selected_tools = configurable.get("selected_tools") or []
    return get_agent(cfg, selected_tools=selected_tools)


