from __future__ import annotations

import os
import sys
import uuid
import logging
import json
from functools import lru_cache
from typing import AsyncIterator
from pathlib import Path

import mimetypes
import boto3
from boto3.dynamodb.conditions import Attr

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
import httpx

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages

from rag.agent import llm_stream as rag_llm_stream
from rag.retriever import verify_faiss_dim_matches_embeddings
from rag.history import build_chat_history

from rag.observability import setup_observability
from opentelemetry.trace import get_current_span
from rag.steps import StepSpanCallbackHandler
from rag.steps import attach_step_sink, detach_step_sink, drain_step_sink # # Prefer step sink utilities from rag.steps (ContextVar-backed).

# Optional: FastAPI auto-instrumentation (safe no-op if deps not installed)
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
except Exception:  # pragma: no cover
    FastAPIInstrumentor = None  # type: ignore

app = FastAPI()
log = logging.getLogger("uvicorn.error")
setup_observability(service_name="llm_code")
_otel_disabled = os.getenv("OTEL_SDK_DISABLED", "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
if (not _otel_disabled) and FastAPIInstrumentor is not None:
    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass # Never break the app if instrumentation fails.


# Fail fast if index built with a different embedding model.
@app.on_event("startup")
def _startup_fail_fast() -> None:
    if os.getenv("SKIP_FAIL_FAST", "false").lower() in ("1", "true"):
        print("[startup] SKIP_FAIL_FAST enabled: skipping FAISS dimension check.")
        return
    try: 
        verify_faiss_dim_matches_embeddings()
    except Exception as e:
        print(f"[startup] Fail-fast check failed: {e}", file=sys.stderr)
        sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# --- File uploads (UI attachments) ---
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/llm_code_uploads"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))  # 25MB total per request
MAX_FILE_TEXT_CHARS = int(os.getenv("MAX_FILE_TEXT_CHARS", "20000"))  # cap text injected into context


def _safe_filename(name: str) -> str:
    # Keep it simple: strip path separators and control chars
    s = (name or "file").replace("\\", "/")
    s = s.split("/")[-1]
    s = "".join(ch for ch in s if ch.isprintable())
    return s[:200] or "file"


def _session_upload_dir(session_id: str) -> Path:
    d = UPLOAD_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_text_best_effort(path: Path, max_chars: int) -> str:
    """Text extraction for attachments."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml"}:
        try:
            data = path.read_bytes()
            try:
                text = data.decode("utf-8")
            except Exception:
                text = data.decode("latin-1", errors="replace")
            return text[:max_chars]
        except Exception:
            return ""
    return ""       # Unsupported types


# Health checks for ALB
@app.get("/health")   
async def health():
    return {"ok": True}
@app.get("/api/health")
async def api_health():
    return await health()


# --- Models (UI Settings dropdown) ---
def _ollama_base_url() -> str:
    # Prefer explicit env; fall back to docker-for-mac host reachability.
    return (os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://host.docker.internal:11434").rstrip("/")


@app.get("/api/models")
async def api_models():
    """Return available local models for the UI.
    Currently implemented for Ollama via /api/tags.
    Response: {"models": ["name:tag", ...], "default": "..."}
    """
    base = _ollama_base_url()
    url = f"{base}/api/tags"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json() if r.content else {}
    except Exception as e:
        # Keep UI functional even if Ollama is unavailable.
        return JSONResponse(status_code=200, content={"models": [], "default": "", "error": str(e)})

    models: list[str] = []
    try:
        for m in (data.get("models") or []):
            name = str(m.get("name") or "").strip()
            if name:
                models.append(name)
    except Exception:
        models = []

    # De-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for m in models:
        if m in seen:
            continue
        seen.add(m)
        uniq.append(m)

    default_model = (os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "").strip()
    return {"models": uniq, "default": default_model}


# --- File uploads endpoint ---
@app.post("/api/files")
async def api_upload_files(session_id: str, files: list[UploadFile] = File(...)):
    """Upload one or more files for a session.
    Returns: {"files": [{"id": "...", "name": "...", "size": 123}]}
    Notes:
    - Files are stored on local disk under UPLOAD_DIR/session_id.
    - Only plain-text-ish files are currently injected into chat context.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    if not files:
        raise HTTPException(status_code=400, detail="No files")

    total = 0
    out: list[dict] = []
    dest_dir = _session_upload_dir(session_id)

    for f in files:
        raw_name = getattr(f, "filename", None) or "file"
        name = _safe_filename(raw_name)
        fid = str(uuid.uuid4())
        dest = dest_dir / f"{fid}__{name}"

        # Stream to disk while enforcing size limits
        written = 0
        try:
            with dest.open("wb") as w:
                while True:
                    chunk = await f.read(1024 * 1024)  # 1MB
                    if not chunk:
                        break
                    written += len(chunk)
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        try:
                            dest.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise HTTPException(status_code=413, detail=f"Upload too large (>{MAX_UPLOAD_BYTES} bytes)")
                    w.write(chunk)
        finally:
            try:
                await f.close()
            except Exception:
                pass
        out.append({"id": fid, "name": name, "size": written, "content_type": (f.content_type or "")})
    return {"files": out}






@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    # Auth is enforced at the ALB listener rule level for /app*
    # full_bleed lets the chat UI take the full viewport (no outer container/topbar spacing).
    return templates.TemplateResponse("app.html", {"request": request, "full_bleed": True})





@lru_cache(maxsize=1)
def _ddb_table():
    """Return DynamoDB table used for chat history.
    Fail fast if region isn't provided.
    Cached to avoid re-creating boto3 resources per request.
    """
    region = (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "").strip()
    if not region:
        raise RuntimeError("AWS_REGION or AWS_DEFAULT_REGION must be set")

    table_name = (os.getenv("DDB_TABLE") or "rag_chat_history").strip()
    if not table_name:
        raise RuntimeError("DDB_TABLE must be set")

    return boto3.resource("dynamodb", region_name=region).Table(table_name)


# --- Helper functions for environment and scoping ---
def _app_env() -> str:
    return (os.getenv("APP_ENV") or "local").strip().lower()


def _principal_id_from_request(request: Request) -> str:
    """Derive principal_id for scoping.
    AWS (ALB/Cognito): use `x-amzn-oidc-identity` (Cognito `sub`) when present.
    Local: fall back to USER_ID or 'local'.
    """
    sub = (request.headers.get("x-amzn-oidc-identity") or "").strip()
    if sub:
        return f"u#{sub}"

    fallback = (os.getenv("USER_ID") or "local").strip()
    if fallback.startswith(("u#", "g#")):
        return fallback
    return f"u#{fallback}"


def _scoped_id(session_id: str, request: Request) -> str:
    """Scope a UI session_id into a storage/checkpointer thread id."""
    return f"{_app_env()}#{_principal_id_from_request(request)}#{session_id}"

# Sidebar title ---> first 5 words
def _five_word_title(text: str) -> str:
    if not text:
        return "Untitled"
    s = " ".join(str(text).strip().split())
    if not s:
        return "Untitled"

    # Keep basic word tokens
    words = []
    for w in s.split(" "):
        w2 = w.strip().strip("\"'`.,:;!?()[]{}<>")
        if w2:
            words.append(w2)
        if len(words) >= 5:
            break

    if not words:
        return "Untitled"
    return " ".join(words)


@app.get("/api/sessions")
async def api_list_sessions(request: Request, limit: int = 50):
    """List recent session_ids for the sidebar.
    Implementation note: uses Scan (OK for now). For scale, add a Sessions table or a GSI.
    """
    # Local mode: do not touch DynamoDB (avoids needing AWS_REGION/AWS creds locally).
    if _app_env() == "local":
        return {"sessions": []}

    table = _ddb_table()
    prefix = f"{_app_env()}#{_principal_id_from_request(request)}#"

    # (DynamoDBChatMessageHistory stores the PK as `SessionId`.)
    resp = table.scan(
        FilterExpression=(
            Attr("SessionId").begins_with(prefix) |
            Attr("session_id").begins_with(prefix)
        )
    )

    # Track: last_ts (for ordering) + first user message (for title)
    sessions: dict[str, dict] = {}

    for item in resp.get("Items", []) or []:
        raw_sid = item.get("SessionId") or item.get("session_id")
        if not raw_sid:
            continue

        # UI should see the unscoped session_id
        sid = raw_sid[len(prefix):] if str(raw_sid).startswith(prefix) else str(raw_sid)

        # ts is stored as Number; be defensive
        try:
            ts = int(item.get("ts", 0))
        except Exception:
            ts = 0

        s = sessions.get(sid)
        if s is None:
            s = {
                "session_id": sid,
                "last_ts": ts,
                "first_user_ts": None,
                "first_user_message": "",
            }
            sessions[sid] = s

        # Update last_ts (max)
        if ts > int(s.get("last_ts", 0) or 0):
            s["last_ts"] = ts

        # Capture first user message (min ts among user messages)
        if (item.get("role") == "user") and item.get("message"):
            cur_first = s.get("first_user_ts")
            if (cur_first is None) or (ts < int(cur_first)):
                s["first_user_ts"] = ts
                s["first_user_message"] = str(item.get("message") or "")

    out = []
    for s in sessions.values():
        title = _five_word_title(s.get("first_user_message") or "")
        out.append(
            {
                "session_id": s["session_id"],
                "last_ts": s.get("last_ts", 0),
                "title": title,
            }
        )

    out = sorted(out, key=lambda x: x.get("last_ts", 0), reverse=True)[:limit]
    return {"sessions": out}


@app.get("/api/sessions/{session_id}")
async def api_get_session(request: Request, session_id: str, limit: int = 200):
    """Load messages for a session (oldest -> newest)."""
    history = build_chat_history(
        session_id=session_id,
        principal_id=_principal_id_from_request(request),
        env=_app_env(),
        history_size=limit,
    )
    msgs = history.messages
    return {
        "session_id": session_id,
        "messages": list(convert_to_openai_messages(msgs)),
    }


@app.get("/chat/stream")
async def chat_stream(
    request: Request,
    message: str = "",
    session_id: str | None = None,
    file_ids: str | None = None,
    tools: str | None = None,
    filters: str | None = None,
    model: str | None = None,
):
    msg = (message or "").strip()
    if not msg:
        async def empty() -> AsyncIterator[dict]:
            yield {"event": "error", "data": "Missing message"}
        return EventSourceResponse(empty())

    sid = session_id or request.headers.get("X-Session-Id") or str(uuid.uuid4())
    history = build_chat_history(
        session_id=sid,
        principal_id=_principal_id_from_request(request),
        env=_app_env(),
        history_size=20,
    )

    # Optional tool gating from UI (comma-separated). Examples: "database,placeholder1"
    selected_tools: list[str] = []
    if tools:
        try:
            selected_tools = [t.strip() for t in str(tools).split(",") if t.strip()]
        except Exception:
            selected_tools = []

    # Optional metadata filters from UI (URL-encoded JSON string)
    doc_filters: dict = {}
    if filters:
        try:
            doc_filters = json.loads(str(filters)) or {}
            if not isinstance(doc_filters, dict):
                doc_filters = {}
        except Exception:
            doc_filters = {}

    # Optional model override from UI Settings
    model_name = (model or "").strip() or None

    # 1) Save user message
    try:
        history.add_user_message(msg)
        log.info(f"DDB write OK session_id={sid}")
    except Exception as e:
        log.exception(f"DDB write FAILED session_id={sid}: {e}")

    # 2) Build the *current turn* messages for the agent.
    # With a LangGraph checkpointer, prior turns are loaded from thread_id=session_id.
    from langchain_core.messages import HumanMessage, SystemMessage

    context_msgs: list[BaseMessage] = [HumanMessage(content=msg)]

    # If UI provided file_ids, best-effort inject plain text contents into this turn.
    ids: list[str] = []
    if file_ids:
        ids = [x.strip() for x in str(file_ids).split(",") if x.strip()]

    if ids:
        dest_dir = _session_upload_dir(sid)
        parts: list[str] = []
        for fid in ids:
            # Files are stored as {fid}__{name}
            matches = list(dest_dir.glob(f"{fid}__*"))
            if not matches:
                continue

            p = matches[0]
            name = p.name.split("__", 1)[-1]
            text = _read_text_best_effort(p, MAX_FILE_TEXT_CHARS)
            if not text:
                parts.append(f"[Attachment: {name}] (not parsed; unsupported file type)")
                continue

            parts.append(f"[Attachment: {name}]\n{text}")

        if parts:
            injected = "\n\n".join(parts)
            context_msgs = [SystemMessage(content="User attached files (best-effort):\n\n" + injected)] + context_msgs

    async def event_gen() -> AsyncIterator[dict]:
        yield {"event": "start", "data": "ok"}
        full: list[str] = []

        try:
            span = get_current_span()
            ctx = span.get_span_context() if span else None
            trace_id = f"{ctx.trace_id:032x}" if ctx and getattr(ctx, "trace_id", 0) else ""
        except Exception:
            trace_id = ""
        meta = json.dumps({"session_id": sid, "trace_id": trace_id, "tools": selected_tools, "filters": doc_filters})
        yield {"event": "meta", "data": meta}

        # Attach a per-request sink for middleware/tool steps.
        sink_token = attach_step_sink([])
        async def _drain_steps() -> AsyncIterator[dict]:
            for step in drain_step_sink():
                yield {"event": "step", "data": json.dumps(step, default=str)}

        try:
            recursion_limit = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "50"))

            stream_kwargs = dict(
                messages=context_msgs,
                selected_tools=selected_tools,
                doc_filters=doc_filters,
                recursion_limit=recursion_limit,
                thread_id=_scoped_id(sid, request),
            )
            if model_name:
                stream_kwargs["model"] = model_name

            try:
                agen = rag_llm_stream(**stream_kwargs)
            except TypeError:
                # Older agent signature: ignore model override
                stream_kwargs.pop("model", None)
                agen = rag_llm_stream(**stream_kwargs)

            async for ev in agen:
                if ev.get("type") == "token":
                    text = str(ev.get("text", ""))
                    if not text:
                        continue
                    full.append(text)
                    safe = text.replace("\r", "\\r").replace("\n", "\\n")
                    yield {"event": "token", "data": safe}
                    async for s_ev in _drain_steps():
                        yield s_ev
                    continue

                if ev.get("type") == "step":
                    payload = ev.get("data")
                    yield {"event": "step", "data": json.dumps(payload, default=str)}
                    async for s_ev in _drain_steps():
                        yield s_ev
                    continue

        finally:
            if full:
                try:
                    history.add_ai_message("".join(full), trace_id=trace_id)
                    log.info(f"DDB assistant write OK session_id={sid}")
                except Exception as e:
                    log.exception(f"DDB assistant write FAILED session_id={sid}: {e}")

            try:
                async for s_ev in _drain_steps():
                    yield s_ev
            finally:
                detach_step_sink(sink_token)

        yield {"event": "end", "data": "ok"}

    return EventSourceResponse(
        event_gen(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": sid,
        },
    )

# Alias streaming chat under /api to match UI calls (/api/chat/stream)
@app.get("/api/chat/stream")
async def api_chat_stream(
    request: Request,
    message: str = "",
    session_id: str | None = None,
    file_ids: str | None = None,
    tools: str | None = None,
    filters: str | None = None,
    model: str | None = None,
):
    return await chat_stream(
        request,
        message=message,
        session_id=session_id,
        file_ids=file_ids,
        tools=tools,
        filters=filters,
        model=model,
    )


@app.get("/api/files/{file_id}")
async def api_get_file(file_id: str, session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    dest_dir = _session_upload_dir(session_id)
    matches = list(dest_dir.glob(f"{file_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="File not found")

    path = matches[0]
    mt, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=mt or "application/octet-stream")