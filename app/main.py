from __future__ import annotations

import os
import sys
import uuid
import logging
from typing import AsyncIterator

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from rag.agent import get_agent
from rag.config import get_settings
from rag.retriever import verify_faiss_dim_matches_embeddings
from rag.utils import extract_text_from_stream_delta
from rag.history import DynamoDBChatMessageHistory

app = FastAPI()
log = logging.getLogger("uvicorn.error")


@app.on_event("startup")
def _startup_fail_fast() -> None:
    # Allow skipping fail-fast via env var (e.g., during FAISS rebuild)
    if os.getenv("SKIP_FAIL_FAST", "false").lower() in ("1", "true"):
        print("[startup] SKIP_FAIL_FAST enabled: skipping FAISS dimension check.")
        return

    # Fail fast on common RAG misconfig: index built with a different embedding model.
    try:
        verify_faiss_dim_matches_embeddings()
    except Exception as e:
        print(f"[startup] Fail-fast check failed: {e}", file=sys.stderr)
        sys.exit(1)


app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# Health check endpoint for ALB
@app.get("/health")
async def health():
    return {"ok": True}


def is_authed(request: Request) -> bool:
    return request.cookies.get("demo_auth") == "1"

async def llm_stream(messages: list[dict]) -> AsyncIterator[str]:
    """Yield text chunks from the real LangChain/LangGraph agent."""
    cfg = get_settings()
    agent = get_agent(cfg)

    # IMPORTANT: This must match what rag/agent.py expects
    inputs = {"messages": messages}

    async for event in agent.astream_events(inputs, version="v2"):
        if event.get("event") != "on_chat_model_stream":
            continue

        chunk = (event.get("data") or {}).get("chunk")
        if chunk is None:
            continue

        delta = getattr(chunk, "content", None)
        text = extract_text_from_stream_delta(delta)
        if not text:
            continue

        yield text


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
async def login_post(request: Request, email: str = Form(default=""), password: str = Form(default="")):
    # Stub: accept anything; set a cookie; redirect to /app
    resp = RedirectResponse(url="/app", status_code=303)
    resp.set_cookie("demo_auth", "1", httponly=True, samesite="lax")
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("demo_auth")
    return resp


@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    if not is_authed(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("app.html", {"request": request})


@app.post("/chat")
async def chat_non_stream(request: Request):
    """
    Optional non-streaming fallback (handy for debugging).
    Body: {"message": "..."}
    """
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    out = []
    async for chunk in llm_stream([{"role": "user", "content": message}]):
        out.append(chunk)
    return {"text": "".join(out)}


def lc_messages_to_dicts(msgs: list[BaseMessage]) -> list[dict]:
    out = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
        elif isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        else:
            # fallback
            out.append({"role": "user", "content": str(m.content)})
    return out


def strip_thinking(text: str) -> str:
    """Remove model 'thinking' blocks from streamed text."""
    if not text:
        return ""

    # Common wrappers seen in some model outputs
    pairs = [
        ("<thinking>", "</thinking>"),
        ("<analysis>", "</analysis>"),
    ]
    for open_tag, close_tag in pairs:
        while True:
            start = text.find(open_tag)
            if start == -1:
                break
            end = text.find(close_tag, start)
            if end == -1:
                # If we only got the opening tag so far, drop from tag onward
                text = text[:start]
                break
            end = end + len(close_tag)
            text = (text[:start] + text[end:])

    return text


@app.get("/chat/stream")
async def chat_stream(request: Request, message: str = "", session_id: str | None = None):
    msg = (message or "").strip()
    if not msg:
        async def empty() -> AsyncIterator[bytes]:
            yield b"event: error\ndata: Missing message\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    sid = session_id or request.headers.get("X-Session-Id") or str(uuid.uuid4())
    history = DynamoDBChatMessageHistory(session_id=sid, limit=20)

    # 1) Save user message
    try:
        history.add_user_message(msg)
        log.info(f"DDB write OK session_id={sid}")
    except Exception as e:
        log.exception(f"DDB write FAILED session_id={sid}: {e}")

    # 2) Load history and build context for the agent
    try:
        past = history.messages
        context_msgs = lc_messages_to_dicts(past)
    except Exception as e:
        log.exception(f"DDB read FAILED session_id={sid}: {e}")
        context_msgs = [{"role": "user", "content": msg}]

    async def event_gen() -> AsyncIterator[bytes]:
        yield b"event: start\ndata: ok\n\n"

        full: list[str] = []
        try:
            async for chunk in llm_stream(context_msgs):
                # Strip thinking/analysis content
                cleaned = strip_thinking(chunk)
                if not cleaned:
                    continue

                full.append(cleaned)

                # SSE data must be line-safe
                safe = cleaned.replace("\r", "\\r").replace("\n", "\\n")
                yield f"data: {safe}\n\n".encode("utf-8")
        finally:
            if full:
                try:
                    history.add_ai_message("".join(full))
                    log.info(f"DDB assistant write OK session_id={sid}")
                except Exception as e:
                    log.exception(f"DDB assistant write FAILED session_id={sid}: {e}")

        yield b"event: end\ndata: ok\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": sid,
        },
    )