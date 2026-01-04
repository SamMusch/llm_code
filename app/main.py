from __future__ import annotations

from typing import AsyncIterator

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rag.agent import get_agent
from rag.config import get_settings


app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# Health check endpoint for ALB
@app.get("/health")
async def health():
    return {"ok": True}


def is_authed(request: Request) -> bool:
    return request.cookies.get("demo_auth") == "1"


async def llm_stream(message: str) -> AsyncIterator[str]:
    """Yield text chunks from the real LangChain/LangGraph agent (Bedrock in ECS via env)."""
    cfg = get_settings()
    agent = get_agent(cfg)

    # Match the same input shape used by rag/agent.py::run_agent
    inputs = {"messages": [{"role": "user", "content": message}]}

    async for event in agent.astream_events(inputs, version="v2"):
        if event.get("event") != "on_chat_model_stream":
            continue

        chunk = (event.get("data") or {}).get("chunk")
        if chunk is None:
            continue

        delta = getattr(chunk, "content", None)
        if not delta:
            continue

        # Some providers return lists of content parts; normalize to string.
        if isinstance(delta, list):
            delta = "".join(str(p) for p in delta)

        yield str(delta)


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
    async for chunk in llm_stream(message):
        out.append(chunk)
    return {"text": "".join(out)}


@app.get("/chat/stream")
async def chat_stream(message: str = ""):
    """
    SSE endpoint.
    Client connects with: /chat/stream?message=...
    """
    msg = (message or "").strip()
    if not msg:
        async def empty() -> AsyncIterator[bytes]:
            yield b"event: error\ndata: Missing message\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    async def event_gen() -> AsyncIterator[bytes]:
        yield b"event: start\ndata: ok\n\n"
        async for chunk in llm_stream(msg):
            # SSE data must be line-safe; keep it simple for now.
            safe = chunk.replace("\n", "\\n")
            yield f"data: {safe}\n\n".encode("utf-8")
        yield b"event: end\ndata: ok\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )