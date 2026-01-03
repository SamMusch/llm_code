from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


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
    """
    Replace this stub with your existing llm_code call that yields tokens/chunks.
    Keep the interface: async iterator of string chunks.
    """
    # Example shape (you will adapt to your real code):
    # from llm_code.rag.agent import chat_stream
    # async for chunk in chat_stream(message=message):
    #     yield chunk

    demo = f"Stub response. You said: {message}"
    for ch in demo:
        await asyncio.sleep(0.01)
        yield ch


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