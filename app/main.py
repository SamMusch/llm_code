from __future__ import annotations

import os
import sys
import uuid
import logging
import urllib.parse
from typing import AsyncIterator
from pathlib import Path

import mimetypes
import boto3

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

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
    """Best-effort text extraction for attachments.

    Current implementation supports plain text-ish files only.
    PDFs/Office docs are not parsed here.
    """
    suffix = path.suffix.lower()

    # Plain text formats
    if suffix in {".txt", ".md", ".json", ".csv", ".log", ".yaml", ".yml"}:
        try:
            data = path.read_bytes()
            # Try utf-8; fall back to latin-1 to avoid hard failures
            try:
                text = data.decode("utf-8")
            except Exception:
                text = data.decode("latin-1", errors="replace")
            return text[:max_chars]
        except Exception:
            return ""

    # Unsupported types: keep empty so we don't mislead
    return ""


# Health check endpoint for ALB
@app.get("/health")
async def health():
    return {"ok": True}

# Alias health check under /api to match ALB routing (/api*)
@app.get("/api/health")
async def api_health():
    return await health()


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




@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    # Auth is enforced at the ALB listener rule level for /app*
    # full_bleed lets the chat UI take the full viewport (no outer container/topbar spacing).
    return templates.TemplateResponse("app.html", {"request": request, "full_bleed": True})



# ALB+Cognito redirects here after successful auth.
# We do not need to process the authorization code in the app when ALB is doing
# authentication. Redirect users to the protected app page.
@app.get("/oauth2/idpresponse")
async def oauth2_idpresponse(request: Request):
    """ALB+Cognito redirects here after successful auth.

    We do not need to process the authorization code in the app when ALB is doing
    authentication. Redirect users to the protected app page.
    """
    return RedirectResponse(url="/app", status_code=302)



# Log out of ALB session + Cognito hosted UI.
# ALB uses its own session cookie; clearing it is handled by the ALB listener rule
# on `/logout*`. We still provide this route so the UI can link to `/logout` and
# consistently land users back on the site.
@app.get("/logout")
async def logout(request: Request):
    """Log out of ALB session + Cognito hosted UI.

    ALB uses its own session cookie; clearing it is handled by the ALB listener rule
    on `/logout*`. We still provide this route so the UI can link to `/logout` and
    consistently land users back on the site.
    """
    # Prefer explicit env vars; fall back to known values used in this deployment.
    domain = os.getenv("COGNITO_DOMAIN", "us-east-1bxivmfrzy.auth.us-east-1.amazoncognito.com")
    client_id = os.getenv("COGNITO_CLIENT_ID", os.getenv("ALB_COGNITO_CLIENT_ID", "2fje718ip1ntli3jocuta07191"))
    post_logout = os.getenv("POST_LOGOUT_REDIRECT", "https://chat.sammusch-ds.com/")

    # Cognito logout endpoint
    qs = urllib.parse.urlencode({"client_id": client_id, "logout_uri": post_logout})
    return RedirectResponse(url=f"https://{domain}/logout?{qs}", status_code=302)


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

# Alias non-streaming chat under /api to match UI calls (/api/chat)
@app.post("/api/chat")
async def api_chat_non_stream(request: Request):
    return await chat_non_stream(request)


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


class _TaggedBlockStripper:
    """Stateful stripper for <thinking>/<analysis> blocks across streamed chunks."""

    _PAIRS = [
        ("<thinking>", "</thinking>"),
        ("<analysis>", "</analysis>"),
    ]

    def __init__(self) -> None:
        self._in_block = False
        self._carry = ""
        self._max_tag = max(max(len(a), len(b)) for a, b in self._PAIRS)
        self._carry_len = max(16, self._max_tag)  # extra safety for partial tags

    def feed(self, text: str) -> str:
        if not text:
            return ""

        s = (self._carry or "") + text
        s_low = s.lower()

        out_parts: list[str] = []
        i = 0

        while i < len(s):
            if not self._in_block:
                # Find earliest opening tag among pairs
                next_pos = None
                next_open = None
                next_close = None
                for open_tag, close_tag in self._PAIRS:
                    p = s_low.find(open_tag, i)
                    if p != -1 and (next_pos is None or p < next_pos):
                        next_pos = p
                        next_open = open_tag
                        next_close = close_tag

                if next_pos is None:
                    break

                # Emit everything before the tag
                out_parts.append(s[i:next_pos])
                i = next_pos + len(next_open)
                self._in_block = True
            else:
                # Skip until we find the corresponding close tag
                found_close = False
                for open_tag, close_tag in self._PAIRS:
                    p = s_low.find(close_tag, i)
                    if p != -1:
                        i = p + len(close_tag)
                        self._in_block = False
                        found_close = True
                        break

                if not found_close:
                    # Need more data to complete the closing tag
                    break

        # Remainder handling with carry to catch partial tags across boundaries
        remainder = s[i:]

        if self._in_block:
            # Discard content inside the block; keep only a tail for partial close-tag matching
            self._carry = remainder[-self._carry_len :]
            return "".join(out_parts)

        # Not in a block: keep a tail as carry to detect partial open tags
        if len(remainder) <= self._carry_len:
            self._carry = remainder
            return "".join(out_parts)

        self._carry = remainder[-self._carry_len :]
        out_parts.append(remainder[: -self._carry_len])
        return "".join(out_parts)

    def flush(self) -> str:
        """Flush any safe remaining text (only when not inside a block)."""
        if self._in_block:
            self._carry = ""
            return ""
        out = self._carry
        self._carry = ""
        return out


def _ddb_table():
    """Return the DynamoDB table used for chat history."""
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    table_name = os.getenv("DDB_TABLE", "rag_chat_history")
    return boto3.resource("dynamodb", region_name=region).Table(table_name)


def _five_word_title(text: str) -> str:
    """Derive a short sidebar title from the first user prompt.

    Note: this is not a semantic summary; it's the first 5 words cleaned.
    """
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
async def api_list_sessions(limit: int = 50):
    """List recent session_ids for the sidebar.

    Implementation note: uses Scan (OK for now). For scale, add a Sessions table or a GSI.
    """
    table = _ddb_table()

    # Small table expected during pilot; scan all items.
    resp = table.scan()

    # Track: last_ts (for ordering) + first user message (for title)
    sessions: dict[str, dict] = {}

    for item in resp.get("Items", []) or []:
        sid = item.get("session_id")
        if not sid:
            continue

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
async def api_get_session(session_id: str, limit: int = 200):
    """Load messages for a session (oldest -> newest)."""
    history = DynamoDBChatMessageHistory(session_id=session_id, limit=limit)
    msgs = history.messages
    return {
        "session_id": session_id,
        "messages": lc_messages_to_dicts(msgs),
    }


@app.get("/chat/stream")
async def chat_stream(
    request: Request,
    message: str = "",
    session_id: str | None = None,
    file_ids: str | None = None,
):
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

        # If UI provided file_ids, best-effort inject plain text contents into context.
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
                    # Don't pretend we parsed unsupported formats
                    parts.append(f"[Attachment: {name}] (not parsed; unsupported file type)")
                    continue

                parts.append(f"[Attachment: {name}]\n{text}")

            if parts:
                injected = "\n\n".join(parts)
                context_msgs = (
                    [{"role": "system", "content": "User attached files (best-effort):\n\n" + injected}]
                    + context_msgs
                )
    except Exception as e:
        log.exception(f"DDB read FAILED session_id={sid}: {e}")
        context_msgs = [{"role": "user", "content": msg}]

    async def event_gen() -> AsyncIterator[bytes]:
        yield b"event: start\ndata: ok\n\n"

        full: list[str] = []
        stripper = _TaggedBlockStripper()
        try:
            async for chunk in llm_stream(context_msgs):
                # Strip tagged blocks across chunk boundaries (e.g., <thinking> ... </thinking>)
                cleaned = stripper.feed(chunk)
                if not cleaned:
                    continue

                full.append(cleaned)

                # SSE data must be line-safe
                safe = cleaned.replace("\r", "\\r").replace("\n", "\\n")
                yield f"data: {safe}\n\n".encode("utf-8")
        finally:
            # Flush any remaining safe text (only when not inside a tagged block)
            tail = stripper.flush()
            if tail:
                full.append(tail)
                safe_tail = tail.replace("\r", "\\r").replace("\n", "\\n")
                yield f"data: {safe_tail}\n\n".encode("utf-8")

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

# Alias streaming chat under /api to match UI calls (/api/chat/stream)
@app.get("/api/chat/stream")
async def api_chat_stream(
    request: Request,
    message: str = "",
    session_id: str | None = None,
    file_ids: str | None = None,
):
    return await chat_stream(request, message=message, session_id=session_id, file_ids=file_ids)


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