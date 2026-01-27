# steps.py: callback handler (OTEL span events + optional Loki-friendly JSON logs)
# wired in app/main.py

from __future__ import annotations

import json
import logging
import time
import contextvars
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace

log = logging.getLogger("rag.steps")

# --- In-memory step sink (per-request) ---
# Used by app/main.py SSE stream to surface middleware/tool steps in the UI.
_STEP_SINK: contextvars.ContextVar[Optional[list[dict[str, Any]]]] = contextvars.ContextVar(
    "STEP_SINK", default=None
)

def attach_step_sink(sink: list[dict[str, Any]]):
    """Attach a per-request sink for step events."""
    return _STEP_SINK.set(sink)

def detach_step_sink(token) -> None:
    """Detach the per-request sink."""
    try:
        _STEP_SINK.reset(token)
    except Exception:
        pass

def emit_step(kind: str, name: str, status: str, payload: Any | None = None) -> None:
    """Append a step event to the active sink (if any).

    kind: 'middleware' | 'tool' | 'model' | ...
    status: 'start' | 'ok' | 'error' | ...
    """
    sink = _STEP_SINK.get()
    if sink is None:
        return
    sink.append(
        {
            "ts": int(time.time() * 1000),
            "kind": kind,
            "name": name,
            "status": status,
            "payload": payload,
        }
    )

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
    """Emits LangChain/LangGraph step markers into the *current* OTEL span as events."""
    def __init__(self, *, max_chars: int = 2000, emit_logs: bool = True):
        self.max_chars = max_chars
        self.emit_logs = emit_logs
        self._t0: dict[str, float] = {}
        self._name: dict[str, str] = {}

    def _span(self):
        span = trace.get_current_span()
        # Avoid exceptions when there is no active/recording span.
        if not span or not span.is_recording():
            return None
        return span

    def _event(self, name: str, attrs: dict[str, Any]):
        span = self._span()
        if span is None:
            return

        span.add_event(
            name,
            {k: str(v) for k, v in attrs.items() if v is not None},
        )

        if self.emit_logs:
            # Loki-friendly structured log; formatter can emit `step` as JSON.
            log.info(name, extra={"step": attrs})

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: str, **kwargs: Any):
        self._t0[run_id] = time.perf_counter()
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or ""
        self._name[run_id] = name
        self._event(
            "lc.tool.start",
            {
                "step_type": "tool",
                "name": name,
                "run_id": run_id,
                "input": _trunc(input_str, self.max_chars),
            },
        )

    def on_tool_end(self, output: str, *, run_id: str, **kwargs: Any):
        dt = int((time.perf_counter() - self._t0.pop(run_id, time.perf_counter())) * 1000)
        name = self._name.pop(run_id, "")
        self._event(
            "lc.tool.end",
            {
                "step_type": "tool",
                "name": name,
                "run_id": run_id,
                "duration_ms": dt,
                "status": "ok",
                "output": _trunc(output, self.max_chars),
            },
        )

    def on_tool_error(self, error: BaseException, *, run_id: str, **kwargs: Any):
        dt = int((time.perf_counter() - self._t0.pop(run_id, time.perf_counter())) * 1000)
        name = self._name.pop(run_id, "")
        self._event(
            "lc.tool.end",
            {
                "step_type": "tool",
                "name": name,
                "run_id": run_id,
                "duration_ms": dt,
                "status": "error",
                "error": repr(error),
            },
        )