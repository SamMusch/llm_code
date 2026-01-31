# steps.py: DEPRECATED shim
from __future__ import annotations

from .agent import (
    StepSpanCallbackHandler,
    attach_step_sink,
    detach_step_sink,
    drain_step_sink,
    emit_step,
)

__all__ = [
    "StepSpanCallbackHandler",
    "attach_step_sink",
    "detach_step_sink",
    "drain_step_sink",
    "emit_step",
]