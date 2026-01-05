"""
utils.py
- includes: helper functions to simplify common patterns (eg math helpers)
"""

import time
def timer(fn):
    def _w(*a, **k):
        t0=time.time(); r=fn(*a,**k); print(f"{fn.__name__}: {time.time()-t0:.2f}s"); return r
    return _w
def clean_text(s: str) -> str:
    return " ".join(s.split())


def extract_text_from_stream_delta(delta) -> str:
    """Normalize provider-specific streaming deltas to user-visible text.

    Providers may return:
      - plain strings
      - dict parts like {"type": "text", "text": "..."}
      - lists of mixed parts (dicts/strings)

    This filters out non-user-visible parts (e.g., "thinking" / tool traces).
    """
    if delta is None:
        return ""

    if isinstance(delta, str):
        return delta

    if isinstance(delta, dict):
        return str(delta.get("text", ""))

    if isinstance(delta, list):
        parts: list[str] = []
        for p in delta:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    txt = p.get("text")
                    if txt:
                        parts.append(str(txt))
                continue
            if isinstance(p, str):
                parts.append(p)
        return "".join(parts)

    return str(delta)