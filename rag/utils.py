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