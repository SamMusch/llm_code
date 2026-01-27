from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from langchain.agents.middleware import AgentState, before_model
from langchain_core.messages import BaseMessage

from .intent import last_human_text


def concat_text(messages: List[BaseMessage]) -> str:
    parts: List[str] = []
    for m in messages:
        c = getattr(m, "content", "")
        parts.append(c if isinstance(c, str) else str(c))
    return "\n".join(parts)


def has_retrieved_sources(messages: List[BaseMessage]) -> bool:
    # Our retriever formats sources like: "Source: Data/docs/<name>" and/or "[1] Source: ..."
    return "source:" in concat_text(messages).lower()


def keyword_overlap_score(a: str, b: str) -> float:
    def toks(s: str) -> set[str]:
        s = s.lower()
        s = re.sub(r"[^a-z0-9_\s]", " ", s)
        raw = [t for t in s.split() if len(t) >= 3]
        stop = {
            "the","and","for","with",
            "that","this","from","into",
            "what","did","does","how",
            "your","you","our","about",
            "when","where","which","who",
        }
        return {t for t in raw if t not in stop}

    ta = toks(a)
    tb = toks(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union


@before_model
def trim_history(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Keep the newest messages within a rough token budget.

    Notes:
    - We avoid LLM-based summarization here to keep middleware side-effect free.
    - If you want true summarization, add a separate summarizer node in LangGraph later.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    # Prefer explicit config if present; otherwise default to a safe cap.
    max_chars = int(os.environ.get("MAX_CONTEXT_CHARS", "60000"))
    max_tokens = max(2000, max_chars // 4)

    def _tokens(msg: BaseMessage) -> int:
        content = getattr(msg, "content", "")
        text = content if isinstance(content, str) else str(content)
        return max(1, len(text) // 4)

    total = sum(_tokens(m) for m in messages)
    if total <= max_tokens:
        return None

    kept: List[BaseMessage] = []
    running = 0
    for msg in reversed(messages):
        t = _tokens(msg)
        if kept and running + t > max_tokens:
            break
        kept.append(msg)
        running += t
    kept.reverse()

    if len(kept) < len(messages):
        kept = [
            {
                "role": "system",
                "content": "Note: earlier conversation history was trimmed for context-length limits.",
            }
        ] + kept

    return {"messages": kept}


@before_model
def context_relevance_hint(state: AgentState, runtime) -> Dict[str, Any] | None:
    """Heuristic relevance check between the last user question and retrieved context.

    Controlled via env var:
      - CONTEXT_RELEVANCE_HINT=true|false
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    enabled = os.environ.get("CONTEXT_RELEVANCE_HINT", "true").lower() in {"1", "true", "yes", "y"}
    if not enabled:
        return None

    human_q = last_human_text(messages)
    if not human_q:
        return None

    if not has_retrieved_sources(messages):
        return None

    score = keyword_overlap_score(human_q, concat_text(messages))
    if score < 0.03:
        return {
            "messages": messages
            + [
                {
                    "role": "system",
                    "content": (
                        f"ContextRelevance=LOW (heuristic score={score:.3f}). "
                        "Prefer another search_docs call with a refined query or ask one clarifying question "
                        "before giving a definitive answer."
                    ),
                }
            ]
        }

    return {
        "messages": messages
        + [
            {
                "role": "system",
                "content": f"ContextRelevance=OK (heuristic score={score:.3f}).",
            }
        ]
    }
