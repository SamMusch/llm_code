from typing import List, Tuple, TypedDict
from langgraph.graph import StateGraph, END
from tenacity import retry, wait_exponential_jitter, stop_after_attempt
from langchain_core.documents import Document
from .config import Settings
from .retriever import load_retriever
from .generator import answer as llm_answer

# Prefer LRAT tool implementation when the template repo is present
try:
    # LRAT layout: src/retrieval_graph/tools.py
    from retrieval_graph.tools import read_doc_by_name as READ_DOC_TOOL
except Exception:
    from .tools import read_doc_by_name as READ_DOC_TOOL

class RAGState(TypedDict, total=False):
    question: str
    cfg: dict
    docs: List[Document]
    answer: str
    tried_alt: bool

def _retrieve(state: RAGState) -> RAGState:
    cfg = Settings(**state["cfg"])
    retriever = load_retriever(cfg)
    docs = retriever.invoke(state["question"])
    return {**state, "docs": docs}

@retry(wait=wait_exponential_jitter(0.5, 2.0), stop=stop_after_attempt(3))
def _generate_with_retry(question: str, docs: List[Document], cfg: Settings) -> str:
    return llm_answer(question, docs, cfg)

def _generate(state: RAGState) -> RAGState:
    cfg = Settings(**state["cfg"])
    ans = _generate_with_retry(state["question"], state.get("docs", []), cfg)
    return {**state, "answer": ans}

def _guard_or_route(state: RAGState):
    cfg = Settings(**state["cfg"])
    if not cfg.hallucination_guard:
        return "ok"

    docs = state.get("docs", [])
    ans = state.get("answer", "").strip()

    # Guard 1: no evidence
    if len(docs) == 0 or not ans:
        return "fallback"

    # Guard 2: oversized context attempt
    ctx_len = sum(len(d.page_content) for d in docs)
    if ctx_len > cfg.max_context_chars and not state.get("tried_alt"):
        return "fallback"
    return "ok"

def _fallback(state: RAGState) -> RAGState:
    # Simple tool hop: try to fetch a specifically named doc by keyword
    cfg = Settings(**state["cfg"])
    q = state["question"]
    snippet = READ_DOC_TOOL.invoke({"name": q, "cfg": cfg.model_dump()})
    if snippet:
        # treat the snippet as a doc and regenerate
        pseudo_doc = Document(page_content=snippet, metadata={"source": "tool:read_doc_by_name"})
        ans = _generate_with_retry(q, [pseudo_doc], cfg)
        return {**state, "answer": ans, "docs": [pseudo_doc], "tried_alt": True}
    # If tool fails, return an abstention
    return {**state, "answer": "I don't know.", "tried_alt": True}

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)
    g.add_node("fallback", _fallback)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", _guard_or_route, {"ok": END, "fallback": "fallback"})
    g.add_edge("fallback", END)
    return g.compile()

def run(question: str, cfg: Settings | None = None) -> Tuple[str, List[Document]]:
    _cfg = (cfg or Settings.load()).model_dump()
    app = build_graph()
    final = app.invoke({"question": question, "cfg": _cfg})
    return final.get("answer", ""), final.get("docs", [])