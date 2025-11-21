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
    docs: List[Document]
    answer: str
    tried_alt: bool


def _retrieve(state: RAGState, cfg: Settings) -> RAGState:
    retriever = load_retriever(k=cfg.k)
    docs = retriever.invoke(state["question"])
    return {**state, "docs": docs}


@retry(wait=wait_exponential_jitter(0.5, 2.0), stop=stop_after_attempt(3))
def _generate_with_retry(question: str, docs: List[Document], cfg: Settings) -> str:
    return llm_answer(question, docs, cfg)


def _generate(state: RAGState, cfg: Settings) -> RAGState:
    ans = _generate_with_retry(state["question"], state.get("docs", []), cfg)
    return {**state, "answer": ans}


def _guard_or_route(state: RAGState, cfg: Settings):
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


def _fallback(state: RAGState, cfg: Settings) -> RAGState:
    # Simple tool hop: try to fetch a specifically named doc by keyword
    q = state["question"]
    snippet = READ_DOC_TOOL.invoke({"name": q, "cfg": cfg.model_dump()})
    if snippet:
        # treat the snippet as a doc and regenerate
        pseudo_doc = Document(page_content=snippet, metadata={"source": "tool:read_doc_by_name"})
        ans = _generate_with_retry(q, [pseudo_doc], cfg)
        return {**state, "answer": ans, "docs": [pseudo_doc], "tried_alt": True}
    # If tool fails, return an abstention
    return {**state, "answer": "I don't know.", "tried_alt": True}


def build_graph(cfg: Settings | None = None):
    cfg = cfg or Settings.load()
    retriever = load_retriever(k=cfg.k)

    g = StateGraph(RAGState)

    def retrieve_with_cfg(state: RAGState) -> RAGState:
        docs = retriever.invoke(state["question"])
        return {**state, "docs": docs}

    def generate_with_cfg(state: RAGState) -> RAGState:
        return _generate(state, cfg)

    def fallback_with_cfg(state: RAGState) -> RAGState:
        return _fallback(state, cfg)

    def guard_with_cfg(state: RAGState):
        return _guard_or_route(state, cfg)

    g.add_node("retrieve", retrieve_with_cfg)
    g.add_node("generate", generate_with_cfg)
    g.add_node("fallback", fallback_with_cfg)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", guard_with_cfg, {"ok": END, "fallback": "fallback"})
    g.add_edge("fallback", END)
    return g.compile()


def run(question: str, cfg: Settings | None = None) -> Tuple[str, List[Document]]:
    _cfg = cfg or Settings.load()
    app = build_graph(_cfg)
    final = app.invoke({"question": question})
    return final.get("answer", ""), final.get("docs", [])