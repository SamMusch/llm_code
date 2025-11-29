"""
generator.py

The RAG pipeline.
"""

from typing import Sequence, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from .config import Settings


# These 3 functions:
    # _build_prompt()     creates prompt template.
    # _llm()              picks LLM client ---> returns the model object
    # _format_context()   concatenates doc text ---> one string.

def _build_prompt() -> ChatPromptTemplate:
    sys = (
        "You are a precise assistant. Answer using only the provided context. "
        "If unsure, say you don't know.\n\n"
        "Context:\n{context}")
    user = "Question: {question}"
    return ChatPromptTemplate.from_messages([("system", sys), ("user", user)])


def _llm(cfg: Settings):
    if cfg.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=cfg.llm_model, temperature=0)
    from langchain_community.llms import HuggingFaceEndpoint            # default: huggingface endpoint
    return HuggingFaceEndpoint(repo_id=cfg.llm_model, temperature=0)


def _format_context(docs: Sequence[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


# --------------------------------------------------------
# implements 3 functions above
def answer(question: str, docs: Sequence[Document], cfg: Settings) -> str:
    prompt = _build_prompt()
    llm = _llm(cfg)
    chain = prompt | llm
    out = chain.invoke({"question": question, "context": _format_context(docs)})
    return out.content if hasattr(out, "content") else str(out)

def generate(question: str, retriever, cfg: Settings) -> Tuple[str, list]:
    docs = retriever.invoke(question)
    ans = answer(question, docs, cfg)
    return ans, docs