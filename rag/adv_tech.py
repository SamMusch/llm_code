# -*- coding: utf-8 -*-

# ------------------------------------------------------------------
# 0. prep
# ------------------------------------------------------------------

import os
from typing import Sequence, List, Optional
from .i_pipe import get_db, vector_store_path, ix_name
from . import HF_MODEL, HF_EMBED_MODEL
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import faiss
from langchain_community.vectorstores import FAISS

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer

from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEndpoint

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever     # Query expansion via MultiQueryRetriever

# ------------------------------------------------------------------
# 0. load
# ------------------------------------------------------------------
def load_html_text(url):
    loader = AsyncHtmlLoader(url)
    html_docs = loader.load()
    text_docs = Html2TextTransformer().transform_documents(html_docs)
    return text_docs[0].page_content if text_docs else ""

def summarize_text(text, llm, max_chars):
    if not llm:
        return ""
    prompt = "Summarize in one concise paragraph. Document:\n" + text[:max_chars]
    out = llm.invoke(prompt)
    return out.strip() if isinstance(out, str) else str(out).strip()

def split_text(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


# ------------------------------------------------------------------
# 1.1.1. Pre-R | Index Optimization | Context Enriched Chunking (method)
# ------------------------------------------------------------------
def build_context_enriched_index(chunks, context, embeddings, sep):
    enriched = [f"{context}{sep}{c}" if context else c for c in chunks]
    dim = len(embeddings.embed_query("probe"))
    index = faiss.IndexFlatIP(dim)
    store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    store.add_texts(list(enriched))
    return store

def retrieve_from_index(vs, query, k):
    return vs.similarity_search(query, k=k)

# ------------------------------------------------------------------
# 1.1.2 Pre-R | Index Optimization | Metadata Enhancement (method)
# ------------------------------------------------------------------
def _empty_meta():
    return {"topic": "", "entities": [], "keywords": [], "summary": ""}

def _meta_chain(llm):
    parser = JsonOutputParser()  # handles JSON parsing; gives format instructions
    prompt = PromptTemplate(
        template=(
            "Extract metadata from the input and return ONLY JSON with keys:\n"
            "topic (string), entities (list), keywords (list), summary (string).\n"
            "{format_instructions}\n\nInput:\n{input}"
        ),
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | llm | parser  # Runnable: dict -> dict

def _normalize_meta(d):
    m = _empty_meta()
    if not isinstance(d, dict):
        return m
    m["topic"] = str(d.get("topic", "")).strip()
    m["summary"] = str(d.get("summary", "")).strip()
    ents = d.get("entities", []) or []
    keys = d.get("keywords", []) or []
    m["entities"] = [str(x).strip() for x in ents if str(x).strip()]
    m["keywords"] = [str(x).strip() for x in keys if str(x).strip()]
    # dedupe, preserve order
    m["entities"] = list(dict.fromkeys(m["entities"]))
    m["keywords"] = list(dict.fromkeys(m["keywords"]))
    return m

def extract_fixed_metadata_from_chunk(chunk_text, llm):
    if not llm or not chunk_text:
        return _empty_meta()
    try:
        chain = _meta_chain(llm)
        out = chain.invoke({"input": chunk_text})
        return _normalize_meta(out)
    except Exception:
        return _empty_meta()

def extract_fixed_metadata_from_query(query_text, llm):
    if not llm or not query_text:
        return _empty_meta()
    try:
        chain = _meta_chain(llm)
        out = chain.invoke({"input": query_text})
        return _normalize_meta(out)
    except Exception:
        return _empty_meta()

def metadata_filter(doc_metadata):
    m = _normalize_meta(doc_metadata or {})
    return {k: v for k, v in m.items() if (isinstance(v, str) and v) or (isinstance(v, list) and v)}


# ------------------------------------------------------------------
# 1.2.1. Pre-R | QUERY Optimization | Query Expansion (method)
# ------------------------------------------------------------------

# Q Expansion: 
    # Q --> *enriched* --> retrieves more relevant information --> increases recall.

# 3 prompts. All make API call --> pass A to LLM.
    # t1 | expansion_prompt
    # t2 | step_back_expansion_prompt | NOT INCLUDING
    # t3 | sub_query_expansion_prompt | NOT INCLUDING

def build_mq_retriever(vs, llm, k):
    return MultiQueryRetriever.from_llm(
        retriever=vs.as_retriever(                          # vs: FAISS (or any) vector store
            search_kwargs={"k": k}),                        # k: top-k per subquery
        llm=llm,)                                           # llm: HuggingFaceEndpoint(...) or any LangChain LLM

def query_expansion(original_query, num, llm, vs, k):
    mqr = build_mq_retriever(vs, llm, k)
    return mqr.get_relevant_documents(original_query)

# ------------------------------------------------------------------
# 1.2.2. Pre-R | QUERY Optimization | Query Transformation (method)
# ------------------------------------------------------------------

# Q Transformation: 
    # Q --> *transformed* --> retrieves more relevant information --> increases recall.

# 2 prompts: All make API call --> pass A to LLM.
    # system_prompt
    # hyde_prompt

def query_transformation_hyde(original_query, llm, embeddings, system_prompt):
    hyde_prompt = (
        f"{system_prompt}\n\n"
        f"Generate a concise, factual answer to the question below.\n"
        f"Return only the answer text.\n\n"
        f"Question: {original_query}")
    answer = llm.invoke(hyde_prompt)
    answer = answer.strip() if isinstance(answer, str) else str(answer).strip()
    hyde_embedding = embeddings.embed_query(answer)
    # embedding_dimension = len(hyde_embedding)  # available if you need it in the caller
    return answer, hyde_embedding


# ------------------------------------------------------------------
# 2.1. R | Hybrid Retrieval
# ------------------------------------------------------------------
# Objective: Align Q & KB with the R algo.

def build_dense_index(chunks, embeddings):
    # embeddings: e.g., HuggingFaceEmbeddings(...)
    return FAISS.from_texts(chunks, embeddings)

def build_sparse_retriever(chunks):
    docs = [Document(page_content=c) for c in chunks]
    return BM25Retriever.from_documents(docs)

def hybrid_search(query, k, vector_store, bm25_retriever):
    dense_results = vector_store.similarity_search(query, k=k)
    sparse_results = bm25_retriever.get_relevant_documents(query)[:k]
    combined_results = [("dense", d.page_content) for d in dense_results] + \
                       [("sparse", d.page_content) for d in sparse_results]
    return combined_results


# ------------------------------------------------------------------
# 3. Post-R | Compression (Minimal)
# ------------------------------------------------------------------

def compress_document(doc_text, llm):
    if not llm or not doc_text:
        return ""
    prompt = (
        "Compress the following document into very short sentences, "
        "keeping only the essential information. Return only the compressed text.\n\n"
        f"{doc_text}"
    )
    out = llm.invoke(prompt)
    return out.strip() if isinstance(out, str) else str(out).strip()

