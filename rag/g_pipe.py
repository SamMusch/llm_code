# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# 1. Load Vector Index
# ------------------------------------------------------------------
import os, re
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from . import HF_MODEL, get_hf_client
from .i_pipe import vector_store_path, ix_name, get_db, mod
from .adv_tech import hybrid_search, metadata_filter

from langchain_community.vectorstores import FAISS  # the vector index --> stores embeddings
from langchain_openai import OpenAIEmbeddings       # the embeddings --> encodes text
from langchain_openai import ChatOpenAI             # G

hf_client = get_hf_client()                         # OK to keep module-level here (it’s the generator)
llm = HuggingFaceEndpoint(                          # single LLM instance for this module
    repo_id=HF_MODEL, temperature=0.2, max_new_tokens=256)

#from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------------------------------------------
# clean text
# ------------------------------------------------------------------
def clean_text(text):
    text = text.replace('\xa0', ' ')        # Replace non-breaking space with regular space
    text = re.sub(r'<[^>]+>', '', text)     # Rm HTML tags
    text = re.sub(r'\[.*?\]', '', text)     # Rm references inside square brackets
    text = ' '.join(text.split())           # Rm extra spaces & newline characters
    return text

# ------------------------------------------------------------------
# tex gen
# ------------------------------------------------------------------
def _hf_generate(prompt: str) -> str:
    """Generate with HF InferenceClient.
    - Try text_generation (for plain TG models)
    - Fallback to chat_completion (for 'conversational' models like Mistral-7B-Instruct)
    """
    try:
        return hf_client.text_generation(
            prompt,
            max_new_tokens=256,
            temperature=0.2,
            # do_sample=False  # optional
        ).strip()
    except ValueError as e:
        if "Supported task: conversational" not in str(e):
            raise
        resp = hf_client.chat_completion(                           # Chat-only model path
            messages=[
                {"role": "system", "content": "Answer concisely using the provided context."},
                {"role": "user", "content": prompt},],
            max_tokens=256,
            temperature=0.2,)
        
        choice = resp.choices[0]                                    # HuggingFace returns OpenAI-like structure
        msg = getattr(choice, "message", None)
        content = getattr(msg, "content", None) if msg else None
        if content:
            return content.strip()
        return str(resp)                                            # Fallback (very old return shapes)


# ------------------------------------------------------------------
# R --> A --> G
# ------------------------------------------------------------------
# k = retrieval size (ie # of docs to pull)
def rag_function(query, k=5, db=None, db_path=vector_store_path, index_name=ix_name):
    db = db or get_db(db_path, index_name)          # allow a preloaded db or load on demand

    # 1) R
    retrieved_docs = db.similarity_search(query, k=k)
    retrieved_context = [clean_text(doc.page_content) for doc in retrieved_docs]

    # 2) A
    augmented_prompt = (
        "Given the context below answer the question.\n"
        f"Question: {query}\n\n"
        "Context:\n"
        + "\n\n".join(retrieved_context)
        + "\n\nAnswer (be concise):")

    # 3) G (HF)
    response = _hf_generate(augmented_prompt).strip()
    return retrieved_context, response