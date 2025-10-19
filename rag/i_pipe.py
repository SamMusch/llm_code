# -*- coding: utf-8 -*-

import os
#folder_path = '/Users/Sam/Downloads/_work/__RAG-Text_Github'
#vector_store_path = folder_path + '/Data'

# Automatically locate the project root (one level above /rag)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # path/to/__RAG-Text_Github/rag
folder_path = os.path.abspath(os.path.join(BASE_DIR, ".."))  # path/to/__RAG-Text_Github
vector_store_path = os.path.join(folder_path, "Data")


ix_name = "CWC_index"
mod = "sentence-transformers/all-mpnet-base-v2"


import warnings
warnings.filterwarnings("ignore")
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# N_chars_display = 100

# ------------------------------------------------------------------
# 0. Functions | summarize_chunk_lengths
# ------------------------------------------------------------------
# All other scripts import this single loader.
# Use HF embeddings --> Load and return the persisted FAISS vector store
def get_db(vs_path = vector_store_path, index_name = ix_name, model_name = mod):
    p = Path(vs_path)
    if not p.exists():
        raise FileNotFoundError(f"FAISS folder not found: {p.resolve()}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    try:                                                            # Prefer newer signature; fall back to older LC versions
        if index_name is not None:
            return FAISS.load_local(
                vs_path, embeddings, index_name=index_name,
                allow_dangerous_deserialization=True)
        return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
    except TypeError:
        if index_name is not None:
            return FAISS.load_local(folder_path=vs_path, embeddings=embeddings, index_name=index_name)
        return FAISS.load_local(folder_path=vs_path, embeddings=embeddings)


# ------------------------------------------------------------------
# 0. Functions | summarize_chunk_lengths
# ------------------------------------------------------------------
def summarize_chunk_lengths(data, title="Chunk Lengths"):
    stats = {
        "Median": round(np.median(data), 0),
        "Mean": round(np.mean(data), 0),
        "Min": round(np.min(data), 0),
        "Max": round(np.max(data), 0),
        "75th %ile": round(np.percentile(data, 75), 0),
        "25th %ile": round(np.percentile(data, 25), 0),}
    summary_text = ""
    for k, v in stats.items():
        summary_text += f"{k:<10}: {v:.2f}\n"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data)
    ax.set_title(title)
    ax.set_xlabel("Chunk Lengths")
    ax.set_ylabel("Values")
    plt.text(1.15,0.95,summary_text,transform=ax.transAxes,
        fontsize=10,verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#999"),)
    plt.tight_layout()
    plt.show()