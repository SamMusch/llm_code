"""
retriever.py
Indexing and retrieval utilities for the RAG pipeline.
"""

from pathlib import Path
from typing import Iterable

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rag.config import cfg
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.email import (
    UnstructuredEmailLoader,
    OutlookMessageLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Simple loader registry by extension
EXT_TO_LOADER = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".rst": TextLoader,
    ".html": UnstructuredFileLoader,
    ".pdf": UnstructuredFileLoader,
    ".docx": UnstructuredFileLoader,
    ".ppt": UnstructuredFileLoader,
    ".pptx": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".eml": UnstructuredEmailLoader,
    ".msg": OutlookMessageLoader,
}


def _load_documents(input_dir: Path) -> list:
    """
    1. Recursively scan for files with known extensions.
    2. Apply the appropriate loader per type.
    3. Return a list of LangChain Document objects ready for chunking and indexing.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Docs dir not found: {input_dir}")

    docs: list = []
    for ext, Loader in EXT_TO_LOADER.items():
        loader = DirectoryLoader(
            str(input_dir),
            glob=f"**/*{ext}",  # recursively search for all files with this extension
            loader_cls=Loader,  # TextLoader, UnstructuredFileLoader, etc.
            show_progress=True,
        )
        try:
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            print(f"[loader-error] Failed to load files with extension {ext}: {e}")
            continue

    return docs


def build_index(docs_dir: Path | None = None) -> None:
    """load → chunk → embed → store"""
    docs_dir = Path(docs_dir) if docs_dir else cfg.docs_dir

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.faiss_dir.mkdir(parents=True, exist_ok=True)

    docs = _load_documents(docs_dir)
    if not docs:
        raise RuntimeError(
            f"No documents found in {docs_dir}. "
            f"Supported extensions: {', '.join(EXT_TO_LOADER.keys())}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    vs = FAISS.from_documents(chunks, embeddings)  # FAISS for now
    vs.save_local(str(cfg.faiss_dir))


def load_retriever(k: int | None = None):
    """
    Reload saved FAISS index → attach embedding model → return retriever interface
    for semantic lookup.
    """
    k = k or cfg.k  # top k docs

    # Guardrails: fail fast with a clear message if the index is missing or incomplete.
    if not cfg.faiss_dir.exists():
        raise RuntimeError(
            f"FAISS index directory not found at {cfg.faiss_dir}. "
            "Build it first with: `python -m rag.cli index`."
        )

    index_file = cfg.faiss_dir / "index.faiss"
    if not index_file.exists():
        raise RuntimeError(
            f"FAISS index file not found at {index_file}. "
            "Build or rebuild it with: `python -m rag.cli index`."
        )

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    try:
        vs = FAISS.load_local(
            str(cfg.faiss_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FAISS index from {cfg.faiss_dir}. "
            "Try rebuilding it with: `python -m rag.cli index`."
        ) from e

    print(f"[retriever] Loaded index from {cfg.faiss_dir} (k={k})")

    return vs.as_retriever(search_kwargs={"k": k})