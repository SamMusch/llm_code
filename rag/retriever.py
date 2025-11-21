"""
retriever.py
indexing pipeline
"""

from pathlib import Path
from typing import Iterable
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import Settings


EXT_TO_LOADER = {               # simple text loader registry
    ".txt": TextLoader,
    ".md": TextLoader,
    ".rst": TextLoader,
    ".html": UnstructuredFileLoader,
    ".pdf": UnstructuredFileLoader,
    ".docx": UnstructuredFileLoader,
}

def _load_documents(input_dir: Path) -> list:   # underscore --> only for this module
    """
    1. recursively scans for files with known extensions (eg .md)
    2. applies loader per type
    3. returns list of LangChain Document objects ready for chunking and indexing
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Docs dir not found: {input_dir}")
    docs = []
    for ext, Loader in EXT_TO_LOADER.items():
        loader = DirectoryLoader(
            str(input_dir), 
            glob=f"**/*{ext}",       # searches recursively for all files w that ext
            loader_cls=Loader,       # TextLoader, UnstructuredFileLoader, etc
            show_progress=True) 
        docs.extend(loader.load())
    return docs


def build_index(cfg: Settings, input_dir: Path | None = None) -> None:
    """load → chunk → embed → store"""
    input_dir = Path(input_dir) if input_dir else cfg.docs_dir
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.faiss_dir.mkdir(parents=True, exist_ok=True)

    docs = _load_documents(input_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, 
        chunk_overlap=cfg.chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    vs = FAISS.from_documents(chunks, embeddings)     # FAISS for now
    vs.save_local(str(cfg.faiss_dir))


def load_retriever(cfg: Settings, k: int | None = None):
    """reload saved FAISS index → attach embedding model → return retriever interface for semantic lookup."""
    k = k or cfg.k  # top k docs
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    vs = FAISS.load_local(str(cfg.faiss_dir), embeddings, allow_dangerous_deserialization=True)
    return vs.as_retriever(search_kwargs={"k": k})