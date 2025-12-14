"""
retriever.py
Indexing and retrieval utilities for the RAG pipeline.

retriever.py is now:
	•	a vector store / retriever module, and
	•	a tool provider for the agent.

"""

from pathlib import Path
from typing import Iterable
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from rag.config import cfg
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import (DirectoryLoader,TextLoader,UnstructuredFileLoader)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.email import (UnstructuredEmailLoader,OutlookMessageLoader,)
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
    loaded_count = 0
    dropped_count = 0
    error_count = 0
    for ext, Loader in EXT_TO_LOADER.items():
        loader = DirectoryLoader(
            str(input_dir),
            glob=f"**/*{ext}",  # recursively search for all files with this extension
            loader_cls=Loader,  # TextLoader, UnstructuredFileLoader, etc.
            show_progress=True,
        )
        try:
            loaded = loader.load()
            loaded_count += len(loaded)
            docs.extend(loaded)
            # Drop empty documents (no text extracted)
            for d in list(docs):
                if not getattr(d, "page_content", "").strip():
                    docs.remove(d)
                    dropped_count += 1
                    print(f"[loader-warning] Dropped empty document: {getattr(d, 'metadata', {}).get('source', 'unknown')}")
        except Exception as e:
            # Detect common Unstructured dependency errors
            msg = str(e).lower()
            if "unstructured" in msg and ("dependency" in msg or "import" in msg or "missing" in msg):
                print(
                    f"[loader-error] Unstructured dependency missing for {ext}. "
                    f"Install with: pip install 'unstructured[all-docs]'"
                )
            else:
                print(f"[loader-error] Failed to load files with extension {ext}: {e}")
            error_count += 1
            continue

    print(
        f"[loader-summary] Loaded: {loaded_count}, Dropped empty: {dropped_count}, Errors: {error_count}"
    )
    return docs


def build_index(docs_dir: Path | None = None, max_docs: int | None = None) -> None:
    """load → chunk → embed → store"""
    docs_dir = Path(docs_dir) if docs_dir else cfg.docs_dir

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.faiss_dir.mkdir(parents=True, exist_ok=True)

    docs = _load_documents(docs_dir)

    # Optional cap on number of documents to index (to avoid OOM on huge corpora).
    if max_docs is None:
        max_docs = getattr(cfg, "max_docs", None)

    if max_docs is not None and len(docs) > max_docs:
        print(
            f"[index-warning] Corpus has {len(docs)} docs; truncating to {max_docs} for indexing.")
        docs = docs[:max_docs]

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

    #embeddings = OpenAIEmbeddings(model=cfg.embedding_model)
    embeddings = OllamaEmbeddings(model=cfg.embedding_model, base_url="http://ollama:11434")

    vs = FAISS.from_documents(chunks, embeddings)  # FAISS for now
    vs.save_local(str(cfg.faiss_dir))


def load_retriever(k: int | None = None):
    """
    Reload saved FAISS index → attach embedding model → return retriever interface
    for semantic lookup.
    """
    k = k or cfg.k  # top k docs

    index_file = cfg.faiss_dir / "index.faiss"

    # Guardrails: fail fast or optionally rebuild if the index is missing.
    if not cfg.faiss_dir.exists() or not index_file.exists():
        auto_rebuild = getattr(cfg, "auto_rebuild_index", False)
        if auto_rebuild:
            print(
                f"[retriever] Index not found in {cfg.faiss_dir}. Auto-rebuilding with build_index()...")
            build_index()  # uses cfg.docs_dir and optional cfg.max_docs
        else:
            raise RuntimeError(
                f"FAISS index not found at {index_file}. "
                "Build it first with: `python -m rag.cli index`."
            )

    #embeddings = OpenAIEmbeddings(model=cfg.embedding_model)
    embeddings = OllamaEmbeddings(model=cfg.embedding_model, base_url="http://ollama:11434")

    try:
        vs = FAISS.load_local(
            str(cfg.faiss_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        # Empty-index guard: make sure we actually have vectors.
        faiss_index = getattr(vs, "index", None)
        if faiss_index is None or getattr(faiss_index, "ntotal", 0) == 0:
            raise RuntimeError(
                f"FAISS index at {cfg.faiss_dir} is empty (0 vectors). "
                "Check docs_dir or rebuild the index in full mode."
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FAISS index from {cfg.faiss_dir}. "
            "Try rebuilding it with: `python -m rag.cli index`."
        ) from e

    print(f"[retriever] Loaded index from {cfg.faiss_dir} (k={k})")

    return vs.as_retriever(search_kwargs={"k": k})

