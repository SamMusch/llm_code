"""
retriever.py | Indexing and retrieval utilities for the RAG pipeline.
retriever.py is now:
	•	a vector store / retriever module, and
	•	a tool provider for the agent.
"""

from pathlib import Path
from collections import defaultdict
from typing import Iterable
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from rag.config import cfg
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import (DirectoryLoader,TextLoader,UnstructuredFileLoader,UnstructuredPowerPointLoader)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.email import (UnstructuredEmailLoader,OutlookMessageLoader,)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Any
import json
import re
from datetime import datetime
try:
    import yaml
except Exception:
    yaml = None

def load_pptx_as_slides(path: str) -> list:
    """Load a PowerPoint and return one LangChain Document per slide."""
    loader = UnstructuredPowerPointLoader(str(path), mode="elements")
    elements = loader.load()

    slides: dict[int, list] = defaultdict(list)
    slide_meta: dict[int, dict[str, Any]] = {}

    for element in elements:
        page_number = element.metadata.get("page_number")
        if page_number is None:
            page_number = 1

        text = (element.page_content or "").strip()
        if text:
            slides[page_number].append(text)

        if page_number not in slide_meta:
            base_meta = dict(element.metadata or {})
            base_meta["source"] = str(path)
            base_meta["page_number"] = page_number
            slide_meta[page_number] = base_meta

    out = []
    for page_number in sorted(slides):
        content = "\n\n".join(slides[page_number]).strip()
        if not content:
            continue
        out.append(
            Document(
                page_content=content,
                metadata=slide_meta.get(page_number, {"source": str(path), "page_number": page_number}),
            )
        )
    return out


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter if present. Return (metadata, body_text)."""
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        return {}, text
    raw, body = m.group(1), m.group(2)
    if yaml is None:
        return {}, body
    try:
        meta = yaml.safe_load(raw) or {}
        return meta, body
    except Exception:
        return {}, body


def _normalize_metadata(meta: dict[str, Any], body: str) -> dict[str, Any]:
    """Keep only the allowed metadata fields and normalize types."""
    out: dict[str, Any] = {}

    # Canonical title: support local/frontmatter `title`, SharePoint/AppFlow `name`,
    # and finally fall back to the source filename.
    title = meta.get("title") or meta.get("name")
    if not (isinstance(title, str) and title.strip()):
        source = meta.get("source")
        if isinstance(source, str) and source.strip():
            title = Path(source).name
    if isinstance(title, str) and title.strip():
        out["title"] = title.strip()

    # Canonical timestamps: support local macOS-style keys and SharePoint/AppFlow keys
    timestamp_key_map = {
        "kMDItemContentCreationDate": ["kMDItemContentCreationDate", "createdDateTime"],
        "kMDItemContentModificationDate": ["kMDItemContentModificationDate", "lastModifiedDateTime"],
    }
    for out_key, candidate_keys in timestamp_key_map.items():
        for candidate_key in candidate_keys:
            v = meta.get(candidate_key)
            if isinstance(v, str):
                try:
                    out[out_key] = int(datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp())
                    break
                except Exception:
                    pass
            elif isinstance(v, (int, float)):
                out[out_key] = int(v)
                break

    # word_count: compute from body
    out["word_count"] = len(body.split())

    # tags: normalize to list[str]
    tags = meta.get("tags")
    if isinstance(tags, list):
        out["tags"] = [str(t) for t in tags]
    elif isinstance(tags, str):
        out["tags"] = [tags]

    # webUrl: SharePoint/AppFlow direct link — preserved verbatim for frontend hyperlinking
    web_url = meta.get("webUrl")
    if isinstance(web_url, str) and web_url.strip():
        out["webUrl"] = web_url.strip()

    return out

# Simple loader registry by extension
EXT_TO_LOADER = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".rst": TextLoader,
    ".html": UnstructuredFileLoader,
    ".pdf": UnstructuredFileLoader,
    ".docx": UnstructuredFileLoader,
    ".ppt": UnstructuredPowerPointLoader, # UnstructuredFileLoader,
    ".pptx": UnstructuredPowerPointLoader, # UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".eml": UnstructuredEmailLoader,
    ".msg": OutlookMessageLoader,
}

def _get_embeddings():
    """Return an embeddings client appropriate for the current runtime."""
    provider = getattr(cfg, "llm_provider", "ollama")
    if provider == "bedrock":
        model_id = getattr(cfg, "embedding_model", None) or "amazon.titan-embed-text-v2:0"
        return BedrockEmbeddings(model_id=model_id)

    # Local/default: Ollama embeddings
    base_url = getattr(cfg, "llm_base_url", None) or os.environ.get("OLLAMA_BASE_URL") or "http://ollama:11434"
    return OllamaEmbeddings(model=cfg.embedding_model, base_url=base_url)


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
        try:
            if ext in {".ppt", ".pptx"}:
                for path in sorted(input_dir.rglob(f"*{ext}")):
                    loaded = load_pptx_as_slides(path)
                    loaded_count += len(loaded)
                    docs.extend(loaded)
            else:
                loader = DirectoryLoader(
                    str(input_dir),
                    glob=f"**/*{ext}",  # recursively search for all files with this extension
                    loader_cls=Loader,  # TextLoader, UnstructuredFileLoader, etc.
                    show_progress=True,
                )
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

    # Parse frontmatter + attach normalized per-document metadata
    for d in docs:
        text = getattr(d, "page_content", "") or ""
        existing_meta = dict(getattr(d, "metadata", {}) or {})

        # Merge sidecar .metadata.json if present (written by Lambda after AppFlow S3 sync).
        # Sidecar supplies SharePoint fields (webUrl, name, createdDateTime, etc.).
        # Loader metadata (source, page_number) takes priority via right-hand merge.
        source_path = existing_meta.get("source", "")
        if source_path:
            sidecar = Path(source_path).parent / (Path(source_path).stem + ".metadata.json")
            if sidecar.exists():
                try:
                    sidecar_meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    if isinstance(sidecar_meta, dict):
                        existing_meta = {**sidecar_meta, **existing_meta}
                except Exception as e:
                    print(f"[loader-warning] Could not read sidecar {sidecar}: {e}")

        meta_raw, body = _parse_frontmatter(text)
        combined_meta = {**existing_meta, **meta_raw}
        norm = _normalize_metadata(combined_meta, body)
        if norm:
            d.metadata = {**existing_meta, **norm}
        else:
            d.metadata = existing_meta
        d.page_content = body

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
    ppt_docs = []
    other_docs = []
    for d in docs:
        source = str((d.metadata or {}).get("source", "")).lower()
        if source.endswith(".ppt") or source.endswith(".pptx"):
            ppt_docs.append(d)
        else:
            other_docs.append(d)

    chunks = list(ppt_docs)
    if other_docs:
        chunks.extend(splitter.split_documents(other_docs))
    embeddings = _get_embeddings()     # old was OpenAIEmbeddings(model=cfg.embedding_model)
    vs = FAISS.from_documents(chunks, embeddings)  # FAISS for now
    vs.save_local(str(cfg.faiss_dir))


def load_retriever(k: int | None = None, filters: dict | None = None, fetch_k: int | None = None):
    """
    Reload saved FAISS index → attach embedding model → return retriever interface for semantic lookup.
    `filters` is a deterministic metadata filter applied at retrieval time.
    """
    k = k or cfg.k  # top k docs
    fetch_k = fetch_k or max(k * 10, 50)
    index_file = cfg.faiss_dir / "index.faiss"

    # Guardrails: fail fast or optionally rebuild if the index is missing.
    if not cfg.faiss_dir.exists() or not index_file.exists():
        auto_rebuild = getattr(cfg, "auto_rebuild_index", False)
        if auto_rebuild:
            print(f"[retriever] Index not found in {cfg.faiss_dir}. Auto-rebuilding with build_index()...")
            build_index()  # uses cfg.docs_dir and optional cfg.max_docs
        else:
            raise RuntimeError(
                f"FAISS index not found at {index_file}. "
                "Build it first with: `python -m rag.cli index`."
            )

    #embeddings = OpenAIEmbeddings(model=cfg.embedding_model)
    embeddings = _get_embeddings()

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

    search_kwargs: dict[str, Any] = {"k": k}
    # FAISS metadata filtering is applied after an initial fetch; fetch_k controls how many candidates are fetched
    # before filtering to k.
    if filters:
        search_kwargs["filter"] = filters
        search_kwargs["fetch_k"] = fetch_k

    return vs.as_retriever(search_kwargs=search_kwargs)


# Make sure dim size matches for FAISS index & embedding model
def verify_faiss_dim_matches_embeddings() -> None:
    index_file = cfg.faiss_dir / "index.faiss"
    if not cfg.faiss_dir.exists() or not index_file.exists():
        raise RuntimeError(
            f"FAISS index not found at {index_file}. Build it first with: `python -m rag.cli index`.")

    embeddings = _get_embeddings()

    # 1) Embedding dimension from a single probe vector
    probe = embeddings.embed_query("ping")
    emb_dim = len(probe) if probe is not None else 0

    # 2) FAISS index dimension
    vs = FAISS.load_local(
        str(cfg.faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    faiss_dim = int(getattr(getattr(vs, "index", None), "d", 0) or 0)

    if not emb_dim or not faiss_dim:
        raise RuntimeError(f"Failed to determine dims (emb_dim={emb_dim}, faiss_dim={faiss_dim}).")

    if emb_dim != faiss_dim:
        raise RuntimeError(
            "Embedding/FAISS dimension mismatch: "
            f"emb_dim={emb_dim} (provider={getattr(cfg, 'llm_provider', None)} embedding_model={getattr(cfg, 'embedding_model', None)}) "
            f"!= faiss_dim={faiss_dim} (faiss_dir={cfg.faiss_dir}). "
            "Rebuild the index with the same embedding model you will query with."
        )

    print(f"[startup] FAISS dim OK (d={faiss_dim})", flush=True)
