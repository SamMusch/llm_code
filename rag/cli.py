
# cli.py 
# a Typer wrapper that calls into other modules
# 
# defines 2 functions:
    # index()   Runs I pipeline
    # ask()     Runs G pipeline

import typer                # lets you use Python functions to define terminal commands
from pathlib import Path

# Importing from other scripts
from rag.config import cfg as base_cfg    # project settings
from rag.retriever import build_index     # load → chunk → embed → store
from rag.graph import run as run_graph    # executes LangGraph pipeline

app = typer.Typer(
    no_args_is_help = True,           # ensures we provide arguments
    add_completion = False            # disables shell autocompletion
    )

# ------------------------------------------------------------
# index()
    # receives optional path (docs to ingest)
    # calls build_index()
    # verifies the index works

@app.command()     # typer subcommand
def index(path: str = typer.Option(None, help="Docs dir; default from config")):
    docs_dir = Path(path) if path else base_cfg.docs_dir
    typer.echo(f"Indexing docs from {docs_dir}")
    build_index(docs_dir)  # from rag.retriever
    
    # verifies the index works
    from rag.retriever import load_retriever  
    try:        
        retriever = load_retriever(k=1)
        _ = retriever.invoke("smoke-test") # LangChain
        typer.echo("👍 [index] Passed: retriever can load/query the index.")
    except Exception as e:
        raise RuntimeError(
            "❌ Index built but failed check; see underlying error.") from e
    typer.echo(f"Indexed into {base_cfg.faiss_dir}")

# ------------------------------------------------------------
# ask()
    # Take Q → run RAG pipeline → return answer & sources

@app.command()
def ask(question: str, k: int = typer.Option(None, help="Top-k docs")):
    cfg = base_cfg
    if k is not None:
        cfg = cfg.model_copy(update={"k": k})

    # run RAG pipeline
    try:
        ans, docs = run_graph(question, cfg)
    except RuntimeError as e:
        typer.echo(f"❌ {e}")      # clean, user-facing errors
        raise typer.Exit(code=1)

    # return answer & sources
    typer.echo(ans)
    typer.echo("\n---\nSources:")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        typer.echo(f"[{i}] {src}")

# ------------------------------------------------------------
def main():
    app()

if __name__ == "__main__":
    main()
