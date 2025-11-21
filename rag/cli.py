import typer
from pathlib import Path

from rag.config import cfg as base_cfg
from rag.retriever import build_index
from rag.graph import run as run_graph

app = typer.Typer(
    no_args_is_help=True,           # if run program w/o arguments ---> will show help message instead of doing nothing
    add_completion=False            # disables shell autocompletion ---> won’t generate scripts for tab completion
    )

@app.command()
def index(path: str = typer.Option(None, help="Docs dir; default from config")):
    docs_dir = Path(path) if path else base_cfg.docs_dir
    typer.echo(f"Indexing docs from {docs_dir}")
    build_index(docs_dir)
    # Health check: try loading the index and performing a dummy query.
    from rag.retriever import load_retriever

    try:
        retriever = load_retriever(k=1)
        _ = retriever.invoke("smoke-test")  # modern LangChain API; avoids deprecation warning
        typer.echo("👍 [index] Health check passed: retriever can load and query the index.")
    except Exception as e:
        raise RuntimeError(
            "❌ Index built but failed health check; see underlying error."
        ) from e
    typer.echo(f"Indexed into {base_cfg.faiss_dir}")

@app.command()
def ask(question: str, k: int = typer.Option(None, help="Top-k docs")):
    cfg = base_cfg
    if k is not None:
        cfg = cfg.model_copy(update={"k": k})

    try:
        ans, docs = run_graph(question, cfg)
    except RuntimeError as e:
        # Surface clean, user-facing errors (e.g., missing/corrupt index) and exit.
        typer.echo(f"❌ {e}")
        raise typer.Exit(code=1)

    typer.echo(ans)
    typer.echo("\n---\nSources:")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        typer.echo(f"[{i}] {src}")

def main():
    app()

if __name__ == "__main__":
    main()
