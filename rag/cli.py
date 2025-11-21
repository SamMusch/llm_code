import typer
from pathlib import Path
from .config import Settings
from .retriever import build_index
from .graph import run

app = typer.Typer(
    no_args_is_help=True,           # if run program w/o arguments ---> will show help message instead of doing nothing
    add_completion=False            # disables shell autocompletion ---> won’t generate scripts for tab completion
    )

@app.command()
def index(path: str = typer.Option(None, help="Docs dir; default from config")):
    cfg = Settings.load()
    build_index(cfg, Path(path) if path else None)
    typer.echo(f"Indexed into {cfg.faiss_dir}")

@app.command()
def ask(question: str, k: int = typer.Option(None, help="Top-k docs")):
    cfg = Settings.load()
    if k is not None:
        cfg = cfg.model_copy(update={"k": k})
    ans, docs = run(question, cfg)
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