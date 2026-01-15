# cli.py 
# a Typer wrapper that calls into other modules
# 
# defines 2 functions:
    # index()   Runs I pipeline
    # ask()     Runs G pipeline

import typer                # lets you use Python functions to define terminal commands
from pathlib import Path
from rag.config import cfg as base_cfg    # project settings
from rag.retriever import build_index     # load → chunk → embed → store
from rag.agent import get_agent          # LangChain agent over tools

app = typer.Typer(
    no_args_is_help = True,           # ensures we provide arguments
    add_completion = False            # disables shell autocompletion
    )

# ------------------------------------------------------------
# index()
    # input: docs to ingest
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
def ask(question: str, k: int = typer.Option(None, help="Top-k docs (currently not wired)")):
    cfg = base_cfg
    if k is not None:
        # k is retained for future wiring; agent currently uses its own defaults.
        cfg = cfg.model_copy(update={"k": k})

    # build agent and run agentic RAG
    try:
        agent = get_agent(cfg)
        state = agent.invoke({"messages": [{"role": "user", "content": question}]})
    except RuntimeError as e:
        typer.echo(f"❌ {e}")      # clean, user-facing errors
        raise typer.Exit(code=1)

    # extract final answer from the last message
    messages = state.get("messages", [])
    if not messages:
        typer.echo("No response from agent.")
        raise typer.Exit(code=1)

    last_msg = messages[-1]
    content = getattr(last_msg, "content", None)
    ans = content if isinstance(content, str) else str(last_msg)

    typer.echo(ans)

# ------------------------------------------------------------
def main():
    app()

if __name__ == "__main__":
    main()
