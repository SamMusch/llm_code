

**Conceptual notes**: https://sammusch.github.io/ds-garden/LLM/00-RAG/

- Initial code based on [textbook](https://learning.oreilly.com/library/view/a-simple-guide/9781633435858/OEBPS/Text/part-1.html) & [github code](https://github.com/abhinav-kimothi/A-Simple-Guide-to-RAG)

- Refactored to adopt [LRAT’s](https://github.com/langchain-ai/retrieval-agent-template) modular design, class hierarchy, and graph-based architecture.



```bash
llm_code/
├─ RAG.py                     # wrapper, fwds to real entrypoint
├─ rag/
│  ├─ __init__.py			  			# Marks dir as a package
│  ├─ config.py               # Pydantic settings (.env, yaml config)
│  ├─ retriever.py            # i_pipe
│  ├─ generator.py            # g_pipe
│  ├─ graph.py                # i_pipe & g_pipe
│  ├─ cli.py                  # command line interface
│  └─ utils.py                # helpers (clean_text, timers)
├─ config/
│  └─ rag.yaml                # settings
├─ Data/                      # vector DB artifacts
├─ .env
└─ requirements.txt
```



```bash
# llm_code/.env

# assuming you name this folder "llm_code"

OPENAI_API_KEY=sk-proj..
HUGGINGFACEHUB_API_TOKEN=hf_...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=llm_code  # FOLDER NAME
```



```bash
cd ... path/to/folder
source .venv/bin/activate
python -m rag.cli index
python -m rag.cli ask "Example question?"
```

