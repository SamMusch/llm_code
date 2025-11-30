

**Conceptual notes**: [sammusch.github.io](https://sammusch.github.io/ds-garden/LLM-RAG/01-RAG-Intro/)

- Initial code based on [textbook](https://learning.oreilly.com/library/view/a-simple-guide/9781633435858/OEBPS/Text/part-1.html) & [github code](https://github.com/abhinav-kimothi/A-Simple-Guide-to-RAG)

- Refactored to adopt LangChain/Graph/Smith.



```bash
llm_code/
├─ RAG.py                     # wrapper, fwds to real entrypoint
├─ rag/
│  ├─ __init__.py			  			# Marks dir as a package
│  ├─ agent.py			  				# 
│  ├─ config.py               # Pydantic settings, .env, yaml config
│  ├─ retriever.py            # i_pipe
│  ├─ generator.py            # g_pipe
│  ├─ cli.py                  # command line interface
│  └─ utils.py                # helpers (clean_text, timers)
├─ config/
│  └─ rag.yaml                # settings
├─ Data/                      # vector DB artifacts
├─ .env
└─ requirements.txt
```


### Install instructions



```bash
cd /path/to/project/llm_code
brew install python@3.11
python3.11 --version

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install -r requirements.txt

# optional UI of langsmith instead of cli
python -m pip install "langgraph-cli[inmem]"

# confirm
python -c "import rag; print('OK')"
```

```bash
# create `.env`
OPENAI_API_KEY=sk-proj-...
LANGCHAIN_API_KEY=lsv2_pt_...
LANGSMITH_API_KEY=lsv2_pt_...   # same as ^
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=llm_code      # folder name
```

```
python -m rag.cli index
python -m rag.cli ask "test question in cli"

langgrah dev
```
