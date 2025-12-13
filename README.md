

**Conceptual notes**: [sammusch.github.io](https://sammusch.github.io/ds-garden/LLM-RAG/01-RAG-Intro/)

- Initial code based on [textbook](https://learning.oreilly.com/library/view/a-simple-guide/9781633435858/OEBPS/Text/part-1.html) & [github code](https://github.com/abhinav-kimothi/A-Simple-Guide-to-RAG)

- Refactored to adopt LangChain/Graph/Smith.


### Architecture (local)

- **Ollama**: Runs a local LLM as an HTTP service.
- **llm_code**: Handles ingestion, retrieval, prompting, and orchestration.
- **Docker Compose**: Defines how services run, connect, and persist data.


### Layout

```bash
llm_code/
├─ rag/
│  ├─ agent.py              # agent + LLM wiring
│  ├─ retriever.py          # retrieval pipeline
│  ├─ generator.py          # generation pipeline
│  ├─ cli.py                # CLI entrypoints
│  └─ utils.py              # helpers
├─ config/
│  └─ rag.yaml              # runtime configuration
├─ Data/
│  ├─ docs/                 # source documents
│  ├─ index/                # vector index artifacts
│  └─ logs/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ .env                     # secrets (not committed)
└─ README.md
```

### Prerequisites

- Docker Desktop
- A LangSmith account (for Studio + tracing)



### Install instructions

```bash
cd /path/to/your/folder
git clone https://github.com/SamMusch/llm_code.git
cd llm_code
```

```bash
# create `.env`
OPENAI_API_KEY=sk-proj-...
LANGCHAIN_API_KEY=lsv2_pt_...
LANGSMITH_API_KEY=lsv2_pt_...   # same as ^
LANGCHAIN_TRACING_V2=true
LANGSMITH_PROJECT=llm_code      # folder name
```

```bash
# start the system (Ollama model server + llm_code RAG app)
docker compose up -d --build

# pull a local model
docker exec -it ollama ollama pull llama3.2:1b
docker exec -it ollama ollama list

# index docs
docker compose exec llm_code python -m rag.cli index

# ask question via cli
docker compose exec llm_code python -m rag.cli ask "test question"

# start dev server (LangGraph dev server + LangSmith Studio)
docker compose exec llm_code langgraph dev --host 0.0.0.0 --port 2024
```

```bash
docker compose exec llm_code python -m rag.cli index
docker compose exec llm_code python -m rag.cli ask "test question"
docker compose exec llm_code langgraph dev --host 0.0.0.0 --port 2024
```
