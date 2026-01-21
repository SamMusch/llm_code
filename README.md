**Conceptual notes**: [sammusch.github.io](https://sammusch.github.io/ds-garden/LLM-RAG/01-RAG-Intro/)
- Initial code based on [textbook](https://learning.oreilly.com/library/view/a-simple-guide/9781633435858/OEBPS/Text/part-1.html) & [github code](https://github.com/abhinav-kimothi/A-Simple-Guide-to-RAG)
- Refactored to adopt LangChain/Graph/Smith.

### Architecture (local)
- **Ollama (host or container)**: Runs the models (LLM & embedding) as an HTTP service.
- **llm_code (Docker)**: Handles ingestion / retrieval / prompting / orchestration.
- **Docker Compose**: Runs app containers, wires them to Ollama.

### Runtime modes (local)
`llm_code` supports two local runtime modes:

**Option A | Ollama.app on host (recommended)**
- Ollama runs natively on macOS (`/Applications/Ollama.app`)
- Docker containers call `host.docker.internal:11434`

**Option B | Ollama in Docker**
- Ollama runs as a Docker container
- Containers call `ollama:11434`

**Pre-reqs**: 
- Docker Desktop
- LangSmith (optional)


```bash
Local runtime (Docker + Mac host)

Browser (you)
  ↓
http://localhost:8080
FastAPI (UI + API)
  ↓
http://localhost:2024
LangGraph (agent / orchestration)
  ↓
http://localhost:11434
Ollama.app (LLM + embeddings)

Observability (airgapped)
FastAPI / LangGraph
  → OTEL → http://otel-collector:4318
  → Tempo (traces) :3200
  → Loki (logs)   :3100
  → Prometheus   :9090
  → Grafana UI   http://localhost:3000
```


### Install instructions

```bash
# STEP 0 | PRE-REQS - DOCKER

cd /path/to/your/folder
docker version    			# confirm Docker engine is reachable
docker compose version	# confirm Compose is available
docker run --rm hello-world # confirm containers can run
```

```bash
# STEP 1 | CLONE REPO

cd /path/to/your/folder
git clone https://github.com/SamMusch/llm_code.git
cd llm_code
```

```bash
# STEP 2 | create a `.env` file (minimal local example)
OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_CHAT_MODEL=llama3.2:1b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

```bash
# STEP 3 | GET MODEL

## Option A — Ollama.app on host (recommended)

# ensure Ollama.app is running
ollama pull llama3.2:1b
ollama pull nomic-embed-text
ollama list

docker compose up -d --build

## Option B — Ollama in Docker

docker compose up -d --build

docker exec -it ollama ollama pull llama3.2:1b
docker exec -it ollama ollama pull nomic-embed-text
docker exec -it ollama ollama list
```

---

### Usage

```bash
cd /path/to/your/folder

# start & stop
docker compose up -d
docker compose down

# if changed reqs, dependencies, etc
docker compose up -d --build

# index docs
docker compose exec llm_code python -m rag.cli index

# ask question via cli
docker compose exec llm_code python -m rag.cli ask "test question"
```
