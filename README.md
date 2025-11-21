```
your-project/
├─ RAG.py              <-- MAIN SCRIPT
├─ rag/
│  ├─ __init__.py
│  ├─ indexing.py
│  ├─ generation.py
│  ├─ evals.py
│  ├─ advanced.py
│  ├─ types.py
│  └─ utils.py
├─ config/
│  └─ rag.yaml
├─ Data/                 <-- put vector DB artifacts here (indexes, parquet, faiss, chroma, etc.)
├─ .env
├─ requirements.txt
└─ README.md
```

- `RAG.py`: a thin **orchestrator** that imports & wires functions.
- `rag/`: a small **internal package**.



---



Use absolute imports, keep functions pure, centralize config, add logging, and avoid any packaging you don’t need. Run it like a normal script.

Why this structure?

- **Importable modules** without making a public library.
- **Stable internal “API”:** `RAG.py` stays project-specific, other py scripts are building blocks
- **Future-proof:** can make a CLI or package without rewrites.
