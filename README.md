# ClinicalRAG

A private, on-premises RAG system for clinical document Q&A. Built for regulated environments where sensitive documents cannot leave the network.

Upload clinical documents, ask questions in natural language, and get accurate answers with source citations. The whole process is running on your own hardware, with no data sent to external services.

---

## Why on-premises

In clinical research, documents like trial protocols, SOPs, and GCP guidelines often contain sensitive or proprietary information. Sending them to external APIs like OpenAI violates data governance requirements under GDPR and ICH E6 GCP.

ClinicalRAG runs entirely on local hardware:

- Documents never leave your machine
- Embeddings are generated locally via Ollama
- The LLM generates answers on your GPU
- The vector database runs in Docker on your machine

---

## Architecture

![ClinicalRAG Architecture](docs/pics/architecture.svg)

ClinicalRAG has three ways to interact, all backed by the same on-prem pipeline:

**REST API**: Call `POST /chat` directly or use Swagger UI at `http://localhost:8000/docs`

**Claude Desktop + MCP**: Claude acts as the AI agent. It automatically launches the MCP server as a background process, discovers the `query_rag` and `ingest_document` tools, and calls them when you ask clinical questions.

**Private terminal agent**: Llama3.1 acts as the agent entirely on your GPU. It also launches the MCP server automatically as a background process, calls `query_rag`, and returns answers. Nothing leaves your machine.

When using Claude Desktop, the generated answer passes through Anthropic's servers. The private agent and REST API are fully on-prem with no external services.

---

### Models

| Model | Role | Size |
|-------|------|------|
| `nomic-embed-text` | Converts text to vectors for semantic search | 300MB |
| `mistral 7B` | Reads retrieved context and generates answers | 4GB |
| `llama3.1 8B` | Agent reasoning, decides which tools to call | 6GB |

All three run via Ollama on your GPU.

---

### RAG pipeline

1. Documents are chunked into overlapping segments
2. Each chunk is embedded using `nomic-embed-text` via Ollama
3. Vectors are stored in Qdrant with the original text as payload
4. At query time, the question is embedded and Qdrant finds the most similar chunks
5. The top chunks are sent to Mistral as context
6. Mistral generates an answer with citations from the source documents

---

### MCP server

An MCP (Model Context Protocol) server exposes the RAG pipeline as tools that AI agents can call. Both Claude Desktop and the private agent launch it automatically as a subprocess, you never run it manually. The MCP server is the standardization layer that allows any MCP-compatible AI client to connect to your on-prem RAG system.

---

### Private agent

A fully on-premises agent using Llama3.1 as the reasoning model. It connects to the MCP server, discovers tools automatically, and decides when to call `query_rag` to answer questions. No external services involved.

Known limitation: Llama3.1 8B has inconsistent tool calling compared to larger models. It gets correct answers but sometimes calls the tool more times than needed. A larger model like Llama3.1 70B would be more reliable but requires more VRAM.

---

## Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| LLM serving | Ollama | Runs models locally on GPU |
| Generation model | Mistral 7B | Answers questions from retrieved context |
| Agent model | Llama3.1 8B | Decides which tools to call |
| Embedding model | nomic-embed-text | Converts text to vectors |
| Vector database | Qdrant | Stores and searches embeddings |
| API | FastAPI | REST endpoints for ingestion and chat |
| MCP server | Python MCP SDK | Exposes RAG as tools for AI agents |
| Infrastructure | Docker Compose | Runs all services locally |

---

## Project structure

```
clinical-rag/
├── api/
│   ├── main.py              # FastAPI app, router registration
│   ├── settings.py          # Config via pydantic-settings
│   ├── core/
│   │   ├── chunking.py      # Split text into overlapping chunks
│   │   ├── embeddings.py    # Embed text via Ollama
│   │   ├── retrieval.py     # Search Qdrant for similar chunks
│   │   └── vector_store.py  # Store chunks and vectors in Qdrant
│   └── routes/
│       ├── health.py        # /healthz and /metrics
│       ├── documents.py     # /documents/ingest
│       └── rag.py          # /chat
├── agent/
│   └── agent.py             # Private on-prem agent using Llama3.1 + MCP
├── mcp_server/
│   └── server.py            # MCP server exposing RAG as tools
├── scripts/
│   └── ingest_all.py        # Batch ingest all files in data/samples/
├── data/
│   └── samples/             # Place .txt documents here
├── infra/
│   ├── docker-compose.yml   # Ollama + Qdrant + API
│   └── k8s/                 # Kubernetes manifests
├── Dockerfile               # FastAPI container
├── requirements.txt
├── setup.ps1                # One-command setup script
├── .env.example
└── .env.docker.example
```

---

### Ingestion pipeline

![Ingestion pipeline](docs/pics/ingestion_flow.svg)

When a document is uploaded to `POST /documents/ingest`:

1. **FastAPI receives the file**: Validates the extension and reads the bytes.
2. **`chunk_text()`**: Splits the text into 500-word overlapping segments.
3. **`embed_text()`**: For each chunk, sends the text to Ollama via `POST /api/embed`.
4. **`nomic-embed-text`**: Converts the text to 768 numbers (the embedding vector).
5. **`store_chunks()`**: Builds a `PointStruct` with a UUID, the vector, and the original text as payload.
6. **Qdrant stores**: `UUID → vector + payload` - the chunk is now searchable.
7. **Response**: `{"status": "indexed", "filename": ..., "chunks": ...}`.

The same `embed_text()` function is used here and in the query pipeline. This is critical as chunks and questions must be embedded by the same model so their vectors are in the same space and similarity search works correctly.

---

### Query pipeline

![Query pipeline](docs/pics/query_flow.svg)

When a user calls `POST /chat`:

1. **FastAPI validates**: Checks the request body via Pydantic.
2. **`embed_text(question)`**: Embeds the question using the same `nomic-embed-text` model that was used during ingestion.
3. **`search_chunks()`**: Sends the question vector to Qdrant, which returns the 3 most similar chunks using cosine similarity.
4. **Build prompt**: Combines the retrieved chunks as context with the original question.
5. **Call Mistral**: Sends the prompt to Ollama via `POST /api/generate`.
6. **Mistral generates**: Reads the context and produces a natural language answer.
7. **Response**: `{"answer": "...", "sources": [...]}`.

Mistral never searches Qdrant. It only reads what FastAPI gives it. The retrieval and the generation are completely separate steps.

---

### MCP server flow

![MCP server flow](docs/pics/mcp_flow.svg)

The MCP server is a thin connector between AI clients and your RAG pipeline. It never runs on its own, it is always launched automatically by whoever needs it.

1. **Client launches the server**: Claude Desktop or the private agent starts `server.py` as a subprocess via stdio.
2. **MCP handshake**: Client and server agree on protocol version, server reports its capabilities.
3. **Tools are exposed**: `query_rag` and `ingest_document` are registered via `@mcp.tool()` decorators. The AI client reads the function name and docstring to understand what each tool does.
4. **Tool call arrives**: The AI client decides to call a tool and sends the request via stdio.
5. **Tool execution**: `query_rag` calls `POST /chat`, `ingest_document` calls `POST /documents/ingest`.
6. **FastAPI runs the pipeline**: The MCP server has no knowledge of Qdrant, Ollama, or embeddings. It just calls FastAPI and returns the result.
7. **Result returned**: A formatted string back to the AI client via stdio.

The MCP server knows nothing about how RAG works internally. If you replace the entire FastAPI implementation, the MCP server does not change. This separation is intentional.

---

### Agent flow

![Agent flow](docs/pics/agent_flow.svg)

When you run `python agent/agent.py`:

1. **User types a goal**: The question or task the agent should solve.
2. **MCP server launches**: `agent.py` starts `mcp_server/server.py` as a subprocess and connects via stdio.
3. **`list_tools()`**: The agent discovers available tools automatically. It filters to `query_rag` only. `ingest_document` is hidden to prevent wrong tool calls with Llama3.1 8B.
4. **Agent loop starts**: The full conversation history plus available tools is sent to Llama3.1 via `POST /api/chat`.
5. **Llama3.1 decides**: Either call a tool or give a final answer.
   - **Tool call** → `call_tool()` sends it to the MCP server via stdio → MCP calls FastAPI `/chat` → result appended to `messages` → loop continues
   - **No tool call** → Llama has enough information → print answer → break
6. **MAX_ITERATIONS safety**: if the loop runs more than 5 times, `force_answer` removes tools from the request. Llama must produce a text answer from whatever it already retrieved.

The `messages` list grows with each iteration. Llama reads the full history every time, it sees its own previous tool calls and their results and uses this to decide what to do next.

---

## Requirements

- Docker Desktop
- Python 3.11+
- NVIDIA GPU with CUDA (tested on RTX 3090)
- NVIDIA Container Toolkit

---

## Setup

Clone the repository and run the setup script:

```powershell
.\setup.ps1
```


This will:
1. Create a Python virtual environment
2. Install dependencies  
3. Copy .env.example to .env
4. Build and start Docker containers (Ollama + Qdrant + FastAPI)
5. Pull Ollama models (Mistral, Llama3.1, nomic-embed-text)
6. Ingest sample clinical documents

---

## Configuration

Copy `.env.example` to `.env` and adjust if needed:

```
OLLAMA_URL=http://localhost:11434
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=clinical_docs
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=mistral
EMBEDDING_SIZE=768
```

For running FastAPI inside Docker, use `.env.docker` with service names instead of localhost:

```
OLLAMA_URL=http://ollama:11434
QDRANT_URL=http://qdrant:6333
```

---

## Running

### Option 1: Development mode (recommended for active development)

FastAPI runs locally with auto-reload. Qdrant and Ollama run in Docker.

```bash
# start infrastructure
docker compose -f infra/docker-compose.yml up -d

# start FastAPI with auto-reload (restarts on code changes)
uvicorn api.main:app --reload
```

Then use whichever interface you want:

**Claude Desktop** -> just open the app. It automatically launches `mcp_server/server.py` 
in the background based on your config file. No manual step needed.

**Private terminal agent** -> run in a new terminal:
```bash
python agent/agent.py
```
This also launches `mcp_server/server.py` automatically as a subprocess. No manual step needed.

You never need to run `mcp_server/server.py` directly, it is always launched automatically 
by whoever needs it.

---

### Option 2: Full Docker deployment

Everything runs in Docker including FastAPI. Use this to test the production setup 
or when you are done developing.

**First time or after code changes** -> rebuild and start:
```bash
docker compose -f infra/docker-compose.yml up -d --build
```

**Daily use** -> start without rebuilding (faster):
```bash
docker compose -f infra/docker-compose.yml up -d
```

**Ingest your documents** -> (skip if you already ran `setup.ps1`, 
data persists in the Docker volume between restarts):
```bash
python scripts/ingest_all.py
```

**Then use whichever interface you want:**

Claude Desktop -> just open the app, it connects automatically.

Private agent -> run in a new terminal:
```bash
python agent/agent.py
```

Note: in this mode FastAPI uses `.env.docker` (Docker service names like 
`http://ollama:11434`) instead of `.env` (localhost URLs). The agent runs 
locally and connects to FastAPI via `localhost:8000`. This works because 
Docker maps port 8000 from the container to your machine.

**Fastest Way:**
You can also start docker containers and the agent by running:
```
run_agent.bat
```


---

### Option 3: Local Kubernetes with minikube

```bash
# start cluster
minikube start --gpus=all

# build and load API image
docker build -t clinical-rag-api:latest .
minikube image load clinical-rag-api:latest

# deploy
minikube kubectl -- apply -f infra/k8s

# expose API (separate terminal)
minikube tunnel

# open
http://localhost/docs
```

Note: Ollama GPU scheduling requires WSL2 minikube on Windows.
Full GPU support on Linux and AKS.

---

## Ingesting documents

Place `.txt` files in `data/samples/` and run:

```bash
python scripts/ingest_all.py
```

The script finds all `.txt` files, chunks them, embeds each chunk, and stores the vectors in Qdrant. Run this once on setup and again whenever you add new documents.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/ingest` | Upload and index a `.txt` file |
| `POST` | `/chat` | Ask a question over indexed documents |
| `GET` | `/healthz` | Health check |
| `GET` | `/metrics` | Basic metrics |

### Example: ingest a document

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@data/samples/adverse_event_sop.txt"
```

Response:
```json
{
  "status": "indexed",
  "filename": "adverse_event_sop.txt",
  "chunks": 4
}
```

### Example: ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How long must trial records be retained?"}'
```

Response:
```json
{
  "answer": "The sponsor must retain all sponsor-specific essential documents for at least 15 years.",
  "sources": [
    {
      "text": "ICH E6 Good Clinical Practice Guidelines...",
      "document": "adverse_event_sop.txt",
      "score": 0.92
    }
  ]
}
```

You can also test all endpoints via the auto-generated Swagger UI at `http://localhost:8000/docs`.

---

## MCP server

The MCP server exposes two tools to AI agents:

`query_rag`: ask a question over indexed clinical documents, returns answer with citations

`ingest_document`: ingest a text file into the system by providing a file path

Both tools are available to Claude Desktop.

The private agent (`agent/agent.py`) only exposes `query_rag`. This is intentional as 
Llama3.1 8B sometimes calls the wrong tool with incorrect arguments when multiple tools 
are available. Since the agent is designed for Q&A, ingestion is handled separately 
via `scripts/ingest_all.py` which is more reliable.

### Connecting Claude Desktop

Add the following to your Claude Desktop config file at:
`C:\Users\<username>\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ClinicalRAG": {
      "command": "C:\\path\\to\\clinical-rag\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\clinical-rag\\mcp_server\\server.py"]
    }
  }
}
```

Restart Claude Desktop. The ClinicalRAG tools will appear under Connectors.

---

## Private agent

For a fully on-premises setup without Claude Desktop:

```bash
python agent/agent.py
```

Enter a goal when prompted:

```
Enter goal: What are the requirements for informed consent in a clinical trial?

Connected to MCP server
MCP tool registered: query_rag

Calling tool: query_rag

Answer: All participants must provide written informed consent before
entering a clinical trial. The investigator is responsible for ensuring
that informed consent is obtained from each subject...
```

The agent uses Llama3.1 as the reasoning model. There are no external services or data leaving your machine.

**Known limitation:** Llama3.1 8B has inconsistent tool calling compared to larger models. It gets the right answer but may call the tool more times than needed. A larger model like Llama3.1 70B would be more reliable but requires more VRAM.

---

## Sample documents

The repository includes sample clinical documents in `data/samples/` based on publicly available guidelines:

- `gcp_basics.txt` — ICH E6 Good Clinical Practice guidelines summary covering informed consent, investigator responsibilities, adverse event reporting, and record retention

These are synthetic summaries for demonstration. In production, replace with your own SOPs, trial protocols, and regulatory documents.

---

## Production path

This project runs locally with Docker Compose. For production deployment:

1. Build and push Docker images to a container registry
2. Deploy to Kubernetes using the manifests in `infra/k8s/`
3. Ollama with GPU access stays on-premises hardware
4. FastAPI and Qdrant can be deployed to a managed Kubernetes cluster (e.g. AKS)

The services are designed to be stateless with health check endpoints, making them Kubernetes-ready.

---

## Tradeoffs and next steps

**Accomplished:**

- Fully private, no data leaves the network.
- GPU-accelerated inference.
- REST API with auto-generated Swagger UI at `/docs`, test all endpoints directly in the browser without any extra tooling.
- MCP integration works reliably with Claude Desktop.
- Private agent works with Llama3.1.
- Kubernetes manifests written and tested locally with minikube.

**Limitations:**

- Only `.txt` files supported. PDF parsing would require adding `pypdf`. 
- Llama3.1 8B tool calling is inconsistent. Larger models are more reliable.
- Single Qdrant collection. No per-user or per-project isolation yet.
- No authentication on the API endpoints.
- GPU scheduling for Ollama in Kubernetes requires a Linux node. 
  On Windows, GPU passthrough to the minikube container is not 
  supported regardless of Docker configuration. 
  Docker Compose with GPU is used for local development instead.

**Future Work:**

- Add PDF support (in progress).
- Add authentication to the API.
- Build a simple chat UI in Next.js.
- Deploy Kubernetes manifests to AKS.
- Add Prometheus metrics with real data (query count, latency, retrieval scores).
- Evaluate larger models for the agent layer.
