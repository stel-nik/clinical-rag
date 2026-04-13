# ClinicalRAG — Architecture Deep Dive

## Part 4 — Design Decisions

This part explains why things were built the way they were — the choices made, the alternatives considered, and the tradeoffs accepted.

---

### Why FastAPI over Flask

Both are Python web frameworks. Flask is older and more widely known. FastAPI is newer and was chosen for three specific reasons.

**Async support.** FastAPI is built on async from the ground up. Every route handler can be `async def`, and every external call can use `await`. This matters for a RAG system because every request makes multiple slow external calls — embedding, vector search, generation. Async allows the server to handle multiple requests simultaneously without blocking. Flask has async support but it was bolted on later and is less natural to use.

**Automatic validation.** FastAPI uses Pydantic for request and response validation. Define a model with type hints and FastAPI validates every incoming request automatically. Wrong type? Returns 422 before your code runs. With Flask you write this validation yourself.

**Auto-generated documentation.** FastAPI generates a Swagger UI at `/docs` automatically from your route definitions. For a project with multiple endpoints this is immediately useful for testing and demonstration — no extra work required.

**Type hints throughout.** FastAPI encourages type hints on everything — route parameters, request bodies, response models. This makes the code self-documenting and catches errors earlier.

Flask would have worked. For a simple project with one or two endpoints the difference is marginal. For a project that makes multiple async calls per request and benefits from auto-documentation, FastAPI is the better fit.

---

### Why `core/` is separated from `routes/`

The project structure separates business logic from HTTP concerns:

```
core/          ← pure Python, no HTTP concepts
  chunking.py
  embeddings.py
  retrieval.py
  vector_store.py

routes/        ← HTTP layer, calls core/
  health.py
  documents.py
  chat.py
```

**The reason:** `core/` functions have no dependency on FastAPI. They take Python values in and return Python values out. This means:

- They can be imported and called from anywhere — the MCP server, the agent, a test script, a Jupyter notebook
- They can be tested without starting a web server
- They can be reused in a future CLI tool or background job without change

If all the logic lived inside the route handlers, you could only call it via HTTP. Separating it means the logic is reusable.

In practice, the agent and MCP server call FastAPI over HTTP rather than importing `core/` directly — keeping components decoupled. But the option exists, and for testing it is valuable.

---

### Why MCP over direct API calls for tool integration

Two ways to connect an AI agent to your RAG system:

**Direct API calls:**
```python
# agent calls FastAPI directly
response = httpx.post("http://localhost:8000/chat", json={"question": question})
```

**MCP:**
```python
# agent discovers tools automatically from MCP server
tools = await session.list_tools()
result = await session.call_tool("query_rag", {"question": question})
```

The direct approach is simpler for one tool. MCP becomes valuable when you have many tools across many systems.

**Tool discovery.** With direct calls the agent has the endpoint hardcoded. With MCP the agent asks "what tools do you have?" and gets the list at runtime. Add a new tool to the MCP server and the agent discovers it automatically on next startup — no code change in the agent.

**Standardization.** MCP is a protocol, not a library. Any MCP client — Claude Desktop, Claude Code, a custom agent — can connect to your MCP server without modification. Direct API calls require each client to know your specific API format.

**Separation of concerns.** The MCP server is a thin connector. It knows nothing about Qdrant, Mistral, or embeddings — it just calls FastAPI. If you replace the entire RAG implementation, the MCP server does not change. If you add a new AI client, the MCP server does not change.

**The tradeoff.** MCP adds a layer of complexity. For a single tool used by a single client, direct API calls are simpler and faster. MCP pays off when you have multiple tools, multiple clients, or when you want the system to be extensible without modifying the agent.

For this project, MCP was chosen specifically because the job it was built for requires MCP server development as a core skill.

---

### Why two separate models

A single model could theoretically handle both embedding and generation. Mistral can produce embeddings and generate text. So why use two models?

**Specialization.** `nomic-embed-text` was specifically trained to produce high-quality semantic embeddings. It outputs 768-dimensional vectors optimized for similarity search. Mistral was trained for text generation. Using each model for what it was trained for produces better results than using one model for both tasks.

**Size and speed.** `nomic-embed-text` is 137MB and embeds text in milliseconds. Mistral is 4GB and takes seconds to generate. During ingestion, you embed hundreds of chunks — this needs to be fast. Using Mistral for embedding would be 30x slower for no quality benefit.

**Resource efficiency.** Both models stay loaded in Ollama's GPU memory. `nomic-embed-text` uses ~0.5GB VRAM. Mistral uses ~5GB. Loading and unloading models for each request would add significant latency.

**The agent model.** Llama3.1 was chosen as a third model for the agent reasoning layer because it has better tool calling support than Mistral. Mistral is optimized for text generation given a prompt — it performs well at RAG generation. Llama3.1 is better at the structured output required for tool calling — deciding which tool to call, with what arguments, and when to stop. Two different strengths, two different jobs.

---

### Why Qdrant over other vector databases

Several options were considered:

**pgvector** — a Postgres extension that adds vector search. Best choice for teams already running Postgres who want to keep everything in one database. Requires more setup than a dedicated vector DB and is slower for pure vector search workloads. Not chosen because this project has no other Postgres use — adding Postgres just for vector storage adds unnecessary infrastructure.

**Pinecone** — a managed cloud vector database. Easiest to set up — just an API key and you are running. Not chosen because it sends your vectors and document chunks to Pinecone's servers. For on-prem clinical data this is not acceptable regardless of convenience.

**ChromaDB** — the simplest option for prototyping. Runs in-process with no separate service. Not chosen because it is not production-grade, has limited Kubernetes support, and running it in Docker alongside other services is less clean than a dedicated container.

**Weaviate** — production-grade with a good Docker image. Considered but Qdrant was preferred for its simpler API, better Python SDK ergonomics, and built-in dashboard at `localhost:6333/dashboard` which is immediately useful for development and debugging.

**Qdrant** — production-grade, has an excellent official Docker image, scales well, clean Python SDK, good Kubernetes story with official Helm charts, and the dashboard makes development easier. The right choice for a production on-prem deployment.

---

### Why `.env` + pydantic-settings over hardcoded config

Config values that change between environments — URLs, model names, collection names — are defined once in `settings.py` with defaults and read from `.env` at startup.

**The alternative** would be hardcoding values throughout the codebase:

```python
# scattered through multiple files — bad
response = httpx.post("http://localhost:11434/api/embed", ...)
client = QdrantClient(url="http://localhost:6333")
```

**The problem** with hardcoding: changing the Ollama URL requires finding and updating every file that references it. Missing one causes a runtime error. In Docker the URL is different from local dev. In Kubernetes it is different again.

**With pydantic-settings:** change one value in `.env` and everything updates. The priority order (env vars override `.env` override defaults) means the same codebase works in development, Docker, and Kubernetes without code changes — just different config.

This is the [twelve-factor app](https://12factor.net/) principle applied: store config in the environment, not in the code.

---

### Kubernetes design decisions

The project was designed with Kubernetes deployment in mind from the start. Several specific decisions reflect this.

**Stateless FastAPI.** Each FastAPI request is independent — no in-memory session state, no local file writes. This means you can run multiple FastAPI instances behind a load balancer and any instance can handle any request. Kubernetes can scale FastAPI horizontally by increasing `replicas` without any code changes.

**Health check endpoint.** `/healthz` exists specifically for Kubernetes liveness probes. When you deploy to K8s, the `livenessProbe` configuration pings this endpoint periodically. If it returns 200 the pod is healthy. If it fails, Kubernetes restarts the pod automatically. Building this endpoint from day one means the app is K8s-ready without modification.

**Environment-based config.** Kubernetes uses ConfigMaps and Secrets to inject environment variables into containers. Because all config is read from environment variables via pydantic-settings, deploying to Kubernetes requires no code changes — just a ConfigMap with the right values.

**Separate services.** Qdrant, Ollama, and FastAPI run as separate containers. In Kubernetes each becomes a separate Deployment with its own Service. This allows independent scaling — if query load increases, scale FastAPI without touching Qdrant or Ollama.

**The GPU tradeoff.** Ollama requires a GPU. In Kubernetes, GPU access requires the NVIDIA device plugin and GPU-enabled node pools. This is available in AKS (Azure Kubernetes Service) but adds cost and complexity. The practical production architecture keeps Ollama on dedicated on-prem GPU hardware and deploys FastAPI and Qdrant to Kubernetes — getting the scalability benefits of K8s for the stateless components while keeping the expensive GPU infrastructure stable and on-prem.

**`emptyDir` vs `PersistentVolumeClaim` for Qdrant.** The current K8s manifests use `emptyDir` for Qdrant storage — data is lost if the pod restarts. For production, Qdrant needs a `PersistentVolumeClaim` — a request for durable storage that survives pod restarts. This is a known limitation of the current manifests, appropriate for a demo but not for production. In AKS, a PVC backed by Azure Disk would be the correct choice.

---

### Tradeoffs and known limitations

Every project makes tradeoffs. Being explicit about them demonstrates engineering maturity.

**Chunk ID collision.** Chunk IDs are just sequential integers (0, 1, 2...). If you ingest two documents, the second document's chunks overwrite the first document's chunks with the same IDs. In production, chunk IDs should be UUIDs or composite keys like `f"{document_name}_{chunk_index}"`. Not fixed because it does not affect single-document demos, but would cause data loss in a multi-document production system.

**Synchronous Qdrant client.** The `qdrant_client` library is used synchronously even though FastAPI is async. This means Qdrant calls technically block the event loop. For production, use `AsyncQdrantClient` which allows proper async Qdrant calls. Not fixed because the sync client is simpler and the performance impact is acceptable at demo scale.

**No authentication.** The API endpoints have no authentication. Anyone who can reach port 8000 can ingest documents or query the system. In production, add JWT authentication or API key validation to all endpoints. Deliberately left out to keep the project focused on AI engineering rather than auth infrastructure.

**Llama3.1 8B tool calling reliability.** As documented in Part 3, Llama3.1 8B has inconsistent tool calling behavior. It gets correct answers but sometimes loops more than necessary. Production agents should use larger models (70B+) or models specifically fine-tuned for tool use. The current implementation includes MAX_ITERATIONS and force_answer as mitigations.

**Sequential embedding.** During ingestion, chunks are embedded one at a time in a loop. Ollama supports batch embedding — sending multiple texts in one request. Batch embedding would be significantly faster for large documents. Not implemented because the current approach is simpler to understand and fast enough for demo-scale documents.

**Single Qdrant collection.** All documents from all users go into one collection. There is no per-user or per-project isolation. In production you would likely have separate collections per tenant or use Qdrant's payload filtering to scope searches to specific document sets.

---

*Continue to Part 5 — Concepts Deep Dive (Decorators, Async, Kubernetes)*