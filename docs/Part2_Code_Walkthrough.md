# ClinicalRAG — Architecture Deep Dive

## Part 2 — Code Walkthrough

This part walks through every file in the project, explaining what each one does, why it is written the way it is, and how it connects to the rest of the system.

---

### Project structure recap

```
api/
├── main.py              # creates the FastAPI app, registers routers
├── settings.py          # reads config from .env
├── core/
│   ├── chunking.py      # splits text into overlapping chunks
│   ├── embeddings.py    # calls Ollama to embed text
│   ├── retrieval.py     # searches Qdrant for similar chunks
│   └── vector_store.py  # creates Qdrant collection, stores vectors
└── routes/
    ├── health.py        # /healthz and /metrics endpoints
    ├── documents.py     # /documents/ingest endpoint
    └── chat.py          # /chat endpoint

agent/
└── agent.py             # private on-prem agent using Llama3.1

mcp_server/
└── server.py            # MCP server exposing RAG as tools

scripts/
└── ingest_all.py        # batch ingest all files in data/samples/
```

The separation between `core/` and `routes/` is intentional. `core/` contains pure business logic with no HTTP concepts — chunking, embedding, retrieval. `routes/` contains the HTTP layer that calls the core logic. This means the core functions are reusable — the agent, the MCP server, and future tools can all import from `core/` without touching the HTTP layer.

---

### `settings.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "clinical_docs"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "mistral"
    embedding_size: int = 768

    class Config:
        env_file = ".env"

settings = Settings()
```

**What it does:** reads configuration from the `.env` file and exposes it as a typed Python object.

**Why pydantic-settings:** Pydantic validates types automatically. If `EMBEDDING_SIZE` is set to `"not-a-number"` in `.env`, the app fails immediately on startup with a clear error rather than crashing later with a confusing message.

**The priority order:** when `Settings()` is created, Pydantic checks in this order:
1. Real environment variables set in the shell
2. Values in `.env`
3. Default values in the class definition

This means you can override any setting without touching the code — just set an environment variable. This is how Kubernetes ConfigMaps work: they inject environment variables into the container, overriding the defaults.

**`settings = Settings()`** creates one instance at module level. Every file that does `from api.settings import settings` gets the same object — no repeated file reads, no inconsistency.

**Why model names are in settings:** to switch from Mistral to Llama3.2, you change one line in `.env`. No hunting through files.

---

### `main.py`

```python
from fastapi import FastAPI
from api.routes.documents import router as documents_router
from api.routes.chat import router as chat_router
from api.routes.health import router as health_router

app = FastAPI(
    title="ClinicalRAG",
    description="Private on-prem RAG system for clinical documents",
    version="0.1.0"
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
```

**What it does:** creates the FastAPI application and registers all route handlers.

**`app = FastAPI(...)`** instantiates the FastAPI class. This is the central object that holds all routes, middleware, and event handlers. Uvicorn receives this object and starts serving HTTP requests.

**Why routers:** instead of defining all routes in `main.py`, each domain gets its own file with its own `APIRouter`. `main.py` just registers them. This keeps `main.py` clean and makes it easy to add new route groups.

**The title, description, and version** appear automatically in the Swagger UI at `/docs`. This matters for demos — it looks professional and gives context to anyone exploring the API.

**What `main.py` does NOT do:** no business logic. No database calls. No HTTP calls to Ollama. It is purely a wiring file.

---

### `core/chunking.py`

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
```

**What it does:** takes a string of text and returns a list of smaller strings (chunks).

**Why split by words not characters:** splitting by characters creates chunks that cut mid-word. Splitting by words gives cleaner boundaries and makes `chunk_size` intuitive.

**The algorithm with a small example:**

```
text = "word1 word2 word3 word4 word5 word6"
chunk_size=3, overlap=1

iteration 1: start=0, end=3 → "word1 word2 word3"
             start = 0 + 3 - 1 = 2

iteration 2: start=2, end=5 → "word3 word4 word5"
             start = 2 + 3 - 1 = 4

iteration 3: start=4, end=7 → "word5 word6"
             start = 4 + 3 - 1 = 6 → loop ends
```

Notice "word3" appears in chunks 1 and 2, "word5" in chunks 2 and 3. That is the overlap — context preserved across boundaries.

**Why this function has no side effects:** it takes text in and returns chunks out. No database calls, no HTTP calls, no global state. Easy to test, easy to reuse anywhere.

**Default values (500 words, 50 overlap):** 500 words is roughly 2-3 paragraphs — enough context for most questions without being too broad.

---

### `core/embeddings.py`

```python
import httpx
from api.settings import settings

async def embed_text(text: str) -> list[float]:
    """Send text to Ollama and get back an embedding vector."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_url}/api/embed",
            json={
                "model": settings.embedding_model,
                "input": text
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
```

**What it does:** sends one chunk of text to Ollama and gets back a list of 768 floats.

**Why `async def`:** this function makes an HTTP call to Ollama which takes time. Marking it async allows FastAPI to handle other requests while waiting for Ollama to respond.

**`async with httpx.AsyncClient() as client`:** opens an HTTP client, uses it, then closes it automatically. The context manager handles cleanup even if an exception occurs. `httpx` is used instead of `requests` because `requests` is synchronous and would block the event loop.

**`response.raise_for_status()`:** if Ollama returns an error, this raises an exception immediately. Fail fast, fail clearly.

**`response.json()["embeddings"][0]`:** Ollama returns a list of embeddings (supports batch). Since you send one text at a time, you take index `[0]`.

**`timeout=30.0`:** if Ollama does not respond within 30 seconds, raise an exception. Without a timeout, a hung Ollama process would cause your request to wait forever.

---

### `core/retrieval.py`

```python
from qdrant_client import QdrantClient
from api.settings import settings

client = QdrantClient(url=settings.qdrant_url)

def search_chunks(query_vector: list[float], top_k: int = 3) -> list[dict]:
    """Search Qdrant for the most similar chunks to the query vector."""
    results = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k
    ).points

    chunks = []
    for result in results:
        chunk = {
            "text": result.payload["text"],
            "document": result.payload["document"],
            "score": result.score
        }
        chunks.append(chunk)
    return chunks
```

**What it does:** takes a query vector and returns the top K most similar chunks from Qdrant.

**`client = QdrantClient(...)` at module level:** created once when the module is imported, not on every function call. The client holds a connection pool and reuses it across requests.

**Why `search_chunks` is not async:** the Qdrant Python client is synchronous. For production you would use the async Qdrant client — this is a known tradeoff.

**`top_k=3`:** return the 3 most relevant chunks. Enough context for most questions without overwhelming the LLM.

**`result.score`:** cosine similarity score between 0 and 1. Closer to 1 means more similar to the query. Scores below 0.5 usually indicate the question is not covered by the indexed documents.

**The payload:** when you stored the chunk in Qdrant, you attached metadata (text, document name, chunk index). This payload comes back with the search result. You needed the vector to find the chunk — now you need the text to send to the LLM.

---

### `core/vector_store.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from api.settings import settings

client = QdrantClient(url=settings.qdrant_url)

def ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if settings.collection_name not in names:
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_size,
                distance=Distance.COSINE
            )
        )

def store_chunks(chunks: list[str], embeddings: list[list[float]], document_name: str):
    """Store chunks and their embeddings in Qdrant."""
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=i,
            vector=vector,
            payload={
                "text": chunk,
                "document": document_name,
                "chunk_index": i
            }
        )
        points.append(point)

    client.upsert(
        collection_name=settings.collection_name,
        points=points
    )
```

**`ensure_collection`:** checks if the collection exists before creating it. Called every time a document is ingested. Safe to call multiple times — if the collection exists, nothing happens. This pattern is called idempotent.

**`Distance.COSINE`:** cosine similarity measures the angle between two vectors regardless of their magnitude. Standard choice for text embeddings.

**`VectorParams(size=settings.embedding_size)`:** tells Qdrant to expect vectors of exactly 768 numbers. Qdrant rejects any vector with a different dimension — useful validation.

**`zip(chunks, embeddings)`:** pairs each chunk with its embedding. More readable than indexing with `[i]`. `enumerate` adds the position index.

**`PointStruct`:** one item in Qdrant — ID, vector, and payload.

**`client.upsert`:** update or insert. Re-ingesting a document updates existing vectors rather than creating duplicates.

**Known limitation:** chunk ID is just the index (0, 1, 2...). If you ingest two documents, the second document's chunks will overwrite the first document's chunks with the same index. In production, use a UUID or composite ID like `f"{document_name}_{chunk_index}"`.

---

### `routes/documents.py`

```python
@router.post("/documents/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files supported for now")

    text = (await file.read()).decode("utf-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    chunks = chunk_text(text)

    embeddings = []
    for chunk in chunks:
        vector = await embed_text(chunk)
        embeddings.append(vector)

    ensure_collection()
    store_chunks(chunks, embeddings, file.filename)

    return {"status": "indexed", "filename": file.filename, "chunks": len(chunks)}
```

**What it does:** receives a file upload, runs the full ingestion pipeline, returns confirmation.

**`UploadFile = File(...)`:** tells FastAPI to expect multipart file upload. `...` means required.

**`await file.read()`:** reads file bytes asynchronously. `.decode("utf-8")` converts bytes to string.

**Two validation checks:** file extension and empty content. Both return HTTP 400 with a clear message — fail early rather than letting the pipeline run on invalid input.

**This function orchestrates but does not implement:** it calls functions from `core/`. Routes handle HTTP concerns, core handles business logic.

---

### `routes/chat.py`

The chat endpoint implements the full RAG query pipeline in 5 clear steps:

```python
# step 1 - embed the question (reuses embed_text from core/)
query_vector = await embed_text(request.question)

# step 2 - search qdrant (reuses search_chunks from core/)
chunks = search_chunks(query_vector, top_k=3)

# step 3 - build prompt
context = "\n\n".join([chunk["text"] for chunk in chunks])
prompt = f"...Context: {context}\n\nQuestion: {request.question}\n\nAnswer:"

# step 4 - send to mistral
response = await client.post(f"{settings.ollama_url}/api/generate", ...)
answer = response.json()["response"]

# step 5 - return answer and citations
return {"answer": answer, "sources": [...]}
```

**`class ChatRequest(BaseModel)`:** Pydantic validates the request body. If the caller sends wrong types, FastAPI returns 422 before your code runs.

**Step 1 reuses `embed_text`:** the same function used during ingestion. Critical — the question and chunks must be embedded by the same model. Different models produce vectors in different spaces — search would fail.

**The prompt structure matters:**
```
System instruction → role and behavior
Context → retrieved chunks separated by blank lines
Question → what the user asked
Answer: → signals where the model should start writing
```

The instruction "If the answer is not in the context, say I don't know" prevents hallucination — Mistral answering from training data instead of your documents.

**`/api/generate` not `/api/embed`:** two different Ollama endpoints. `/api/embed` converts text to vectors. `/api/generate` generates text given a prompt.

**`"stream": False`:** wait for the complete answer rather than streaming word by word. Streaming would require more complex code.

**Citations:** `chunk["text"][:200]` returns only the first 200 characters as a preview. The full text is in Qdrant — no need to return it all in every response.

---

### `routes/health.py`

```python
@router.get("/healthz")
def health():
    return {"status": "ok"}

@router.get("/metrics")
def metrics():
    return {"status": "ok", "ollama_url": settings.ollama_url}
```

**`/healthz`:** the Kubernetes health check endpoint. K8s pings this periodically — if it returns 200 the container is healthy, if it fails K8s restarts the container. This is why you built it from day one.

**Why `healthz` not `health`:** the `z` is a convention from Google's internal systems. It distinguishes the technical health check endpoint from any user-facing route.

**Neither function is `async`:** they do not call any external services. No need for async.

---

### What is reusable across the project

`core/` modules have no FastAPI dependencies — pure Python functions. They are reused in multiple places:

| Function | Used in |
|----------|---------|
| `embed_text` | `documents.py` (embed chunks) + `chat.py` (embed query) |
| `search_chunks` | `chat.py` |
| `chunk_text` | `documents.py` |
| `ensure_collection` | `documents.py` |
| `store_chunks` | `documents.py` |

The agent and MCP server do NOT import from `core/` directly — they call the FastAPI endpoints over HTTP. This keeps them decoupled. If you change the RAG implementation, the MCP server does not need to change.

---

*Continue to Part 3 — The Agent and MCP Server*