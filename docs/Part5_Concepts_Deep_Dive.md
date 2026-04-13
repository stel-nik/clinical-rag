# ClinicalRAG — Architecture Deep Dive

## Part 5 — Concepts Deep Dive

This part explains three concepts that appear throughout the codebase and are worth understanding deeply: Python decorators, async/await, and Kubernetes.

---

## Decorators

### What a decorator is

A decorator is a function that wraps another function to add behavior without modifying the original function's code.

Start with a simple example that has nothing to do with FastAPI:

```python
def my_decorator(func):
    def wrapper():
        print("before the function runs")
        func()
        print("after the function runs")
    return wrapper

def say_hello():
    print("hello")

say_hello = my_decorator(say_hello)
say_hello()
```

Output:
```
before the function runs
hello
after the function runs
```

`my_decorator` takes `say_hello` as input, wraps it in a new function called `wrapper`, and returns `wrapper`. When you call `say_hello()` you are actually calling `wrapper()` which calls the original `say_hello()` in the middle.

The `@` syntax is shorthand for exactly this:

```python
@my_decorator
def say_hello():
    print("hello")
```

This is identical to writing `say_hello = my_decorator(say_hello)`. The `@` is just cleaner syntax — it makes it clear at the point of definition that this function is being decorated, rather than a separate assignment line later.

---

### Decorators with arguments

Decorators can also take arguments. This requires one more level of nesting:

```python
def repeat(times):
    def decorator(func):
        def wrapper():
            for _ in range(times):
                func()
        return wrapper
    return decorator

@repeat(times=3)
def say_hello():
    print("hello")

say_hello()
```

Output:
```
hello
hello
hello
```

`repeat(times=3)` returns a decorator. That decorator wraps `say_hello`. This pattern — a function that returns a decorator — is what `@app.get("/healthz")` uses.

---

### Decorators in FastAPI

FastAPI uses decorators to register route handlers. When you write:

```python
@app.get("/healthz")
def health():
    return {"status": "ok"}
```

`app.get("/healthz")` is a method call that returns a decorator. That decorator registers `health` as the handler for `GET /healthz` requests. When a request arrives at that path, FastAPI looks up the registered handler and calls it.

Under the hood, FastAPI maintains a routing table — a dictionary mapping (method, path) pairs to handler functions. The decorator adds an entry to that table.

---

### What the code would look like without decorators

Without decorators, you would register routes manually:

```python
app = FastAPI()

def health():
    return {"status": "ok"}

def metrics():
    return {"status": "ok", "ollama_url": settings.ollama_url}

async def ingest_document(file: UploadFile):
    ...

async def chat(request: ChatRequest):
    ...

# register all routes manually at the bottom
app.add_api_route("/healthz", health, methods=["GET"])
app.add_api_route("/metrics", metrics, methods=["GET"])
app.add_api_route("/documents/ingest", ingest_document, methods=["POST"])
app.add_api_route("/chat", chat, methods=["POST"])
```

This works — FastAPI supports `add_api_route` directly. But the problems are clear:

- The route and its handler are defined in separate places — you have to scroll to the bottom to understand which path each function handles
- Easy to forget to register a new function
- The file becomes cluttered with registration code

With decorators the route is declared at the point of definition. You can read the file top to bottom and immediately understand what each function does and where it is mounted. This is called declarative style — you declare what something is rather than imperatively telling the system what to do.

---

### Decorators elsewhere in the project

**`@mcp.tool()`** in `mcp_server/server.py`:

```python
@mcp.tool()
async def query_rag(question: str) -> str:
    """Ask a question over indexed clinical documents."""
    ...
```

Same pattern. `mcp.tool()` returns a decorator that registers `query_rag` as an MCP tool. The MCP SDK reads the function name, docstring, and type hints to build the tool definition that gets sent to clients.

**Other Python decorators you will encounter:**

`@staticmethod` — marks a method as static (no `self` parameter).

`@property` — makes a method callable as an attribute. `obj.value` instead of `obj.value()`.

`@classmethod` — a method that receives the class itself as the first argument instead of an instance.

`@functools.lru_cache` — caches the return value of a function. Repeated calls with the same arguments return the cached result.

---

### The mental model

Think of a decorator as a label you attach to a function that says "this function has this role". `@app.get("/healthz")` means "this function handles GET requests to /healthz". `@mcp.tool()` means "this function is an MCP tool". The framework reads these labels and knows what to do with each function.

---

## Async/Await

### The problem async solves

A web server handles many requests simultaneously. Consider two users calling `POST /chat` at the same time. Each request needs to:

1. Call Ollama to embed the question (~50ms)
2. Call Qdrant to search vectors (~20ms)
3. Call Ollama again to generate an answer (~5000ms)

Total: ~5 seconds per request.

**Without async — synchronous:**

```
User 1 request arrives
  → embed question → WAIT 50ms (server doing nothing)
  → search Qdrant  → WAIT 20ms (server doing nothing)
  → generate answer → WAIT 5000ms (server doing nothing)
  → respond to User 1

User 2 request arrives (was waiting the entire 5 seconds)
  → embed question → WAIT 50ms
  → ...
```

User 2 waited 5 seconds before their request even started. The server spent almost all of its time doing nothing — just waiting for external services to respond. Total time for both users: ~10 seconds.

**With async:**

```
User 1 request arrives
  → embed question → await (hand control back, not blocking)

User 2 request arrives (immediately handled)
  → embed question → await (hand control back)

Ollama responds for User 1
  → search Qdrant → await

Ollama responds for User 2
  → search Qdrant → await

Qdrant responds for User 1
  → generate answer → await

Qdrant responds for User 2
  → generate answer → await

Both answers ready → both users respond
```

Both requests run concurrently. Total time for both users: ~5 seconds instead of ~10.

---

### The event loop

The event loop is the engine that makes async work. It is a single loop that runs continuously, checking what work is ready to be done.

```
event loop:
  while True:
      for each task that is ready:
          run it until it hits an await
          when it hits await, pause it and move on
      check if any awaited operations completed
      if yes, mark those tasks as ready
```

When your code hits `await client.post(...)`, it tells the event loop: "I am waiting for a network response. Run something else while I wait. Come back to me when the response arrives."

The event loop is single-threaded — only one piece of Python code runs at a time. But because most of the waiting is I/O (network calls, disk reads), async allows the thread to stay busy handling other requests instead of sitting idle.

---

### Concurrency vs parallelism

These are often confused:

**Parallelism** — multiple things happening at exactly the same time on multiple CPU cores. True simultaneous execution. Python's `multiprocessing` module does this.

**Concurrency** — multiple things making progress, but not necessarily at the exact same time. One thread switches between tasks. Python's async does this.

For a web server making network calls, concurrency is what you need. Network calls spend most of their time waiting — the CPU is not busy, it is just waiting for bytes to arrive. Concurrency allows the CPU to do other work during that waiting time.

Parallelism would help for CPU-bound work — image processing, machine learning training, heavy computation. For this project, the GPU handles the heavy computation (Ollama runs on GPU, not CPU) so the Python code is mostly doing I/O — network calls to Ollama and Qdrant. Async/concurrency is exactly the right tool.

---

### `async def` vs `def`

**Use `async def` when the function:**
- Calls an external service (Ollama, Qdrant, a database)
- Uses `await` anywhere inside it
- Is called from another async function with `await`

**Use regular `def` when the function:**
- Does pure computation (chunking, string manipulation, math)
- Does not need to wait for anything external
- Is a simple utility function

In this project:

```python
# async — calls Ollama over the network
async def embed_text(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(...)

# sync — pure computation, no external calls
def chunk_text(text: str, ...) -> list[str]:
    words = text.split()
    ...

# sync — health check does no I/O
def health():
    return {"status": "ok"}

# async — calls both Ollama and Qdrant
async def chat(request: ChatRequest):
    query_vector = await embed_text(request.question)
    chunks = search_chunks(query_vector)
    ...
```

---

### What `await` actually does

`await` is a yield point. It says: "start this operation, pause this function, let the event loop run other things, and resume this function when the operation completes."

```python
async def embed_text(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(    # ← pause here
            f"{settings.ollama_url}/api/embed",
            json={"model": settings.embedding_model, "input": text},
            timeout=30.0
        )
        # ← resumed here when Ollama responds
        response.raise_for_status()
        return response.json()["embeddings"][0]
```

When the event loop sees `await client.post(...)`, it:
1. Starts the HTTP request (sends bytes to Ollama)
2. Pauses `embed_text` — suspends it at this line
3. Runs other ready tasks (other requests, other awaits that completed)
4. When Ollama's response arrives, marks `embed_text` as ready to continue
5. Resumes `embed_text` from where it paused

---

### What happens if you forget `await`

Forgetting `await` is a common bug. Without `await`, you get the coroutine object back instead of the result:

```python
# WRONG — missing await
response = client.post(f"{settings.ollama_url}/api/embed", ...)
# response is now a coroutine object, not an HTTP response
data = response.json()  # AttributeError: 'coroutine' has no attribute 'json'
```

Python will sometimes warn you with `RuntimeWarning: coroutine was never awaited` but not always. The error is confusing because `response` looks like it should work but is actually an unawaited coroutine.

The fix is always `await`:

```python
# CORRECT
response = await client.post(f"{settings.ollama_url}/api/embed", ...)
data = response.json()  # works correctly
```

---

### Why `httpx` instead of `requests`

`requests` is the most popular Python HTTP library but it is synchronous. Using `requests` inside an async function blocks the entire event loop:

```python
# WRONG — blocks the event loop
async def embed_text(text: str) -> list[float]:
    import requests
    response = requests.post(...)  # blocks everything while waiting
    ...
```

When `requests.post()` runs, it blocks the Python thread entirely. No other async tasks can run. The concurrency benefit is completely lost — it is as if you wrote synchronous code.

`httpx` is an async-compatible HTTP library. It has the same API as `requests` but supports async:

```python
# CORRECT — yields control while waiting
async def embed_text(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(...)  # yields control while waiting
```

The rule: in async code, every I/O operation must use an async-compatible library. `httpx` for HTTP, `asyncpg` for Postgres, `motor` for MongoDB, `aiofiles` for file I/O.

---

### `async with` — context managers in async code

A context manager handles setup and teardown automatically. The synchronous version:

```python
with open("file.txt") as f:
    content = f.read()
# file is automatically closed here, even if an exception occurred
```

`async with` is the same pattern for async resources:

```python
async with httpx.AsyncClient() as client:
    response = await client.post(...)
# client is automatically closed here
```

Why not just create a client directly?

```python
# works but resource leak risk
client = httpx.AsyncClient()
response = await client.post(...)
# if an exception occurs above, client.aclose() is never called
```

`async with` guarantees the client is closed properly even if an exception occurs. For network connections this matters — unclosed connections accumulate and eventually exhaust the connection pool.

---

### Walking through `chat.py` with two simultaneous requests

User 1 and User 2 both send `POST /chat` at the same moment. Here is exactly what happens:

```
t=0ms   User 1 request arrives → FastAPI starts async chat() for User 1
t=0ms   User 2 request arrives → FastAPI starts async chat() for User 2

t=1ms   User 1: await embed_text("What is informed consent?")
        → sends HTTP to Ollama, pauses, event loop runs User 2

t=1ms   User 2: await embed_text("How long must records be retained?")
        → sends HTTP to Ollama, pauses, event loop checks for completed I/O

t=51ms  Ollama responds to User 1's embed request
        → User 1 resumes: query_vector = [0.21, -0.85, ...]
        → User 1: search_chunks(query_vector) — synchronous Qdrant call
        → blocks briefly (known tradeoff — sync Qdrant client)

t=71ms  User 1: search complete, has chunks
        → await client.post(ollama_url/api/generate, ...)
        → sends generation request to Ollama, pauses

t=72ms  Ollama responds to User 2's embed request
        → User 2 resumes, searches Qdrant
        → User 2: await client.post(ollama_url/api/generate, ...)
        → sends generation request to Ollama, pauses

t=5071ms  Ollama responds with User 1's answer
          → User 1 resumes, builds response, returns to client

t=5072ms  Ollama responds with User 2's answer
          → User 2 resumes, builds response, returns to client
```

Both users receive their answers around 5 seconds. Without async they would have received them at 5 and 10 seconds.

The sync Qdrant client at ~t=51ms is the known bottleneck — it blocks the event loop briefly. For this project it is acceptable. In production, the async Qdrant client would allow Qdrant searches to run concurrently too.

---

### When NOT to use async

**Scripts.** `scripts/ingest_all.py` runs sequentially from the command line. There is no concurrent workload to manage — just process files one by one. Regular synchronous `httpx.post()` is simpler and correct here.

**CPU-bound work.** `chunk_text` splits text and builds lists. This is pure Python computation. Async would add overhead without benefit because there is no I/O to yield during. Regular `def` is correct.

**Simple functions.** `health()` and `metrics()` return dictionaries immediately. No I/O, no waiting. Regular `def` is correct.

**The rule in one sentence:** use `async def` when the function waits for something external. Use `def` for everything else.

---

## Kubernetes

### What problem Kubernetes solves

You have your application running in Docker containers. On one machine this works well. In production you need:

- Multiple machines running your containers for availability
- Automatic restart when a container crashes
- Scaling up when load increases, scaling down when it drops
- Rolling updates with no downtime
- Health monitoring across all containers

Kubernetes is the system that manages all of this. You describe what you want — "3 copies of FastAPI always running, restart if any crash, replace them one at a time when I deploy a new version" — and Kubernetes makes it happen and keeps it that way.

---

### The cluster

A Kubernetes cluster is a group of machines working together:

```
Kubernetes cluster
├── Control plane (master node)
│   ├── API server      ← kubectl talks to this
│   ├── Scheduler       ← decides which node gets each pod
│   ├── Controller      ← watches pods, restarts crashed ones
│   └── etcd            ← stores all cluster state
│
├── Worker node 1
│   ├── kubelet         ← agent that runs pods on this node
│   ├── kube-proxy      ← handles networking
│   └── Pods            ← your running containers
│
└── Worker node 2
    ├── kubelet
    ├── kube-proxy
    └── Pods
```

The control plane is the brain — it makes decisions. Worker nodes are the muscles — they run your containers. In a cloud deployment (AKS, EKS, GKE) the control plane is managed for you.

---

### The five resources you need to know

**Pod** — the smallest deployable unit. One or more containers running together on the same node. You almost never create pods directly — you let Deployments manage them.

**Deployment** — manages pods. Ensures the right number are always running. If a pod crashes, the Deployment controller creates a new one. If you update the container image, it replaces pods one by one with no downtime.

```yaml
kind: Deployment
spec:
  replicas: 3  # always keep 3 pods running
```

**Service** — gives pods a stable network address. Pods have dynamic IPs that change every time they restart. A Service gives them a permanent address and load balances traffic across all matching pods.

```yaml
kind: Service
spec:
  selector:
    app: api     # routes to pods with this label
  ports:
    - port: 8000
```

**ConfigMap** — stores non-secret configuration as key-value pairs. Pods read these as environment variables. The Kubernetes equivalent of your `.env` file.

```yaml
kind: ConfigMap
data:
  OLLAMA_URL: "http://ollama-service:11434"
  QDRANT_URL: "http://qdrant-service:6333"
```

**Secret** — like ConfigMap but for sensitive data. Values are base64-encoded. In production use external secret management like Azure Key Vault.

---

### How labels connect everything

Labels are key-value tags attached to any Kubernetes resource. They are how resources find each other.

A Deployment creates pods with a label:
```yaml
template:
  metadata:
    labels:
      app: api     # pods get this label
```

A Service selects pods by that label:
```yaml
selector:
  app: api         # route traffic to pods with this label
```

The Deployment and Service are completely independent resources connected only through labels. This means you can have multiple versions of a service running simultaneously just by changing which labels the Service selects.

---

### Health checks — how `/healthz` connects to Kubernetes

When you deploy to Kubernetes you configure a liveness probe:

```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
  failureThreshold: 3
```

Kubernetes calls `GET /healthz` every 30 seconds. If it returns 200, the pod is healthy. If it fails 3 times in a row, Kubernetes kills the pod and starts a new one automatically. This is why `/healthz` was built from day one.

---

### The manifest files for this project

**`configmap.yaml`:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: clinicalrag-config
data:
  OLLAMA_URL: "http://ollama-service:11434"
  QDRANT_URL: "http://qdrant-service:6333"
  COLLECTION_NAME: "clinical_docs"
  EMBEDDING_MODEL: "nomic-embed-text"
  LLM_MODEL: "mistral"
  EMBEDDING_SIZE: "768"
```

URLs use Service names — `ollama-service`, `qdrant-service` — not localhost. Inside a Kubernetes cluster, services find each other by their Service resource names, the same concept as Docker Compose service names.

---

**`qdrant-deployment.yaml`:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest
          ports:
            - containerPort: 6333
          volumeMounts:
            - name: qdrant-storage
              mountPath: /qdrant/storage
      volumes:
        - name: qdrant-storage
          emptyDir: {}
```

`replicas: 1` — Qdrant is a database. Multiple replicas without clustering configuration would cause data inconsistency.

`emptyDir: {}` — temporary storage, lost if the pod restarts. For production replace with a PersistentVolumeClaim backed by Azure Disk.

`selector.matchLabels` and `template.metadata.labels` must match — this is how the Deployment finds and manages its pods.

---

**`qdrant-service.yaml`:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
spec:
  selector:
    app: qdrant
  ports:
    - port: 6333
      targetPort: 6333
```

`ClusterIP` type (default) — only accessible inside the cluster. Qdrant should not be exposed to the internet.

---

**`ollama-deployment.yaml`** — the GPU-specific part:

```yaml
containers:
  - name: ollama
    image: ollama/ollama:latest
    resources:
      limits:
        nvidia.com/gpu: 1
    volumeMounts:
      - name: ollama-models
        mountPath: /root/.ollama
volumes:
  - name: ollama-models
    emptyDir: {}
```

`nvidia.com/gpu: 1` — requests one GPU. Kubernetes will only schedule this pod on a node that has an available GPU. In AKS this requires a GPU node pool (NC-series VMs). The NVIDIA device plugin must be installed in the cluster.

---

**`api-deployment.yaml`:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
        - name: api
          image: your-registry/clinical-rag-api:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: clinicalrag-config
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
```

`replicas: 2` — FastAPI is stateless so two copies give availability without coordination.

`envFrom.configMapRef` — reads all key-value pairs from the ConfigMap and injects them as environment variables. This is how Kubernetes replaces your `.env` file.

`image: your-registry/clinical-rag-api:latest` — the Docker image built from your Dockerfile, pushed to a container registry.

---

### Docker Compose vs Kubernetes — side by side

| Concept | Docker Compose | Kubernetes |
|---------|---------------|------------|
| Define a service | `services: api:` | `kind: Deployment` |
| Container image | `image: ollama/ollama` | `image: ollama/ollama` |
| Port mapping | `ports: "8000:8000"` | `kind: Service` |
| Environment variables | `env_file: .env` | `kind: ConfigMap` + `envFrom` |
| Persistent storage | `volumes: qdrant_data:` | `kind: PersistentVolumeClaim` |
| Health check | not standard | `livenessProbe` |
| Scaling | `--scale api=3` | `replicas: 3` in Deployment |
| Service discovery | service names in compose | service names via Service |
| Start everything | `docker compose up` | `kubectl apply -f infra/k8s/` |

The concepts map directly. Docker Compose is for one machine. Kubernetes is for many machines. The container images are identical — the same Dockerfile works in both.

---

### Key kubectl commands

```bash
# apply manifests
kubectl apply -f infra/k8s/

# see what is running
kubectl get pods
kubectl get services
kubectl get deployments

# see details
kubectl describe pod api-7d9f8b-xk2j4
kubectl describe service qdrant-service

# see logs
kubectl logs api-7d9f8b-xk2j4
kubectl logs api-7d9f8b-xk2j4 --follow

# run a command inside a pod
kubectl exec -it api-7d9f8b-xk2j4 -- bash

# delete resources
kubectl delete -f infra/k8s/

# scale a deployment
kubectl scale deployment api --replicas=3

# resource usage
kubectl top pods
kubectl top nodes
```

Pod names include a random suffix — `api-7d9f8b-xk2j4`. Use `kubectl get pods` to find current names.

---

### How the app is already K8s-ready

**Stateless FastAPI** — no in-memory state between requests. Multiple replicas can run without coordination. Scale horizontally by increasing `replicas`.

**Health endpoint** — `/healthz` works as a liveness probe without modification.

**Environment-based config** — pydantic-settings reads environment variables. Kubernetes ConfigMaps inject environment variables. No code changes needed.

**Containerized** — a Dockerfile exists. Build, push, reference in manifests.

**Separate services** — Qdrant, Ollama, and FastAPI are separate containers. Each becomes an independent Deployment in Kubernetes, scalable and updatable independently.

The only additions needed for production: PersistentVolumeClaim for Qdrant, GPU node pool for Ollama, container registry for the API image.

---

### Namespaces

A namespace divides a cluster into logical groups:

```bash
kubectl get pods --namespace production
kubectl get pods --namespace staging
```

All manifests use `namespace: default`. In a real deployment create a dedicated namespace:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: clinical-rag
```

This keeps resources separate from other applications in the same cluster, with independent access controls and resource quotas.

---

*End of Architecture Deep Dive*