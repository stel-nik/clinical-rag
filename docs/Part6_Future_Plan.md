# Part 6 — Future Plan

This document outlines planned improvements to ClinicalRAG, grouped by theme. It covers what the system currently does well, what is missing for production readiness, and the priority order for future development.

---

## What ClinicalRAG already covers

- On-prem LLM deployment — Ollama running Mistral 7B, Llama3.1 8B, and nomic-embed-text locally on an RTX 3090
- RAG pipeline from scratch — chunking, embedding, vector search, and generation without external frameworks
- Vector database — Qdrant with HNSW indexing and cosine similarity
- MCP server — exposes `query_rag` and `ingest_document` as tools for AI agents
- Private agent — Llama3.1 with tool calling via MCP, fully on-prem
- FastAPI — REST API layer with health checks and metrics endpoints
- Docker and Docker Compose — full containerisation of all services
- Kubernetes manifests — written and partially tested locally
- Architecture documentation — five-part documentation covering architecture, code walkthrough, MCP server, design decisions, and concepts

---

## Planned improvements

### Observability and monitoring

The system currently has basic health checks and a metrics endpoint. The next step is proper observability across all components.

**Structured logging** — every request should log the model used, request latency, token count, retrieval scores, and document sources. Logs should be structured as JSON so they can be queried and analysed. This replaces the current mix of print statements and basic logging.

**MLflow tracking** — track every ingestion run and query run as an MLflow experiment. Log parameters (chunk size, overlap, embedding model, top_k) and metrics (ingestion time, number of chunks, retrieval scores). This enables comparison across different configurations and provides an audit trail of system behaviour.

**Prometheus metrics** — extend the existing `/metrics` endpoint to expose request counts, latency histograms, error rates, and model load times in Prometheus format. This enables integration with standard monitoring stacks.

**Readiness and liveness probes** — extend existing health checks to properly distinguish between liveness (is the service running) and readiness (is the service ready to handle requests, are models loaded).

Covers: monitoring and continuous improvement requirements, and the compliance evidence request in enterprise deployments.

---

### Testing and CI/CD

The system currently has no automated tests and no CI/CD pipeline. This is the most visible gap for production readiness.

**Unit tests** — test core functions independently: `chunk_text()` with various input sizes, `embed_text()` with mock Ollama responses, `search_chunks()` with a small test collection, and `store_chunks()` with UUID ID generation.

**Integration tests** — spin up the full stack and test end-to-end: ingest a test document, query it, verify the answer contains relevant content. These tests run against real services and verify the full pipeline works.

**GitHub Actions CI** — on every pull request: install dependencies, run linting with flake8 or ruff, run unit tests, build the Docker image to verify it compiles. Failures block merging.

**GitHub Actions CD** — on merge to main: build and push the Docker image to GitHub Container Registry (ghcr.io), tag with the commit SHA and latest. This makes every main branch commit a deployable artefact.

Covers: CI/CD pipelines and version control best practices from the job description.

---

### Guardrails

The system currently has no input or output filtering. In a regulated clinical environment this is a significant gap — the model can be asked anything and will attempt to answer from its training data rather than the indexed documents.

**Input filtering** — a middleware layer that validates requests before they reach the model. Blocks queries outside the clinical domain, enforces maximum query length, and rejects requests that match known adversarial patterns (prompt injection attempts).

**Output filtering** — validates model responses before returning them. Flags responses that contain hallucinated citations, responses that contradict the retrieved context, or responses that contain personally identifiable information.

**System prompt hardening** — enforce via the system prompt that the model must only answer from the provided context and must not use its own training data. The current system prompt does this at a basic level — this would make it more robust.

**Configurable policy** — allow guardrail rules to be defined in a configuration file rather than hardcoded. This enables different rule sets for different document collections or customer environments without code changes. This follows the pattern of OpenAI's gpt-oss-safeguard which evaluates requests against a developer-provided natural language policy at inference time.

Covers: custom guardrails request in enterprise deployments and secure AI systems requirement.

---

### Evaluation

The system currently has no way to measure whether it is answering correctly. This makes it impossible to compare different configurations or detect quality degradation over time.

**Evaluation dataset** — a small set of clinical questions with known correct answers, sourced from the indexed documents. Even 20-30 questions provides a meaningful baseline.

**Retrieval quality metrics** — for each question, measure whether the correct document chunk appears in the top-k results. Metrics: precision@k, recall@k, mean reciprocal rank.

**Answer quality metrics** — measure whether the generated answer is faithful to the retrieved context (does not hallucinate) and relevant to the question. Tools: RAGAS or DeepEval provide these metrics without requiring human annotation.

**MLflow experiment tracking** — log all evaluation results to MLflow. This enables comparison across chunk sizes, embedding models, top_k values, and system prompt variations. Running the evaluation before and after any configuration change shows whether the change improved or degraded quality.

**Drift detection** — monitor the distribution of retrieval scores over time. A shift in the score distribution may indicate that new documents are semantically different from the training distribution of the embedding model, or that the query patterns have changed.

Covers: model behaviour benchmarks for compliance evidence, and continuous improvement practices.

---

### Security and compliance

The system currently has no authentication and no audit trail. Any request to the API is accepted without verification.

**API authentication** — add API key authentication to all endpoints. Each client gets a unique key that is passed in the request header. Invalid or missing keys return a 401 response. Keys are stored as environment variables, not in code.

**Audit logging** — log every request with timestamp, client identifier, query text hash (not the full query, to protect privacy), which documents were retrieved, and response time. This provides a traceable record of system usage for compliance purposes.

**Per-user document isolation** — allow different API keys to access different Qdrant collections. This enables multi-tenant deployments where different users or teams can only query their own documents.

**Data retention policy** — define how long logs and audit records are retained. In regulated environments this is often mandated (for example, 15 years for clinical trial records).

**Secrets management** — all secrets currently stored in `.env` files. The next step is integration with a secrets manager (Azure Key Vault or HashiCorp Vault) so secrets are never stored on disk in any environment.

Covers: data governance, security, and compliance requirements from the job description.

---

### Multi-provider LLM abstraction

The system currently only supports Ollama. Adding a provider abstraction layer enables switching between on-prem and cloud models without changing application code.

**Provider interface** — a Python abstract class that defines the embedding and generation interface. Ollama, OpenAI, and Claude implement this interface. The active provider is selected via an environment variable.

**OpenAI and Claude API support** — implement the provider interface for OpenAI and Anthropic APIs. This enables the system to run in two modes: fully on-prem with Ollama for sensitive data, or cloud API for less sensitive workloads.

**Token counting and cost tracking** — when using cloud APIs, log token counts per request and accumulate cost estimates. This enables cost analysis and prompt optimisation.

**Retry logic and timeout handling** — cloud APIs can fail or be slow. Add exponential backoff retry logic and configurable timeouts so the system degrades gracefully rather than returning errors.

Covers: flexibility across deployment models and understanding of cost management in AI systems.

---

### Infrastructure and cloud deployment

The system currently runs locally via Docker Compose. The Kubernetes manifests are written but Ollama cannot run on GPU in Kubernetes on Windows due to Docker Desktop virtualisation limitations.

**PersistentVolumeClaims** — replace emptyDir volumes in Kubernetes manifests with PersistentVolumeClaims for Qdrant and Ollama. This ensures data survives pod restarts and is production-safe.

**Terraform** — provision cloud infrastructure as code. Modules for Azure Container Registry, AKS cluster, storage accounts, and Key Vault. This enables reproducible deployments and documents the infrastructure alongside the application code.

**Azure deployment** — deploy FastAPI and Qdrant to Azure Container Apps or AKS. Ollama stays on-prem on the RTX 3090 for GPU inference, connected to the cloud services via a secure tunnel. This hybrid pattern reflects how enterprises typically deploy AI systems — cloud for the stateless API layer, on-prem for GPU inference.

**GPU on Kubernetes** — the current Windows limitation prevents Ollama from running on GPU inside a Kubernetes pod. On a Linux node with native Docker Engine and the NVIDIA device plugin, this works correctly. The Kubernetes manifests are already written and production-ready for a Linux deployment.

Covers: scalable AI workloads in Kubernetes, cloud deployment experience, and infrastructure as code.

---

## Priority order

Given the current state of the project and its primary purpose as a portfolio piece for AI engineering roles in regulated environments, the recommended implementation order is:

1. **Basic guardrails** — highest relevance to regulated environments, directly demonstrable
2. **Structured logging and MLflow** — observability story, connects to compliance evidence use case
3. **Evaluation dataset and metrics** — makes quality claims verifiable
4. **CI/CD with GitHub Actions** — closes the most visible production readiness gap
5. **API authentication** — minimum viable security for any real deployment
6. **Multi-provider LLM abstraction** — enables demonstration of cost and flexibility tradeoffs
7. **Terraform and Azure deployment** — completes the cloud deployment story
8. **PersistentVolumeClaims in Kubernetes** — production-ready stateful deployment

Items 1-3 are high priority for demonstrating production thinking in interviews. Items 4-8 are medium priority for long-term portfolio strength.

---

## Known limitations (current state)

- Chunk ID collision on re-ingestion — fixed with UUID, documented
- No authentication on API endpoints
- No input or output filtering
- No evaluation framework — no way to measure answer quality
- Sync Qdrant client — should use AsyncQdrantClient in production
- Single Qdrant collection — no per-user isolation
- GPU scheduling in Kubernetes on Windows — requires native Linux, documented
- PDF special character handling — branch in progress
- No CI/CD pipeline
- No structured logging or audit trail
