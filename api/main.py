from fastapi import FastAPI
from api.routes.documents import router as documents_router
from api.settings import settings

app = FastAPI(
    title="ClinicalRAG",
    description="Private on-prem RAG system for clinical documents",
    version="0.1.0"
)

app.include_router(documents_router)

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return {"status": "ok", "ollama_url": settings.ollama_url}