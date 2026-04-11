from fastapi import APIRouter
from api.settings import settings

router = APIRouter()

@router.get("/healthz")
def health():
    return {"status": "ok"}

@router.get("/metrics")
def metrics():
    return {"status": "ok", "ollama_url": settings.ollama_url}