from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "clinical_docs"

    class Config:
        env_file = ".env"

settings = Settings()