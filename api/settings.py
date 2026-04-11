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