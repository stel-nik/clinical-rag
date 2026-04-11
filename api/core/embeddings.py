import httpx
from api.settings import settings

async def embed_text(
    text: str
    ) -> list[float]:
    '''
    Send text to Ollama and get back an embedding vector.
    '''
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_url}/api/embed",
            json={
                'model': 'nomic-embed-text',
                'input': text
            },
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()['embeddings'][0]