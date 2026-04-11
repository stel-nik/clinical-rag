from qdrant_client import QdrantClient
from api.settings import settings

client = QdrantClient(url=settings.qdrant_url)

def search_chunks(
    query_vector: list[float],
    top_k: int = 3
    ) -> list[dict]:
    '''
    Search Qdrant for the most similar chunks to the query vector.
    Returns the top_k most relevant chunks with their text.
    '''
    results = client.query_points(
        collection_name=settings.collection_name,
        query_vector=query_vector,
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