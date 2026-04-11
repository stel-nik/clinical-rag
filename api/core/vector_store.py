from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from api.settings import settings

client = QdrantClient(url=settings.qdrant_url)

def ensure_collection():
    '''
    Create the Qdrant collection if it doesn't exist.
    '''
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
        
def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    document_name: str
    ): 
    '''
    Store chunks and their embeddings in Qdrant.
    '''
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={
                'text': chunks[i],
                'document': document_name,
                'chunk_index': i
            }
        )
        for i in range(len(chunks))
    ]
    
    client.upsert(
        collection_name=settings.collection_name,
        points=points
    )   