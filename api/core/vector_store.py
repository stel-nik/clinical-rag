import uuid
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
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=str(uuid.uuid4()), # unique ID per chunk — prevents overwriting when ingesting multiple documents 
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