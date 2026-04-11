from fastapi import APIRouter, UploadFile, File, HTTPException
from api.core.chunking import chunk_text
from api.core.embeddings import embed_text
from api.core.vector_store import ensure_collection, store_chunks

router = APIRouter()

@router.post('/documents/ingest')
async def ingest_document(file: UploadFile = File(...)):
    '''
    Upload the text file, chunk it, embed it and store in Qdrant.
    '''
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail='Only .txt files supported for now')
    
    text = (await file.read()).decode('utf-8')
    
    if not text.strip():
        raise HTTPException(status_code=400, detail='File is empty')
    
    chunks = chunk_text(text)
    
    embeddings = []
    for chunk in chunks:
        vector = await embed_text(chunk)
        embeddings.append(vector)
        
    ensure_collection()
    store_chunks(chunks, embeddings, file.filename)
    
    return {
        'status': 'indexed',
        'filename': file.filename,
        'chunks': len(chunks)
    }
    