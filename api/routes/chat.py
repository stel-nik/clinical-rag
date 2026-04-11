import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.core.embeddings import embed_text
from api.core.retrieval import search_chunks
from api.settings import settings

router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    
@router.post('/chat')
async def chat(request: ChatRequest):
    '''
    Answer a question using RAG over indexed clinical documents.
    '''
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail='Question cannot be empty'
        )
        
    # step 1 - embed the question
    query_vector = await embed_text(request.question)
    
    # step 2 - search quadrant
    chunks = search_chunks(query_vector, top_k=3)
    
    if not chunks:
        raise HTTPException(status_code=404, detail='No relevant documents found')
    
    # step 3 - build prompt
    texts = []
    for chunk in chunks:
        texts.append(chunk['text'])
    context = '\n\n'.join(texts)        
     
    prompt = f'''You are a clinical research assistant. 
        Answer the question below using only the context provided.
        If the answer is not in the context, say "I don't know".
        Context: {context}
        Question: {request.question}
        Answer:'''
    
    # step 4 - send to mistral
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f'{settings.ollama_url}/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False
            },
            timeout=60.0
        )
        response.raise_for_status()
        
    answer = response.json()['response']
    
    # step 5 - return answer and citations
    sources = []
    for chunk in chunks:
        source = {
            'text': chunk['text'][:200],
            'document': chunk['document'],
            'score': chunk['score']
        }
        sources.append(source)
        
    return {
        'answer': answer,
        'sources': sources   
    }