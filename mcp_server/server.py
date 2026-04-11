import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('ClinicalRAG')

API_URL = 'http://localhost:8000'

@mcp.tool()
async def query_rag(question: str) -> str:
    '''
    Ask a question over indexed clinical documents.
    Returns an answer with citations from the source documents.
    '''
    async with httpx.AssyncClient() as client:
        response = await client.post(
            f'{API_URL}/chat',
            json={'question': question},
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        
        answer = data['answer']
        sources = data['sources']
        
        source_texts = []
        for source in sources:
            source_texts.append(
                f'- {source['document']} (score: {source['score']:.2f}): {source['text'][:100]}'
            )
        sources_formatted = '\n'.join(source_texts)
        
        return f'Answer: {answer}\n\nSources:\n{sources_formatted}'
    
@mcp.tool()
async def ingest_document(file_path: str) -> str:
    '''
    Ingest a text file into the clinical RAG system.
    Chunks, embeds and stores the document for future queries.
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    filename = file_path.split('/')[-1].split('\\')[-1]
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f'{API_URL}/documents/ingest',
            files={'file': (filename, content.encode(), 'text/plain')},
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()
        
        return f'Successfully indexed {data['filename']} into {data['chunks']} chunks.'
    
if __name__ == '__main__':
    mcp.run()