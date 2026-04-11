import httpx
from mcp.server.fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP('ClinicalRAG')
API_URL = "http://localhost:8000"

@mcp.tool()
async def query_rag(question: str) -> str:
    """Ask a question over indexed clinical documents."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f'{API_URL}/chat',
            json={'question': question},
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()

    answer = data['answer']
    sources = data['sources']

    source_lines = []
    for source in sources:
        doc = source['document']
        preview = source['text'][:100]
        source_lines.append(f"{doc}: {preview}")

    sources_text = "\n".join(source_lines)

    return f"{answer}\nSources:\n{sources_text}"

@mcp.tool()
async def ingest_document(file_path: str) -> str:
    """Ingest a text file into the clinical RAG system."""
    path = Path(file_path)

    async with httpx.AsyncClient() as client:
        with open(path, 'rb') as f:
            upload = (path.name, f, 'text/plain')
            response = await client.post(
                f'{API_URL}/documents/ingest',
                files={'file': upload},
                timeout=60.0
            )
        response.raise_for_status()
        data = response.json()

    return f"Indexed {data['filename']} into {data['chunks']} chunks."

if __name__ == '__main__':
    mcp.run()