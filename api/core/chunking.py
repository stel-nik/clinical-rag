def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
    ) -> list[str]:
    '''
    Split text into overlapping chunks.
    
    Args:
        text: The input text to be chunked.
        chunk_size: Number of words per chunk.
        overlap: Number of words to repeat between chunks.
    
    Returns:
        List: List of text chunks.
    
    '''
    
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks